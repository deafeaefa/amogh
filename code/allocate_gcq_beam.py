#!/usr/bin/env python3
"""Pure, resumable conditional beam search for mixed-precision allocation.

The allocator deliberately knows nothing about models or metrics.  It consumes a
catalog of named candidates with exact integer byte costs and emits JSON plans of
unique candidate sets for an external evaluator (for example, a GPU grounding
job).  Scores are then recorded, possibly in partial or arbitrary order.  Once a
plan is complete, the allocator advances a width-N beam using *conditional*
marginals relative to every parent state.

Typical workflow::

    python allocate_gcq_beam.py init \
        --candidates shortlist24.json --budget-bytes 88080384 \
        --context-sha256 "$BOUND_PROTOCOL_SHA256" \
        --run-dir runs/gcq_beam --beam-width 4
    python allocate_gcq_beam.py plan --run-dir runs/gcq_beam
    # Evaluate every pending state in the emitted plan JSON on a GPU.
    python allocate_gcq_beam.py record \
        --run-dir runs/gcq_beam --results scores_round_000.json
    # Repeat plan/record until status reports "complete".

Candidate input accepts either a top-level list or
``{"candidates": [{"module_name": ..., "delta_bytes": ...}, ...]}``.  ``name``
is accepted as a generic alias.  Result input
accepts ``{"scores": {STATE_ID: SCORE}}``, a direct mapping, or
``{"results": [{"state_id": ..., "score": ...}, ...]}``.

Run artifacts are self-contained:

* ``run.json`` is the authoritative resumable state machine;
* ``cache.json`` is a stable, sorted score-cache view;
* ``trace.jsonl`` is an audit trace regenerated from authoritative events; and
* ``plan_round_*.json`` files are immutable-in-content external-evaluation plans.

No model packages are imported.  All comparisons and tie-breaks are exact and
deterministic: higher score, then fewer bytes, then lexicographic member tuple.
Pareto pruning is optional and disabled by default: with an interacting
objective, a currently dominated parent can still have the best future child.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
RUN_FILE = "run.json"
CACHE_FILE = "cache.json"
TRACE_FILE = "trace.jsonl"


class BeamAllocationError(ValueError):
    """Raised when an artifact violates the allocator contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BeamAllocationError(f"{label} must be an integer")
    if value < minimum:
        raise BeamAllocationError(f"{label} must be >= {minimum}")
    return value


def _require_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BeamAllocationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise BeamAllocationError(f"{label} must be a finite number")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise BeamAllocationError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise BeamAllocationError(f"{label} must be a SHA-256 hex digest") from exc
    return value.lower()


@dataclass(frozen=True, order=True)
class Candidate:
    """One optional promotion and its exact incremental serialized cost."""

    name: str
    delta_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise BeamAllocationError("candidate name must be a non-empty string")
        _require_plain_int(self.delta_bytes, f"delta_bytes for {self.name}", minimum=1)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "delta_bytes": self.delta_bytes}


def load_candidates(path: str | Path) -> tuple[list[Candidate], int | None]:
    """Load and validate a candidate catalog.

    Returns the sorted candidates and an optional top-level ``budget_bytes``.
    """

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "shortlist_sha256" in payload:
        recorded = _require_sha256(payload["shortlist_sha256"], "shortlist_sha256")
        unhashed = dict(payload)
        unhashed.pop("shortlist_sha256", None)
        artifact_payload = (_canonical_json(unhashed) + "\n").encode("utf-8")
        if hashlib.sha256(artifact_payload).hexdigest() != recorded:
            raise BeamAllocationError("frozen shortlist content hash mismatch")
        if payload.get("top_k") != len(payload.get("candidates", [])):
            raise BeamAllocationError("frozen shortlist top_k/count mismatch")
    embedded_budget = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        rows = payload["candidates"]
        if "budget_bytes" in payload:
            embedded_budget = _require_plain_int(
                payload["budget_bytes"], "budget_bytes"
            )
    else:
        raise BeamAllocationError(
            "candidate file must be a list or an object containing a candidates list"
        )

    candidates: list[Candidate] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BeamAllocationError(f"candidate row {index} must be an object")
        name = row.get("name", row.get("module_name"))
        if name is None or "delta_bytes" not in row:
            raise BeamAllocationError(
                f"candidate row {index} requires name/module_name and delta_bytes"
            )
        candidates.append(Candidate(name, row["delta_bytes"]))
    return _normalize_candidates(candidates), embedded_budget


def load_results(path: str | Path) -> dict[str, float]:
    """Load one external-evaluator result shard."""

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "results" in payload:
        rows = payload["results"]
        if not isinstance(rows, list):
            raise BeamAllocationError("results must be a list")
        raw: dict[str, Any] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(("state_id", "score")) - set(row):
                raise BeamAllocationError(
                    f"result row {index} requires state_id and score"
                )
            state_id = row["state_id"]
            if state_id in raw:
                raise BeamAllocationError(f"duplicate result for {state_id}")
            raw[state_id] = row["score"]
    elif isinstance(payload, dict) and "scores" in payload:
        raw = payload["scores"]
        if not isinstance(raw, dict):
            raise BeamAllocationError("scores must be an object")
    elif isinstance(payload, dict):
        raw = payload
    else:
        raise BeamAllocationError("result file must contain a JSON object")

    results: dict[str, float] = {}
    for state_id, score in raw.items():
        if not isinstance(state_id, str) or not state_id:
            raise BeamAllocationError("result state IDs must be non-empty strings")
        results[state_id] = _require_score(score, f"score for {state_id}")
    return results


def _normalize_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    result = sorted(candidates, key=lambda candidate: candidate.name)
    if not result:
        raise BeamAllocationError("at least one candidate is required")
    names = [candidate.name for candidate in result]
    if len(names) != len(set(names)):
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        raise BeamAllocationError(f"duplicate candidate names: {duplicates}")
    return result


def _state_id(members: Sequence[str]) -> str:
    canonical_members = sorted(members)
    return "state-" + _sha256_json(canonical_members)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


class BeamRun:
    """Persistent plan/record state machine for conditional beam search."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        run_path = self.run_dir / RUN_FILE
        if not run_path.is_file():
            raise BeamAllocationError(f"run does not exist: {run_path}")
        with run_path.open(encoding="utf-8") as handle:
            self.data = json.load(handle)
        self._validate_loaded_state()
        # Repair derived artifacts after an interruption between atomic writes.
        self._write_cache_artifact()
        self._write_trace_artifact()
        if self.data.get("current_plan") is not None:
            self._write_plan_artifact(self.data["current_plan"])

    @classmethod
    def initialize(
        cls,
        run_dir: str | Path,
        candidates: Iterable[Candidate],
        budget_bytes: int,
        *,
        beam_width: int = 4,
        context_sha256: str | None = None,
        pareto_prune: bool = False,
    ) -> "BeamRun":
        catalog = _normalize_candidates(candidates)
        budget_bytes = _require_plain_int(budget_bytes, "budget_bytes")
        beam_width = _require_plain_int(beam_width, "beam_width", minimum=1)
        if context_sha256 is not None:
            context_sha256 = _require_sha256(context_sha256, "context_sha256")
        if not isinstance(pareto_prune, bool):
            raise BeamAllocationError("pareto_prune must be boolean")
        target = Path(run_dir)
        target.mkdir(parents=True, exist_ok=True)
        run_path = target / RUN_FILE
        if run_path.exists():
            raise BeamAllocationError(f"refusing to overwrite existing run: {run_path}")

        catalog_rows = [candidate.as_dict() for candidate in catalog]
        catalog_hash = _sha256_json(catalog_rows)
        empty_id = _state_id(())
        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "catalog_hash": catalog_hash,
            "context_sha256": context_sha256,
            "run_fingerprint": _sha256_json(
                {
                    "catalog_hash": catalog_hash,
                    "budget_bytes": budget_bytes,
                    "beam_width": beam_width,
                    "context_sha256": context_sha256,
                    "pareto_prune": pareto_prune,
                }
            ),
            "candidates": catalog_rows,
            "budget_bytes": budget_bytes,
            "beam_width": beam_width,
            "pareto_prune": pareto_prune,
            "round_index": 0,
            "status": "active",
            "stop_reason": None,
            "beam": [empty_id],
            "archive": [],
            "final_state_id": None,
            "states": {
                empty_id: {
                    "members": [],
                    "cost_bytes": 0,
                    "score": None,
                }
            },
            "current_plan": None,
            "events": [],
        }
        instance = object.__new__(cls)
        instance.run_dir = target
        instance.data = data
        instance._event(
            "initialized",
            empty_state_id=empty_id,
            candidate_count=len(catalog),
            budget_bytes=budget_bytes,
            beam_width=beam_width,
            context_sha256=context_sha256,
            pareto_prune=pareto_prune,
        )
        instance._persist()
        return instance

    @property
    def candidates(self) -> list[Candidate]:
        return [
            Candidate(row["name"], row["delta_bytes"])
            for row in self.data["candidates"]
        ]

    @property
    def is_complete(self) -> bool:
        return self.data["status"] == "complete"

    def state(self, state_id: str) -> dict[str, Any]:
        try:
            return self.data["states"][state_id]
        except KeyError as exc:
            raise BeamAllocationError(f"unknown state: {state_id}") from exc

    def plan(self) -> dict[str, Any] | None:
        """Create or return the current unique external-evaluation plan.

        Returns ``None`` when the run is complete.  Repeated calls before all
        scores are recorded return byte-for-byte equivalent plan content.
        """

        if self.is_complete:
            return None
        if self.data["current_plan"] is not None:
            self._write_plan_artifact(self.data["current_plan"])
            return copy.deepcopy(self.data["current_plan"])

        unscored_parents = [
            state_id
            for state_id in self.data["beam"]
            if self.state(state_id)["score"] is None
        ]
        if unscored_parents:
            planned_states = []
            for state_id in sorted(unscored_parents, key=self._member_key):
                row = self.state(state_id)
                planned_states.append(
                    {
                        "state_id": state_id,
                        "members": list(row["members"]),
                        "cost_bytes": row["cost_bytes"],
                        "parents": [],
                    }
                )
            plan = self._new_plan("baseline", planned_states)
        else:
            plan = self._build_expansion_plan()
            if plan is None:
                return None

        self.data["current_plan"] = plan
        self._event(
            "plan_created",
            round_index=plan["round_index"],
            plan_kind=plan["kind"],
            plan_file=plan["artifact_name"],
            unique_state_count=len(plan["states"]),
            transition_count=sum(len(row["parents"]) for row in plan["states"]),
        )
        self._persist()
        self._write_plan_artifact(plan)
        return copy.deepcopy(plan)

    def record(self, results: Mapping[str, float]) -> dict[str, Any]:
        """Record a partial result shard and advance when the plan is complete."""

        if self.is_complete:
            if results:
                raise BeamAllocationError("cannot record results for a completed run")
            return self.status_summary()
        plan = self.data.get("current_plan")
        if plan is None:
            raise BeamAllocationError("no active plan; call plan before record")

        allowed = {row["state_id"] for row in plan["states"]}
        normalized: dict[str, float] = {}
        for state_id, raw_score in results.items():
            if state_id not in allowed:
                raise BeamAllocationError(
                    f"result {state_id} is not part of the active plan"
                )
            normalized[state_id] = _require_score(raw_score, f"score for {state_id}")

        newly_recorded: list[str] = []
        for state_id in sorted(normalized, key=self._member_key):
            score = normalized[state_id]
            prior = self.state(state_id)["score"]
            if prior is None:
                self.state(state_id)["score"] = score
                newly_recorded.append(state_id)
            elif float(prior) != score:
                raise BeamAllocationError(
                    f"conflicting score for {state_id}: cached={prior}, new={score}"
                )

        pending = self._pending_ids(plan)
        self._event(
            "results_recorded",
            round_index=plan["round_index"],
            recorded_state_ids=newly_recorded,
            pending_state_count=len(pending),
        )
        if pending:
            self._persist()
            return self.status_summary()

        self._advance_complete_plan(plan)
        self._persist()
        return self.status_summary()

    def status_summary(self) -> dict[str, Any]:
        plan = self.data.get("current_plan")
        final = None
        if self.data.get("final_state_id") is not None:
            final_id = self.data["final_state_id"]
            final = {"state_id": final_id, **copy.deepcopy(self.state(final_id))}
        return {
            "status": self.data["status"],
            "round_index": self.data["round_index"],
            "beam_width": self.data["beam_width"],
            "pareto_prune": self.data.get("pareto_prune", False),
            "context_sha256": self.data.get("context_sha256"),
            "budget_bytes": self.data["budget_bytes"],
            "beam": [self._state_projection(state_id) for state_id in self.data["beam"]],
            "active_plan": plan["artifact_name"] if plan else None,
            "pending_state_ids": self._pending_ids(plan) if plan else [],
            "cached_score_count": sum(
                row["score"] is not None for row in self.data["states"].values()
            ),
            "stop_reason": self.data["stop_reason"],
            "final": final,
        }

    def select_scored_state(self, max_bytes: int | None = None) -> dict[str, Any]:
        """Select the best archived state at an optional smaller byte cap.

        This is how the frozen B=4.25 secondary result is derived from the
        primary B=4.5 trace without launching or tuning a second search.
        Selection is only allowed after the primary run is complete.
        """
        if not self.is_complete:
            raise BeamAllocationError("state selection requires a completed run")
        cap = self.data["budget_bytes"] if max_bytes is None else _require_plain_int(
            max_bytes, "max_bytes"
        )
        if cap > self.data["budget_bytes"]:
            raise BeamAllocationError("max_bytes exceeds the frozen primary budget")
        eligible = [
            state_id
            for state_id in self.data["archive"]
            if self.state(state_id)["score"] is not None
            and self.state(state_id)["cost_bytes"] <= cap
        ]
        if not eligible:
            raise BeamAllocationError(f"no archived scored state fits {cap} bytes")
        state_id = min(eligible, key=self._rank_key)
        result = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "gcq_frozen_beam_selection",
            "run_fingerprint": self.data["run_fingerprint"],
            "catalog_hash": self.data["catalog_hash"],
            "context_sha256": self.data.get("context_sha256"),
            "primary_budget_bytes": self.data["budget_bytes"],
            "selection_cap_bytes": cap,
            "selection_rule": "higher score; fewer bytes; lexicographic members",
            "state": self._state_projection(state_id),
        }
        result["selection_sha256"] = _sha256_json(result)
        return result

    def _validate_loaded_state(self) -> None:
        if not isinstance(self.data, dict):
            raise BeamAllocationError("run.json must contain an object")
        if self.data.get("schema_version") != SCHEMA_VERSION:
            raise BeamAllocationError("unsupported run schema version")
        required = {
            "catalog_hash",
            "candidates",
            "budget_bytes",
            "beam_width",
            "round_index",
            "status",
            "beam",
            "archive",
            "states",
            "events",
        }
        missing = required - set(self.data)
        if missing:
            raise BeamAllocationError(f"run.json is missing fields: {sorted(missing)}")
        catalog = _normalize_candidates(
            Candidate(row["name"], row["delta_bytes"])
            for row in self.data["candidates"]
        )
        if [candidate.as_dict() for candidate in catalog] != self.data["candidates"]:
            raise BeamAllocationError("candidate catalog is not canonically sorted")
        if _sha256_json(self.data["candidates"]) != self.data["catalog_hash"]:
            raise BeamAllocationError("candidate catalog hash mismatch")
        _require_plain_int(self.data["budget_bytes"], "budget_bytes")
        _require_plain_int(self.data["beam_width"], "beam_width", minimum=1)
        _require_plain_int(self.data["round_index"], "round_index")
        if not isinstance(self.data.get("pareto_prune", False), bool):
            raise BeamAllocationError("pareto_prune must be boolean")
        if self.data.get("context_sha256") is not None:
            _require_sha256(self.data["context_sha256"], "context_sha256")
        if self.data["status"] not in ("active", "complete"):
            raise BeamAllocationError("invalid run status")
        candidate_cost = {candidate.name: candidate.delta_bytes for candidate in catalog}
        for state_id, row in self.data["states"].items():
            if not isinstance(row, dict) or not isinstance(row.get("members"), list):
                raise BeamAllocationError(f"invalid state row: {state_id}")
            members = row["members"]
            if members != sorted(members) or len(members) != len(set(members)):
                raise BeamAllocationError(f"state members are not canonical: {state_id}")
            if any(member not in candidate_cost for member in members):
                raise BeamAllocationError(f"state contains unknown candidate: {state_id}")
            if _state_id(members) != state_id:
                raise BeamAllocationError(f"state ID mismatch: {state_id}")
            expected_cost = sum(candidate_cost[member] for member in members)
            if row.get("cost_bytes") != expected_cost:
                raise BeamAllocationError(f"state cost mismatch: {state_id}")
            if expected_cost > self.data["budget_bytes"]:
                raise BeamAllocationError(f"state exceeds byte budget: {state_id}")
            if row.get("score") is not None:
                _require_score(row["score"], f"cached score for {state_id}")
        for field in ("beam", "archive"):
            if any(state_id not in self.data["states"] for state_id in self.data[field]):
                raise BeamAllocationError(f"{field} references an unknown state")

    def _new_plan(
        self, kind: str, planned_states: list[dict[str, Any]]
    ) -> dict[str, Any]:
        round_index = self.data["round_index"]
        artifact_name = f"plan_round_{round_index:03d}_{kind}.json"
        return {
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": self.data["run_fingerprint"],
            "catalog_hash": self.data["catalog_hash"],
            "context_sha256": self.data.get("context_sha256"),
            "round_index": round_index,
            "kind": kind,
            "objective": "maximize_score",
            "strict_positive_conditional_marginal": True,
            "budget_bytes": self.data["budget_bytes"],
            "beam_width": self.data["beam_width"],
            "pareto_prune": self.data.get("pareto_prune", False),
            "parent_beam": [
                self._state_projection(state_id) for state_id in self.data["beam"]
            ],
            "states": planned_states,
            "artifact_name": artifact_name,
        }

    def _build_expansion_plan(self) -> dict[str, Any] | None:
        candidate_cost = {
            candidate.name: candidate.delta_bytes for candidate in self.candidates
        }
        planned: dict[str, dict[str, Any]] = {}
        any_remaining = False
        for parent_id in sorted(self.data["beam"], key=self._member_key):
            parent = self.state(parent_id)
            parent_members = set(parent["members"])
            for candidate in self.candidates:
                if candidate.name in parent_members:
                    continue
                any_remaining = True
                child_members = tuple(sorted((*parent_members, candidate.name)))
                child_cost = sum(candidate_cost[name] for name in child_members)
                if child_cost > self.data["budget_bytes"]:
                    continue
                child_id = _state_id(child_members)
                if child_id not in self.data["states"]:
                    self.data["states"][child_id] = {
                        "members": list(child_members),
                        "cost_bytes": child_cost,
                        "score": None,
                    }
                entry = planned.setdefault(
                    child_id,
                    {
                        "state_id": child_id,
                        "members": list(child_members),
                        "cost_bytes": child_cost,
                        "parents": [],
                    },
                )
                entry["parents"].append(
                    {"state_id": parent_id, "added_candidate": candidate.name}
                )

        if not planned:
            reason = "candidate_exhausted" if not any_remaining else "budget_exhausted"
            self._finish(reason)
            self._persist()
            return None

        planned_states = sorted(
            planned.values(), key=lambda row: tuple(row["members"])
        )
        for row in planned_states:
            row["parents"].sort(
                key=lambda parent: (
                    self._member_key(parent["state_id"]),
                    parent["added_candidate"],
                )
            )
        return self._new_plan("expansion", planned_states)

    def _advance_complete_plan(self, plan: dict[str, Any]) -> None:
        self.data["current_plan"] = None
        if plan["kind"] == "baseline":
            baseline_ids = [row["state_id"] for row in plan["states"]]
            self.data["archive"] = self._sorted_state_ids(
                set(self.data["archive"]) | set(baseline_ids)
            )
            self.data["round_index"] += 1
            self._event(
                "baseline_scored",
                state_ids=self._sorted_state_ids(baseline_ids),
            )
            return

        eligible: set[str] = set()
        transition_rows: list[dict[str, Any]] = []
        for child_row in plan["states"]:
            child_id = child_row["state_id"]
            child_score = float(self.state(child_id)["score"])
            positive_parent_count = 0
            for transition in child_row["parents"]:
                parent_id = transition["state_id"]
                parent_score = float(self.state(parent_id)["score"])
                marginal = child_score - parent_score
                positive = marginal > 0.0
                positive_parent_count += int(positive)
                transition_rows.append(
                    {
                        "parent_state_id": parent_id,
                        "child_state_id": child_id,
                        "added_candidate": transition["added_candidate"],
                        "marginal": marginal,
                        "strictly_positive": positive,
                    }
                )
            if positive_parent_count:
                eligible.add(child_id)

        transition_rows.sort(
            key=lambda row: (
                self._member_key(row["child_state_id"]),
                self._member_key(row["parent_state_id"]),
                row["added_candidate"],
            )
        )
        if not eligible:
            self.data["round_index"] += 1
            self._event(
                "round_rejected",
                round_index=plan["round_index"],
                reason="non_positive_marginal",
                transitions=transition_rows,
            )
            self._finish("non_positive_marginal")
            return

        if self.data.get("pareto_prune", False):
            pareto, dominated_by = self._pareto_prune(eligible)
        else:
            pareto, dominated_by = self._sorted_state_ids(eligible), {}
        selected = sorted(pareto, key=self._rank_key)[: self.data["beam_width"]]
        self.data["archive"] = self._sorted_state_ids(
            set(self.data["archive"]) | eligible
        )
        self.data["beam"] = selected
        self.data["round_index"] += 1
        self._event(
            "round_advanced",
            round_index=plan["round_index"],
            eligible_state_ids=self._sorted_state_ids(eligible),
            pareto_state_ids=self._sorted_state_ids(pareto),
            dominated_by=dominated_by,
            selected_beam_state_ids=selected,
            transitions=transition_rows,
        )

    def _pareto_prune(
        self, state_ids: Iterable[str]
    ) -> tuple[list[str], dict[str, str]]:
        ordered = sorted(set(state_ids), key=self._rank_key)
        kept: list[str] = []
        dominated_by: dict[str, str] = {}
        for target_id in ordered:
            target = self.state(target_id)
            target_score = float(target["score"])
            dominators: list[str] = []
            for other_id in ordered:
                if other_id == target_id:
                    continue
                other = self.state(other_id)
                other_score = float(other["score"])
                no_more_bytes = other["cost_bytes"] <= target["cost_bytes"]
                no_worse_score = other_score >= target_score
                one_strict = (
                    other["cost_bytes"] < target["cost_bytes"]
                    or other_score > target_score
                )
                if no_more_bytes and no_worse_score and one_strict:
                    dominators.append(other_id)
            if dominators:
                dominated_by[target_id] = min(dominators, key=self._rank_key)
            else:
                kept.append(target_id)
        return kept, {
            state_id: dominated_by[state_id]
            for state_id in sorted(dominated_by, key=self._member_key)
        }

    def _finish(self, reason: str) -> None:
        if not self.data["archive"]:
            scored = [
                state_id
                for state_id, row in self.data["states"].items()
                if row["score"] is not None
            ]
            if not scored:
                raise BeamAllocationError("cannot finish before the baseline is scored")
            self.data["archive"] = self._sorted_state_ids(scored)
        final_id = min(self.data["archive"], key=self._rank_key)
        self.data["status"] = "complete"
        self.data["stop_reason"] = reason
        self.data["final_state_id"] = final_id
        self.data["current_plan"] = None
        self._event(
            "run_completed",
            stop_reason=reason,
            final_state=self._state_projection(final_id),
        )

    def _rank_key(self, state_id: str) -> tuple[Any, ...]:
        row = self.state(state_id)
        if row["score"] is None:
            raise BeamAllocationError(f"cannot rank unscored state: {state_id}")
        return (-float(row["score"]), row["cost_bytes"], tuple(row["members"]))

    def _member_key(self, state_id: str) -> tuple[str, ...]:
        return tuple(self.state(state_id)["members"])

    def _sorted_state_ids(self, state_ids: Iterable[str]) -> list[str]:
        return sorted(set(state_ids), key=self._member_key)

    def _state_projection(self, state_id: str) -> dict[str, Any]:
        return {"state_id": state_id, **copy.deepcopy(self.state(state_id))}

    def _pending_ids(self, plan: dict[str, Any] | None) -> list[str]:
        if plan is None:
            return []
        return [
            row["state_id"]
            for row in plan["states"]
            if self.state(row["state_id"])["score"] is None
        ]

    def _event(self, kind: str, **payload: Any) -> None:
        self.data["events"].append(
            {"event_index": len(self.data["events"]), "kind": kind, **payload}
        )

    def _persist(self) -> None:
        _atomic_write_json(self.run_dir / RUN_FILE, self.data)
        self._write_cache_artifact()
        self._write_trace_artifact()

    def _write_cache_artifact(self) -> None:
        entries = []
        for state_id in sorted(self.data["states"], key=self._member_key):
            entries.append(self._state_projection(state_id))
        _atomic_write_json(
            self.run_dir / CACHE_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": self.data["run_fingerprint"],
                "catalog_hash": self.data["catalog_hash"],
                "entries": entries,
            },
        )

    def _write_trace_artifact(self) -> None:
        text = "".join(_canonical_json(event) + "\n" for event in self.data["events"])
        _atomic_write_text(self.run_dir / TRACE_FILE, text)

    def _write_plan_artifact(self, plan: dict[str, Any]) -> None:
        _atomic_write_json(self.run_dir / plan["artifact_name"], plan)


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create a new allocation run")
    init.add_argument("--candidates", required=True)
    init.add_argument("--run-dir", required=True)
    init.add_argument("--budget-bytes", type=int)
    init.add_argument("--beam-width", type=int, default=4)
    init.add_argument(
        "--context-sha256",
        help="bound protocol digest copied into the run fingerprint and every plan",
    )
    init.add_argument(
        "--pareto-prune",
        action="store_true",
        help="opt in to current-score/cost Pareto pruning (unsafe for arbitrary interactions)",
    )

    plan = subparsers.add_parser("plan", help="emit or resume the next evaluation plan")
    plan.add_argument("--run-dir", required=True)

    record = subparsers.add_parser("record", help="record one result JSON shard")
    record.add_argument("--run-dir", required=True)
    record.add_argument("--results", required=True)

    status = subparsers.add_parser("status", help="show current run status")
    status.add_argument("--run-dir", required=True)

    select = subparsers.add_parser(
        "select", help="freeze the best state from a completed trace at a byte cap"
    )
    select.add_argument("--run-dir", required=True)
    select.add_argument("--max-bytes", type=int)
    select.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "init":
        candidates, embedded_budget = load_candidates(args.candidates)
        budget = args.budget_bytes
        if budget is None:
            budget = embedded_budget
        elif embedded_budget is not None and budget != embedded_budget:
            raise BeamAllocationError(
                "--budget-bytes conflicts with candidate-file budget_bytes"
            )
        if budget is None:
            raise BeamAllocationError(
                "budget is required via --budget-bytes or candidate-file budget_bytes"
            )
        run = BeamRun.initialize(
            args.run_dir,
            candidates,
            budget,
            beam_width=args.beam_width,
            context_sha256=args.context_sha256,
            pareto_prune=args.pareto_prune,
        )
        _print_json(run.status_summary())
        return 0

    run = BeamRun(args.run_dir)
    if args.command == "plan":
        plan = run.plan()
        summary = run.status_summary()
        if plan is not None:
            summary.update(
                {
                    "plan_file": str(run.run_dir / plan["artifact_name"]),
                    "plan_kind": plan["kind"],
                    "unique_state_count": len(plan["states"]),
                }
            )
        _print_json(summary)
        return 0
    if args.command == "record":
        _print_json(run.record(load_results(args.results)))
        return 0
    if args.command == "status":
        _print_json(run.status_summary())
        return 0
    if args.command == "select":
        selection = run.select_scored_state(args.max_bytes)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(selection, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _print_json({"out": str(output), **selection})
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

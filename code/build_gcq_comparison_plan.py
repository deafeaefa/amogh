#!/usr/bin/env python3
"""Build one immutable evaluation plan for frozen GCQ selections and controls.

The plan deduplicates identical projection sets while retaining every reporting
label.  It accepts the completed primary beam selection, an optional secondary
selection derived from the same trace, and the exact-cost controls manifest.
All artifacts must share the same allocator catalog and raw launch-protocol
hash.  No score is read and no model package is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ARTIFACT_KIND = "gcq_frozen_comparison_plan"


class ComparisonPlanError(ValueError):
    """Raised when frozen selections and controls cannot be joined safely."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_id_for_members(members: Sequence[str]) -> str:
    return "state-" + canonical_sha256(sorted(members))


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ComparisonPlanError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ComparisonPlanError(f"{label} must be a JSON object")
    return value


def _members(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(name, str) and name for name in value):
        raise ComparisonPlanError(f"{label} members must be non-empty strings")
    if value != sorted(value) or len(value) != len(set(value)):
        raise ComparisonPlanError(f"{label} members must be sorted and unique")
    return list(value)


def _cost(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ComparisonPlanError(f"{label} cost must be a nonnegative integer")
    return value


def validate_selection(
    value: Mapping[str, Any],
    *,
    label: str,
    context_sha256: str,
    catalog_hash: str | None = None,
    run_fingerprint: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    if value.get("schema_version") != SCHEMA_VERSION or value.get("artifact_kind") != (
        "gcq_frozen_beam_selection"
    ):
        raise ComparisonPlanError(f"{label} is not a frozen beam selection")
    stored_hash = _require_sha256(value.get("selection_sha256"), f"{label}.selection_sha256")
    unhashed = dict(value)
    unhashed.pop("selection_sha256", None)
    if canonical_sha256(unhashed) != stored_hash:
        raise ComparisonPlanError(f"{label} content hash mismatch")
    if value.get("context_sha256") != context_sha256:
        raise ComparisonPlanError(f"{label} is bound to another launch protocol")
    selected_catalog = _require_sha256(value.get("catalog_hash"), f"{label}.catalog_hash")
    selected_run = _require_sha256(value.get("run_fingerprint"), f"{label}.run_fingerprint")
    if catalog_hash is not None and selected_catalog != catalog_hash:
        raise ComparisonPlanError("primary and secondary selections use different catalogs")
    if run_fingerprint is not None and selected_run != run_fingerprint:
        raise ComparisonPlanError("primary and secondary selections use different beam runs")
    state = value.get("state")
    if not isinstance(state, dict):
        raise ComparisonPlanError(f"{label} has no selected state")
    members = _members(state.get("members"), label)
    cost = _cost(state.get("cost_bytes"), label)
    state_id = state.get("state_id")
    if state_id != state_id_for_members(members):
        raise ComparisonPlanError(f"{label} state ID does not match its members")
    cap = _cost(value.get("selection_cap_bytes"), f"{label}.selection_cap_bytes")
    primary_budget = _cost(value.get("primary_budget_bytes"), f"{label}.primary_budget_bytes")
    if cap > primary_budget:
        raise ComparisonPlanError(f"{label} selection cap exceeds the primary budget")
    if cost > cap:
        raise ComparisonPlanError(f"{label} selected state exceeds its cap")
    return {"state_id": state_id, "members": members, "cost_bytes": cost}, selected_catalog, selected_run


def _control_rows(controls: Mapping[str, Any], target: int) -> list[tuple[str, list[str], int]]:
    rows: list[tuple[str, list[str], int]] = []
    score_controls = controls.get("score_driven_controls")
    if not isinstance(score_controls, dict):
        raise ComparisonPlanError("controls manifest has no score-driven controls")
    expected = ("additive_gcq", "vqa_driven", "maba_style_additive")
    if set(score_controls) != set(expected):
        raise ComparisonPlanError("controls manifest has unexpected score-driven policies")
    for label in expected:
        row = score_controls[label]
        if not isinstance(row, dict):
            raise ComparisonPlanError(f"control {label} is not an object")
        members = _members(row.get("members"), label)
        cost = _cost(row.get("cost_bytes"), label)
        if cost != target or row.get("matches_target") is not True:
            raise ComparisonPlanError(f"control {label} does not match the primary cost")
        rows.append((label, members, cost))

    random_block = controls.get("random_controls")
    if not isinstance(random_block, dict) or random_block.get("no_best_seed_selection") is not True:
        raise ComparisonPlanError("random controls do not freeze all seeds")
    seeds = random_block.get("frozen_seeds")
    samples = random_block.get("samples")
    if not isinstance(seeds, list) or not isinstance(samples, list) or len(seeds) != len(samples):
        raise ComparisonPlanError("random-control seeds/samples are inconsistent")
    for index, (seed, row) in enumerate(zip(seeds, samples)):
        if type(seed) is not int or seed < 0 or not isinstance(row, dict) or row.get("seed") != seed:
            raise ComparisonPlanError(f"random control {index} has an invalid seed")
        members = _members(row.get("members"), f"random seed {seed}")
        cost = _cost(row.get("cost_bytes"), f"random seed {seed}")
        if cost != target or row.get("matches_target") is not True:
            raise ComparisonPlanError(f"random seed {seed} does not match the primary cost")
        rows.append((f"random_seed_{seed}", members, cost))

    coarse = controls.get("coarse_historical_control")
    if coarse is not None:
        if not isinstance(coarse, dict):
            raise ComparisonPlanError("historical coarse control is not an object")
        rows.append(
            (
                "historical_coarse",
                _members(coarse.get("members"), "historical coarse"),
                _cost(coarse.get("actual_cost_bytes"), "historical coarse"),
            )
        )
    return rows


def build_comparison_plan(
    primary_selection: Mapping[str, Any],
    controls: Mapping[str, Any],
    *,
    context_sha256: str,
    secondary_selection: Mapping[str, Any] | None = None,
    source_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate, deduplicate, and label every frozen comparison state."""
    context_hash = _require_sha256(context_sha256, "context_sha256")
    primary, catalog_hash, run_fingerprint = validate_selection(
        primary_selection, label="primary", context_sha256=context_hash
    )
    if controls.get("schema_version") != SCHEMA_VERSION or controls.get("artifact_kind") != (
        "gcq_exact_cost_matched_controls"
    ):
        raise ComparisonPlanError("controls artifact kind/schema is invalid")
    if controls.get("protocol_context_sha256") != context_hash:
        raise ComparisonPlanError("controls are bound to another launch protocol")
    if controls.get("projection_catalog_sha256") != catalog_hash:
        raise ComparisonPlanError("controls and beam selection use different catalogs")
    target = _cost(controls.get("target_delta_bytes"), "controls target")
    if target != primary["cost_bytes"]:
        raise ComparisonPlanError("matched-control target differs from primary selected cost")

    labeled: list[tuple[str, list[str], int]] = [
        ("all_w4", [], 0),
        ("gcq_primary", primary["members"], primary["cost_bytes"]),
    ]
    if secondary_selection is not None:
        secondary, _, _ = validate_selection(
            secondary_selection,
            label="secondary",
            context_sha256=context_hash,
            catalog_hash=catalog_hash,
            run_fingerprint=run_fingerprint,
        )
        if secondary["cost_bytes"] > primary["cost_bytes"]:
            raise ComparisonPlanError("secondary selected state is costlier than primary")
        labeled.append(
            ("gcq_secondary_b4_25", secondary["members"], secondary["cost_bytes"])
        )
    labeled.extend(_control_rows(controls, target))

    label_to_state_id: dict[str, str] = {}
    states_by_members: dict[tuple[str, ...], dict[str, Any]] = {}
    for label, members, cost in labeled:
        if label in label_to_state_id:
            raise ComparisonPlanError(f"duplicate comparison label {label}")
        key = tuple(members)
        state_id = state_id_for_members(members)
        incumbent = states_by_members.get(key)
        if incumbent is None:
            incumbent = {
                "state_id": state_id,
                "members": list(members),
                "cost_bytes": cost,
                "parents": [],
                "labels": [],
            }
            states_by_members[key] = incumbent
        elif incumbent["cost_bytes"] != cost:
            raise ComparisonPlanError(
                f"identical member set has conflicting costs for label {label}"
            )
        incumbent["labels"].append(label)
        label_to_state_id[label] = state_id

    states = []
    for key in sorted(states_by_members):
        row = states_by_members[key]
        row["labels"].sort()
        states.append(row)
    fingerprint_payload = {
        "run_fingerprint": run_fingerprint,
        "catalog_hash": catalog_hash,
        "context_sha256": context_hash,
        "label_to_state_id": dict(sorted(label_to_state_id.items())),
        "states": states,
    }
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "run_fingerprint": run_fingerprint,
        "catalog_hash": catalog_hash,
        "context_sha256": context_hash,
        "comparison_fingerprint": canonical_sha256(fingerprint_payload),
        "round_index": 0,
        "kind": "comparison",
        "objective": "compare_frozen_states",
        "strict_positive_conditional_marginal": False,
        "budget_bytes": _cost(
            primary_selection.get("primary_budget_bytes"), "primary budget"
        ),
        "pareto_prune": False,
        "beam_width": 4,
        "deduplicate_identical_member_sets": True,
        "label_to_state_id": dict(sorted(label_to_state_id.items())),
        "states": states,
        "evaluation_policy": {
            "manifest": "same frozen training-only decode manifest used by the beam",
            "decode_each_unique_state_once": True,
            "random_controls": "report every frozen seed and their mean; never choose the best seed",
        },
    }
    if source_sha256 is not None:
        plan["source_sha256"] = {
            name: _require_sha256(digest, f"source_sha256[{name}]")
            for name, digest in sorted(source_sha256.items())
        }
    return plan


def write_plan_exclusive(path: str | Path, value: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ComparisonPlanError(f"refusing to overwrite comparison plan: {destination}") from error
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-selection", type=Path, required=True)
    parser.add_argument("--secondary-selection", type=Path)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--protocol-context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    primary = _load_object(args.primary_selection, "primary selection")
    controls = _load_object(args.controls, "controls")
    secondary = (
        _load_object(args.secondary_selection, "secondary selection")
        if args.secondary_selection is not None
        else None
    )
    sources = {
        "primary_selection": sha256_file(args.primary_selection),
        "controls": sha256_file(args.controls),
        "protocol_context": sha256_file(args.protocol_context),
    }
    if args.secondary_selection is not None:
        sources["secondary_selection"] = sha256_file(args.secondary_selection)
    plan = build_comparison_plan(
        primary,
        controls,
        context_sha256=sources["protocol_context"],
        secondary_selection=secondary,
        source_sha256=sources,
    )
    digest = write_plan_exclusive(args.out, plan)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": digest,
                "unique_states": len(plan["states"]),
                "labels": len(plan["label_to_state_id"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

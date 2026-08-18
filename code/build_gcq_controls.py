#!/usr/bin/env python3
"""Build exact-byte matched allocation controls for the GCQ upgrade.

This module consumes the same projection catalog as ``allocate_gcq_beam.py``:
each candidate has a unique ``name`` and a strictly positive integer
``delta_bytes``.  Three score-driven controls (additive GCQ, VQA-driven, and a
MABA-style additive score) are solved as deterministic exact-target 0/1
knapsacks.  Random controls are sampled uniformly from *all* subsets whose
cost is exactly the target, using dynamic-programming completion counts and one
frozen PRNG seed per reported sample.

The CLI publishes exactly one manifest with exclusive, atomic create semantics;
it refuses to replace an existing result.  All inputs and the required protocol
or context file are SHA-256 bound into that manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from allocate_gcq_beam import Candidate, load_candidates


SCHEMA_VERSION = 1
POLICIES = ("additive_gcq", "vqa_driven", "maba_style_additive")


class ControlBuildError(ValueError):
    """Raised when exact-cost controls cannot be built safely."""


class InfeasibleTargetError(ControlBuildError):
    """Raised when no candidate subset has exactly the requested cost."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(_canonical_json(value).encode("utf-8"))


def _require_plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlBuildError(f"{label} must be an integer")
    if value < minimum:
        raise ControlBuildError(f"{label} must be >= {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ControlBuildError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _require_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlBuildError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ControlBuildError(f"{label} must be a finite number")
    return result


def _normalized_candidates(candidates: Iterable[Candidate]) -> tuple[Candidate, ...]:
    result = tuple(sorted(candidates, key=lambda candidate: candidate.name))
    if not result:
        raise ControlBuildError("at least one candidate is required")
    names = [candidate.name for candidate in result]
    if len(names) != len(set(names)):
        raise ControlBuildError("candidate names must be unique")
    for candidate in result:
        _require_plain_int(
            candidate.delta_bytes,
            f"delta_bytes for {candidate.name}",
            minimum=1,
        )
    return result


def load_score_map(path: str | Path) -> dict[str, float]:
    """Load a JSON score map.

    Accepted forms are a direct ``{name: score}`` mapping,
    ``{"scores": {name: score}}``, or
    ``{"scores": [{"name": ..., "score": ...}, ...]}``.
    """

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    raw: Any
    if isinstance(payload, dict) and "scores" in payload:
        raw = payload["scores"]
    elif isinstance(payload, dict) and "candidates" in payload:
        raw = payload["candidates"]
    else:
        raw = payload

    if isinstance(raw, list):
        mapping: dict[str, Any] = {}
        for index, row in enumerate(raw):
            if not isinstance(row, dict):
                raise ControlBuildError(f"score row {index} must be an object")
            name = row.get("name", row.get("module_name"))
            score = row.get("score", row.get("repair_macro"))
            if name is None or score is None:
                raise ControlBuildError(
                    f"score row {index} requires name/module_name and score/repair_macro"
                )
            if not isinstance(name, str) or not name:
                raise ControlBuildError(f"score row {index} has invalid name")
            if name in mapping:
                raise ControlBuildError(f"duplicate score for {name}")
            mapping[name] = score
        raw = mapping
    if not isinstance(raw, dict):
        raise ControlBuildError("score file must contain an object or score-row list")

    result: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ControlBuildError("score names must be non-empty strings")
        result[name] = _require_score(value, f"score for {name}")
    return result


def load_fixed_members(
    path: str | Path, *, candidate_names: Sequence[str] | None = None
) -> list[str]:
    """Load exact members or expand a historical substring promotion file."""

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if "members" in payload:
            payload = payload["members"]
        elif "projections" in payload:
            payload = payload["projections"]
        elif "substrings" in payload:
            if candidate_names is None:
                raise ControlBuildError(
                    "candidate_names are required to expand coarse substrings"
                )
            substrings = payload["substrings"]
            if (
                not isinstance(substrings, list)
                or not substrings
                or not all(isinstance(value, str) and value for value in substrings)
            ):
                raise ControlBuildError("coarse substrings must be non-empty strings")
            if len(substrings) != len(set(substrings)):
                raise ControlBuildError("coarse control contains duplicate substrings")
            unmatched = [
                substring
                for substring in substrings
                if not any(substring in name for name in candidate_names)
            ]
            if unmatched:
                raise ControlBuildError(
                    f"coarse substrings match no projection: {unmatched}"
                )
            payload = [
                name
                for name in candidate_names
                if any(substring in name for substring in substrings)
            ]
        else:
            raise ControlBuildError(
                "coarse control object requires members, projections, or substrings"
            )
    if not isinstance(payload, list):
        raise ControlBuildError("coarse control must be a JSON list")
    members: list[str] = []
    for index, name in enumerate(payload):
        if not isinstance(name, str) or not name:
            raise ControlBuildError(f"coarse member {index} must be a string")
        members.append(name)
    if len(members) != len(set(members)):
        raise ControlBuildError("coarse control contains duplicate members")
    return sorted(members)


def validate_score_map(
    candidates: Sequence[Candidate], scores: Mapping[str, float], label: str
) -> dict[str, float]:
    candidate_names = {candidate.name for candidate in candidates}
    score_names = set(scores)
    missing = sorted(candidate_names - score_names)
    extra = sorted(score_names - candidate_names)
    if missing or extra:
        raise ControlBuildError(
            f"{label} score-map/catalog mismatch; missing={missing}, extra={extra}"
        )
    return {
        candidate.name: _require_score(scores[candidate.name], f"{label}:{candidate.name}")
        for candidate in candidates
    }


@dataclass(frozen=True)
class KnapsackSolution:
    members: tuple[str, ...]
    cost_bytes: int
    objective: Decimal

    def as_dict(self) -> dict[str, Any]:
        return {
            "members": list(self.members),
            "cost_bytes": self.cost_bytes,
            "objective_score": float(self.objective),
            "objective_score_decimal": str(self.objective),
        }


def solve_exact_knapsack(
    candidates: Iterable[Candidate],
    scores: Mapping[str, float],
    target_bytes: int,
) -> KnapsackSolution:
    """Maximize additive score subject to cost being exactly ``target_bytes``.

    Ties are resolved by the lexicographically smallest sorted member tuple.
    Decimal values created from the canonical float strings make equality and
    tie behavior independent of binary summation order.
    """

    catalog = _normalized_candidates(candidates)
    target = _require_plain_int(target_bytes, "target_bytes")
    score_map = validate_score_map(catalog, scores, "knapsack")
    decimal_scores = {
        name: Decimal(str(value)) for name, value in score_map.items()
    }

    # cost -> (objective, canonical member tuple)
    dynamic: dict[int, tuple[Decimal, tuple[str, ...]]] = {
        0: (Decimal(0), ())
    }
    for candidate in catalog:
        previous = list(dynamic.items())
        updated = dict(dynamic)
        for prior_cost, (prior_score, prior_members) in previous:
            new_cost = prior_cost + candidate.delta_bytes
            if new_cost > target:
                continue
            proposal = (
                prior_score + decimal_scores[candidate.name],
                prior_members + (candidate.name,),
            )
            incumbent = updated.get(new_cost)
            if (
                incumbent is None
                or proposal[0] > incumbent[0]
                or (proposal[0] == incumbent[0] and proposal[1] < incumbent[1])
            ):
                updated[new_cost] = proposal
        dynamic = updated

    if target not in dynamic:
        raise InfeasibleTargetError(
            f"no candidate subset has exact cost {target} bytes"
        )
    objective, members = dynamic[target]
    return KnapsackSolution(members, target, objective)


class ExactSubsetCounter:
    """Count and uniformly unrank all exact-cost feasible subsets."""

    def __init__(self, candidates: Iterable[Candidate], target_bytes: int):
        self.candidates = _normalized_candidates(candidates)
        self.target_bytes = _require_plain_int(target_bytes, "target_bytes")
        count = len(self.candidates)
        self.suffix_counts: list[dict[int, int]] = [{} for _ in range(count + 1)]
        self.suffix_counts[count] = {0: 1}
        for index in range(count - 1, -1, -1):
            candidate = self.candidates[index]
            following = self.suffix_counts[index + 1]
            current = dict(following)  # Exclude this candidate.
            for suffix_cost, completions in following.items():
                cost = suffix_cost + candidate.delta_bytes
                if cost <= self.target_bytes:
                    current[cost] = current.get(cost, 0) + completions
            self.suffix_counts[index] = current
        self.support_size = self.suffix_counts[0].get(self.target_bytes, 0)
        if self.support_size == 0:
            raise InfeasibleTargetError(
                f"no candidate subset has exact cost {self.target_bytes} bytes"
            )

    def unrank(self, rank: int) -> tuple[str, ...]:
        """Return the unique feasible subset associated with ``rank``.

        At each candidate, feasible inclusion completions form the first block
        and exclusion completions the second.  This is a bijection from
        ``range(support_size)`` to the exact-cost support.
        """

        rank = _require_plain_int(rank, "rank")
        if rank >= self.support_size:
            raise ControlBuildError(
                f"rank {rank} is outside support of size {self.support_size}"
            )
        remaining = self.target_bytes
        selected: list[str] = []
        for index, candidate in enumerate(self.candidates):
            after = self.suffix_counts[index + 1]
            include_remaining = remaining - candidate.delta_bytes
            include_count = (
                after.get(include_remaining, 0) if include_remaining >= 0 else 0
            )
            if rank < include_count:
                selected.append(candidate.name)
                remaining = include_remaining
            else:
                rank -= include_count
        if remaining != 0 or rank != 0:
            raise AssertionError("exact-subset unranking invariant failed")
        return tuple(selected)

    def sample(self, seed: int) -> tuple[int, tuple[str, ...]]:
        seed = _require_plain_int(seed, "random seed", minimum=0)
        sample_rank = random.Random(seed).randrange(self.support_size)
        return sample_rank, self.unrank(sample_rank)


def subset_cost(candidates: Iterable[Candidate], members: Iterable[str]) -> int:
    catalog = _normalized_candidates(candidates)
    cost_by_name = {candidate.name: candidate.delta_bytes for candidate in catalog}
    selected = list(members)
    if len(selected) != len(set(selected)):
        raise ControlBuildError("subset contains duplicate members")
    unknown = sorted(set(selected) - set(cost_by_name))
    if unknown:
        raise ControlBuildError(f"subset contains unknown members: {unknown}")
    return sum(cost_by_name[name] for name in selected)


def build_controls_manifest(
    candidates: Iterable[Candidate],
    target_bytes: int,
    score_maps: Mapping[str, Mapping[str, float]],
    random_seeds: Sequence[int],
    context_sha256: str,
    *,
    source_sha256: Mapping[str, str] | None = None,
    coarse_members: Sequence[str] | None = None,
    coarse_candidates: Iterable[Candidate] | None = None,
) -> dict[str, Any]:
    """Build and validate the complete matched-control manifest in memory."""

    catalog = _normalized_candidates(candidates)
    target = _require_plain_int(target_bytes, "target_bytes")
    context_digest = _require_sha256(context_sha256, "context_sha256")
    if set(score_maps) != set(POLICIES):
        raise ControlBuildError(
            f"score_maps must contain exactly {list(POLICIES)}"
        )
    seeds = [
        _require_plain_int(seed, f"random_seeds[{index}]", minimum=0)
        for index, seed in enumerate(random_seeds)
    ]
    if not seeds:
        raise ControlBuildError("at least one frozen random seed is required")
    if len(seeds) != len(set(seeds)):
        raise ControlBuildError("random seeds must be unique")

    normalized_scores = {
        policy: validate_score_map(catalog, score_maps[policy], policy)
        for policy in POLICIES
    }
    policies: dict[str, Any] = {}
    for policy in POLICIES:
        solution = solve_exact_knapsack(catalog, normalized_scores[policy], target)
        row = solution.as_dict()
        row.update(
            {
                "policy": policy,
                "solver": "deterministic_exact_target_0_1_knapsack",
                "tie_break": "max_score_then_lexicographically_smallest_members",
                "score_map_sha256": sha256_json(normalized_scores[policy]),
                "matches_target": solution.cost_bytes == target,
            }
        )
        policies[policy] = row

    counter = ExactSubsetCounter(catalog, target)
    random_samples: list[dict[str, Any]] = []
    for seed in seeds:
        rank, members = counter.sample(seed)
        cost = subset_cost(catalog, members)
        random_samples.append(
            {
                "seed": seed,
                "sample_rank": rank,
                "members": list(members),
                "cost_bytes": cost,
                "matches_target": cost == target,
            }
        )

    catalog_rows = [candidate.as_dict() for candidate in catalog]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "gcq_exact_cost_matched_controls",
        "protocol_context_sha256": context_digest,
        "projection_catalog_sha256": sha256_json(catalog_rows),
        "candidate_count": len(catalog),
        "target_delta_bytes": target,
        "score_driven_controls": policies,
        "random_controls": {
            "sampling_method": "uniform_exact_cost_subset_by_dp_completion_count_unranking",
            "support_size": counter.support_size,
            "frozen_seeds": list(seeds),
            "no_best_seed_selection": True,
            "samples": random_samples,
        },
        "validation": {
            "all_score_driven_costs_equal_target": all(
                row["cost_bytes"] == target for row in policies.values()
            ),
            "all_random_costs_equal_target": all(
                row["cost_bytes"] == target for row in random_samples
            ),
            "all_random_seeds_preserved_in_input_order": [
                row["seed"] for row in random_samples
            ]
            == seeds,
        },
    }
    if source_sha256 is not None:
        manifest["source_sha256"] = {
            name: _require_sha256(digest, f"source_sha256[{name}]")
            for name, digest in sorted(source_sha256.items())
        }
    if coarse_members is not None:
        resolved = sorted(coarse_members)
        coarse_catalog = (
            _normalized_candidates(coarse_candidates)
            if coarse_candidates is not None
            else catalog
        )
        actual_cost = subset_cost(coarse_catalog, resolved)
        manifest["coarse_historical_control"] = {
            "policy": "supplied_fixed_members_no_force_matching",
            "members": resolved,
            "actual_cost_bytes": actual_cost,
            "target_delta_bytes": target,
            "matches_target": actual_cost == target,
            "projection_catalog_sha256": sha256_json(
                [candidate.as_dict() for candidate in coarse_catalog]
            ),
        }

    checks = manifest["validation"]
    if not all(checks.values()):
        raise AssertionError(f"matched-control validation failed: {checks}")
    return manifest


def write_manifest_exclusive(path: str | Path, manifest: Mapping[str, Any]) -> str:
    """Publish one JSON manifest atomically and without overwrite.

    A fully fsynced temporary file is hard-linked into place.  The link is an
    atomic create operation and fails if the destination already exists.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{random.SystemRandom().randrange(1 << 63)}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ControlBuildError(
                f"refusing to overwrite existing manifest: {destination}"
            ) from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return sha256_bytes(text.encode("utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--target-bytes", required=True, type=int)
    parser.add_argument("--gcq-scores", required=True)
    parser.add_argument("--vqa-scores", required=True)
    parser.add_argument("--maba-scores", required=True)
    parser.add_argument("--random-seeds", required=True, nargs="+", type=int)
    parser.add_argument(
        "--protocol-context",
        required=True,
        help="frozen protocol/context artifact whose raw SHA-256 is embedded",
    )
    parser.add_argument("--coarse-members", default="")
    parser.add_argument(
        "--coarse-catalog",
        default="",
        help="optional full 196-projection catalog for a historical coarse state",
    )
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    candidates, embedded_budget = load_candidates(args.candidates)
    if embedded_budget is not None and embedded_budget != args.target_bytes:
        raise ControlBuildError(
            "candidate-file budget_bytes conflicts with --target-bytes"
        )
    score_paths = {
        "additive_gcq": args.gcq_scores,
        "vqa_driven": args.vqa_scores,
        "maba_style_additive": args.maba_scores,
    }
    score_maps = {
        policy: load_score_map(path) for policy, path in score_paths.items()
    }
    context_digest = sha256_file(args.protocol_context)
    source_hashes = {
        "projection_catalog": sha256_file(args.candidates),
        "protocol_context": context_digest,
        **{
            f"{policy}_scores": sha256_file(path)
            for policy, path in score_paths.items()
        },
    }
    coarse_members = None
    coarse_candidates = None
    if args.coarse_members:
        if args.coarse_catalog:
            coarse_candidates, _ = load_candidates(args.coarse_catalog)
            source_hashes["coarse_catalog"] = sha256_file(args.coarse_catalog)
        else:
            coarse_candidates = candidates
        coarse_members = load_fixed_members(
            args.coarse_members,
            candidate_names=[candidate.name for candidate in coarse_candidates],
        )
        source_hashes["coarse_members"] = sha256_file(args.coarse_members)

    manifest = build_controls_manifest(
        candidates,
        args.target_bytes,
        score_maps,
        args.random_seeds,
        context_digest,
        source_sha256=source_hashes,
        coarse_members=coarse_members,
        coarse_candidates=coarse_candidates,
    )
    output_digest = write_manifest_exclusive(args.out, manifest)
    print(
        json.dumps(
            {
                "manifest": str(Path(args.out)),
                "manifest_sha256": output_digest,
                "target_delta_bytes": args.target_bytes,
                "random_seeds": args.random_seeds,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

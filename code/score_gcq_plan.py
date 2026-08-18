"""Turn strict per-state decoded JSONL files into allocator score input.

This CPU-only bridge validates that every state in one immutable beam plan has
exactly one result log, that every row is bound to the expected training-only
decode manifest and allocation state, and then emits the absolute eight-cell
macro mean GIoU consumed by ``allocate_gcq_beam.py record``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from gcq_profile_metrics import decoded_macro_summary


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def manifest_records(value: object) -> list[dict]:
    if isinstance(value, dict):
        value = value.get("records", value.get("examples"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("decode manifest must be a JSON record list")
    if not value:
        raise ValueError("decode manifest is empty")
    return value


def validate_exact_manifest_order(
    records: Sequence[Mapping[str, object]],
    expected: Sequence[Mapping[str, object]],
) -> None:
    if len(records) != len(expected):
        raise ValueError(
            f"decoded result length differs from manifest: {len(records)} != {len(expected)}"
        )
    identity_fields = ("uid", "image_id", "task", "area_quartile")
    for index, (actual, source) in enumerate(zip(records, expected)):
        for field in identity_fields:
            if actual.get(field) != source.get(field):
                raise ValueError(
                    f"decoded result/manifest {field} mismatch at row {index}: "
                    f"{actual.get(field)!r} != {source.get(field)!r}"
                )


def score_plan(
    plan: dict,
    results: dict[str, list[dict]],
    *,
    manifest_sha256: str,
    context_sha256: str,
    expected_manifest_records: Sequence[Mapping[str, object]] | None = None,
) -> dict:
    if plan.get("context_sha256") != context_sha256:
        raise ValueError("beam plan does not match the bound protocol context")
    plan_rows = plan.get("states")
    if not isinstance(plan_rows, list):
        raise ValueError("beam plan has no states list")
    state_ids = [str(row["state_id"]) for row in plan_rows]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("beam plan contains duplicate state IDs")
    if set(results) != set(state_ids):
        raise ValueError(
            f"result set differs from plan; missing={sorted(set(state_ids)-set(results))}, "
            f"extra={sorted(set(results)-set(state_ids))}"
        )
    summaries = {}
    for state_id in state_ids:
        if expected_manifest_records is not None:
            validate_exact_manifest_order(results[state_id], expected_manifest_records)
        summaries[state_id] = decoded_macro_summary(
            results[state_id],
            expected_manifest_sha256=manifest_sha256,
            expected_state_id=state_id,
        )
    return {
        "schema_version": 1,
        "run_fingerprint": plan.get("run_fingerprint"),
        "round_index": plan.get("round_index"),
        "context_sha256": context_sha256,
        "decode_manifest_sha256": manifest_sha256,
        "scores": {state_id: summaries[state_id]["mean_giou_macro"] for state_id in state_ids},
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    result_group = parser.add_mutually_exclusive_group(required=True)
    result_group.add_argument(
        "--result", action="append", default=[], metavar="STATE_ID=JSONL",
        help="repeat once per plan state",
    )
    result_group.add_argument(
        "--results-dir", type=Path,
        help="directory written by eval_gcq_plan.py (STATE_ID.decoded.jsonl)",
    )
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="frozen decode manifest; its file hash and exact row order are enforced",
    )
    parser.add_argument("--context-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with open(args.plan) as handle:
        plan = json.load(handle)
    with open(args.manifest) as handle:
        expected_manifest_records = manifest_records(json.load(handle))
    manifest_sha256 = sha256_file(args.manifest)
    results = {}
    result_hashes = {}
    specifications = list(args.result)
    if args.results_dir is not None:
        plan_rows = plan.get("states")
        if not isinstance(plan_rows, list):
            raise ValueError("beam plan has no states list")
        specifications = [
            f"{row['state_id']}={args.results_dir / (str(row['state_id']) + '.decoded.jsonl')}"
            for row in plan_rows
        ]
    for spec in specifications:
        if "=" not in spec:
            raise ValueError(f"--result must be STATE_ID=JSONL, got {spec!r}")
        state_id, path = spec.split("=", 1)
        if state_id in results:
            raise ValueError(f"duplicate --result for {state_id}")
        results[state_id] = read_jsonl(path)
        result_hashes[state_id] = sha256_file(path)
    output = score_plan(
        plan,
        results,
        manifest_sha256=manifest_sha256,
        context_sha256=args.context_sha256,
        expected_manifest_records=expected_manifest_records,
    )
    output["plan_sha256"] = sha256_file(args.plan)
    output["result_sha256"] = dict(sorted(result_hashes.items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"out": str(args.out), "states": len(results)}, indent=2))


if __name__ == "__main__":
    main()

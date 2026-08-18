"""Pure scoring utilities for the projection-level GCQ upgrade.

This module deliberately contains no model code.  GPU workers may emit raw
teacher-forced coordinate KL records and decoded REC JSONL files, while these
functions perform the outcome-independent aggregation, fixed-shortlist
selection, and paired decoded reranking on CPU.

The aggregation order is important:

1. average token KL within each of the four numeric coordinates;
2. average the four coordinates within a row;
3. average rows within each task x relative-area-quartile cell; and
4. macro-average the eight cells.

That prevents a coordinate split into more BPE tokens, or a more populous
stratum, from receiving extra weight.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TASKS = ("rec", "coco_grounding")
QUARTILES = (1, 2, 3, 4)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _cell(record: Mapping[str, object]) -> tuple[str, int]:
    task = str(record["task"])
    quartile = int(record["area_quartile"])
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    if quartile not in QUARTILES:
        raise ValueError(f"invalid area_quartile {quartile}; expected 1..4")
    return task, quartile


def coordinate_row_kl(coordinates: Sequence[Sequence[float]]) -> float:
    """Return a four-coordinate mean after first averaging each token group."""
    if len(coordinates) != 4:
        raise ValueError(f"expected exactly four coordinate groups, got {len(coordinates)}")
    coordinate_means = []
    for index, tokens in enumerate(coordinates):
        if not tokens:
            raise ValueError(f"coordinate {index} has no token KL values")
        values = [_finite(value, f"coordinate {index} token KL") for value in tokens]
        coordinate_means.append(sum(values) / len(values))
    return sum(coordinate_means) / 4.0


def aggregate_coordinate_candidate(
    records: Iterable[Mapping[str, object]],
    *,
    require_all_cells: bool = True,
) -> dict:
    """Aggregate raw W4/W8 coordinate-token KL records for one projection.

    Every record needs ``uid``, ``task``, ``area_quartile``,
    ``w4_coordinate_token_kl`` and ``w8_coordinate_token_kl``.  The two KL
    fields are lists of four lists: one inner list per numeric coordinate.
    """
    rows = list(records)
    if not rows:
        raise ValueError("candidate has no profiling records")
    seen: set[str] = set()
    cell_rows: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for record in rows:
        uid = str(record["uid"])
        if uid in seen:
            raise ValueError(f"duplicate profiling uid {uid!r}")
        seen.add(uid)
        w4 = coordinate_row_kl(record["w4_coordinate_token_kl"])  # type: ignore[arg-type]
        w8 = coordinate_row_kl(record["w8_coordinate_token_kl"])  # type: ignore[arg-type]
        cell_rows[_cell(record)].append((w4, w8))

    expected = {(task, q) for task in TASKS for q in QUARTILES}
    missing = sorted(expected - set(cell_rows))
    if require_all_cells and missing:
        raise ValueError(f"profiling records are missing task/quartile cells: {missing}")

    cells = {}
    for key in sorted(cell_rows):
        pairs = cell_rows[key]
        w4 = sum(pair[0] for pair in pairs) / len(pairs)
        w8 = sum(pair[1] for pair in pairs) / len(pairs)
        cells[f"{key[0]}:q{key[1]}"] = {
            "n": len(pairs),
            "kl_w4": w4,
            "kl_w8": w8,
            "repair": w4 - w8,
        }
    macro_w4 = sum(cell["kl_w4"] for cell in cells.values()) / len(cells)
    macro_w8 = sum(cell["kl_w8"] for cell in cells.values()) / len(cells)
    return {
        "n": len(rows),
        "n_cells": len(cells),
        "kl_w4_macro": macro_w4,
        "kl_w8_macro": macro_w8,
        "repair_macro": macro_w4 - macro_w8,
        "cells": cells,
    }


def build_shortlist(
    candidates: Iterable[Mapping[str, object]],
    *,
    top_k: int = 24,
    source_hashes: Mapping[str, str] | None = None,
) -> dict:
    """Freeze the top positive coordinate-KL repairs per exact added byte."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    normalized = []
    seen: set[str] = set()
    for candidate in candidates:
        name = str(candidate["module_name"])
        if name in seen:
            raise ValueError(f"duplicate candidate {name!r}")
        seen.add(name)
        repair = _finite(candidate["repair_macro"], f"{name} repair_macro")
        delta_bytes = int(candidate["delta_bytes"])
        if delta_bytes <= 0:
            raise ValueError(f"{name} delta_bytes must be positive, got {delta_bytes}")
        if repair <= 0:
            continue
        normalized.append({
            "module_name": name,
            "repair_macro": repair,
            "delta_bytes": delta_bytes,
            "repair_per_byte": repair / delta_bytes,
            "w4_sha256": candidate.get("w4_sha256"),
            "w8_sha256": candidate.get("w8_sha256"),
        })
    normalized.sort(
        key=lambda row: (-row["repair_per_byte"], -row["repair_macro"], row["module_name"])
    )
    if len(normalized) < top_k:
        raise ValueError(
            f"only {len(normalized)} candidates have positive repair; cannot freeze top-{top_k}"
        )
    selected = normalized[:top_k]
    payload = {
        "schema_version": 1,
        "selection_rule": "positive repair_macro / exact delta_bytes; descending; repair; module_name",
        "top_k": top_k,
        "source_hashes": dict(sorted((source_hashes or {}).items())),
        "candidates": selected,
    }
    payload["shortlist_sha256"] = canonical_sha256(payload)
    return payload


def _read_jsonl(path: str | Path) -> list[dict]:
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


def _decoded_value(record: Mapping[str, object], key: str, default: float) -> float:
    return _finite(record.get(key, default), key)


def decoded_macro_summary(
    records: Sequence[Mapping[str, object]],
    *,
    expected_manifest_sha256: str | None = None,
    expected_state_id: str | None = None,
) -> dict:
    """Compute absolute decoded metrics with equal task/quartile cell weight."""
    if not records:
        raise ValueError("decoded result is empty")
    seen: set[str] = set()
    cells_raw: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
    for record in records:
        uid = str(record["uid"])
        if uid in seen:
            raise ValueError(f"duplicate decoded uid {uid!r}")
        seen.add(uid)
        if expected_manifest_sha256 and record.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError(f"manifest hash mismatch for {uid}")
        if expected_state_id and record.get("allocation_state_id") != expected_state_id:
            raise ValueError(f"allocation state mismatch for {uid}")
        parse_failed = float(bool(record.get("parse_failed", record.get("box1000") is None)))
        cells_raw[_cell(record)].append((
            _decoded_value(record, "giou", -1.0),
            parse_failed,
            _decoded_value(record, "precise_iou", 0.0),
        ))
    expected = {(task, q) for task in TASKS for q in QUARTILES}
    missing = sorted(expected - set(cells_raw))
    if missing:
        raise ValueError(f"decoded records are missing task/quartile cells: {missing}")
    cells = {}
    for key in sorted(cells_raw):
        values = cells_raw[key]
        cells[f"{key[0]}:q{key[1]}"] = {
            "n": len(values),
            "mean_giou": sum(v[0] for v in values) / len(values),
            "parse_fail": sum(v[1] for v in values) / len(values),
            "precise_iou": sum(v[2] for v in values) / len(values),
        }
    return {
        "n": len(records),
        "mean_giou_macro": sum(v["mean_giou"] for v in cells.values()) / len(cells),
        "parse_fail_macro": sum(v["parse_fail"] for v in cells.values()) / len(cells),
        "precise_iou_macro": sum(v["precise_iou"] for v in cells.values()) / len(cells),
        "cells": cells,
    }


def paired_decoded_summary(
    baseline: Sequence[Mapping[str, object]],
    promoted: Sequence[Mapping[str, object]],
    *,
    expected_manifest_sha256: str | None = None,
) -> dict:
    """Strict paired, eight-cell-macro decoded effect for one promotion."""
    if len(baseline) != len(promoted):
        raise ValueError(f"paired decoded length mismatch: {len(baseline)} != {len(promoted)}")
    if not baseline:
        raise ValueError("decoded result is empty")
    cell_deltas: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
    seen: set[str] = set()
    for index, (base, candidate) in enumerate(zip(baseline, promoted)):
        base_uid = str(base["uid"])
        candidate_uid = str(candidate["uid"])
        if base_uid != candidate_uid:
            raise ValueError(
                f"paired decoded UID/order mismatch at row {index}: {base_uid!r} != {candidate_uid!r}"
            )
        if base_uid in seen:
            raise ValueError(f"duplicate decoded uid {base_uid!r}")
        seen.add(base_uid)
        if base.get("image_id") != candidate.get("image_id"):
            raise ValueError(f"paired image_id mismatch for {base_uid}")
        if _cell(base) != _cell(candidate):
            raise ValueError(f"paired task/quartile mismatch for {base_uid}")
        for record in (base, candidate):
            manifest_hash = record.get("manifest_sha256")
            if expected_manifest_sha256 and manifest_hash != expected_manifest_sha256:
                raise ValueError(
                    f"manifest hash mismatch for {base_uid}: {manifest_hash!r} != "
                    f"{expected_manifest_sha256!r}"
                )
        giou_delta = _decoded_value(candidate, "giou", -1.0) - _decoded_value(base, "giou", -1.0)
        parse_delta = float(bool(candidate.get("parse_failed", candidate.get("box1000") is None))) - float(
            bool(base.get("parse_failed", base.get("box1000") is None))
        )
        precise_delta = _decoded_value(candidate, "precise_iou", 0.0) - _decoded_value(
            base, "precise_iou", 0.0
        )
        cell_deltas[_cell(base)].append((giou_delta, parse_delta, precise_delta))

    expected = {(task, q) for task in TASKS for q in QUARTILES}
    missing = sorted(expected - set(cell_deltas))
    if missing:
        raise ValueError(f"decoded records are missing task/quartile cells: {missing}")
    cells = {}
    for key in sorted(cell_deltas):
        values = cell_deltas[key]
        cells[f"{key[0]}:q{key[1]}"] = {
            "n": len(values),
            "giou_delta": sum(v[0] for v in values) / len(values),
            "parse_fail_delta": sum(v[1] for v in values) / len(values),
            "precise_iou_delta": sum(v[2] for v in values) / len(values),
        }
    return {
        "n": len(baseline),
        "giou_delta_macro": sum(v["giou_delta"] for v in cells.values()) / len(cells),
        "parse_fail_delta_macro": sum(v["parse_fail_delta"] for v in cells.values()) / len(cells),
        "precise_iou_delta_macro": sum(v["precise_iou_delta"] for v in cells.values()) / len(cells),
        "cells": cells,
    }


def rerank_shortlist(
    shortlist: Mapping[str, object],
    baseline: Sequence[Mapping[str, object]],
    promoted_by_name: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    expected_manifest_sha256: str | None = None,
) -> dict:
    """Rerank exactly the frozen shortlist; membership cannot change here."""
    entries = list(shortlist["candidates"])  # type: ignore[arg-type]
    names = [str(entry["module_name"]) for entry in entries]
    if set(promoted_by_name) != set(names):
        missing = sorted(set(names) - set(promoted_by_name))
        extra = sorted(set(promoted_by_name) - set(names))
        raise ValueError(f"decoded candidate set differs from frozen shortlist; missing={missing}, extra={extra}")
    proxy = {str(entry["module_name"]): float(entry["repair_per_byte"]) for entry in entries}
    rows = []
    for name in names:
        row = {"module_name": name, **paired_decoded_summary(
            baseline,
            promoted_by_name[name],
            expected_manifest_sha256=expected_manifest_sha256,
        )}
        row["proxy_repair_per_byte"] = proxy[name]
        rows.append(row)
    rows.sort(key=lambda row: (
        -row["giou_delta_macro"],
        row["parse_fail_delta_macro"],
        -row["precise_iou_delta_macro"],
        -row["proxy_repair_per_byte"],
        row["module_name"],
    ))
    payload = {
        "schema_version": 1,
        "shortlist_sha256": shortlist.get("shortlist_sha256"),
        "decode_manifest_sha256": expected_manifest_sha256,
        "ranking_rule": "paired macro GIoU; parse-fail; precise-IoU; proxy; module_name",
        "candidates": rows,
    }
    payload["rerank_sha256"] = canonical_sha256(payload)
    return payload


def _load_candidate_summaries(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    with open(path) as handle:
        value = json.load(handle)
    return value["candidates"] if isinstance(value, dict) else value


def _write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as handle:
        handle.write(canonical_json_bytes(value).decode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    shortlist_parser = sub.add_parser("shortlist", help="freeze a top-k proxy shortlist")
    shortlist_parser.add_argument(
        "--summaries", type=Path, action="append", required=True,
        help="repeat for disjoint profiler slices; all expected candidates are required",
    )
    shortlist_parser.add_argument("--top-k", type=int, default=24)
    shortlist_parser.add_argument("--expected-candidates", type=int, default=196)
    shortlist_parser.add_argument("--out", type=Path, required=True)

    rerank_parser = sub.add_parser("rerank", help="paired decoded rerank of the frozen shortlist")
    rerank_parser.add_argument("--shortlist", type=Path, required=True)
    rerank_parser.add_argument("--baseline", type=Path, required=True)
    rerank_parser.add_argument(
        "--candidate", action="append", default=[], metavar="MODULE=JSONL",
        help="repeat exactly once per frozen shortlist module",
    )
    rerank_parser.add_argument("--manifest-sha256")
    rerank_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "shortlist":
        summaries = [
            row
            for path in args.summaries
            for row in _load_candidate_summaries(path)
        ]
        if len(summaries) != args.expected_candidates:
            raise ValueError(
                f"received {len(summaries)} candidate summaries; "
                f"expected {args.expected_candidates}"
            )
        source_hashes = {
            f"summaries_{index:03d}": sha256_file(path)
            for index, path in enumerate(args.summaries)
        }
        result = build_shortlist(
            summaries,
            top_k=args.top_k,
            source_hashes=source_hashes,
        )
    else:
        with open(args.shortlist) as handle:
            shortlist = json.load(handle)
        candidates = {}
        for spec in args.candidate:
            if "=" not in spec:
                raise ValueError(f"--candidate must be MODULE=JSONL, got {spec!r}")
            name, path = spec.split("=", 1)
            if name in candidates:
                raise ValueError(f"duplicate --candidate module {name!r}")
            candidates[name] = _read_jsonl(path)
        result = rerank_shortlist(
            shortlist,
            _read_jsonl(args.baseline),
            candidates,
            expected_manifest_sha256=args.manifest_sha256,
        )
    _write_exclusive(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

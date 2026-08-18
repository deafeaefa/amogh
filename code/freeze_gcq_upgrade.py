"""Bind the score-blind GCQ upgrade design to immutable launch artifacts.

The checked-in ``gcq_upgrade_protocol.json`` intentionally has unbound input
hashes: it freezes methodological choices before upgraded scores exist.  This
tool is the second, one-way step.  It validates the two training-only profiling
manifests and the 196-projection candidate-bank manifest, hashes every launch
input and implementation file, and exclusively writes a launch-frozen derived
protocol.  It never reads model scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from build_gcq_vqa_control_data import validate_manifest as validate_vqa_control_manifest
from recovery_utils import coordinate_number_spans


EXPECTED_TASKS = ("rec", "coco_grounding")
EXPECTED_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode() + b"\n"
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path):
    with open(path) as handle:
        return json.load(handle)


def _records(value, label: str) -> list[dict]:
    if isinstance(value, dict):
        value = value.get("records", value.get("examples"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{label} must be a JSON list of record objects")
    return value


def validate_profile_pair(proxy_value, decode_value, expected_per_cell: int = 64) -> dict:
    """Validate the frozen 2 x 4 x 64 manifests and their image disjointness."""
    expected_counts = Counter({(task, quartile): expected_per_cell
                               for task in EXPECTED_TASKS for quartile in range(1, 5)})
    summaries = {}
    image_sets = {}
    for label, value in (("proxy", proxy_value), ("decode", decode_value)):
        rows = _records(value, label)
        uids = [str(row.get("uid", "")) for row in rows]
        if not all(uids) or len(uids) != len(set(uids)):
            raise ValueError(f"{label} manifest requires unique non-empty UIDs")
        image_ids = [int(row["image_id"]) for row in rows]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"{label} manifest must use one unique image per row")
        counts = Counter((str(row["task"]), int(row["area_quartile"])) for row in rows)
        if counts != expected_counts:
            raise ValueError(
                f"{label} task/quartile counts differ from {dict(expected_counts)}: {dict(counts)}"
            )
        absolute_sizes = set()
        for row in rows:
            relative_area = float(row["relative_area"])
            if not 0.0 < relative_area <= 1.0:
                raise ValueError(f"{label} {row['uid']} has invalid relative_area {relative_area}")
            if len(coordinate_number_spans(str(row["answer"]))) != 4:
                raise ValueError(f"{label} {row['uid']} answer lacks four coordinates")
            if str(row["task"]) == "coco_grounding":
                absolute_sizes.add(str(row["absolute_size"]))
        if absolute_sizes != {"small", "medium", "large"}:
            raise ValueError(
                f"{label} category rows must include small/medium/large; got {absolute_sizes}"
            )
        image_sets[label] = set(image_ids)
        summaries[label] = {
            "examples": len(rows),
            "unique_images": len(image_ids),
            "task_quartile_counts": {
                f"{task}:q{quartile}": counts[(task, quartile)]
                for task in EXPECTED_TASKS for quartile in range(1, 5)
            },
            "category_absolute_sizes": sorted(absolute_sizes),
        }
    overlap = image_sets["proxy"] & image_sets["decode"]
    if overlap:
        raise ValueError(f"proxy/decode manifests overlap on {len(overlap)} images")
    return summaries


def _candidate_rows(value) -> list[dict]:
    if not isinstance(value, dict):
        raise ValueError("candidate-bank manifest must be an object")
    rows = value.get("candidates", value.get("projections"))
    if isinstance(rows, dict):
        normalized = []
        for name, row in rows.items():
            if not isinstance(row, dict):
                raise ValueError(f"candidate-bank entry {name!r} is not an object")
            normalized.append({"module_name": name, **row})
        return normalized
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("candidate-bank manifest needs a candidates/projections list or map")
    return rows


def validate_candidate_bank(
    value,
    expected_count: int = 196,
    *,
    design: Mapping[str, object] | None = None,
    design_file_sha256: str | None = None,
    calibration_file_sha256: str | None = None,
) -> dict:
    rows = _candidate_rows(value)
    if len(rows) != expected_count:
        raise ValueError(f"candidate bank has {len(rows)} projections, expected {expected_count}")
    names = [str(row.get("module_name", row.get("name", ""))) for row in rows]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("candidate-bank module names must be unique and non-empty")
    suffix_counts = Counter()
    total_delta = 0
    for name, row in zip(names, rows):
        matches = [suffix for suffix in EXPECTED_SUFFIXES if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"candidate {name!r} is not one of the seven allowed projections")
        suffix_counts[matches[0]] += 1
        delta = int(row.get("delta_bytes", row.get("logical_delta_bytes", 0)))
        if delta <= 0:
            raise ValueError(f"candidate {name!r} has invalid added bytes {delta}")
        total_delta += delta
        for bits in (4, 8):
            digest = row.get(f"w{bits}_sha256")
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
                raise ValueError(f"candidate {name!r} has invalid W{bits} digest")
    expected_suffix_counts = Counter({suffix: expected_count // len(EXPECTED_SUFFIXES)
                                      for suffix in EXPECTED_SUFFIXES})
    if suffix_counts != expected_suffix_counts:
        raise ValueError(f"candidate projection-role counts are wrong: {dict(suffix_counts)}")
    if design is not None and isinstance(design.get("model"), Mapping):
        if value.get("schema_version") != 1 or value.get("artifact_kind") != "gcq_packed_gptq_projection_bank":
            raise ValueError("candidate bank is not the persisted packed GPTQ artifact")
        stored_hash = value.get("manifest_content_sha256")
        unhashed = dict(value)
        unhashed.pop("manifest_content_sha256", None)
        if not isinstance(stored_hash, str) or canonical_sha256(unhashed) != stored_hash:
            raise ValueError("candidate-bank manifest content hash mismatch")
        model = design["model"]
        quantization = design.get("quantization")
        if not isinstance(quantization, Mapping):
            raise ValueError("design protocol lacks quantization settings")
        recipe = value.get("recipe")
        if not isinstance(recipe, Mapping):
            raise ValueError("candidate bank lacks its GPTQ recipe")
        expected_recipe = {
            "base_model": model.get("id"),
            "revision": model.get("revision"),
            "bits": quantization.get("candidate_bits"),
            "group_size": quantization.get("group_size"),
            "block_size": quantization.get("block_size"),
            "percdamp": quantization.get("percdamp"),
            "scale_dtype": quantization.get("scale_dtype"),
            "prefix_policy": "earlier_decoder_layers_cached_w4",
        }
        for field, expected_value in expected_recipe.items():
            if recipe.get(field) != expected_value:
                raise ValueError(f"candidate-bank recipe {field} differs from design")
        architecture = value.get("architecture")
        if not isinstance(architecture, Mapping):
            raise ValueError("candidate bank lacks strict architecture validation")
        if architecture.get("projections") != model.get("expected_projection_count"):
            raise ValueError("candidate-bank architecture projection count differs from design")
        if architecture.get("weights") != model.get("expected_decoder_weight_count"):
            raise ValueError("candidate-bank architecture weight count differs from design")
        provenance = value.get("provenance")
        sources = provenance.get("source_files") if isinstance(provenance, Mapping) else None
        if not isinstance(sources, Mapping):
            raise ValueError("candidate bank lacks source-file provenance")
        expected_sources = {
            "protocol_file_sha256": design_file_sha256,
            "calibration_manifest_file_sha256": calibration_file_sha256,
        }
        for field, expected_value in expected_sources.items():
            if expected_value is not None and sources.get(field) != expected_value:
                raise ValueError(f"candidate-bank provenance {field} mismatch")
    return {
        "projections": len(rows),
        "role_counts": dict(sorted(suffix_counts.items())),
        "sum_promotion_delta_bytes": total_delta,
    }


def implementation_tree_hash(paths: Iterable[str | Path]) -> tuple[str, dict[str, str]]:
    resolved = sorted({Path(path).resolve() for path in paths}, key=str)
    if not resolved:
        raise ValueError("at least one implementation file must be bound")
    files = {path.name: sha256_file(path) for path in resolved}
    if len(files) != len(resolved):
        raise ValueError("implementation files must have unique basenames")
    return canonical_sha256(files), files


def bind_protocol(
    design_path: str | Path,
    *,
    calibration_manifest: str | Path,
    proxy_manifest: str | Path,
    decode_manifest: str | Path,
    vqa_control_manifest: str | Path,
    candidate_bank_manifest: str | Path,
    packing_spec: str | Path,
    implementation_files: Iterable[str | Path],
    bound_at_utc: str | None = None,
) -> dict:
    design = _load_json(design_path)
    if design.get("status") != "design_frozen_inputs_unbound" or design.get("bound_hashes") is not None:
        raise ValueError("design protocol is not the pristine unbound protocol")
    profile_validation = validate_profile_pair(
        _load_json(proxy_manifest), _load_json(decode_manifest)
    )
    vqa_records = validate_vqa_control_manifest(_load_json(vqa_control_manifest))
    bank_validation = validate_candidate_bank(
        _load_json(candidate_bank_manifest),
        design=design,
        design_file_sha256=sha256_file(design_path),
        calibration_file_sha256=sha256_file(calibration_manifest),
    )
    tree_digest, tree_files = implementation_tree_hash(implementation_files)
    input_paths = {
        "calibration_manifest": Path(calibration_manifest),
        "proxy_manifest": Path(proxy_manifest),
        "decode_manifest": Path(decode_manifest),
        "vqa_control_manifest": Path(vqa_control_manifest),
        "candidate_bank_manifest": Path(candidate_bank_manifest),
        "packing_spec": Path(packing_spec),
    }
    hashes = {f"{name}_sha256": sha256_file(path) for name, path in input_paths.items()}
    hashes["implementation_tree_sha256"] = tree_digest
    required = set(design["required_bound_hashes"])
    if set(hashes) != required:
        raise ValueError(f"bound hash keys do not match design: {sorted(set(hashes) ^ required)}")
    result = dict(design)
    result["status"] = "launch_frozen"
    result["status_note"] = (
        "Launch inputs and implementation are hash-bound. No upgraded score was read by the binder."
    )
    result["bound_at_utc"] = bound_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result["bound_hashes"] = dict(sorted(hashes.items()))
    result["bound_inputs"] = {
        name: {"file_name": path.name, "sha256": hashes[f"{name}_sha256"]}
        for name, path in input_paths.items()
    }
    result["implementation_files"] = tree_files
    result["validation"] = {
        "profiling_manifests": profile_validation,
        "vqa_control_manifest": {
            "examples": len(vqa_records),
            "unique_images": len({row["image_id"] for row in vqa_records}),
        },
        "candidate_bank": bank_validation,
    }
    result["protocol_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=Path(__file__).with_name("gcq_upgrade_protocol.json"))
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--proxy-manifest", type=Path, required=True)
    parser.add_argument("--decode-manifest", type=Path, required=True)
    parser.add_argument("--vqa-control-manifest", type=Path, required=True)
    parser.add_argument("--candidate-bank-manifest", type=Path, required=True)
    parser.add_argument("--packing-spec", type=Path, required=True)
    parser.add_argument(
        "--implementation", type=Path, action="append", default=[],
        help="implementation file to bind; repeat as needed",
    )
    parser.add_argument(
        "--implementation-dir", type=Path, action="append", default=[],
        help="bind every top-level .py file in this directory (recommended for code/)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    implementation_files = list(args.implementation)
    for directory in args.implementation_dir:
        if not directory.is_dir():
            raise ValueError(f"implementation directory does not exist: {directory}")
        implementation_files.extend(sorted(directory.glob("*.py")))
    if not implementation_files:
        raise ValueError("provide --implementation or --implementation-dir")
    result = bind_protocol(
        args.design,
        calibration_manifest=args.calibration_manifest,
        proxy_manifest=args.proxy_manifest,
        decode_manifest=args.decode_manifest,
        vqa_control_manifest=args.vqa_control_manifest,
        candidate_bank_manifest=args.candidate_bank_manifest,
        packing_spec=args.packing_spec,
        implementation_files=implementation_files,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "out": str(args.out),
        "protocol_sha256": result["protocol_sha256"],
        "status": result["status"],
    }, indent=2))


if __name__ == "__main__":
    main()

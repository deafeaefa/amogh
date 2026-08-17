#!/usr/bin/env python3
"""Validate and summarize the frozen RefCOCO+ grounding confirmation.

The confirmation is authorized only when the one-time fresh-VQAv2 gate passed,
and it is hard-bound to that exact adapter (path plus all artifact hashes).  An
integrity failure raises and exits nonzero.  A scientifically valid gate
failure is still a final result: the summary is published and the process exits
zero so a failed result cannot be mistaken for permission to try another model.

``--preflight`` validates every frozen input, the VQA authorization chain, the
image inventory, and the no-peeking wording without reading prediction files.
It is used by both the launcher and each scheduler task before inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from recovery_utils import BASE_MODEL, BASE_REVISION, IOU_THRESHOLDS, precise_iou_score, sha256_file


RECIPE_ID = "gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
CANDIDATE_STEPS = (200, 300, 400, 500, 600, 750)
GROUND_PROTOCOL_SHA256 = (
    "70e828254c3bdbe75a4b2ace8fb3f6b305675b31cd1a3708560d503ec9ec75a8"
)
REPLAY_PROTOCOL_SHA256 = (
    "fd938d0d39116b989ffdcde4dd5ce64bbb419a1292e4ab9f32864416953e5e6d"
)
PROMOTION_SHA256 = (
    "78b3138da1c70f2f944e3c63d5cfc754f59bf6062c27564d8cc6b2792ba0c1e6"
)
FRESH_VQA_MANIFEST_SHA256 = (
    "416aea5f9dd6f4a4cfa7061c3b5ca0af88647e49d5f17ed00a8d83885ea21038"
)
FRESH_VQA_METADATA_SHA256 = (
    "a7777a0199f7fb432deeee94d478ca092474dd8a5e47cffe92a93d62b57601e8"
)
VQA_NONINFERIORITY_MARGIN = -0.015
MINIMUM_STORAGE_HEADROOM_BYTES = 1_073_741_824
SPLITS = {
    "testA": {
        "task_id": 1,
        "subset": "refcocoplus_testA_confirm_full",
        "manifest": "refcocoplus_testA_confirm_full.json",
        "manifest_sha256": "542fbbf73fe0623ed79ddba19167fce34647f97613d14e48b12ca5d96457ff83",
        "ordered_uid_sha256": "298cd11f84e8acf8098eed3c9232ac90b7156938abdd87d1d6fbd7d8cfdbc409",
        "expressions": 5_726,
        "images": 750,
        "source_arrow_sha256": "cca834e4b0c880c4ceac39a5bf48e017ee7eefeb5ce50d869fa5deccd6ff6d75",
    },
    "testB": {
        "task_id": 2,
        "subset": "refcocoplus_testB_confirm_full",
        "manifest": "refcocoplus_testB_confirm_full.json",
        "manifest_sha256": "fafda8c81957baef1417c26e84fe225f51100eb2261ae61d123325a2b1c44462",
        "ordered_uid_sha256": "ad420b65cc28cef2b0c717977c2067f6ec51154e88e59e10dcc1d4d751d67d02",
        "expressions": 4_889,
        "images": 750,
        "source_arrow_sha256": "00861b22908f8f0e897784548b0b510344ea351fab65734ef641afb14524fff4",
    },
}
META_SHA256 = "af4b9e3b7f5c5848e4bacb7218240ef3b736582e879a4ff48353cefc41a5615a"
FORBIDDEN_MANIFESTS = {
    "recovery_train": (
        "recovery_train_vqa_replay_12k.json",
        "8bf3b6a1589527f5847ea28a7c5f0daeb89f6e0d7fa220451db87c52314aec4a",
    ),
    "recovery_dev": (
        "recovery_dev_1k.json",
        "e40a1374bdc18c6639615e82dab67cbd0ae9c8d63524b34db4170e159d13b23b",
    ),
    "allocation_probe": (
        "dprobe_refcoco_train_512.json",
        "e0103a90d76b00aa6bb0604951f1924d96a2f4763df7a824e299b18d7608920d",
    ),
}
ARMS = ("bf16", "uniform_rtn_w4", "untrained_gcq4.25", "selected_adapter")
ARM_TAG_TEMPLATES = {
    "bf16": "bf16_refcocoplus_{split}_confirm",
    "uniform_rtn_w4": "w4rtn_refcocoplus_{split}_confirm",
    "untrained_gcq4.25": "gcq425_untrained_refcocoplus_{split}_confirm",
    "selected_adapter": "gcq425_vqa_selected_refcocoplus_{split}_confirm",
}
COMPARISONS = {
    "selected_adapter_minus_untrained_gcq4.25": (
        "selected_adapter",
        "untrained_gcq4.25",
    ),
    "untrained_gcq4.25_minus_uniform_rtn_w4": (
        "untrained_gcq4.25",
        "uniform_rtn_w4",
    ),
    "bf16_minus_untrained_gcq4.25": ("bf16", "untrained_gcq4.25"),
}
METRICS = ("rec", "giou", "precise_iou", "parse_fail")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_BASE_SEED = 20260860
PARSE_MARGIN = 0.005
CANDIDATE_FIELDS = (
    "step",
    "recipe_id",
    "adapter_dir",
    "adapter_sha256",
    "adapter_config_sha256",
    "manifest_sha256",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require_close(
    actual: float,
    expected: float,
    label: str,
    *,
    atol: float = 1e-12,
) -> None:
    require(math.isfinite(actual), f"{label} is not finite: {actual!r}")
    require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol),
        f"{label} changed: {actual!r} != {expected!r}",
    )


def require_hash(path: Path, expected: object, label: str) -> str:
    require(
        isinstance(expected, str) and len(expected) == 64,
        f"missing or invalid frozen hash for {label}",
    )
    require(path.is_file(), f"missing hash-locked file for {label}: {path}")
    actual = sha256_file(path)
    require(actual == expected, f"{label} changed after grounding launch: {path}")
    return actual


def require_read_only(path: Path, label: str) -> None:
    require(path.is_file(), f"missing frozen {label}: {path}")
    require(
        path.stat().st_mode & 0o222 == 0,
        f"frozen {label} is still writable: {path}",
    )


def audit_prior_grounding_predictions(
    runs: Path,
    manifests: dict[str, Path],
    *,
    requested_splits: tuple[str, ...],
) -> dict:
    """Reject evidence that a confirmation expression was already evaluated.

    Fixed names catch empty/crashed attempts.  The subset identity in metrics
    catches alternate tags after a completed run, and UID scanning catches an
    alternate-tag prediction log even when its metrics file was never written.
    Only identities are inspected; no prediction values or scores are returned.
    """
    require(runs.is_dir(), f"missing runs directory for no-peeking audit: {runs}")
    require(requested_splits, "no grounding splits requested for no-peeking audit")
    require(
        len(set(requested_splits)) == len(requested_splits)
        and all(split in SPLITS for split in requested_splits),
        f"invalid no-peeking split inventory: {requested_splits!r}",
    )
    uids_by_split: dict[str, set[str]] = {}
    for split in requested_splits:
        path = manifests[split]
        require_hash(path, SPLITS[split]["manifest_sha256"], f"RefCOCO+ {split} manifest")
        rows = load_json(path)
        require(isinstance(rows, list), f"RefCOCO+ {split} manifest is not a list")
        uids = set()
        for index, row in enumerate(rows):
            require(isinstance(row, dict), f"invalid RefCOCO+ {split} row {index}")
            uid = row.get("uid")
            require(isinstance(uid, str) and uid, f"missing RefCOCO+ {split} UID {index}")
            require(uid not in uids, f"duplicate RefCOCO+ {split} UID {uid!r}")
            uids.add(uid)
        require(
            len(uids) == SPLITS[split]["expressions"],
            f"RefCOCO+ {split} UID count changed during no-peeking audit",
        )
        uids_by_split[split] = uids

    fixed_names = {
        f"{ARM_TAG_TEMPLATES[arm].format(split=split)}{suffix}"
        for split in requested_splits
        for arm in ARMS
        for suffix in (".rec.jsonl", ".rec.metrics.json")
    }
    metric_files = sorted(runs.rglob("*.rec.metrics.json"))
    prediction_files = sorted(runs.rglob("*.rec.jsonl"))
    for path in (*metric_files, *prediction_files):
        require(path.is_file(), f"non-file grounding audit path: {path}")
        require(
            path.name not in fixed_names,
            f"grounding-confirmation fixed tag was already evaluated: {path}",
        )

    subsets = {split: SPLITS[split]["subset"] for split in requested_splits}
    for path in metric_files:
        # Read only files that mention one of the target subset identities. This
        # avoids racing an unrelated concurrently written metrics file.
        raw = path.read_text(encoding="utf-8")
        matching = [split for split, subset in subsets.items() if subset in raw]
        if not matching:
            continue
        try:
            metrics = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"cannot prove untouched status: invalid target metrics JSON in {path}"
            ) from exc
        require(isinstance(metrics, dict), f"grounding metrics are not an object: {path}")
        require(
            metrics.get("subset") not in {subsets[split] for split in matching},
            f"RefCOCO+ confirmation subset was already evaluated according to {path}",
        )

    all_target_uids = set().union(*(uids_by_split[split] for split in requested_splits))
    uid_prefixes = tuple(f"refcocoplus_{split}:" for split in requested_splits)
    for path in prediction_files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not any(prefix in line for prefix in uid_prefixes):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"cannot prove untouched status: invalid target row in {path}:{line_number}"
                    ) from exc
                require(isinstance(row, dict), f"non-object target row in {path}:{line_number}")
                require(
                    row.get("uid") not in all_target_uids,
                    f"RefCOCO+ confirmation UID {row.get('uid')!r} was already evaluated "
                    f"in {path}:{line_number}",
                )
    return {
        "manifest_sha256": {
            split: SPLITS[split]["manifest_sha256"] for split in requested_splits
        },
        "splits": list(requested_splits),
        "fixed_tag_outputs_found": 0,
        "matching_subset_metrics_found": 0,
        "confirmation_uid_prediction_logs_found": 0,
        "rec_metrics_files_scanned": len(metric_files),
        "rec_prediction_files_scanned": len(prediction_files),
    }


def candidate_projection(value: object) -> dict:
    require(isinstance(value, dict), "selected candidate is not an object")
    candidate = {field: value.get(field) for field in CANDIDATE_FIELDS}
    require(
        isinstance(candidate["step"], int)
        and not isinstance(candidate["step"], bool)
        and candidate["step"] in CANDIDATE_STEPS,
        f"invalid selected checkpoint step: {candidate['step']!r}",
    )
    require(candidate["recipe_id"] == RECIPE_ID, "selected recipe changed")
    require(
        isinstance(candidate["adapter_dir"], str) and candidate["adapter_dir"],
        "selected adapter path is missing",
    )
    for field in CANDIDATE_FIELDS[3:]:
        require(
            isinstance(candidate[field], str) and len(candidate[field]) == 64,
            f"selected candidate has an invalid {field}",
        )
    return candidate


def canonical_adapter_dir(root: Path, step: int) -> Path:
    base = root / "adapters" / RECIPE_ID
    return base if step == 750 else base / f"checkpoint-{step:06d}"


def ordered_uid_sha256(rows: list[dict]) -> str:
    payload = "".join(f"{row['uid']}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def image_inventory(rows: list[dict], data: Path, *, verify_images: bool) -> dict:
    inventory: dict[int, tuple[str, int, int]] = {}
    for row in rows:
        image_id = int(row["image_id"])
        value = (str(row["file_name"]), int(row["width"]), int(row["height"]))
        require(
            image_id not in inventory or inventory[image_id] == value,
            f"inconsistent image metadata for COCO image {image_id}",
        )
        inventory[image_id] = value
    digest = hashlib.sha256()
    total_bytes = 0
    for image_id, (file_name, width, height) in sorted(inventory.items()):
        subdir = "train2014" if "train2014" in file_name else "val2014"
        path = data / "images" / subdir / file_name
        require(path.is_file() and path.stat().st_size > 0, f"missing image: {path}")
        file_sha = sha256_file(path)
        total_bytes += path.stat().st_size
        digest.update(f"{image_id}\t{file_name}\t{width}\t{height}\t{file_sha}\n".encode())
        if verify_images:
            with Image.open(path) as image:
                require(
                    image.size == (width, height),
                    f"image dimensions changed for {path}: {image.size} != {(width, height)}",
                )
                image.verify()
    return {
        "unique_images": len(inventory),
        "total_file_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def validate_manifest(
    path: Path,
    split: str,
    data: Path,
    *,
    verify_images: bool,
) -> tuple[list[dict], dict]:
    spec = SPLITS[split]
    require_hash(path, spec["manifest_sha256"], f"RefCOCO+ {split} manifest")
    rows = load_json(path)
    require(isinstance(rows, list), f"RefCOCO+ {split} manifest is not a list")
    require(
        len(rows) == spec["expressions"],
        f"RefCOCO+ {split} expression count changed",
    )
    seen: set[str] = set()
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"invalid RefCOCO+ {split} row {index}")
        expected_uid = f"refcocoplus_{split}:{index:05d}"
        require(row.get("uid") == expected_uid, f"unexpected UID at {split}:{index}")
        require(expected_uid not in seen, f"duplicate UID {expected_uid!r}")
        expected = {
            "dataset": "refcocoplus",
            "source": "refcocoplus",
            "task": "rec",
            "split": split,
        }
        for key, value in expected.items():
            require(row.get(key) == value, f"unexpected {key} for {expected_uid}")
        for key in ("image_id", "ref_id", "width", "height"):
            require(
                isinstance(row.get(key), int) and not isinstance(row[key], bool),
                f"invalid {key} for {expected_uid}",
            )
        require(row["width"] > 0 and row["height"] > 0, f"invalid geometry for {expected_uid}")
        require(
            isinstance(row.get("file_name"), str)
            and row["file_name"] == f"COCO_train2014_{row['image_id']:012d}.jpg",
            f"invalid file name for {expected_uid}",
        )
        require(
            isinstance(row.get("expression"), str) and row["expression"].strip(),
            f"empty expression for {expected_uid}",
        )
        box = row.get("bbox_xywh")
        require(
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in box)
            and float(box[2]) > 0
            and float(box[3]) > 0,
            f"invalid box for {expected_uid}",
        )
        relative_area = float(box[2]) * float(box[3]) / (row["width"] * row["height"])
        require_close(float(row.get("relative_area")), relative_area, f"relative area {expected_uid}")
        seen.add(expected_uid)
    require(
        ordered_uid_sha256(rows) == spec["ordered_uid_sha256"],
        f"RefCOCO+ {split} ordered UID digest changed",
    )
    inventory = image_inventory(rows, data, verify_images=verify_images)
    require(inventory["unique_images"] == spec["images"], f"{split} image count changed")
    return rows, inventory


def validate_authorization(runs: Path, code_dir: Path) -> tuple[dict, dict[str, Path]]:
    root = runs / "recovery_vqa_replay"
    paths = {
        "vqa_confirmation_summary": root / "vqa_confirmation_summary.json",
        "vqa_confirmation_launch": root / "confirmation" / "confirmation_launch_manifest.json",
        "development_summary": root / "development_summary.json",
        "development_launch": root / "development_launch_manifest.json",
        "artifact_validation": root / "artifact_validation.json",
        "training_launch": root / "training_launch_manifest.json",
        "replay_protocol": code_dir / "recovery_vqa_replay_protocol.json",
        "vqa_confirmation_summarizer": code_dir / "summarize_recovery_vqa_confirmation.py",
    }
    for label in (
        "vqa_confirmation_summary",
        "vqa_confirmation_launch",
        "development_summary",
        "development_launch",
    ):
        require_read_only(paths[label], label.replace("_", " "))
    vqa = load_json(paths["vqa_confirmation_summary"])
    expected_top = {
        "schema_version": 1,
        "recipe_id": RECIPE_ID,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "confirmation_pass": True,
        "recovery_pipeline_pass": True,
        "scientific_outcome": "PASS",
        "integrity_validation_pass": True,
    }
    for key, expected in expected_top.items():
        require(vqa.get(key) == expected, f"fresh-VQA authorization {key} is not {expected!r}")
    gate = "VQA_fresh_paired_CI95_lower_at_least_minus_1.5pt"
    require(vqa.get("gates") == {gate: True}, "fresh-VQA confirmation gate did not pass")
    candidate = candidate_projection(vqa.get("selected"))

    adapter_dir = canonical_adapter_dir(root, int(candidate["step"]))
    require(candidate["adapter_dir"] == str(adapter_dir), "confirmed adapter path is noncanonical")
    paths.update(
        {
            "selected_adapter": adapter_dir / "adapter_model.safetensors",
            "selected_adapter_config": adapter_dir / "adapter_config.json",
            "selected_adapter_manifest": adapter_dir / "gcq_recovery_manifest.json",
        }
    )
    require_hash(paths["selected_adapter"], candidate["adapter_sha256"], "selected adapter")
    require_hash(
        paths["selected_adapter_config"],
        candidate["adapter_config_sha256"],
        "selected adapter config",
    )
    require_hash(
        paths["selected_adapter_manifest"],
        candidate["manifest_sha256"],
        "selected adapter manifest",
    )

    vqa_launch_hash = sha256_file(paths["vqa_confirmation_launch"])
    require(
        vqa.get("confirmation_launch_manifest") == str(paths["vqa_confirmation_launch"])
        and vqa.get("confirmation_launch_manifest_sha256") == vqa_launch_hash,
        "fresh-VQA summary does not bind its launch manifest",
    )
    vqa_launch = load_json(paths["vqa_confirmation_launch"])
    require(
        vqa_launch.get("schema_version") == 1
        and vqa_launch.get("recipe_id") == RECIPE_ID
        and vqa_launch.get("base_model") == BASE_MODEL
        and vqa_launch.get("base_revision") == BASE_REVISION,
        "fresh-VQA launch identity changed",
    )
    candidate_tag = f"vqa50_step{candidate['step']}_vqa_fresh5k"
    require(
        vqa_launch.get("evaluation")
        == {
            "task": "vqa",
            "questions": 5_000,
            "unique_images": 4_571,
            "start": 0,
            "limit": 5_000,
            "batch_size": 24,
            "device": "cuda:0",
            "baseline_tag": "gcq425_untrained_vqa_fresh5k",
            "candidate_tag": candidate_tag,
            "execution_order": [
                "untrained_gcq_baseline",
                "single_selected_candidate",
            ],
            "same_job_and_gpu_required": True,
        },
        "fresh-VQA launch evaluation contract changed",
    )
    require(
        vqa_launch.get("bootstrap")
        == {
            "unit": "image-clustered paired candidate-minus-untrained-GCQ",
            "resamples": 10_000,
            "seed": 20260850,
        },
        "fresh-VQA launch bootstrap contract changed",
    )
    require(
        vqa_launch.get("only_scientific_gate", {}).get("minimum")
        == VQA_NONINFERIORITY_MARGIN,
        "fresh-VQA launch noninferiority margin changed",
    )
    require(
        vqa_launch.get("trained_candidates") == [candidate],
        "fresh-VQA launch did not evaluate exactly the confirmed adapter",
    )

    # Revalidate the complete immutable input inventory used to create the VQA
    # result, then bind those same bytes into the grounding launch.  This makes
    # the authorization a chain of concrete artifacts rather than a Boolean in
    # a standalone summary.
    launched_paths = vqa_launch.get("paths")
    launched_hashes = vqa_launch.get("hashes")
    require(
        isinstance(launched_paths, dict)
        and isinstance(launched_hashes, dict)
        and set(launched_paths) == set(launched_hashes),
        "fresh-VQA launch path/hash inventory changed",
    )
    for label, raw_path in launched_paths.items():
        require(isinstance(label, str) and label, "invalid fresh-VQA launch path key")
        require(isinstance(raw_path, str) and raw_path, f"invalid fresh-VQA path {label}")
        input_path = Path(raw_path)
        require_hash(input_path, launched_hashes[label], f"fresh-VQA input {label}")
        paths[f"vqa_launch_input_{label}"] = input_path
    expected_vqa_inputs = {
        "development_summary": paths["development_summary"],
        "development_launch": paths["development_launch"],
        "artifact_validation": paths["artifact_validation"],
        "training_launch": paths["training_launch"],
        "protocol": paths["replay_protocol"],
        "selected_adapter": paths["selected_adapter"],
        "selected_adapter_config": paths["selected_adapter_config"],
        "selected_adapter_manifest": paths["selected_adapter_manifest"],
        "confirmation_summarizer": paths["vqa_confirmation_summarizer"],
    }
    for label, expected_path in expected_vqa_inputs.items():
        require(
            launched_paths.get(label) == str(expected_path),
            f"fresh-VQA launch {label} path changed",
        )
    require(
        launched_hashes.get("fresh_manifest") == FRESH_VQA_MANIFEST_SHA256
        and launched_hashes.get("fresh_metadata") == FRESH_VQA_METADATA_SHA256
        and launched_hashes.get("protocol") == REPLAY_PROTOCOL_SHA256
        and launched_hashes.get("promotion") == PROMOTION_SHA256,
        "fresh-VQA launch frozen data/protocol/quantization hashes changed",
    )

    fresh_vqa = vqa.get("fresh_vqa")
    require(
        isinstance(fresh_vqa, dict)
        and fresh_vqa.get("manifest_sha256") == FRESH_VQA_MANIFEST_SHA256
        and fresh_vqa.get("metadata_sha256") == FRESH_VQA_METADATA_SHA256
        and fresh_vqa.get("questions") == 5_000
        and fresh_vqa.get("unique_images") == 4_571,
        "fresh-VQA summary data identity changed",
    )
    require(vqa.get("bootstrap") == vqa_launch["bootstrap"],
            "fresh-VQA summary bootstrap differs from its launch")
    require(vqa.get("frozen_gate_threshold") == VQA_NONINFERIORITY_MARGIN,
            "fresh-VQA summary gate threshold changed")
    vqa_result = vqa.get("vqa")
    require(isinstance(vqa_result, dict), "fresh-VQA scientific result is missing")
    paired = vqa_result.get("paired_delta")
    require(isinstance(paired, dict), "fresh-VQA paired result is missing")
    ci95 = paired.get("ci95")
    require(
        isinstance(ci95, list)
        and len(ci95) == 2
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in ci95
        )
        and float(ci95[0]) <= float(ci95[1])
        and float(ci95[0]) >= VQA_NONINFERIORITY_MARGIN
        and paired.get("n_examples") == 5_000
        and paired.get("n_images") == 4_571
        and paired.get("resamples") == 10_000
        and paired.get("seed") == 20260850,
        "fresh-VQA paired result does not satisfy the frozen authorization gate",
    )
    baseline_score = _finite_float(vqa_result.get("untrained_gcq"), "fresh-VQA baseline")
    candidate_score = _finite_float(
        vqa_result.get("selected_candidate"), "fresh-VQA selected candidate"
    )
    point_delta = _finite_float(vqa_result.get("point_delta"), "fresh-VQA point delta")
    require(0.0 <= baseline_score <= 1.0 and 0.0 <= candidate_score <= 1.0,
            "fresh-VQA accuracy is outside [0,1]")
    require_close(point_delta, candidate_score - baseline_score, "fresh-VQA point delta")
    require_close(float(paired.get("observed")), point_delta, "fresh-VQA paired delta")

    vqa_confirmation = paths["vqa_confirmation_launch"].parent
    output_paths = {
        "runtime_provenance": vqa_confirmation / "runtime_provenance.json",
        "baseline_jsonl": vqa_confirmation / "gcq425_untrained_vqa_fresh5k.vqa.jsonl",
        "baseline_metrics": vqa_confirmation / "gcq425_untrained_vqa_fresh5k.vqa.metrics.json",
        "candidate_jsonl": vqa_confirmation / f"{candidate_tag}.vqa.jsonl",
        "candidate_metrics": vqa_confirmation / f"{candidate_tag}.vqa.metrics.json",
        "results_csv": vqa_confirmation / "results.csv",
        "execution_ledger": vqa_confirmation / "execution_ledger.json",
    }
    output_hashes = vqa.get("output_hashes")
    require(
        isinstance(output_hashes, dict) and set(output_hashes) == set(output_paths),
        "fresh-VQA output hash inventory changed",
    )
    for label, output_path in output_paths.items():
        require_hash(output_path, output_hashes[label], f"fresh-VQA output {label}")
        paths[f"vqa_output_{label}"] = output_path
    runtime = load_json(output_paths["runtime_provenance"])
    require(
        runtime == vqa.get("execution", {}).get("runtime_provenance")
        and runtime.get("selected_step") == candidate["step"]
        and runtime.get("confirmation_launch_manifest_sha256") == vqa_launch_hash
        and runtime.get("hardware_gate_pass") is True,
        "fresh-VQA runtime provenance changed",
    )
    ledger = load_json(output_paths["execution_ledger"])
    evaluations = ledger.get("evaluations") if isinstance(ledger, dict) else None
    require(
        isinstance(evaluations, list)
        and len(evaluations) == 2
        and evaluations[0].get("role") == "untrained_gcq_baseline"
        and evaluations[0].get("adapter") is None
        and evaluations[1].get("role") == "single_selected_candidate"
        and evaluations[1].get("adapter")
        == {"path": candidate["adapter_dir"], "sha256": candidate["adapter_sha256"]}
        and all(row.get("exit_code") == 0 for row in evaluations),
        "fresh-VQA execution ledger does not prove the exact adapter evaluation",
    )

    development = load_json(paths["development_summary"])
    require(
        development.get("schema_version") == 1
        and development.get("recipe_id") == RECIPE_ID
        and development.get("base_model") == BASE_MODEL
        and development.get("base_revision") == BASE_REVISION
        and development.get("selection_succeeded") is True,
        "development selection did not authorize a candidate",
    )
    require(development.get("protocol_sha256") == REPLAY_PROTOCOL_SHA256,
            "development summary used a different recovery protocol")
    selected = development.get("selected")
    require(
        isinstance(selected, dict)
        and selected.get("eligible") is True
        and candidate_projection(selected) == candidate,
        "fresh-VQA-confirmed adapter differs from development selection",
    )
    require(
        vqa.get("development_summary") == str(paths["development_summary"])
        and vqa.get("development_summary_sha256") == sha256_file(paths["development_summary"]),
        "fresh-VQA summary does not bind the current development summary",
    )
    require(
        development.get("development_launch_manifest") == str(paths["development_launch"])
        and development.get("development_launch_manifest_sha256")
        == sha256_file(paths["development_launch"]),
        "development summary does not bind its launch manifest",
    )
    development_launch = load_json(paths["development_launch"])
    checkpoint = development_launch.get("checkpoints", {}).get(str(candidate["step"]))
    require(
        isinstance(checkpoint, dict)
        and checkpoint.get("adapter_dir") == candidate["adapter_dir"]
        and checkpoint.get("adapter_sha256") == candidate["adapter_sha256"]
        and checkpoint.get("adapter_config_sha256") == candidate["adapter_config_sha256"]
        and checkpoint.get("manifest_sha256") == candidate["manifest_sha256"],
        "confirmed adapter differs from the frozen development launch",
    )
    validation = load_json(paths["artifact_validation"])
    validated = validation.get("checkpoints", {}).get(str(candidate["step"]))
    require(
        isinstance(validated, dict)
        and validated.get("directory") == candidate["adapter_dir"]
        and validated.get("artifact", {}).get("sha256") == candidate["adapter_sha256"]
        and validated.get("adapter_config", {}).get("sha256")
        == candidate["adapter_config_sha256"]
        and validated.get("manifest_sha256") == candidate["manifest_sha256"],
        "confirmed adapter differs from artifact validation",
    )
    adapter_manifest = load_json(paths["selected_adapter_manifest"])
    require(
        adapter_manifest.get("schema_version") == 1
        and adapter_manifest.get("base_model") == BASE_MODEL
        and adapter_manifest.get("base_revision") == BASE_REVISION
        and adapter_manifest.get("artifact", {}).get("sha256") == candidate["adapter_sha256"],
        "selected adapter manifest identity changed",
    )
    quant = adapter_manifest.get("quantization", {})
    require(
        quant.get("method") == "rtn_quantize_dequantize"
        and quant.get("bits") == 4
        and quant.get("group_size") == 128
        and quant.get("promote_sha256") == PROMOTION_SHA256,
        "selected adapter quantization contract changed",
    )
    training = load_json(paths["training_launch"])
    require(
        training.get("schema_version") == 1
        and training.get("recipe_id") == RECIPE_ID
        and training.get("data_sha256")
        == "8bf3b6a1589527f5847ea28a7c5f0daeb89f6e0d7fa220451db87c52314aec4a"
        and training.get("promotion_sha256") == PROMOTION_SHA256
        and training.get("objective") == "cwce"
        and training.get("coordinate_weight") == 5
        and training.get("learning_rate") == 0.00005
        and training.get("effective_batch_size") == 16
        and training.get("epochs") == 1
        and training.get("planned_optimizer_steps") == 750
        and training.get("seed") == 0
        and candidate["step"] in training.get("checkpoint_steps", []),
        "training launch does not contain the confirmed checkpoint",
    )
    require(
        training.get("protocol_sha256") == REPLAY_PROTOCOL_SHA256,
        "training launch used a different recovery protocol",
    )
    require_hash(paths["replay_protocol"], REPLAY_PROTOCOL_SHA256, "recovery protocol")
    return candidate, paths


def preflight(
    runs: Path,
    data: Path,
    code_dir: Path,
    *,
    verify_images: bool = True,
) -> dict:
    candidate, auth_paths = validate_authorization(runs, code_dir)
    protocol_path = code_dir / "recovery_grounding_confirmation_protocol.json"
    meta_path = data / "subsets" / "refcocoplus_confirmation.meta.json"
    require_hash(protocol_path, GROUND_PROTOCOL_SHA256, "grounding protocol")
    require_hash(meta_path, META_SHA256, "RefCOCO+ metadata")
    protocol = load_json(protocol_path)
    meta = load_json(meta_path)
    require(
        protocol.get("schema_version") == 1
        and protocol.get("recipe_id") == RECIPE_ID
        and protocol.get("base_model") == BASE_MODEL
        and protocol.get("base_revision") == BASE_REVISION,
        "grounding protocol identity changed",
    )
    scientific = protocol.get("data", {}).get("scientific_status", {})
    require(scientific.get("manifest_frozen_before_selection") is True,
            "protocol no longer states that the manifest was frozen")
    require(scientific.get("model_predictions_and_metrics_unseen_before_selection") is True,
            "protocol no longer states that predictions/metrics were unseen")
    require("labels_unread" not in json.dumps(protocol).lower(),
            "protocol incorrectly claims that labels were unread")
    require(scientific.get("global_coco_image_unseen_claim") is False,
            "protocol makes an invalid globally image-unseen claim")
    authorization = protocol.get("authorization", {})
    require(
        authorization.get("required_summary") == "vqa_confirmation_summary.json"
        and authorization.get("required_field") == "confirmation_pass"
        and authorization.get("required_value") is True
        and authorization.get("allowed_trained_candidates") == 1,
        "grounding authorization contract changed",
    )
    evaluation = protocol.get("evaluation", {})
    require(
        evaluation.get("scheduler_array_tasks") == {"1": "testA", "2": "testB"}
        and evaluation.get("hardware") == "one NVIDIA L40S per split"
        and evaluation.get("within_task_execution")
        == "all four arms run sequentially on the same physical GPU"
        and evaluation.get("arm_order")
        == ["bf16", "uniform_rtn_w4", "untrained_gcq4.25", "vqa_confirmed_selected_adapter"],
        "grounding execution contract changed",
    )
    require(
        evaluation.get("max_pixels") == 1_003_520
        and evaluation.get("batch_size") == 16
        and evaluation.get("decoding") == "greedy",
        "grounding evaluator settings changed",
    )
    quant = evaluation.get("quantization", {})
    require(
        quant.get("method") == "rtn_quantize_dequantize"
        and quant.get("bits") == 4
        and quant.get("group_size") == 128
        and quant.get("gcq_promotion_manifest_sha256") == PROMOTION_SHA256
        and quant.get("gcq_average_decoder_bits") == 4.25,
        "grounding quantization contract changed",
    )
    statistics = protocol.get("statistics", {})
    require(
        statistics.get("bootstrap_resamples") == BOOTSTRAP_RESAMPLES
        and statistics.get("bootstrap_base_seed") == BOOTSTRAP_BASE_SEED
        and statistics.get("comparison_unit")
        == "paired referring expressions clustered by COCO image",
        "grounding statistics contract changed",
    )
    gates = protocol.get("gates", {})
    require(
        gates.get("each_split_separate") is True
        and gates.get("selected_minus_untrained_gcq_rec_ci95_lower_strictly_above_zero") is True
        and gates.get("selected_minus_untrained_gcq_giou_point_strictly_above_zero") is True
        and gates.get("selected_minus_untrained_gcq_precise_iou_point_strictly_above_zero") is True
        and gates.get("selected_parse_fail_max_increase") == PARSE_MARGIN,
        "grounding gate contract changed",
    )
    require(
        protocol.get("failure_policy")
        == {
            "scientific_gate_failure": (
                "write the complete summary and exit successfully; do not substitute another checkpoint"
            ),
            "integrity_failure": "do not write a scientific result; exit nonzero",
        },
        "grounding failure policy changed",
    )

    require(isinstance(meta, dict) and meta.get("schema_version") == 1,
            "RefCOCO+ metadata schema changed")
    require(meta.get("forbidden_image_overlap") == 0,
            "RefCOCO+ metadata reports forbidden-image overlap")
    require(meta.get("unique_confirmation_images") == 1_500,
            "RefCOCO+ combined image count changed")
    rows_by_split: dict[str, list[dict]] = {}
    inventories: dict[str, dict] = {}
    manifest_paths: dict[str, Path] = {}
    for split, spec in SPLITS.items():
        manifest_path = data / "subsets" / spec["manifest"]
        rows, inventory = validate_manifest(
            manifest_path, split, data, verify_images=verify_images
        )
        rows_by_split[split] = rows
        inventories[split] = inventory
        manifest_paths[split] = manifest_path
        meta_split = meta.get("splits", {}).get(split, {})
        require(
            meta_split.get("manifest") == spec["manifest"]
            and meta_split.get("manifest_sha256") == spec["manifest_sha256"]
            and meta_split.get("ordered_uid_sha256") == spec["ordered_uid_sha256"]
            and meta_split.get("expressions") == spec["expressions"]
            and meta_split.get("images") == spec["images"]
            and meta_split.get("source_arrow_sha256") == spec["source_arrow_sha256"],
            f"RefCOCO+ {split} metadata changed",
        )
        source_path = Path(meta_split.get("source_arrow", ""))
        require_hash(source_path, spec["source_arrow_sha256"], f"RefCOCO+ {split} source Arrow")
        auth_paths[f"source_arrow_{split}"] = source_path
    image_sets = {
        split: {int(row["image_id"]) for row in rows}
        for split, rows in rows_by_split.items()
    }
    require(not (image_sets["testA"] & image_sets["testB"]),
            "RefCOCO+ testA/testB confirmation images overlap")
    forbidden_images: set[int] = set()
    for label, (name, expected_hash) in FORBIDDEN_MANIFESTS.items():
        path = data / "subsets" / name
        require_hash(path, expected_hash, label)
        require(meta.get("forbidden_manifests", {}).get(name) == expected_hash,
                f"metadata does not bind {name}")
        rows = load_json(path)
        require(isinstance(rows, list), f"forbidden manifest is not a list: {path}")
        forbidden_images.update(int(row["image_id"]) for row in rows)
        auth_paths[label] = path
    require(
        not ((image_sets["testA"] | image_sets["testB"]) & forbidden_images),
        "RefCOCO+ confirmation overlaps recovery/profiling images",
    )
    auth_paths.update(
        {
            "grounding_protocol": protocol_path,
            "grounding_metadata": meta_path,
            "manifest_testA": manifest_paths["testA"],
            "manifest_testB": manifest_paths["testB"],
            "promotion": runs / "promote_gcq_b4.25.json",
        }
    )
    require_hash(auth_paths["promotion"], PROMOTION_SHA256, "GCQ promotion manifest")
    return {
        "schema_version": 1,
        "authorization_pass": True,
        "selected": candidate,
        "splits": {
            split: {
                "manifest": str(manifest_paths[split]),
                "manifest_sha256": SPLITS[split]["manifest_sha256"],
                "ordered_uid_sha256": SPLITS[split]["ordered_uid_sha256"],
                "expressions": SPLITS[split]["expressions"],
                "images": SPLITS[split]["images"],
                "image_inventory": inventories[split],
            }
            for split in SPLITS
        },
        "input_paths": {key: str(path) for key, path in auth_paths.items()},
    }


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {label}: {value!r}") from exc
    require(math.isfinite(result), f"non-finite value for {label}: {value!r}")
    return result


def iou_giou(a: list[float], b: list[float]) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0.0
    cx1, cy1 = min(ax1, bx1), min(ay1, by1)
    cx2, cy2 = max(ax2, bx2), max(ay2, by2)
    hull = (cx2 - cx1) * (cy2 - cy1)
    giou = iou - (hull - union) / hull if hull > 0 else iou
    return iou, giou


def load_prediction_rows(path: Path, manifest: list[dict]) -> list[dict]:
    require(path.is_file(), f"missing grounding prediction log: {path}")
    rows: list[dict] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.strip() != "", f"blank row in {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            require(isinstance(row, dict), f"non-object row in {path}:{line_number}")
            require(len(rows) < len(manifest), f"too many prediction rows in {path}")
            expected = manifest[len(rows)]
            uid = row.get("uid")
            require(uid == expected["uid"], f"UID/order mismatch in {path}:{line_number}")
            require(uid not in seen, f"duplicate UID {uid!r} in {path}")
            for field in ("image_id", "task", "source"):
                require(row.get(field) == expected.get(field),
                        f"{field} mismatch for {uid!r} in {path}")
            require(isinstance(row.get("pred_raw"), str), f"missing prediction for {uid!r}")
            box = row.get("box1000")
            require(
                box is None
                or (
                    isinstance(box, list)
                    and len(box) == 4
                    and all(isinstance(value, int) and not isinstance(value, bool) for value in box)
                    and box[0] < box[2]
                    and box[1] < box[3]
                ),
                f"invalid parsed box for {uid!r}",
            )
            reported_iou = _finite_float(row.get("iou"), f"{uid} IoU")
            reported_giou = _finite_float(row.get("giou"), f"{uid} GIoU")
            if box is None:
                expected_iou, expected_giou = 0.0, -1.0
            else:
                width, height = float(expected["width"]), float(expected["height"])
                prediction = [
                    box[0] * width / 1000.0,
                    box[1] * height / 1000.0,
                    box[2] * width / 1000.0,
                    box[3] * height / 1000.0,
                ]
                gt = [float(value) for value in expected["bbox_xywh"]]
                target = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]
                expected_iou, expected_giou = iou_giou(prediction, target)
            require_close(reported_iou, expected_iou, f"{uid} recomputed IoU")
            require_close(reported_giou, expected_giou, f"{uid} recomputed GIoU")
            require(isinstance(row.get("hit"), bool), f"invalid hit flag for {uid!r}")
            require(row["hit"] == (reported_iou >= 0.5), f"hit mismatch for {uid!r}")
            row["iou"] = reported_iou
            row["giou"] = reported_giou
            row["parse_fail"] = float(box is None)
            rows.append(row)
            seen.add(uid)
    require(len(rows) == len(manifest),
            f"expected {len(manifest)} rows in {path}, found {len(rows)}")
    return rows


def aggregate(rows: list[dict]) -> dict:
    require(rows, "cannot aggregate empty grounding rows")
    count = len(rows)
    threshold_accuracy = {
        f"{threshold:.2f}": sum(float(row["iou"] >= threshold) for row in rows) / count
        for threshold in IOU_THRESHOLDS
    }
    return {
        "n": count,
        "rec": threshold_accuracy["0.50"],
        "giou": sum(float(row["giou"]) for row in rows) / count,
        "precise_iou": sum(threshold_accuracy.values()) / len(threshold_accuracy),
        "parse_fail": sum(float(row["parse_fail"]) for row in rows) / count,
        "accuracy_by_iou": threshold_accuracy,
    }


def compare_reported_group(actual: dict, reported: object, label: str) -> None:
    require(isinstance(reported, dict), f"missing reported group {label}")
    require(reported.get("n") == actual["n"], f"{label} count changed")
    mapping = {
        "rec": "acc_iou_0.5",
        "giou": "mean_giou",
        "precise_iou": "mean_acc_iou_0.50_0.95",
        "parse_fail": "parse_fail",
    }
    for key, reported_key in mapping.items():
        require_close(float(reported.get(reported_key)), float(actual[key]), f"{label} {key}")
    reported_thresholds = reported.get("accuracy_by_iou")
    require(isinstance(reported_thresholds, dict), f"missing threshold metrics for {label}")
    require(set(reported_thresholds) == set(actual["accuracy_by_iou"]),
            f"threshold keys changed for {label}")
    for threshold, value in actual["accuracy_by_iou"].items():
        require_close(float(reported_thresholds[threshold]), value,
                      f"{label} accuracy@{threshold}")


def validate_metrics(path: Path, rows: list[dict], *, split: str, tag: str) -> tuple[dict, dict]:
    metrics = load_json(path)
    spec = SPLITS[split]
    expected = {
        "tag": tag,
        "model": BASE_MODEL,
        "subset": spec["subset"],
        "n": spec["expressions"],
        "base_revision": BASE_REVISION,
    }
    for key, value in expected.items():
        require(metrics.get(key) == value, f"unexpected {key} in {path}")
    scores = aggregate(rows)
    overall_reported = {
        "n": metrics.get("n"),
        "acc_iou_0.5": metrics.get("acc_iou_0.5"),
        "mean_giou": metrics.get("mean_giou"),
        "mean_acc_iou_0.50_0.95": metrics.get("mean_acc_iou_0.50_0.95"),
        "parse_fail": metrics.get("parse_fail"),
        "accuracy_by_iou": metrics.get("accuracy_by_iou"),
    }
    compare_reported_group(scores, overall_reported, f"{path} overall")
    compare_reported_group(scores, metrics.get("by_task", {}).get("rec"), f"{path} task rec")
    compare_reported_group(
        scores,
        metrics.get("by_source", {}).get("refcocoplus"),
        f"{path} source refcocoplus",
    )
    return metrics, {key: scores[key] for key in ("n", *METRICS)}


def example_metric(row: dict, metric: str) -> float:
    if metric == "rec":
        return float(row["iou"] >= 0.5)
    if metric == "giou":
        return float(row["giou"])
    if metric == "precise_iou":
        return precise_iou_score(float(row["iou"]))
    if metric == "parse_fail":
        return float(row["parse_fail"])
    raise ValueError(f"unknown metric: {metric}")


def paired_image_delta(
    reference: list[dict],
    candidate: list[dict],
    metric: str,
    *,
    resamples: int,
    seed: int,
) -> dict:
    require(len(reference) == len(candidate), "paired result counts differ")
    clusters: dict[int, list[float]] = defaultdict(list)
    for reference_row, candidate_row in zip(reference, candidate):
        require(reference_row["uid"] == candidate_row["uid"], "paired UID/order mismatch")
        require(reference_row["image_id"] == candidate_row["image_id"],
                "paired image metadata mismatch")
        clusters[int(reference_row["image_id"])].append(
            example_metric(candidate_row, metric) - example_metric(reference_row, metric)
        )
    require(clusters, "paired bootstrap has no image clusters")
    cluster_sums = np.asarray([sum(values) for values in clusters.values()], dtype=np.float64)
    cluster_counts = np.asarray([len(values) for values in clusters.values()], dtype=np.int64)
    observed = float(cluster_sums.sum() / cluster_counts.sum())
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=np.float64)
    cluster_count = len(cluster_sums)
    for start in range(0, resamples, 512):
        end = min(start + 512, resamples)
        chosen = rng.integers(0, cluster_count, size=(end - start, cluster_count))
        bootstrap[start:end] = (
            cluster_sums[chosen].sum(axis=1) / cluster_counts[chosen].sum(axis=1)
        )
    return {
        "observed": observed,
        "ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "n_expressions": len(reference),
        "n_images": len(clusters),
        "resamples": resamples,
        "seed": seed,
    }


def gates_for_split(comparisons: dict) -> dict[str, bool]:
    selected = comparisons["selected_adapter_minus_untrained_gcq4.25"]
    return {
        "selected_minus_untrained_GCQ_REC_CI95_lower_strictly_above_zero": (
            float(selected["rec"]["ci95"][0]) > 0.0
        ),
        "selected_minus_untrained_GCQ_GIoU_point_strictly_above_zero": (
            float(selected["giou"]["observed"]) > 0.0
        ),
        "selected_minus_untrained_GCQ_precise_IoU_point_strictly_above_zero": (
            float(selected["precise_iou"]["observed"]) > 0.0
        ),
        "selected_parse_fail_increase_at_most_0.5pt": (
            float(selected["parse_fail"]["observed"]) <= PARSE_MARGIN
        ),
    }


def validate_runtime_provenance(
    path: Path,
    launch_path: Path,
    split: str,
    candidate: dict,
) -> dict:
    provenance = load_json(path)
    expected = {
        "schema_version": 1,
        "evaluation_role": "one-time frozen RefCOCO+ grounding confirmation",
        "recipe_id": RECIPE_ID,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "split": split,
        "scheduler_array_task": SPLITS[split]["task_id"],
        "grounding_launch_manifest": str(launch_path),
        "grounding_launch_manifest_sha256": sha256_file(launch_path),
        "selected": candidate,
        "arm_order": list(ARMS),
        "hardware_contract": "exactly one visible NVIDIA L40S",
        "hardware_gate_pass": True,
    }
    for key, value in expected.items():
        require(provenance.get(key) == value, f"runtime provenance {key} changed in {path}")
    scheduler = provenance.get("scheduler", {})
    require(
        isinstance(scheduler.get("job_id"), str) and scheduler["job_id"]
        and scheduler.get("task_id") == str(SPLITS[split]["task_id"])
        and isinstance(scheduler.get("hostname"), str) and scheduler["hostname"]
        and isinstance(scheduler.get("queue"), str)
        and "l40s" in scheduler["queue"].lower(),
        f"invalid scheduler provenance in {path}",
    )
    python = provenance.get("python", {})
    require(
        isinstance(python.get("executable"), str)
        and python["executable"]
        and isinstance(python.get("version"), str)
        and python["version"],
        f"invalid Python provenance in {path}",
    )
    packages = provenance.get("packages")
    require(isinstance(packages, dict), f"missing package provenance in {path}")
    for package in ("torch", "transformers", "peft", "safetensors", "numpy", "pillow"):
        require(isinstance(packages.get(package), str) and packages[package],
                f"missing {package} version in {path}")
    cuda = provenance.get("cuda", {})
    devices = cuda.get("devices")
    require(
        cuda.get("available") is True
        and cuda.get("device_count") == 1
        and isinstance(cuda.get("visible_devices"), str)
        and cuda["visible_devices"]
        and isinstance(devices, list)
        and len(devices) == 1,
        f"runtime did not expose exactly one CUDA device in {path}",
    )
    device = devices[0]
    require(device.get("index") == 0 and "L40S" in str(device.get("name", "")),
            f"non-L40S device in {path}")
    capability = device.get("compute_capability")
    require(
        isinstance(capability, list)
        and len(capability) == 2
        and tuple(capability) >= (8, 0),
        f"invalid CUDA compute capability in {path}: {capability!r}",
    )
    require(int(device.get("total_memory_bytes", 0)) >= 40 * 2**30,
            f"CUDA device has less than 40 GiB in {path}")
    require(isinstance(device.get("uuid"), str) and device["uuid"],
            f"missing physical GPU UUID in {path}")
    require(isinstance(cuda.get("driver_versions"), list) and cuda["driver_versions"],
            f"missing driver provenance in {path}")
    storage = provenance.get("storage_headroom")
    require(isinstance(storage, dict), f"missing storage provenance in {path}")
    require(
        storage.get("minimum_available_bytes") == MINIMUM_STORAGE_HEADROOM_BYTES
        and storage.get("gate_pass") is True,
        f"storage-headroom gate changed in {path}",
    )
    probes = storage.get("probes")
    require(
        isinstance(probes, dict) and set(probes) == {"output", "model_cache"},
        f"storage-headroom probe inventory changed in {path}",
    )
    launch = load_json(launch_path)
    launcher_probes = launch.get("storage_headroom", {}).get("launcher_probes", {})
    for label, probe in probes.items():
        require(isinstance(probe, dict), f"invalid runtime {label} storage probe")
        available = probe.get("available_bytes")
        require(
            probe.get("path") == launcher_probes.get(label, {}).get("path")
            and isinstance(available, int)
            and not isinstance(available, bool)
            and available >= MINIMUM_STORAGE_HEADROOM_BYTES,
            f"runtime {label} storage headroom was insufficient in {path}",
        )
    return provenance


def expected_arm_config(arm: str, candidate: dict) -> dict:
    return {
        "bf16": {
            "rtn_bits": 0,
            "rtn_group": 128,
            "promotion_sha256": None,
            "adapter_sha256": None,
        },
        "uniform_rtn_w4": {
            "rtn_bits": 4,
            "rtn_group": 128,
            "promotion_sha256": None,
            "adapter_sha256": None,
        },
        "untrained_gcq4.25": {
            "rtn_bits": 4,
            "rtn_group": 128,
            "promotion_sha256": PROMOTION_SHA256,
            "adapter_sha256": None,
        },
        "selected_adapter": {
            "rtn_bits": 4,
            "rtn_group": 128,
            "promotion_sha256": PROMOTION_SHA256,
            "adapter_sha256": candidate["adapter_sha256"],
        },
    }[arm]


def validate_execution_log(
    path: Path,
    launch_path: Path,
    split: str,
    candidate: dict,
    output_dir: Path,
    gpu_uuid: str,
) -> list[dict]:
    require(path.is_file(), f"missing arm execution log: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.strip(), f"blank arm execution row in {path}:{line_number}")
            rows.append(json.loads(line))
    require(len(rows) == len(ARMS), f"expected four arm records in {path}")
    launch_sha = sha256_file(launch_path)
    for index, (arm, record) in enumerate(zip(ARMS, rows), 1):
        tag = ARM_TAG_TEMPLATES[arm].format(split=split)
        jsonl = output_dir / f"{tag}.rec.jsonl"
        metrics = output_dir / f"{tag}.rec.metrics.json"
        expected = {
            "sequence_index": index,
            "arm": arm,
            "tag": tag,
            "split": split,
            "launch_sha256": launch_sha,
            "gpu_uuid": gpu_uuid,
            "configuration": expected_arm_config(arm, candidate),
            "output_hashes": {
                "rec_jsonl": sha256_file(jsonl),
                "rec_metrics": sha256_file(metrics),
            },
        }
        require(record == expected, f"arm execution record changed at {path}:{index}")
    return rows


def validate_results_csv(path: Path, split: str, arm_scores: dict[str, dict]) -> list[str]:
    require(path.is_file(), f"missing grounding results CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames
            == [
                "tag", "model", "task", "subset", "n", "acc", "mean_giou",
                "parse_fail", "acc_small", "acc_medium", "acc_large",
                "blank_image", "seconds",
            ],
            f"grounding results CSV schema changed for {split}",
        )
        rows = list(reader)
    expected_tags = [ARM_TAG_TEMPLATES[arm].format(split=split) for arm in ARMS]
    require([row.get("tag") for row in rows] == expected_tags,
            f"grounding arms ran in the wrong order for {split}")
    for arm, row in zip(ARMS, rows):
        require(row.get("model") == BASE_MODEL and row.get("task") == "rec",
                f"invalid results CSV identity for {split}")
        require(row.get("subset") == SPLITS[split]["subset"],
                f"invalid results CSV subset for {split}")
        require(row.get("n") == str(SPLITS[split]["expressions"]),
                f"invalid results CSV count for {split}")
        require(row.get("blank_image") == "False",
                f"grounding CSV reports a blank-image run for {split} {arm}")
        try:
            rec = float(row["acc"])
            giou = float(row["mean_giou"])
            parse_fail = float(row["parse_fail"])
            seconds = int(row["seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid grounding CSV numeric field for {split} {arm}") from exc
        for label, actual, expected in (
            ("REC", rec, arm_scores[arm]["rec"]),
            ("GIoU", giou, arm_scores[arm]["giou"]),
            ("parse-fail", parse_fail, arm_scores[arm]["parse_fail"]),
        ):
            require(
                math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=5.1e-5),
                f"grounding CSV {label} disagrees with recomputed rows for {split} {arm}",
            )
        require(seconds >= 0, f"negative grounding runtime for {split} {arm}")
    return expected_tags


def validate_completion_marker(
    path: Path,
    *,
    split: str,
    launch_path: Path,
    provenance_path: Path,
    execution_path: Path,
    results_path: Path,
) -> dict:
    marker = load_json(path)
    expected = {
        "schema_version": 1,
        "split": split,
        "grounding_launch_manifest_sha256": sha256_file(launch_path),
        "runtime_provenance_sha256": sha256_file(provenance_path),
        "arm_execution_sha256": sha256_file(execution_path),
        "results_csv_sha256": sha256_file(results_path),
        "all_four_arms_complete": True,
    }
    require(marker == expected, f"task completion marker changed for {split}")
    return marker


def expected_launch_paths(
    runs: Path,
    data: Path,
    code_dir: Path,
    candidate: dict,
    preflight_report: dict,
) -> dict[str, Path]:
    root = runs / "recovery_vqa_replay"
    paths = {key: Path(value) for key, value in preflight_report["input_paths"].items()}
    paths.update(
        {
            "environment": code_dir / "env.sh",
            "eval_rec": code_dir / "eval_rec.py",
            "recovery_utils": code_dir / "recovery_utils.py",
            "quant_utils": code_dir / "quant_utils.py",
            "gcq_patches": code_dir / "gcq_patches.py",
            "grounding_builder": code_dir / "build_refcocoplus_confirmation.py",
            "grounding_launcher": code_dir / "launch_recovery_grounding_confirmation.sh",
            "grounding_batch": code_dir / "batch_recovery_grounding_confirmation.sh",
            "grounding_summarizer": Path(__file__).resolve(),
        }
    )
    require(paths["selected_adapter"] == Path(candidate["adapter_dir"]) / "adapter_model.safetensors",
            "preflight selected adapter path changed")
    return paths


def verify_launch_and_inputs(
    runs: Path,
    data: Path,
    code_dir: Path,
) -> tuple[dict, dict, dict[str, Path]]:
    root = runs / "recovery_vqa_replay"
    launch_path = root / "grounding_confirmation" / "grounding_confirmation_launch_manifest.json"
    require_read_only(launch_path, "grounding launch manifest")
    launch = load_json(launch_path)
    preflight_report = preflight(runs, data, code_dir, verify_images=True)
    candidate = preflight_report["selected"]
    expected_top = {
        "schema_version": 1,
        "evaluation_role": "one-time frozen RefCOCO+ grounding confirmation",
        "recipe_id": RECIPE_ID,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "selected": candidate,
        "arm_order": list(ARMS),
        "scheduler_array_tasks": {"1": "testA", "2": "testB"},
        "evaluation": {
            "hardware": "one NVIDIA L40S per split",
            "within_task_execution": "all four arms sequentially on the same physical GPU",
            "max_pixels": 1_003_520,
            "batch_size": 16,
            "device": "cuda:0",
            "decoding": "greedy",
        },
        "bootstrap": {
            "unit": "paired referring expressions clustered by COCO image",
            "resamples": BOOTSTRAP_RESAMPLES,
            "base_seed": BOOTSTRAP_BASE_SEED,
            "seed_mapping": (
                "base_seed + split_index*12 + comparison_index*4 + metric_index; "
                "split order testA,testB; comparison/metric order as stored"
            ),
        },
        "failure_policy": {
            "scientific_gate_failure": "publish complete FAIL summary and exit zero",
            "integrity_failure": "publish no scientific summary and exit nonzero",
            "checkpoint_substitution_after_failure": False,
        },
        "no_peeking": {
            "manifest_frozen_before_selection": True,
            "model_predictions_and_metrics_unseen_before_selection": True,
            "expected_output_tags_absent_when_launch_frozen": True,
            "alternate_tag_identity_scan_before_launch": True,
            "recheck_each_split_at_job_start": True,
            "globally_image_unseen_claim": False,
        },
    }
    for key, value in expected_top.items():
        require(launch.get(key) == value, f"grounding launch {key} changed")
    untouched_audit = launch.get("untouched_audit")
    require(isinstance(untouched_audit, dict), "grounding untouched audit is missing")
    require(
        untouched_audit.get("manifest_sha256")
        == {split: SPLITS[split]["manifest_sha256"] for split in SPLITS}
        and untouched_audit.get("splits") == list(SPLITS)
        and untouched_audit.get("fixed_tag_outputs_found") == 0
        and untouched_audit.get("matching_subset_metrics_found") == 0
        and untouched_audit.get("confirmation_uid_prediction_logs_found") == 0
        and untouched_audit.get("checked_before_launch") is True
        and untouched_audit.get("recheck_each_split_at_job_start") is True,
        "grounding untouched-audit contract changed",
    )
    for key in ("rec_metrics_files_scanned", "rec_prediction_files_scanned"):
        require(
            isinstance(untouched_audit.get(key), int)
            and not isinstance(untouched_audit[key], bool)
            and untouched_audit[key] >= 0,
            f"grounding untouched-audit {key} is invalid",
        )
    storage = launch.get("storage_headroom")
    require(isinstance(storage, dict), "grounding storage-headroom contract is missing")
    require(
        storage.get("minimum_available_bytes") == MINIMUM_STORAGE_HEADROOM_BYTES
        and storage.get("recheck_at_job_start") is True,
        "grounding storage-headroom policy changed",
    )
    probes = storage.get("launcher_probes")
    require(
        isinstance(probes, dict) and set(probes) == {"output", "model_cache"},
        "grounding launcher storage probe inventory changed",
    )
    expected_probe_paths = {
        "output": str((runs / "recovery_vqa_replay").resolve()),
        "model_cache": str(Path(os.environ["HF_HOME"]).resolve()),
    }
    for label, probe in probes.items():
        require(isinstance(probe, dict), f"invalid grounding {label} storage probe")
        available = probe.get("available_bytes")
        require(
            probe.get("path") == expected_probe_paths[label]
            and isinstance(available, int)
            and not isinstance(available, bool)
            and available >= MINIMUM_STORAGE_HEADROOM_BYTES,
            f"grounding launcher {label} storage probe changed",
        )
    launched_splits = launch.get("splits")
    require(isinstance(launched_splits, dict) and set(launched_splits) == set(SPLITS),
            "grounding launch split inventory changed")
    for split, spec in SPLITS.items():
        expected = {
            "task_id": spec["task_id"],
            "subset": spec["subset"],
            "manifest": preflight_report["splits"][split]["manifest"],
            "manifest_sha256": spec["manifest_sha256"],
            "ordered_uid_sha256": spec["ordered_uid_sha256"],
            "expressions": spec["expressions"],
            "images": spec["images"],
            "image_inventory": preflight_report["splits"][split]["image_inventory"],
            "tags": {
                arm: ARM_TAG_TEMPLATES[arm].format(split=split) for arm in ARMS
            },
        }
        require(launched_splits.get(split) == expected, f"launch contract changed for {split}")
    paths = expected_launch_paths(runs, data, code_dir, candidate, preflight_report)
    require(
        launch.get("paths") == {key: str(path) for key, path in paths.items()},
        "grounding launch path inventory changed",
    )
    hashes = launch.get("hashes")
    require(isinstance(hashes, dict) and set(hashes) == set(paths),
            "grounding launch hash inventory changed")
    for label, path in paths.items():
        require_hash(path, hashes[label], label)
    require(hashes["grounding_protocol"] == GROUND_PROTOCOL_SHA256,
            "grounding launch protocol hash changed")
    require(hashes["promotion"] == PROMOTION_SHA256,
            "grounding launch promotion hash changed")
    require(hashes["selected_adapter"] == candidate["adapter_sha256"],
            "grounding launch adapter hash changed")
    require(hashes["selected_adapter_config"] == candidate["adapter_config_sha256"],
            "grounding launch adapter-config hash changed")
    require(hashes["selected_adapter_manifest"] == candidate["manifest_sha256"],
            "grounding launch adapter-manifest hash changed")
    return launch, preflight_report, paths


def publish_summary(path: Path, summary: dict) -> None:
    require(
        summary.get("integrity_validation_pass") is True,
        "refusing to publish a grounding summary without passed integrity validation",
    )
    require(
        isinstance(summary.get("confirmation_pass"), bool),
        "grounding summary is missing its Boolean scientific outcome",
    )
    expected_outcome = "PASS" if summary["confirmation_pass"] else "FAIL"
    require(
        summary.get("scientific_outcome") == expected_outcome,
        "grounding Boolean and named scientific outcomes disagree",
    )
    require(not path.exists(), f"refusing to overwrite grounding summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".grounding_confirmation_summary.", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        os.chmod(path, 0o444)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--preflight",
        action="store_true",
        help="validate frozen inputs/authorization only; never read predictions",
    )
    modes.add_argument(
        "--audit-pristine",
        action="store_true",
        help="scan only output identities for evidence of a prior confirmation run",
    )
    parser.add_argument(
        "--audit-split",
        choices=("all", *SPLITS),
        help="split identity to scan with --audit-pristine",
    )
    parser.add_argument(
        "--skip-image-decode",
        action="store_true",
        help="test-only speed option for preflight; launchers must not use it",
    )
    args = parser.parse_args()
    runs = Path(os.environ["GCQ_RUNS"])
    data = Path(os.environ["GCQ_DATA"])
    code_dir = Path(__file__).resolve().parent
    if args.audit_pristine:
        require(args.audit_split is not None,
                "--audit-pristine requires --audit-split")
        require(not args.skip_image_decode,
                "--skip-image-decode is invalid with --audit-pristine")
        requested = tuple(SPLITS) if args.audit_split == "all" else (args.audit_split,)
        report = audit_prior_grounding_predictions(
            runs,
            {
                split: data / "subsets" / SPLITS[split]["manifest"]
                for split in requested
            },
            requested_splits=requested,
        )
        print(json.dumps(report, sort_keys=True))
        return
    require(args.audit_split is None,
            "--audit-split requires --audit-pristine")
    if args.preflight:
        report = preflight(
            runs, data, code_dir, verify_images=not args.skip_image_decode
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    require(not args.skip_image_decode, "--skip-image-decode is valid only with --preflight")

    root = runs / "recovery_vqa_replay"
    confirmation_root = root / "grounding_confirmation"
    launch_path = confirmation_root / "grounding_confirmation_launch_manifest.json"
    output = root / "grounding_confirmation_summary.json"
    launch, preflight_report, _ = verify_launch_and_inputs(runs, data, code_dir)
    candidate = preflight_report["selected"]
    summary_splits: dict[str, dict] = {}
    output_hashes: dict[str, dict] = {}
    for split_index, (split, spec) in enumerate(SPLITS.items()):
        manifest = load_json(Path(preflight_report["splits"][split]["manifest"]))
        output_dir = confirmation_root / split
        expected_output_names = {
            "runtime_provenance.json",
            "arm_execution.jsonl",
            "results.csv",
            "task_complete.json",
            *{
                f"{ARM_TAG_TEMPLATES[arm].format(split=split)}{suffix}"
                for arm in ARMS
                for suffix in (".rec.jsonl", ".rec.metrics.json")
            },
        }
        require(output_dir.is_dir(), f"missing grounding output directory: {output_dir}")
        actual_output_names = {path.name for path in output_dir.iterdir()}
        require(
            actual_output_names == expected_output_names,
            f"grounding output inventory changed for {split}: "
            f"missing={sorted(expected_output_names - actual_output_names)}, "
            f"unexpected={sorted(actual_output_names - expected_output_names)}",
        )
        provenance_path = output_dir / "runtime_provenance.json"
        execution_path = output_dir / "arm_execution.jsonl"
        provenance = validate_runtime_provenance(
            provenance_path, launch_path, split, candidate
        )
        gpu_uuid = provenance["cuda"]["devices"][0]["uuid"]
        arm_rows: dict[str, list[dict]] = {}
        arm_scores: dict[str, dict] = {}
        arm_hashes: dict[str, dict] = {}
        for arm in ARMS:
            tag = ARM_TAG_TEMPLATES[arm].format(split=split)
            prediction_path = output_dir / f"{tag}.rec.jsonl"
            metrics_path = output_dir / f"{tag}.rec.metrics.json"
            rows = load_prediction_rows(prediction_path, manifest)
            _, scores = validate_metrics(metrics_path, rows, split=split, tag=tag)
            arm_rows[arm] = rows
            arm_scores[arm] = scores
            arm_hashes[arm] = {
                "rec_jsonl": sha256_file(prediction_path),
                "rec_metrics": sha256_file(metrics_path),
            }
        execution = validate_execution_log(
            execution_path,
            launch_path,
            split,
            candidate,
            output_dir,
            gpu_uuid,
        )
        results_csv = output_dir / "results.csv"
        evaluation_order = validate_results_csv(results_csv, split, arm_scores)
        completion_path = output_dir / "task_complete.json"
        completion = validate_completion_marker(
            completion_path,
            split=split,
            launch_path=launch_path,
            provenance_path=provenance_path,
            execution_path=execution_path,
            results_path=results_csv,
        )
        comparisons: dict[str, dict] = {}
        for comparison_index, (name, (candidate_arm, reference_arm)) in enumerate(
            COMPARISONS.items()
        ):
            comparisons[name] = {}
            for metric_index, metric in enumerate(METRICS):
                seed = (
                    BOOTSTRAP_BASE_SEED
                    + split_index * len(COMPARISONS) * len(METRICS)
                    + comparison_index * len(METRICS)
                    + metric_index
                )
                paired = paired_image_delta(
                    arm_rows[reference_arm],
                    arm_rows[candidate_arm],
                    metric,
                    resamples=BOOTSTRAP_RESAMPLES,
                    seed=seed,
                )
                require_close(
                    paired["observed"],
                    arm_scores[candidate_arm][metric] - arm_scores[reference_arm][metric],
                    f"{split} {name} {metric} point delta",
                )
                comparisons[name][metric] = paired
        gates = gates_for_split(comparisons)
        split_pass = all(gates.values())
        summary_splits[split] = {
            "manifest": preflight_report["splits"][split],
            "runtime_provenance": provenance,
            "arm_execution": execution,
            "task_completion": completion,
            "evaluation_order": evaluation_order,
            "scores": arm_scores,
            "paired_comparisons": comparisons,
            "gates": gates,
            "split_pass": split_pass,
        }
        output_hashes[split] = {
            "runtime_provenance": sha256_file(provenance_path),
            "arm_execution": sha256_file(execution_path),
            "results_csv": sha256_file(results_csv),
            "task_completion": sha256_file(completion_path),
            "arms": arm_hashes,
        }

    provenance_a = summary_splits["testA"]["runtime_provenance"]
    provenance_b = summary_splits["testB"]["runtime_provenance"]
    require(
        provenance_a["scheduler"]["job_id"] == provenance_b["scheduler"]["job_id"],
        "testA/testB did not run as tasks of the same frozen scheduler array",
    )
    software_signature_a = {
        "python": provenance_a["python"],
        "packages": provenance_a["packages"],
        "cuda_runtime_version": provenance_a["cuda"].get("runtime_version"),
        "cudnn_version": provenance_a["cuda"].get("cudnn_version"),
        "driver_versions": provenance_a["cuda"].get("driver_versions"),
        "device_name": provenance_a["cuda"]["devices"][0].get("name"),
        "compute_capability": provenance_a["cuda"]["devices"][0].get(
            "compute_capability"
        ),
        "total_memory_bytes": provenance_a["cuda"]["devices"][0].get(
            "total_memory_bytes"
        ),
    }
    software_signature_b = {
        "python": provenance_b["python"],
        "packages": provenance_b["packages"],
        "cuda_runtime_version": provenance_b["cuda"].get("runtime_version"),
        "cudnn_version": provenance_b["cuda"].get("cudnn_version"),
        "driver_versions": provenance_b["cuda"].get("driver_versions"),
        "device_name": provenance_b["cuda"]["devices"][0].get("name"),
        "compute_capability": provenance_b["cuda"]["devices"][0].get(
            "compute_capability"
        ),
        "total_memory_bytes": provenance_b["cuda"]["devices"][0].get(
            "total_memory_bytes"
        ),
    }
    require(
        software_signature_a == software_signature_b,
        "testA/testB software or L40S model signatures differ",
    )
    confirmation_pass = all(value["split_pass"] for value in summary_splits.values())
    summary = {
        "schema_version": 1,
        "evaluation_role": "one-time frozen RefCOCO+ grounding confirmation",
        "recipe_id": RECIPE_ID,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "grounding_launch_manifest": str(launch_path),
        "grounding_launch_manifest_sha256": sha256_file(launch_path),
        "grounding_protocol_sha256": GROUND_PROTOCOL_SHA256,
        "vqa_confirmation_summary": str(
            Path(preflight_report["input_paths"]["vqa_confirmation_summary"])
        ),
        "vqa_confirmation_summary_sha256": launch["hashes"]["vqa_confirmation_summary"],
        "selected": candidate,
        "scientific_status": {
            "manifest_frozen_before_selection": True,
            "model_predictions_and_metrics_unseen_before_selection": True,
            "globally_image_unseen_claim": False,
            "interpretation": (
                "untouched RefCOCO+ expressions/results; COCO image partitions overlap "
                "the earlier RefCOCO test evaluation"
            ),
        },
        "bootstrap": launch["bootstrap"],
        "frozen_gates": {
            "each_split_separate": True,
            "selected_minus_untrained_gcq_rec_ci95_lower_strictly_above_zero": True,
            "selected_minus_untrained_gcq_giou_point_strictly_above_zero": True,
            "selected_minus_untrained_gcq_precise_iou_point_strictly_above_zero": True,
            "selected_parse_fail_max_increase": PARSE_MARGIN,
        },
        "splits": summary_splits,
        "cross_split_runtime": {
            "same_scheduler_array_job": True,
            "identical_software_and_gpu_model_signature": True,
            "signature": software_signature_a,
        },
        "confirmation_pass": confirmation_pass,
        "recovery_pipeline_pass": confirmation_pass,
        "scientific_outcome": "PASS" if confirmation_pass else "FAIL",
        "integrity_validation_pass": True,
        "output_hashes": output_hashes,
        "next_if_pass": "report the frozen grounding confirmation; do not tune on it",
        "next_if_fail": (
            "report the frozen grounding failure; do not substitute another checkpoint"
        ),
    }
    publish_summary(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"GROUNDING CONFIRMATION SUMMARY: {output}")
    print(f"SCIENTIFIC OUTCOME: {'PASS' if confirmation_pass else 'FAIL'}")
    # A valid scientific FAIL deliberately returns normally. Integrity failures
    # above raise before publication and therefore exit nonzero.


if __name__ == "__main__":
    main()

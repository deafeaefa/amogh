"""Audit and select the balanced VQA-replay recovery checkpoint.

This script is deliberately development-only.  It verifies the frozen launch
manifest and every baseline/checkpoint artifact, checks that all predictions
have exactly the manifest UIDs in manifest order, recomputes the reported
scores, applies the predeclared gates, and writes ``development_summary.json``.
It never reads or evaluates the fresh confirmation set.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from recovery_utils import BASE_MODEL, BASE_REVISION, precise_iou_score, sha256_file
from summarize_recovery_checkpoint_sweep import load_vqa_rows, mean_vqa
from summarize_recovery_pilot import (
    group_scores,
    load_jsonl_unique,
    paired_cluster_contrast,
)
from summarize_recovery_selected_eval import paired_image_delta, require_same_uids


RECIPE_ID = "gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
CANDIDATE_STEPS = (200, 300, 400, 500, 600, 750)
REC_EXAMPLES = 1_000
PRIMARY_REC_EXAMPLES = 750
VQA_EXAMPLES = 5_000
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260850


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON file: {path}")
    with open(path) as f:
        return json.load(f)


def require_close(actual: float, expected: float, label: str, *, atol: float = 1e-12) -> None:
    require(math.isfinite(actual), f"{label} is not finite: {actual!r}")
    require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol),
        f"{label} changed: {actual!r} != {expected!r}",
    )


def require_hash(path: Path, expected: object, label: str) -> str:
    require(isinstance(expected, str) and len(expected) == 64,
            f"missing or invalid frozen hash for {label}")
    require(path.is_file(), f"missing hash-locked file for {label}: {path}")
    actual = sha256_file(path)
    require(actual == expected, f"{label} changed after development launch: {path}")
    return actual


def validate_runtime_provenance(path: Path, launch_path: Path, step: int) -> dict:
    provenance = load_json(path)
    require(isinstance(provenance, dict), f"runtime provenance is not an object: {path}")
    for key, expected in {
        "schema_version": 1,
        "recipe_id": RECIPE_ID,
        "checkpoint_step": step,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "development_launch_manifest": str(launch_path),
        "development_launch_manifest_sha256": sha256_file(launch_path),
        "hardware_contract": "exactly one visible NVIDIA L40S",
        "hardware_gate_pass": True,
    }.items():
        require(provenance.get(key) == expected,
                f"runtime provenance {key} mismatch in {path}: "
                f"{provenance.get(key)!r} != {expected!r}")
    scheduler = provenance.get("scheduler")
    require(isinstance(scheduler, dict), f"missing scheduler provenance in {path}")
    require(isinstance(scheduler.get("job_id"), str) and scheduler["job_id"],
            f"missing scheduler job ID in {path}")
    expected_task_id = str(CANDIDATE_STEPS.index(step) + 1)
    require(scheduler.get("task_id") == expected_task_id,
            f"wrong scheduler task ID in {path}: {scheduler.get('task_id')!r}")
    require(isinstance(scheduler.get("hostname"), str) and scheduler["hostname"],
            f"missing scheduler hostname in {path}")
    python = provenance.get("python")
    require(isinstance(python, dict), f"missing Python provenance in {path}")
    for key in ("executable", "version"):
        require(isinstance(python.get(key), str) and python[key],
                f"missing Python {key} in {path}")
    packages = provenance.get("packages")
    require(isinstance(packages, dict), f"missing package provenance in {path}")
    for package in ("torch", "transformers", "peft", "safetensors", "numpy"):
        require(isinstance(packages.get(package), str) and packages[package],
                f"missing {package} version in {path}")
    cuda = provenance.get("cuda")
    require(isinstance(cuda, dict), f"missing CUDA provenance in {path}")
    require(cuda.get("available") is True and cuda.get("device_count") == 1,
            f"runtime did not expose exactly one CUDA device in {path}")
    require(isinstance(cuda.get("visible_devices"), str) and cuda["visible_devices"],
            f"CUDA_VISIBLE_DEVICES is missing in {path}")
    driver_versions = cuda.get("driver_versions")
    require(isinstance(driver_versions, list) and driver_versions
            and all(isinstance(version, str) and version for version in driver_versions),
            f"NVIDIA driver provenance is missing in {path}")
    devices = cuda.get("devices")
    require(isinstance(devices, list) and len(devices) == 1,
            f"invalid CUDA device inventory in {path}")
    device = devices[0]
    require(device.get("index") == 0 and "L40S" in str(device.get("name", "")),
            f"non-L40S CUDA device in {path}: {device!r}")
    capability = device.get("compute_capability")
    require(isinstance(capability, list) and len(capability) == 2
            and tuple(capability) >= (8, 0),
            f"invalid CUDA compute capability in {path}: {capability!r}")
    require(int(device.get("total_memory_bytes", 0)) >= 40 * 2**30,
            f"CUDA device has less than 40 GiB in {path}")
    return provenance


def load_rec_manifest(path: Path) -> list[dict]:
    rows = load_json(path)
    require(isinstance(rows, list), f"REC manifest is not a JSON list: {path}")
    require(len(rows) == REC_EXAMPLES,
            f"expected {REC_EXAMPLES} REC manifest rows in {path}, found {len(rows)}")
    seen: set[str] = set()
    primary = 0
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"invalid REC manifest row {index} in {path}")
        uid = row.get("uid")
        require(isinstance(uid, str) and uid, f"missing REC UID at {path}:{index}")
        require(uid not in seen, f"duplicate REC manifest UID {uid!r} in {path}")
        require(row.get("image_id") is not None, f"missing image_id for REC UID {uid!r}")
        require(row.get("task") in {"rec", "coco_grounding"},
                f"invalid task for REC UID {uid!r}")
        require(isinstance(row.get("source"), str) and row["source"],
                f"missing source for REC UID {uid!r}")
        seen.add(uid)
        primary += int(row["task"] == "rec")
    require(primary == PRIMARY_REC_EXAMPLES,
            f"expected {PRIMARY_REC_EXAMPLES} primary REC rows, found {primary}")
    return rows


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {label}: {value!r}") from exc
    require(math.isfinite(result), f"non-finite numeric value for {label}: {value!r}")
    return result


def load_rec_rows(path: Path, manifest: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Load REC rows while preserving and enforcing the frozen manifest order."""
    require(path.is_file(), f"missing REC prediction log: {path}")
    ordered: list[dict] = []
    seen: set[str] = set()
    with open(path) as f:
        for line_number, line in enumerate(f, 1):
            require(line.strip() != "", f"blank REC row in {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            require(isinstance(row, dict), f"non-object REC row in {path}:{line_number}")
            uid = row.get("uid")
            require(isinstance(uid, str) and uid,
                    f"missing REC UID in {path}:{line_number}")
            require(uid not in seen, f"duplicate REC UID {uid!r} in {path}")
            require(len(ordered) < len(manifest),
                    f"too many REC rows in {path}; first extra UID is {uid!r}")
            expected = manifest[len(ordered)]
            require(uid == expected["uid"],
                    f"REC UID/order mismatch in {path}:{line_number}: "
                    f"{uid!r} != {expected['uid']!r}")
            for field in ("image_id", "task", "source"):
                require(row.get(field) == expected.get(field),
                        f"REC {field} mismatch for UID {uid!r} in {path}")
            iou = _finite_float(row.get("iou"), f"{path}:{line_number} iou")
            giou = _finite_float(row.get("giou"), f"{path}:{line_number} giou")
            require(0.0 <= iou <= 1.0, f"IoU outside [0,1] for UID {uid!r} in {path}")
            require(-1.0 <= giou <= 1.0, f"GIoU outside [-1,1] for UID {uid!r} in {path}")
            require(isinstance(row.get("hit"), bool),
                    f"invalid hit flag for UID {uid!r} in {path}")
            require(row["hit"] == (iou >= 0.5),
                    f"hit flag disagrees with IoU for UID {uid!r} in {path}")
            require("box1000" in row, f"missing parsed box field for UID {uid!r} in {path}")
            require(isinstance(row.get("pred_raw"), str),
                    f"missing raw prediction for UID {uid!r} in {path}")
            box = row.get("box1000")
            require(
                box is None or (
                    isinstance(box, list) and len(box) == 4
                    and all(isinstance(value, int) and not isinstance(value, bool) for value in box)
                    and box[0] < box[2] and box[1] < box[3]
                ),
                f"invalid parsed box for UID {uid!r} in {path}",
            )
            if box is None:
                require_close(iou, 0.0, f"parse-failed IoU for UID {uid!r}")
                require_close(giou, -1.0, f"parse-failed GIoU for UID {uid!r}")
            row["iou"] = iou
            row["giou"] = giou
            row["parse_fail"] = float(box is None)
            ordered.append(row)
            seen.add(uid)
    require(len(ordered) == len(manifest),
            f"expected exactly {len(manifest)} REC rows in {path}, found {len(ordered)}")
    # Reuse the audited loader too, and assert its UID set agrees with the
    # order-preserving validation above.
    unique = load_jsonl_unique(path)
    require(set(unique) == seen, f"REC unique-loader disagreement in {path}")
    return ordered, unique


def aggregate_rec(rows: list[dict], *, field: str | None = None, value: str | None = None) -> dict:
    selected = rows if field is None else [row for row in rows if row.get(field) == value]
    require(selected, f"cannot aggregate an empty REC group {field}={value!r}")
    return {
        "n": len(selected),
        "rec": sum(float(row["iou"] >= 0.5) for row in selected) / len(selected),
        "giou": sum(float(row["giou"]) for row in selected) / len(selected),
        "precise_iou": sum(precise_iou_score(float(row["iou"])) for row in selected)
        / len(selected),
        "parse_fail": sum(float(row.get("box1000") is None) for row in selected)
        / len(selected),
    }


def compare_rec_scores(recomputed: dict, reported: dict, label: str) -> None:
    require(recomputed["n"] == reported["n"],
            f"{label} count changed: {recomputed['n']} != {reported['n']}")
    for name in ("rec", "giou", "precise_iou", "parse_fail"):
        require_close(float(recomputed[name]), float(reported[name]), f"{label} {name}")


def validate_rec_metrics(
    path: Path,
    rows: list[dict],
    *,
    expected_tag: str,
) -> tuple[dict, dict]:
    metrics = load_json(path)
    require(isinstance(metrics, dict), f"REC metrics are not an object: {path}")
    expected_fields = {
        "tag": expected_tag,
        "model": BASE_MODEL,
        "subset": "recovery_dev_1k",
        "n": REC_EXAMPLES,
        "base_revision": BASE_REVISION,
    }
    for key, expected in expected_fields.items():
        require(metrics.get(key) == expected,
                f"unexpected {key} in {path}: {metrics.get(key)!r} != {expected!r}")

    overall = aggregate_rec(rows)
    primary = aggregate_rec(rows, field="task", value="rec")
    compare_rec_scores(overall, group_scores(metrics, "overall"), f"{path} overall")
    compare_rec_scores(primary, group_scores(metrics, "rec"), f"{path} primary REC")
    for task in sorted({row["task"] for row in rows}):
        compare_rec_scores(
            aggregate_rec(rows, field="task", value=task),
            group_scores(metrics, task),
            f"{path} task {task}",
        )
    for source in sorted({row["source"] for row in rows}):
        reported_source = metrics.get("by_source", {}).get(source)
        require(isinstance(reported_source, dict), f"missing source {source!r} in {path}")
        reported = {
            "n": int(reported_source["n"]),
            "rec": float(reported_source["acc_iou_0.5"]),
            "giou": float(reported_source["mean_giou"]),
            "precise_iou": float(reported_source["mean_acc_iou_0.50_0.95"]),
            "parse_fail": float(reported_source["parse_fail"]),
        }
        compare_rec_scores(
            aggregate_rec(rows, field="source", value=source),
            reported,
            f"{path} source {source}",
        )
    return metrics, primary


def expected_vqa_uids_and_images(path: Path) -> tuple[list[str], dict[str, str]]:
    manifest = load_json(path)
    require(isinstance(manifest, list), f"VQA manifest is not a JSON list: {path}")
    require(len(manifest) == VQA_EXAMPLES,
            f"expected {VQA_EXAMPLES} VQA rows in {path}, found {len(manifest)}")
    uids: list[str] = []
    image_by_uid: dict[str, str] = {}
    question_ids: set[int] = set()
    for index, row in enumerate(manifest):
        require(isinstance(row, dict), f"invalid VQA manifest row {index} in {path}")
        question_id = row.get("question_id")
        require(isinstance(question_id, int) and not isinstance(question_id, bool),
                f"invalid VQA question_id at {path}:{index}")
        require(question_id not in question_ids,
                f"duplicate VQA question_id {question_id!r} in {path}")
        require(row.get("image_id") is not None,
                f"missing VQA image_id for question {question_id}")
        uid = f"vqa:{question_id}"
        uids.append(uid)
        image_by_uid[uid] = str(row["image_id"])
        question_ids.add(question_id)
    require(len(image_by_uid) == VQA_EXAMPLES, "VQA UID mapping is not one-to-one")
    return uids, image_by_uid


def validate_vqa_metrics(
    path: Path,
    rows: list[dict],
    *,
    expected_tag: str,
    vqa_manifest_path: Path,
    require_hardened_schema: bool,
) -> dict:
    metrics = load_json(path)
    require(isinstance(metrics, dict), f"VQA metrics are not an object: {path}")
    expected_fields = {
        "tag": expected_tag,
        "model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "task": "vqa",
        "n": VQA_EXAMPLES,
        "vqa_evaluator": "official_normalization_leave_one_annotator_out",
    }
    for key, expected in expected_fields.items():
        require(metrics.get(key) == expected,
                f"unexpected {key} in {path}: {metrics.get(key)!r} != {expected!r}")
    require_close(float(metrics.get("accuracy")), mean_vqa(rows), f"{path} accuracy")
    require(metrics.get("parse_fail") is None and metrics.get("pope_variants") is None,
            f"unexpected non-VQA fields in {path}")
    if require_hardened_schema:
        require(metrics.get("start") == 0, f"unexpected VQA start in {path}")
        require(metrics.get("requested_limit") == 0,
                f"unexpected VQA requested_limit in {path}")
        inputs = metrics.get("input_files")
        require(isinstance(inputs, list) and len(inputs) == 1,
                f"expected one frozen VQA input file in {path}")
        expected_input = {
            "path": str(vqa_manifest_path.resolve()),
            "sha256": sha256_file(vqa_manifest_path),
        }
        require(inputs[0] == expected_input,
                f"VQA input file/hash mismatch in {path}: {inputs[0]!r}")
    return metrics


def load_and_validate_vqa_rows(path: Path, expected_uids: list[str]) -> list[dict]:
    rows = load_vqa_rows(path, expected_count=VQA_EXAMPLES)
    actual_uids = [row["uid"] for row in rows]
    require(actual_uids == expected_uids,
            f"VQA UIDs/order do not match the frozen development manifest: {path}")
    with open(path) as f:
        for line_number, line in enumerate(f, 1):
            raw = json.loads(line)
            require(isinstance(raw.get("pred"), str),
                    f"missing VQA prediction in {path}:{line_number}")
    return rows


def verify_launch_and_inputs(
    runs: Path,
    data: Path,
    code_dir: Path,
) -> tuple[dict, dict, dict[str, Path]]:
    root = runs / "recovery_vqa_replay"
    launch_path = root / "development_launch_manifest.json"
    launch = load_json(launch_path)
    require(isinstance(launch, dict), f"development launch is not an object: {launch_path}")
    require(launch.get("schema_version") == 1, "unexpected development launch schema")
    require(launch.get("recipe_id") == RECIPE_ID, "development launch recipe changed")
    require(launch.get("base_model") == BASE_MODEL, "development launch base model changed")
    require(launch.get("base_revision") == BASE_REVISION,
            "development launch base revision changed")
    require(launch.get("candidate_steps") == list(CANDIDATE_STEPS),
            "development candidate steps changed")
    expected_evaluation = {
        "grounding_subset": "recovery_dev_1k",
        "grounding_examples": REC_EXAMPLES,
        "primary_task": "rec",
        "primary_examples": PRIMARY_REC_EXAMPLES,
        "vqa_examples": VQA_EXAMPLES,
    }
    require(launch.get("evaluation") == expected_evaluation,
            f"development evaluation contract changed: {launch.get('evaluation')!r}")

    base_dir = runs / "recovery_pilot" / "eval" / "gcq425_lora_ce_s0"
    w4_dir = runs / "recovery_pilot" / "eval" / "w4rtn_lora_cwce_g5_s0"
    paths = {
        "validation": root / "artifact_validation.json",
        "environment": code_dir / "env.sh",
        "artifact_validator": code_dir / "validate_recovery_vqa_replay.py",
        "protocol": code_dir / "recovery_vqa_replay_protocol.json",
        "eval_rec": code_dir / "eval_rec.py",
        "eval_vqa": code_dir / "eval_vqa.py",
        "recovery_utils": code_dir / "recovery_utils.py",
        "quant_utils": code_dir / "quant_utils.py",
        "gcq_patches": code_dir / "gcq_patches.py",
        "batch_script": code_dir / "batch_recovery_vqa_development.sh",
        "development_summarizer": Path(__file__).resolve(),
        "summary_pilot_support": code_dir / "summarize_recovery_pilot.py",
        "summary_checkpoint_support": code_dir / "summarize_recovery_checkpoint_sweep.py",
        "summary_selected_support": code_dir / "summarize_recovery_selected_eval.py",
        "promotion": runs / "promote_gcq_b4.25.json",
        "recovery_dev": data / "subsets" / "recovery_dev_1k.json",
        "vqa_dev": data / "subsets" / "vqa_val_5k.json",
        "baseline_rec": base_dir / "gcq425_untrained_recoverydev.rec.jsonl",
        "baseline_rec_metrics": base_dir / "gcq425_untrained_recoverydev.rec.metrics.json",
        "baseline_vqa": base_dir / "gcq425_untrained_recoverypilot_vqa5k.vqa.jsonl",
        "baseline_vqa_metrics": base_dir
        / "gcq425_untrained_recoverypilot_vqa5k.vqa.metrics.json",
        "w4_rec_metrics": w4_dir
        / "w4rtn_lora_cwce_g5_s0_recoverydev.rec.metrics.json",
    }
    launch_hashes = launch.get("hashes")
    require(isinstance(launch_hashes, dict), "development launch hashes are missing")
    require(set(launch_hashes) == set(paths),
            "development launch hash keys changed: "
            f"missing={sorted(set(paths) - set(launch_hashes))}, "
            f"unexpected={sorted(set(launch_hashes) - set(paths))}")
    for label, path in paths.items():
        require_hash(path, launch_hashes[label], label)
    launched_paths = launch.get("paths")
    path_keys = {
        "validation",
        "environment",
        "protocol",
        "promotion",
        "recovery_dev",
        "vqa_dev",
        "baseline_rec",
        "baseline_rec_metrics",
        "baseline_vqa",
        "baseline_vqa_metrics",
        "w4_rec_metrics",
    }
    require(
        launched_paths == {key: str(paths[key]) for key in path_keys},
        f"development launch paths changed: {launched_paths!r}",
    )

    protocol = load_json(paths["protocol"])
    require(isinstance(protocol, dict), "frozen protocol is not a JSON object")
    require(protocol.get("recipe_id") == RECIPE_ID, "frozen protocol recipe changed")
    require(protocol.get("base_model") == BASE_MODEL, "frozen protocol base model changed")
    require(protocol.get("base_revision") == BASE_REVISION,
            "frozen protocol base revision changed")
    development = protocol.get("development", {})
    require(development.get("candidate_steps") == list(CANDIDATE_STEPS),
            "frozen protocol candidate steps changed")

    validation = load_json(paths["validation"])
    require(isinstance(validation, dict), "artifact validation report is not an object")
    require(validation.get("recipe_id") == RECIPE_ID, "artifact validation recipe changed")
    require(validation.get("base_model") == BASE_MODEL, "artifact validation model changed")
    require(validation.get("base_revision") == BASE_REVISION,
            "artifact validation revision changed")
    require(validation.get("protocol_sha256") == launch_hashes["protocol"],
            "artifact validation used a different protocol")

    checkpoints = launch.get("checkpoints")
    validation_checkpoints = validation.get("checkpoints")
    require(isinstance(checkpoints, dict) and set(checkpoints) == {str(x) for x in CANDIDATE_STEPS},
            "development checkpoint map changed")
    require(isinstance(validation_checkpoints, dict),
            "artifact validation checkpoint map is missing")
    for step in CANDIDATE_STEPS:
        key = str(step)
        launched = checkpoints[key]
        validated = validation_checkpoints.get(key)
        require(isinstance(launched, dict) and isinstance(validated, dict),
                f"missing checkpoint metadata for step {step}")
        canonical = (
            root / "adapters" / RECIPE_ID
            if step == 750
            else root / "adapters" / RECIPE_ID / f"checkpoint-{step:06d}"
        )
        require(launched.get("adapter_dir") == str(canonical),
                f"noncanonical adapter path for step {step}")
        require(validated.get("directory") == str(canonical),
                f"validation path disagrees for step {step}")
        adapter_path = canonical / "adapter_model.safetensors"
        adapter_config_path = canonical / "adapter_config.json"
        manifest_path = canonical / "gcq_recovery_manifest.json"
        require(validated.get("artifact", {}).get("path") == str(adapter_path),
                f"validation artifact path disagrees for step {step}")
        require_hash(adapter_path, launched.get("adapter_sha256"),
                     f"checkpoint {step} adapter")
        require_hash(adapter_config_path, launched.get("adapter_config_sha256"),
                     f"checkpoint {step} adapter config")
        require_hash(manifest_path, launched.get("manifest_sha256"),
                     f"checkpoint {step} manifest")
        require(launched.get("adapter_sha256") == validated.get("artifact", {}).get("sha256"),
                f"launch/validation adapter hash disagreement at step {step}")
        require(launched.get("manifest_sha256") == validated.get("manifest_sha256"),
                f"launch/validation manifest hash disagreement at step {step}")
        require(validated.get("adapter_config", {}).get("path") == str(adapter_config_path),
                f"validation adapter config path disagrees for step {step}")
        require(
            launched.get("adapter_config_sha256")
            == validated.get("adapter_config", {}).get("sha256"),
            f"launch/validation adapter config hash disagreement at step {step}",
        )
    return launch, protocol, paths


def gates_for_candidate(
    primary: dict,
    baseline_primary: dict,
    w4_primary: dict,
    candidate_vqa: float,
    baseline_vqa: float,
    vqa_pair: dict,
    frozen_gates: dict,
) -> dict[str, bool]:
    rec_gain = float(frozen_gates["primary_rec_gain_over_untrained_gcq_min"])
    parse_margin = float(frozen_gates["primary_parse_fail_max_increase"])
    vqa_point_margin = float(frozen_gates["vqa_point_drop_max"])
    vqa_ci_margin = float(frozen_gates["vqa_paired_ci95_lower_bound_min"])
    require(frozen_gates.get("primary_giou_must_improve") is True,
            "protocol disabled the GIoU improvement gate")
    require(frozen_gates.get("primary_precise_iou_must_improve") is True,
            "protocol disabled the precise-IoU improvement gate")
    require(frozen_gates.get("primary_rec_must_exceed_w4_cwce") is True,
            "protocol disabled the W4+CWCE control gate")
    return {
        "primary_REC_gain_at_least_1pt": (
            primary["rec"] - baseline_primary["rec"] >= rec_gain
        ),
        "primary_GIoU_above_untrained_GCQ": primary["giou"] > baseline_primary["giou"],
        "primary_precise_IoU_above_untrained_GCQ": (
            primary["precise_iou"] > baseline_primary["precise_iou"]
        ),
        "primary_REC_above_W4_weighted_control": primary["rec"] > w4_primary["rec"],
        "primary_parse_fail_within_0.5pt": (
            primary["parse_fail"] <= baseline_primary["parse_fail"] + parse_margin
        ),
        "VQA_dev_point_drop_within_0.5pt": candidate_vqa >= baseline_vqa - vqa_point_margin,
        "VQA_dev_paired_CI95_lower_at_least_minus_1.5pt": (
            float(vqa_pair["ci95"][0]) >= vqa_ci_margin
        ),
    }


def selection_key(step: int, candidate: dict, recipe_id: str = RECIPE_ID) -> tuple:
    primary = candidate["primary_rec"]
    return (
        primary["rec"],
        primary["precise_iou"],
        primary["giou"],
        -step,
        recipe_id,
    )


def main() -> None:
    runs = Path(os.environ["GCQ_RUNS"])
    data = Path(os.environ["GCQ_DATA"])
    code_dir = Path(__file__).resolve().parent
    root = runs / "recovery_vqa_replay"
    launch_path = root / "development_launch_manifest.json"
    launch, protocol, paths = verify_launch_and_inputs(runs, data, code_dir)

    rec_manifest = load_rec_manifest(paths["recovery_dev"])
    vqa_uids, vqa_images = expected_vqa_uids_and_images(paths["vqa_dev"])

    base_rec_ordered, base_rec_unique = load_rec_rows(paths["baseline_rec"], rec_manifest)
    _, base_primary = validate_rec_metrics(
        paths["baseline_rec_metrics"],
        base_rec_ordered,
        expected_tag="gcq425_untrained_recoverydev",
    )
    base_vqa_rows = load_and_validate_vqa_rows(paths["baseline_vqa"], vqa_uids)
    validate_vqa_metrics(
        paths["baseline_vqa_metrics"],
        base_vqa_rows,
        expected_tag="gcq425_untrained_recoverypilot_vqa5k",
        vqa_manifest_path=paths["vqa_dev"],
        require_hardened_schema=False,
    )
    base_vqa_score = mean_vqa(base_vqa_rows)

    w4_metrics = load_json(paths["w4_rec_metrics"])
    require(isinstance(w4_metrics, dict), "W4+CWCE metrics are not an object")
    for key, expected in {
        "tag": "w4rtn_lora_cwce_g5_s0_recoverydev",
        "model": BASE_MODEL,
        "subset": "recovery_dev_1k",
        "n": REC_EXAMPLES,
        "base_revision": BASE_REVISION,
    }.items():
        require(w4_metrics.get(key) == expected,
                f"unexpected {key} in {paths['w4_rec_metrics']}: "
                f"{w4_metrics.get(key)!r} != {expected!r}")
    w4_primary = group_scores(w4_metrics, "rec")
    require(w4_primary["n"] == PRIMARY_REC_EXAMPLES,
            "W4+CWCE primary REC count changed")

    development_protocol = protocol["development"]
    references = development_protocol["reference_scores"]
    for metric, reference_key in {
        "rec": "untrained_gcq_primary_rec",
        "giou": "untrained_gcq_primary_giou",
        "precise_iou": "untrained_gcq_primary_precise_iou",
        "parse_fail": "untrained_gcq_primary_parse_fail",
    }.items():
        require_close(float(base_primary[metric]), float(references[reference_key]),
                      f"protocol baseline {metric}")
    require_close(float(w4_primary["rec"]), float(references["w4_cwce_primary_rec"]),
                  "protocol W4+CWCE REC")
    require_close(base_vqa_score, float(references["untrained_gcq_vqa5k"]),
                  "protocol untrained-GCQ VQA")

    frozen_gates = development_protocol["gates"]
    candidates: dict[str, dict] = {}
    runtime_signatures: dict[str, dict] = {}
    for candidate_index, step in enumerate(CANDIDATE_STEPS):
        directory = root / "development" / f"step{step}"
        tag = f"vqa50_step{step}"
        provenance_path = directory / "runtime_provenance.json"
        rec_log_path = directory / f"{tag}_recoverydev.rec.jsonl"
        rec_metrics_path = directory / f"{tag}_recoverydev.rec.metrics.json"
        vqa_log_path = directory / f"{tag}_vqa5k_dev.vqa.jsonl"
        vqa_metrics_path = directory / f"{tag}_vqa5k_dev.vqa.metrics.json"

        provenance = validate_runtime_provenance(provenance_path, launch_path, step)
        device = provenance["cuda"]["devices"][0]
        runtime_signatures[str(step)] = {
            "python": provenance["python"],
            "packages": provenance["packages"],
            "cuda_runtime_version": provenance["cuda"]["runtime_version"],
            "cudnn_version": provenance["cuda"]["cudnn_version"],
            "driver_versions": provenance["cuda"]["driver_versions"],
            "device_name": device["name"],
            "compute_capability": device["compute_capability"],
            "total_memory_bytes": device["total_memory_bytes"],
        }

        rec_ordered, rec_unique = load_rec_rows(rec_log_path, rec_manifest)
        _, primary = validate_rec_metrics(
            rec_metrics_path,
            rec_ordered,
            expected_tag=f"{tag}_recoverydev",
        )
        vqa_rows = load_and_validate_vqa_rows(vqa_log_path, vqa_uids)
        validate_vqa_metrics(
            vqa_metrics_path,
            vqa_rows,
            expected_tag=f"{tag}_vqa5k_dev",
            vqa_manifest_path=paths["vqa_dev"],
            require_hardened_schema=True,
        )
        require_same_uids(base_vqa_rows, vqa_rows, f"checkpoint {step} VQA development")

        seed = BOOTSTRAP_SEED + candidate_index * 5
        paired_rec = {
            metric: paired_cluster_contrast(
                {"baseline": base_rec_unique, "candidate": rec_unique},
                {"candidate": 1.0, "baseline": -1.0},
                metric,
                task="rec",
                resamples=BOOTSTRAP_RESAMPLES,
                seed=seed + metric_index,
            )
            for metric_index, metric in enumerate(("rec", "giou", "precise_iou"))
        }
        rec_images = {row["uid"]: str(row["image_id"]) for row in rec_manifest}
        paired_rec["parse_fail"] = paired_image_delta(
            base_rec_ordered,
            rec_ordered,
            rec_images,
            field="parse_fail",
            resamples=BOOTSTRAP_RESAMPLES,
            seed=seed + 3,
        )
        vqa_pair = paired_image_delta(
            base_vqa_rows,
            vqa_rows,
            vqa_images,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=seed + 4,
        )
        candidate_vqa_score = mean_vqa(vqa_rows)
        require_close(
            float(paired_rec["rec"]["observed"]),
            float(primary["rec"] - base_primary["rec"]),
            f"checkpoint {step} paired REC delta",
        )
        require_close(
            float(paired_rec["giou"]["observed"]),
            float(primary["giou"] - base_primary["giou"]),
            f"checkpoint {step} paired GIoU delta",
        )
        require_close(
            float(paired_rec["precise_iou"]["observed"]),
            float(primary["precise_iou"] - base_primary["precise_iou"]),
            f"checkpoint {step} paired precise-IoU delta",
        )
        require_close(
            float(paired_rec["parse_fail"]["observed"]),
            float(primary["parse_fail"] - base_primary["parse_fail"]),
            f"checkpoint {step} paired parse-failure delta",
        )
        require_close(
            float(vqa_pair["observed"]),
            candidate_vqa_score - base_vqa_score,
            f"checkpoint {step} paired VQA delta",
        )
        gates = gates_for_candidate(
            primary,
            base_primary,
            w4_primary,
            candidate_vqa_score,
            base_vqa_score,
            vqa_pair,
            frozen_gates,
        )
        checkpoint = launch["checkpoints"][str(step)]
        candidates[str(step)] = {
            "step": step,
            "recipe_id": RECIPE_ID,
            "adapter_dir": checkpoint["adapter_dir"],
            "adapter_sha256": checkpoint["adapter_sha256"],
            "adapter_config_sha256": checkpoint["adapter_config_sha256"],
            "manifest_sha256": checkpoint["manifest_sha256"],
            "runtime_provenance": provenance,
            "primary_rec": primary,
            "paired_primary_vs_untrained_gcq": paired_rec,
            "vqa_dev_5k": {
                "untrained_gcq": base_vqa_score,
                "candidate": candidate_vqa_score,
                "point_delta": candidate_vqa_score - base_vqa_score,
                "paired_delta": vqa_pair,
            },
            "gates": gates,
            "eligible": all(gates.values()),
            "output_hashes": {
                "runtime_provenance": sha256_file(provenance_path),
                "rec_jsonl": sha256_file(rec_log_path),
                "rec_metrics": sha256_file(rec_metrics_path),
                "vqa_jsonl": sha256_file(vqa_log_path),
                "vqa_metrics": sha256_file(vqa_metrics_path),
            },
        }

    reference_runtime = runtime_signatures[str(CANDIDATE_STEPS[0])]
    for step in CANDIDATE_STEPS[1:]:
        require(runtime_signatures[str(step)] == reference_runtime,
                f"runtime software/hardware differs at checkpoint {step}")

    eligible_steps = [
        step for step in CANDIDATE_STEPS if candidates[str(step)]["eligible"]
    ]
    selected_step = max(
        eligible_steps,
        key=lambda step: selection_key(step, candidates[str(step)]),
        default=None,
    )
    selected = None
    if selected_step is not None:
        winner = candidates[str(selected_step)]
        selected = {
            "step": selected_step,
            "recipe_id": RECIPE_ID,
            "adapter_dir": winner["adapter_dir"],
            "adapter_sha256": winner["adapter_sha256"],
            "adapter_config_sha256": winner["adapter_config_sha256"],
            "manifest_sha256": winner["manifest_sha256"],
            "selection_key": list(selection_key(selected_step, winner)),
            "scores": {
                "primary_rec": winner["primary_rec"],
                "vqa_dev_5k": winner["vqa_dev_5k"],
            },
            "gates": winner["gates"],
            "eligible": True,
        }

    summary = {
        "schema_version": 1,
        "evaluation_role": "development-only checkpoint gating and selection",
        "recipe_id": RECIPE_ID,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "protocol_sha256": launch["hashes"]["protocol"],
        "development_launch_manifest": str(launch_path),
        "development_launch_manifest_sha256": sha256_file(launch_path),
        "development_data": {
            "grounding_manifest": str(paths["recovery_dev"]),
            "grounding_manifest_sha256": launch["hashes"]["recovery_dev"],
            "grounding_examples": REC_EXAMPLES,
            "primary_rec_examples": PRIMARY_REC_EXAMPLES,
            "vqa_manifest": str(paths["vqa_dev"]),
            "vqa_manifest_sha256": launch["hashes"]["vqa_dev"],
            "vqa_examples": VQA_EXAMPLES,
            "vqa_role": development_protocol["vqa_role"],
        },
        "bootstrap": {
            "unit": "image-clustered paired candidate-minus-untrained-GCQ",
            "resamples": BOOTSTRAP_RESAMPLES,
            "base_seed": BOOTSTRAP_SEED,
        },
        "selection_rule": (
            "among candidates passing every frozen gate, lexicographically maximize "
            "primary REC, primary precise-IoU, primary GIoU, negative optimizer step, "
            "then stable recipe identifier"
        ),
        "references": {
            "untrained_gcq": {
                "primary_rec": base_primary,
                "vqa_dev_5k": base_vqa_score,
            },
            "w4_cwce": {"primary_rec": w4_primary},
        },
        "frozen_gate_thresholds": frozen_gates,
        "candidates": candidates,
        "eligible_steps": eligible_steps,
        "selected": selected,
        "selection_succeeded": selected is not None,
        "confirmation_authorization": (
            "evaluate exactly this one selected checkpoint once on the frozen fresh VQA set"
            if selected is not None
            else "none; no checkpoint passed the frozen development gates"
        ),
    }
    output = root / "development_summary.json"
    with open(output, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"DEVELOPMENT SUMMARY: {output}")
    if selected is None:
        raise SystemExit("no checkpoint passed every frozen balanced-replay development gate")


if __name__ == "__main__":
    main()

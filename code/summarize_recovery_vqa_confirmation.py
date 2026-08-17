"""Audit the one-time fresh-VQAv2 confirmation and apply its frozen gate.

The untrained GCQ baseline and the single development-selected checkpoint must
have been evaluated sequentially in the same non-array L40S job.  Integrity
failures raise and exit nonzero.  A scientifically valid noninferiority failure
is still a completed experiment: the summary is written and this script exits
zero so the failed result cannot be mistaken for a retryable infrastructure
error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from recovery_utils import BASE_MODEL, BASE_REVISION, sha256_file
from summarize_recovery_checkpoint_sweep import load_vqa_rows, mean_vqa
from summarize_recovery_pilot import group_scores
from summarize_recovery_selected_eval import paired_image_delta


RECIPE_ID = "gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
CANDIDATE_STEPS = (200, 300, 400, 500, 600, 750)
FRESH_EXAMPLES = 5_000
FRESH_IMAGES = 4_571
FRESH_MANIFEST_SHA256 = (
    "416aea5f9dd6f4a4cfa7061c3b5ca0af88647e49d5f17ed00a8d83885ea21038"
)
FRESH_METADATA_SHA256 = (
    "a7777a0199f7fb432deeee94d478ca092474dd8a5e47cffe92a93d62b57601e8"
)
ORDERED_QUESTION_IDS_SHA256 = (
    "238e349350af36cd22a3c251d7c71ceda0152d20399ebf361c4373823ce2e383"
)
SORTED_IMAGE_IDS_SHA256 = (
    "1c2b9c1b25e358d3c3741cac975d512971d9a5dd2d6c449e5cc8d0ef07de7618"
)
PROTOCOL_SHA256 = (
    "fd938d0d39116b989ffdcde4dd5ce64bbb419a1292e4ab9f32864416953e5e6d"
)
PROMOTION_SHA256 = (
    "78b3138da1c70f2f944e3c63d5cfc754f59bf6062c27564d8cc6b2792ba0c1e6"
)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260850
NONINFERIORITY_MARGIN = -0.015
BASELINE_TAG = "gcq425_untrained_vqa_fresh5k"
MINIMUM_STORAGE_HEADROOM_BYTES = 1_073_741_824


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON file: {path}")
    with open(path, encoding="utf-8") as handle:
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
    require(actual == expected, f"{label} changed after confirmation launch: {path}")
    return actual


def _newline_int_hash(values: list[int]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def audit_prior_fresh_predictions(
    runs: Path,
    fresh_manifest: Path,
    *,
    baseline_tag: str,
    candidate_tag: str,
) -> dict:
    """Reject evidence that any frozen confirmation question was evaluated.

    Fixed output names catch even empty or crashed attempts. Metrics are matched
    by manifest hash or resolved path, so alternate tags cannot evade the audit.
    Prediction logs are also scanned by UID to catch attempts that stopped before
    writing their metrics file.
    """
    require(runs.is_dir(), f"missing runs directory for no-peeking audit: {runs}")
    require_hash(fresh_manifest, FRESH_MANIFEST_SHA256, "fresh VQA manifest")
    raw_manifest = load_json(fresh_manifest)
    require(isinstance(raw_manifest, list), "fresh VQA manifest is not a JSON list")
    fresh_uids = set()
    for index, row in enumerate(raw_manifest):
        require(isinstance(row, dict), f"fresh VQA row {index} is not an object")
        uid = row.get("uid")
        require(isinstance(uid, str) and uid, f"fresh VQA row {index} has no UID")
        require(uid not in fresh_uids, f"duplicate fresh VQA UID {uid!r}")
        fresh_uids.add(uid)
    require(
        len(fresh_uids) == FRESH_EXAMPLES,
        f"fresh VQA manifest has {len(fresh_uids)} unique UIDs, expected {FRESH_EXAMPLES}",
    )

    fixed_names = {
        f"{baseline_tag}.vqa.jsonl",
        f"{baseline_tag}.vqa.metrics.json",
        f"{candidate_tag}.vqa.jsonl",
        f"{candidate_tag}.vqa.metrics.json",
    }
    resolved_manifest = fresh_manifest.resolve()
    metric_files = sorted(runs.rglob("*.vqa.metrics.json"))
    prediction_files = sorted(runs.rglob("*.vqa.jsonl"))
    for path in (*metric_files, *prediction_files):
        require(path.is_file(), f"non-file VQA audit path: {path}")
        require(
            path.name not in fixed_names,
            f"fresh-confirmation fixed tag was already evaluated: {path}",
        )

    for path in metric_files:
        metrics = load_json(path)
        require(isinstance(metrics, dict), f"VQA metrics are not an object: {path}")
        inputs = metrics.get("input_files")
        if inputs is None:
            continue
        require(isinstance(inputs, list), f"invalid input_files inventory in {path}")
        for item in inputs:
            require(isinstance(item, dict), f"invalid input_files entry in {path}")
            input_hash = item.get("sha256")
            input_path = item.get("path")
            path_match = False
            if isinstance(input_path, str) and input_path:
                path_match = Path(input_path).resolve() == resolved_manifest
            require(
                input_hash != FRESH_MANIFEST_SHA256 and not path_match,
                f"fresh VQA manifest was already evaluated according to metrics: {path}",
            )

    for path in prediction_files:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                require(line.strip() != "", f"blank VQA row in audit file {path}:{line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"cannot prove untouched status: invalid JSON in {path}:{line_number}"
                    ) from exc
                require(
                    isinstance(row, dict),
                    f"cannot prove untouched status: non-object row in {path}:{line_number}",
                )
                uid = row.get("uid")
                require(
                    uid not in fresh_uids,
                    f"fresh VQA UID {uid!r} was already evaluated in {path}:{line_number}",
                )
    results_files = sorted(runs.rglob("results.csv"))
    for path in results_files:
        require(path.is_file(), f"non-file results audit path: {path}")
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            require(
                reader.fieldnames is not None and "tag" in reader.fieldnames,
                f"cannot audit fixed confirmation tags in results CSV: {path}",
            )
            for row_number, row in enumerate(reader, 2):
                require(
                    row.get("tag") not in {baseline_tag, candidate_tag},
                    f"fresh-confirmation fixed tag was already recorded in {path}:{row_number}",
                )
    return {
        "fresh_manifest_sha256": FRESH_MANIFEST_SHA256,
        "fixed_tag_outputs_found": 0,
        "matching_input_metrics_found": 0,
        "fresh_uid_prediction_logs_found": 0,
        "fixed_tag_results_rows_found": 0,
        "vqa_metrics_files_scanned": len(metric_files),
        "vqa_prediction_files_scanned": len(prediction_files),
        "results_csv_files_scanned": len(results_files),
    }


def compute_image_inventory(fresh_manifest: Path, image_dir: Path) -> dict:
    """Cryptographically bind every JPEG byte consumed by the confirmation."""
    manifest = load_json(fresh_manifest)
    require(isinstance(manifest, list), "fresh VQA manifest is not a JSON list")
    filenames = set()
    for index, row in enumerate(manifest):
        require(isinstance(row, dict), f"fresh VQA row {index} is not an object")
        filename = row.get("file_name")
        require(
            isinstance(filename, str)
            and filename
            and Path(filename).name == filename,
            f"invalid fresh VQA image filename at row {index}: {filename!r}",
        )
        filenames.add(filename)
    digest = hashlib.sha256()
    total_bytes = 0
    for filename in sorted(filenames):
        path = image_dir / filename
        require(path.is_file(), f"missing fresh VQA image: {path}")
        size = path.stat().st_size
        require(size > 0, f"empty fresh VQA image: {path}")
        file_hash = sha256_file(path)
        digest.update(f"{filename}\t{size}\t{file_hash}\n".encode("utf-8"))
        total_bytes += size
    return {
        "schema": "sorted-filename-tab-size-tab-sha256-newline-v1",
        "image_directory": str(image_dir.resolve()),
        "images": len(filenames),
        "total_bytes": total_bytes,
        "aggregate_sha256": digest.hexdigest(),
    }


def canonical_adapter_dir(root: Path, step: int) -> Path:
    require(step in CANDIDATE_STEPS, f"selected step is outside frozen candidates: {step}")
    adapter_root = root / "adapters" / RECIPE_ID
    return adapter_root if step == 750 else adapter_root / f"checkpoint-{step:06d}"


def _candidate_projection(candidate: dict) -> dict:
    keys = (
        "step",
        "recipe_id",
        "adapter_dir",
        "adapter_sha256",
        "adapter_config_sha256",
        "manifest_sha256",
    )
    return {key: candidate.get(key) for key in keys}


def validate_development_selection(
    development_summary: dict,
    development_launch: dict,
    root: Path,
) -> dict:
    """Recompute the frozen winner and bind every candidate to its outputs."""
    expected_thresholds = {
        "primary_rec_gain_over_untrained_gcq_min": 0.01,
        "primary_giou_must_improve": True,
        "primary_precise_iou_must_improve": True,
        "primary_rec_must_exceed_w4_cwce": True,
        "primary_parse_fail_max_increase": 0.005,
        "vqa_point_drop_max": 0.005,
        "vqa_paired_ci95_lower_bound_min": -0.015,
    }
    require(
        development_summary.get("frozen_gate_thresholds") == expected_thresholds,
        "development frozen gate thresholds changed",
    )
    references = development_summary.get("references")
    require(isinstance(references, dict), "development references are missing")
    baseline_primary = references.get("untrained_gcq", {}).get("primary_rec")
    require(isinstance(baseline_primary, dict), "development REC baseline is missing")
    for metric, expected in {
        "rec": 0.8053333333333333,
        "giou": 0.7087078091647582,
        "precise_iou": 0.6674666666666667,
        "parse_fail": 0.0013333333333333333,
    }.items():
        require_close(float(baseline_primary.get(metric)), expected,
                      f"development baseline {metric}")
    baseline_vqa = float(references.get("untrained_gcq", {}).get("vqa_dev_5k"))
    require_close(baseline_vqa, 0.7737, "development baseline VQA")
    w4_primary = references.get("w4_cwce", {}).get("primary_rec")
    require(isinstance(w4_primary, dict), "development W4 reference is missing")
    require_close(float(w4_primary.get("rec")), 0.8173333333333334,
                  "development W4 REC reference")
    expected_gate_names = {
        "primary_REC_gain_at_least_1pt",
        "primary_GIoU_above_untrained_GCQ",
        "primary_precise_IoU_above_untrained_GCQ",
        "primary_REC_above_W4_weighted_control",
        "primary_parse_fail_within_0.5pt",
        "VQA_dev_point_drop_within_0.5pt",
        "VQA_dev_paired_CI95_lower_at_least_minus_1.5pt",
    }
    candidates = development_summary.get("candidates")
    expected_candidate_keys = {str(step) for step in CANDIDATE_STEPS}
    require(
        isinstance(candidates, dict) and set(candidates) == expected_candidate_keys,
        "development summary candidate inventory changed",
    )
    eligible_steps = []
    for step in CANDIDATE_STEPS:
        key = str(step)
        candidate = candidates[key]
        require(isinstance(candidate, dict), f"development candidate {step} is invalid")
        require(
            candidate.get("step") == step and candidate.get("recipe_id") == RECIPE_ID,
            f"development candidate {step} identity changed",
        )
        checkpoint = development_launch.get("checkpoints", {}).get(key)
        require(isinstance(checkpoint, dict), f"development launch checkpoint {step} is missing")
        require(
            _candidate_projection(candidate)
            == {
                "step": step,
                "recipe_id": RECIPE_ID,
                "adapter_dir": checkpoint.get("adapter_dir"),
                "adapter_sha256": checkpoint.get("adapter_sha256"),
                "adapter_config_sha256": checkpoint.get("adapter_config_sha256"),
                "manifest_sha256": checkpoint.get("manifest_sha256"),
            },
            f"development candidate {step} differs from the frozen launch",
        )
        gates = candidate.get("gates")
        require(
            isinstance(gates, dict)
            and set(gates) == expected_gate_names
            and all(isinstance(value, bool) for value in gates.values()),
            f"development candidate {step} gate inventory changed",
        )
        eligible = all(gates.values())
        require(
            candidate.get("eligible") is eligible,
            f"development candidate {step} eligibility disagrees with its gates",
        )
        primary = candidate.get("primary_rec")
        require(isinstance(primary, dict), f"development candidate {step} REC scores are missing")
        for metric in ("rec", "precise_iou", "giou"):
            value = primary.get(metric)
            require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)),
                f"development candidate {step} has invalid {metric}",
            )
        output_hashes = candidate.get("output_hashes")
        expected_hash_names = {
            "runtime_provenance",
            "rec_jsonl",
            "rec_metrics",
            "vqa_jsonl",
            "vqa_metrics",
        }
        require(
            isinstance(output_hashes, dict) and set(output_hashes) == expected_hash_names,
            f"development candidate {step} output hash inventory changed",
        )
        tag = f"vqa50_step{step}"
        directory = root / "development" / f"step{step}"
        output_paths = {
            "runtime_provenance": directory / "runtime_provenance.json",
            "rec_jsonl": directory / f"{tag}_recoverydev.rec.jsonl",
            "rec_metrics": directory / f"{tag}_recoverydev.rec.metrics.json",
            "vqa_jsonl": directory / f"{tag}_vqa5k_dev.vqa.jsonl",
            "vqa_metrics": directory / f"{tag}_vqa5k_dev.vqa.metrics.json",
        }
        for label, path in output_paths.items():
            require_hash(path, output_hashes[label], f"development step {step} {label}")
        actual_primary = group_scores(load_json(output_paths["rec_metrics"]), "rec")
        require(
            primary == actual_primary,
            f"development candidate {step} REC scores disagree with its metrics",
        )
        actual_vqa = mean_vqa(load_vqa_rows(output_paths["vqa_jsonl"], FRESH_EXAMPLES))
        vqa = candidate.get("vqa_dev_5k")
        require(isinstance(vqa, dict), f"development candidate {step} VQA scores are missing")
        require_close(float(vqa.get("untrained_gcq")), baseline_vqa,
                      f"development candidate {step} VQA baseline")
        require_close(float(vqa.get("candidate")), actual_vqa,
                      f"development candidate {step} VQA score")
        require_close(float(vqa.get("point_delta")), actual_vqa - baseline_vqa,
                      f"development candidate {step} VQA point delta")
        paired_vqa = vqa.get("paired_delta")
        require(isinstance(paired_vqa, dict),
                f"development candidate {step} paired VQA result is missing")
        ci95 = paired_vqa.get("ci95")
        require(
            isinstance(ci95, list)
            and len(ci95) == 2
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in ci95
            )
            and float(ci95[0]) <= float(ci95[1]),
            f"development candidate {step} paired VQA CI is invalid",
        )
        expected_gates = {
            "primary_REC_gain_at_least_1pt": (
                primary["rec"] - baseline_primary["rec"] >= 0.01
            ),
            "primary_GIoU_above_untrained_GCQ": (
                primary["giou"] > baseline_primary["giou"]
            ),
            "primary_precise_IoU_above_untrained_GCQ": (
                primary["precise_iou"] > baseline_primary["precise_iou"]
            ),
            "primary_REC_above_W4_weighted_control": (
                primary["rec"] > w4_primary["rec"]
            ),
            "primary_parse_fail_within_0.5pt": (
                primary["parse_fail"] <= baseline_primary["parse_fail"] + 0.005
            ),
            "VQA_dev_point_drop_within_0.5pt": actual_vqa >= baseline_vqa - 0.005,
            "VQA_dev_paired_CI95_lower_at_least_minus_1.5pt": (
                float(ci95[0]) >= -0.015
            ),
        }
        require(gates == expected_gates,
                f"development candidate {step} gates disagree with its scores")
        if eligible:
            eligible_steps.append(step)

    require(
        development_summary.get("eligible_steps") == eligible_steps,
        "development eligible_steps disagrees with candidate gates",
    )
    winner_step = max(
        eligible_steps,
        key=lambda step: (
            float(candidates[str(step)]["primary_rec"]["rec"]),
            float(candidates[str(step)]["primary_rec"]["precise_iou"]),
            float(candidates[str(step)]["primary_rec"]["giou"]),
            -step,
            RECIPE_ID,
        ),
        default=None,
    )
    require(winner_step is not None, "development summary authorized no confirmation candidate")
    require(
        development_summary.get("selection_succeeded") is True,
        "development selection_succeeded disagrees with the eligible candidates",
    )
    selected = development_summary.get("selected")
    require(isinstance(selected, dict), "development selected candidate is missing")
    winner = candidates[str(winner_step)]
    require(
        _candidate_projection(selected) == _candidate_projection(winner),
        "development selected candidate is not the frozen winner",
    )
    expected_selection_key = [
        winner["primary_rec"]["rec"],
        winner["primary_rec"]["precise_iou"],
        winner["primary_rec"]["giou"],
        -winner_step,
        RECIPE_ID,
    ]
    require(
        selected.get("selection_key") == expected_selection_key,
        "development selected candidate has the wrong selection key",
    )
    require(
        selected.get("scores")
        == {
            "primary_rec": winner.get("primary_rec"),
            "vqa_dev_5k": winner.get("vqa_dev_5k"),
        },
        "development selected scores differ from the winning candidate",
    )
    require(selected.get("gates") == winner.get("gates"),
            "development selected gates differ from the winning candidate")
    require(selected.get("eligible") is True,
            "development selected candidate is not eligible")
    require(
        development_summary.get("confirmation_authorization")
        == "evaluate exactly this one selected checkpoint once on the frozen fresh VQA set",
        "development confirmation authorization changed",
    )
    return selected


def validate_fresh_manifest(
    path: Path,
    metadata_path: Path,
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    require_hash(path, FRESH_MANIFEST_SHA256, "fresh VQA manifest")
    require_hash(metadata_path, FRESH_METADATA_SHA256, "fresh VQA metadata")
    metadata = load_json(metadata_path)
    require(isinstance(metadata, dict), "fresh VQA metadata is not an object")
    expected_output = {
        "path": "subsets/vqa_fresh_confirm_5k.json",
        "format": "UTF-8 canonical JSON plus trailing newline",
        "questions": FRESH_EXAMPLES,
        "unique_images": FRESH_IMAGES,
        "manifest_sha256": FRESH_MANIFEST_SHA256,
        "ordered_question_ids_sha256": ORDERED_QUESTION_IDS_SHA256,
        "sorted_unique_image_ids_sha256": SORTED_IMAGE_IDS_SHA256,
    }
    require(metadata.get("schema_version") == 1, "fresh VQA metadata schema changed")
    require(metadata.get("output") == expected_output, "fresh VQA metadata output changed")

    manifest = load_json(path)
    require(isinstance(manifest, list), "fresh VQA manifest is not a JSON list")
    require(
        len(manifest) == FRESH_EXAMPLES,
        f"fresh VQA manifest has {len(manifest)} rows, expected {FRESH_EXAMPLES}",
    )
    expected_uids: list[str] = []
    image_by_uid: dict[str, str] = {}
    answers_by_uid: dict[str, list[str]] = {}
    question_ids: list[int] = []
    image_ids: set[int] = set()
    for index, row in enumerate(manifest):
        require(isinstance(row, dict), f"fresh VQA row {index} is not an object")
        question_id = row.get("question_id")
        image_id = row.get("image_id")
        require(
            isinstance(question_id, int) and not isinstance(question_id, bool),
            f"invalid fresh VQA question_id at row {index}",
        )
        require(
            isinstance(image_id, int) and not isinstance(image_id, bool),
            f"invalid fresh VQA image_id at row {index}",
        )
        uid = f"vqa:{question_id}"
        require(row.get("uid") == uid, f"fresh VQA UID mismatch at row {index}")
        require(uid not in image_by_uid, f"duplicate fresh VQA UID {uid!r}")
        require(
            row.get("file_name") == f"COCO_val2014_{image_id:012d}.jpg",
            f"fresh VQA filename mismatch for {uid!r}",
        )
        require(
            isinstance(row.get("question"), str) and row["question"],
            f"missing fresh VQA question for {uid!r}",
        )
        answers = row.get("answers")
        require(
            isinstance(answers, list)
            and len(answers) == 10
            and all(isinstance(answer, str) for answer in answers),
            f"fresh VQA answers are invalid for {uid!r}",
        )
        expected_uids.append(uid)
        image_by_uid[uid] = str(image_id)
        answers_by_uid[uid] = answers
        question_ids.append(question_id)
        image_ids.add(image_id)
    require(
        len(image_ids) == FRESH_IMAGES,
        f"fresh VQA manifest has {len(image_ids)} images, expected {FRESH_IMAGES}",
    )
    require(
        _newline_int_hash(question_ids) == ORDERED_QUESTION_IDS_SHA256,
        "fresh VQA ordered-question hash changed",
    )
    require(
        _newline_int_hash(sorted(image_ids)) == SORTED_IMAGE_IDS_SHA256,
        "fresh VQA sorted-image hash changed",
    )
    return expected_uids, image_by_uid, answers_by_uid


def load_vqa_predictions(
    path: Path,
    expected_uids: list[str],
    answers_by_uid: dict[str, list[str]] | None = None,
) -> list[dict]:
    require(path.is_file(), f"missing VQA prediction log: {path}")
    rows: list[dict] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.strip() != "", f"blank VQA row in {path}:{line_number}")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            require(isinstance(raw, dict), f"non-object VQA row in {path}:{line_number}")
            uid = raw.get("uid")
            require(isinstance(uid, str) and uid, f"missing VQA UID in {path}:{line_number}")
            require(uid not in seen, f"duplicate VQA UID {uid!r} in {path}")
            require(
                len(rows) < len(expected_uids),
                f"too many VQA rows in {path}; first extra UID is {uid!r}",
            )
            require(
                uid == expected_uids[len(rows)],
                f"VQA UID/order mismatch in {path}:{line_number}: "
                f"{uid!r} != {expected_uids[len(rows)]!r}",
            )
            prediction = raw.get("pred")
            require(
                isinstance(prediction, str),
                f"missing normalized VQA prediction in {path}:{line_number}",
            )
            try:
                score = float(raw["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid VQA score in {path}:{line_number}") from exc
            require(
                math.isfinite(score) and 0.0 <= score <= 1.0,
                f"VQA score outside [0,1] in {path}:{line_number}",
            )
            if answers_by_uid is not None:
                require(uid in answers_by_uid, f"missing reference answers for {uid!r}")
                # Import lazily so lightweight contract tests do not load the model stack.
                from eval_vqa import vqa_soft_score

                recomputed_score, normalized_prediction = vqa_soft_score(
                    prediction, answers_by_uid[uid]
                )
                require(
                    prediction == normalized_prediction,
                    f"stored VQA prediction is not normalized in {path}:{line_number}",
                )
                require_close(
                    score,
                    recomputed_score,
                    f"recomputed VQA score in {path}:{line_number}",
                )
            rows.append({"uid": uid, "score": score})
            seen.add(uid)
    require(
        len(rows) == len(expected_uids),
        f"expected exactly {len(expected_uids)} VQA rows in {path}, found {len(rows)}",
    )
    return rows


def validate_vqa_metrics(
    path: Path,
    rows: list[dict],
    *,
    expected_tag: str,
    manifest_path: Path,
) -> dict:
    metrics = load_json(path)
    require(isinstance(metrics, dict), f"VQA metrics are not an object: {path}")
    expected = {
        "tag": expected_tag,
        "model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "task": "vqa",
        "n": FRESH_EXAMPLES,
        "start": 0,
        "requested_limit": FRESH_EXAMPLES,
        "vqa_evaluator": "official_normalization_leave_one_annotator_out",
        "pope_variants": None,
        "parse_fail": None,
    }
    for key, value in expected.items():
        require(
            metrics.get(key) == value,
            f"unexpected {key} in {path}: {metrics.get(key)!r} != {value!r}",
        )
    require_close(float(metrics.get("accuracy")), mean_vqa(rows), f"{path} accuracy")
    expected_input = [{
        "path": str(manifest_path.resolve()),
        "sha256": FRESH_MANIFEST_SHA256,
    }]
    require(
        metrics.get("input_files") == expected_input,
        f"VQA input file/hash mismatch in {path}: {metrics.get('input_files')!r}",
    )
    return metrics


def validate_runtime_provenance(
    path: Path,
    launch_path: Path,
    candidate: dict,
    candidate_tag: str,
) -> dict:
    provenance = load_json(path)
    require(isinstance(provenance, dict), f"runtime provenance is not an object: {path}")
    expected = {
        "schema_version": 1,
        "recipe_id": RECIPE_ID,
        "selected_step": candidate["step"],
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "confirmation_launch_manifest": str(launch_path),
        "confirmation_launch_manifest_sha256": sha256_file(launch_path),
        "hardware_contract": "exactly one visible NVIDIA L40S",
        "hardware_gate_pass": True,
    }
    for key, value in expected.items():
        require(
            provenance.get(key) == value,
            f"runtime provenance {key} mismatch: {provenance.get(key)!r} != {value!r}",
        )
    scheduler = provenance.get("scheduler")
    require(isinstance(scheduler, dict), "runtime scheduler provenance is missing")
    require(
        isinstance(scheduler.get("job_id"), str) and scheduler["job_id"],
        "runtime scheduler job ID is missing",
    )
    require(
        scheduler.get("task_id") in (None, "", "undefined"),
        "confirmation must run as one non-array scheduler job",
    )
    require(
        isinstance(scheduler.get("hostname"), str) and scheduler["hostname"],
        "runtime scheduler hostname is missing",
    )
    require(
        isinstance(scheduler.get("queue"), str)
        and "l40s" in scheduler["queue"].lower(),
        f"confirmation did not run in the L40S queue: {scheduler.get('queue')!r}",
    )
    require(
        isinstance(scheduler.get("batch_shell_pid"), int)
        and scheduler["batch_shell_pid"] > 0,
        "runtime batch shell PID is missing",
    )
    python = provenance.get("python")
    require(isinstance(python, dict), "runtime Python provenance is missing")
    for key in ("executable", "version"):
        require(
            isinstance(python.get(key), str) and python[key],
            f"runtime Python {key} is missing",
        )
    packages = provenance.get("packages")
    require(isinstance(packages, dict), "runtime package provenance is missing")
    for package in ("torch", "transformers", "peft", "safetensors", "numpy"):
        require(
            isinstance(packages.get(package), str) and packages[package],
            f"runtime {package} version is missing",
        )
    cuda = provenance.get("cuda")
    require(isinstance(cuda, dict), "runtime CUDA provenance is missing")
    require(
        cuda.get("available") is True and cuda.get("device_count") == 1,
        "runtime did not expose exactly one CUDA device",
    )
    require(
        isinstance(cuda.get("visible_devices"), str) and cuda["visible_devices"],
        "runtime CUDA_VISIBLE_DEVICES is missing",
    )
    require(
        isinstance(cuda.get("driver_versions"), list)
        and cuda["driver_versions"]
        and all(isinstance(version, str) and version for version in cuda["driver_versions"]),
        "runtime NVIDIA driver provenance is missing",
    )
    devices = cuda.get("devices")
    require(isinstance(devices, list) and len(devices) == 1, "invalid CUDA inventory")
    device = devices[0]
    require(
        device.get("index") == 0 and "L40S" in str(device.get("name", "")),
        f"confirmation ran on a non-L40S CUDA device: {device!r}",
    )
    capability = device.get("compute_capability")
    require(
        isinstance(capability, list)
        and len(capability) == 2
        and tuple(capability) >= (8, 0),
        f"invalid confirmation CUDA compute capability: {capability!r}",
    )
    require(
        int(device.get("total_memory_bytes", 0)) >= 40 * 2**30,
        "confirmation CUDA device has less than 40 GiB",
    )
    expected_execution = {
        "mode": "single non-array scheduler job",
        "order": ["untrained_gcq_baseline", "single_selected_candidate"],
        "baseline_tag": BASELINE_TAG,
        "candidate_tag": candidate_tag,
        "device_argument_for_both": "cuda:0",
        "same_process_environment_for_both": True,
    }
    require(
        provenance.get("execution") == expected_execution,
        f"runtime execution contract changed: {provenance.get('execution')!r}",
    )
    storage = provenance.get("storage_headroom")
    require(isinstance(storage, dict), "runtime storage-headroom provenance is missing")
    require(
        storage.get("minimum_available_bytes") == MINIMUM_STORAGE_HEADROOM_BYTES,
        "runtime storage-headroom threshold changed",
    )
    require(storage.get("gate_pass") is True, "runtime storage-headroom gate did not pass")
    probes = storage.get("probes")
    require(
        isinstance(probes, dict) and set(probes) == {"output", "model_cache"},
        "runtime storage-headroom probe inventory changed",
    )
    for label, probe in probes.items():
        require(isinstance(probe, dict), f"runtime {label} storage probe is invalid")
        require(
            isinstance(probe.get("path"), str) and probe["path"],
            f"runtime {label} storage path is missing",
        )
        available = probe.get("available_bytes")
        require(
            isinstance(available, int)
            and not isinstance(available, bool)
            and available >= MINIMUM_STORAGE_HEADROOM_BYTES,
            f"runtime {label} storage headroom was below the frozen minimum: {available!r}",
        )
    launch = load_json(launch_path)
    launcher_probes = launch.get("storage_headroom", {}).get("launcher_probes", {})
    require(
        {label: probe["path"] for label, probe in probes.items()}
        == {label: probe.get("path") for label, probe in launcher_probes.items()},
        "runtime storage probe paths differ from the frozen launch",
    )
    return provenance


def validate_execution_ledger(
    path: Path,
    launch_path: Path,
    provenance: dict,
    candidate: dict,
    candidate_tag: str,
    *,
    fresh_manifest: Path,
    evaluator: Path,
    promotion: Path,
    confirmation: Path,
) -> dict:
    ledger = load_json(path)
    require(isinstance(ledger, dict), "confirmation execution ledger is not an object")
    require(ledger.get("schema_version") == 1, "execution ledger schema changed")
    require(
        ledger.get("confirmation_launch_manifest") == str(launch_path)
        and ledger.get("confirmation_launch_manifest_sha256") == sha256_file(launch_path),
        "execution ledger does not bind the confirmation launch",
    )
    require(
        ledger.get("scheduler") == provenance.get("scheduler"),
        "execution ledger scheduler identity differs from runtime provenance",
    )
    expected_shared = {
        "model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "task": "vqa",
        "fresh_manifest": str(fresh_manifest),
        "fresh_manifest_sha256": FRESH_MANIFEST_SHA256,
        "evaluator": str(evaluator),
        "evaluator_sha256": sha256_file(evaluator),
        "quantization": {
            "method": "rtn_quantize_dequantize",
            "bits": 4,
            "group_size": 128,
            "promotion_manifest": str(promotion),
            "promotion_manifest_sha256": PROMOTION_SHA256,
            "average_decoder_bits": 4.25,
        },
        "max_pixels": 1003520,
        "batch_size": 24,
        "device": "cuda:0",
        "blank_image": False,
        "generation": {"max_new_tokens": 16, "do_sample": False},
    }
    require(
        ledger.get("shared_inference_contract") == expected_shared,
        "execution ledger inference contract changed",
    )
    evaluations = ledger.get("evaluations")
    require(
        isinstance(evaluations, list) and len(evaluations) == 2,
        "execution ledger must contain exactly two evaluations",
    )
    expected_rows = (
        ("untrained_gcq_baseline", BASELINE_TAG, None),
        ("single_selected_candidate", candidate_tag, candidate),
    )
    times = []
    for row, (role, tag, adapter) in zip(evaluations, expected_rows):
        require(isinstance(row, dict), f"execution ledger {role} row is invalid")
        require(row.get("role") == role and row.get("tag") == tag,
                f"execution ledger {role} identity changed")
        for key in ("started_ns", "ended_ns", "python_pid"):
            require(
                isinstance(row.get(key), int)
                and not isinstance(row[key], bool)
                and row[key] > 0,
                f"execution ledger {role} {key} is invalid",
            )
        require(row.get("started_ns") < row.get("ended_ns"),
                f"execution ledger {role} timestamps are invalid")
        require(row.get("exit_code") == 0,
                f"execution ledger {role} did not exit successfully")
        times.extend((row["started_ns"], row["ended_ns"]))
        expected_argv = [
            provenance["python"]["executable"], str(evaluator),
            "--model", BASE_MODEL, "--revision", BASE_REVISION,
            "--task", "vqa", "--vqa-file", str(fresh_manifest),
            "--tag", tag, "--start", "0", "--limit", "5000",
            "--rtn-bits", "4", "--rtn-group", "128",
            "--promote-file", str(promotion),
        ]
        if adapter is not None:
            expected_argv += ["--adapter-dir", adapter["adapter_dir"]]
        expected_argv += [
            "--max-pixels", "1003520", "--batch", "24", "--device", "cuda:0"
        ]
        require(row.get("argv") == expected_argv,
                f"execution ledger {role} command changed")
        expected_adapter = None if adapter is None else {
            "path": adapter["adapter_dir"],
            "sha256": adapter["adapter_sha256"],
        }
        require(row.get("adapter") == expected_adapter,
                f"execution ledger {role} adapter changed")
        jsonl = confirmation / f"{tag}.vqa.jsonl"
        metrics = confirmation / f"{tag}.vqa.metrics.json"
        expected_outputs = {
            "jsonl": {"path": str(jsonl), "sha256": sha256_file(jsonl)},
            "metrics": {"path": str(metrics), "sha256": sha256_file(metrics)},
        }
        require(row.get("outputs") == expected_outputs,
                f"execution ledger {role} output hashes changed")
    require(times == sorted(times) and times[1] <= times[2],
            "execution ledger contradicts sequential baseline-first evaluation")
    return ledger


def confirmation_gate(paired_delta: dict, threshold: float = NONINFERIORITY_MARGIN) -> bool:
    ci95 = paired_delta.get("ci95")
    require(
        isinstance(ci95, list)
        and len(ci95) == 2
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in ci95
        ),
        "paired VQA delta has an invalid CI95",
    )
    require(float(ci95[0]) <= float(ci95[1]), "paired VQA CI95 bounds are reversed")
    require(math.isfinite(threshold), "paired VQA gate threshold is not finite")
    return float(ci95[0]) >= threshold


def scientific_result(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    image_by_uid: dict[str, str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    threshold: float = NONINFERIORITY_MARGIN,
) -> dict:
    paired = paired_image_delta(
        baseline_rows,
        candidate_rows,
        image_by_uid,
        resamples=resamples,
        seed=seed,
    )
    baseline_score = mean_vqa(baseline_rows)
    candidate_score = mean_vqa(candidate_rows)
    require_close(
        float(paired["observed"]),
        candidate_score - baseline_score,
        "fresh VQA paired point delta",
    )
    passed = confirmation_gate(paired, threshold)
    return {
        "untrained_gcq": baseline_score,
        "selected_candidate": candidate_score,
        "point_delta": candidate_score - baseline_score,
        "paired_delta": paired,
        "gate_pass": passed,
    }


def validate_results_csv(
    path: Path,
    candidate_tag: str,
    *,
    baseline_accuracy: float,
    candidate_accuracy: float,
) -> list[str]:
    require(path.is_file(), f"missing confirmation results CSV: {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_fields = [
            "tag", "model", "task", "subset", "n", "acc", "mean_giou",
            "parse_fail", "acc_small", "acc_medium", "acc_large", "blank_image",
            "seconds",
        ]
        require(reader.fieldnames == expected_fields,
                f"confirmation CSV schema changed: {reader.fieldnames!r}")
        rows = list(reader)
    require(len(rows) == 2, f"confirmation results CSV must have exactly two rows: {path}")
    tags = [row.get("tag") for row in rows]
    require(
        tags == [BASELINE_TAG, candidate_tag],
        f"confirmation evaluations ran in the wrong order: {tags!r}",
    )
    for row, expected_accuracy in zip(rows, (baseline_accuracy, candidate_accuracy)):
        require(row.get("model") == BASE_MODEL, "confirmation CSV model changed")
        require(row.get("task") == "vqa", "confirmation CSV task changed")
        require(row.get("subset") == "vqa", "confirmation CSV subset changed")
        require(row.get("n") == str(FRESH_EXAMPLES), "confirmation CSV count changed")
        require(row.get("blank_image") == "False", "confirmation used blank images")
        for field in (
            "mean_giou", "parse_fail", "acc_small", "acc_medium", "acc_large"
        ):
            require(row.get(field) == "", f"unexpected confirmation CSV {field}")
        try:
            csv_accuracy = float(row["acc"])
            seconds = int(row["seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid numeric field in confirmation CSV") from exc
        require(
            math.isclose(csv_accuracy, expected_accuracy, rel_tol=0.0, abs_tol=5.1e-5),
            "confirmation CSV accuracy disagrees with recomputed predictions",
        )
        require(seconds >= 0, "confirmation CSV runtime is negative")
    return tags


def verify_launch_and_inputs(
    runs: Path,
    data: Path,
    code_dir: Path,
) -> tuple[dict, dict, dict[str, Path]]:
    root = runs / "recovery_vqa_replay"
    confirmation = root / "confirmation"
    launch_path = confirmation / "confirmation_launch_manifest.json"
    launch = load_json(launch_path)
    require(
        launch_path.stat().st_mode & 0o777 == 0o444,
        "confirmation launch manifest is not immutable mode 0444",
    )
    require(isinstance(launch, dict), "confirmation launch is not a JSON object")
    expected_top = {
        "schema_version": 1,
        "evaluation_role": (
            "one-time untouched VQAv2 confirmation of the single "
            "development-selected candidate"
        ),
        "recipe_id": RECIPE_ID,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "quantization": {
            "method": "rtn_quantize_dequantize",
            "bits": 4,
            "group_size": 128,
            "average_decoder_bits": 4.25,
            "max_pixels": 1003520,
        },
        "bootstrap": {
            "unit": "image-clustered paired candidate-minus-untrained-GCQ",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "only_scientific_gate": {
            "name": "VQA_fresh_paired_CI95_lower_at_least_minus_1.5pt",
            "statistic": "paired candidate-minus-untrained-GCQ CI95 lower bound",
            "minimum": NONINFERIORITY_MARGIN,
        },
    }
    for key, value in expected_top.items():
        require(launch.get(key) == value, f"confirmation launch {key} changed")

    candidates = launch.get("trained_candidates")
    require(
        isinstance(candidates, list) and len(candidates) == 1,
        "confirmation launch must contain exactly one trained candidate",
    )
    candidate = candidates[0]
    required_candidate_keys = {
        "step",
        "recipe_id",
        "adapter_dir",
        "adapter_sha256",
        "adapter_config_sha256",
        "manifest_sha256",
    }
    require(
        isinstance(candidate, dict) and set(candidate) == required_candidate_keys,
        "confirmation candidate fields changed",
    )
    step = candidate.get("step")
    require(
        isinstance(step, int) and not isinstance(step, bool) and step in CANDIDATE_STEPS,
        f"invalid confirmation candidate step: {step!r}",
    )
    require(candidate.get("recipe_id") == RECIPE_ID, "confirmation candidate recipe changed")
    adapter_dir = canonical_adapter_dir(root, step)
    require(
        candidate.get("adapter_dir") == str(adapter_dir),
        "confirmation candidate adapter path is noncanonical",
    )
    for key in ("adapter_sha256", "adapter_config_sha256", "manifest_sha256"):
        value = candidate.get(key)
        require(
            isinstance(value, str) and len(value) == 64,
            f"invalid confirmation candidate hash: {key}",
        )
    candidate_tag = f"vqa50_step{step}_vqa_fresh5k"
    expected_evaluation = {
        "task": "vqa",
        "questions": FRESH_EXAMPLES,
        "unique_images": FRESH_IMAGES,
        "start": 0,
        "limit": FRESH_EXAMPLES,
        "batch_size": 24,
        "device": "cuda:0",
        "baseline_tag": BASELINE_TAG,
        "candidate_tag": candidate_tag,
        "execution_order": ["untrained_gcq_baseline", "single_selected_candidate"],
        "same_job_and_gpu_required": True,
    }
    require(
        launch.get("evaluation") == expected_evaluation,
        f"confirmation evaluation contract changed: {launch.get('evaluation')!r}",
    )
    untouched_audit = launch.get("untouched_audit")
    require(isinstance(untouched_audit, dict), "confirmation untouched audit is missing")
    for key, value in {
        "fresh_manifest_sha256": FRESH_MANIFEST_SHA256,
        "fixed_tag_outputs_found": 0,
        "matching_input_metrics_found": 0,
        "fresh_uid_prediction_logs_found": 0,
        "fixed_tag_results_rows_found": 0,
    }.items():
        require(
            untouched_audit.get(key) == value,
            f"confirmation untouched audit {key} changed",
        )
    for key in (
        "vqa_metrics_files_scanned",
        "vqa_prediction_files_scanned",
        "results_csv_files_scanned",
    ):
        require(
            isinstance(untouched_audit.get(key), int)
            and not isinstance(untouched_audit[key], bool)
            and untouched_audit[key] >= 0,
            f"confirmation untouched audit {key} is invalid",
        )
    require(
        untouched_audit.get("checked_before_launch") is True
        and untouched_audit.get("recheck_at_job_start") is True,
        "confirmation untouched audit timing contract changed",
    )
    storage_launch = launch.get("storage_headroom")
    require(isinstance(storage_launch, dict), "confirmation storage-headroom contract is missing")
    require(
        storage_launch.get("minimum_available_bytes") == MINIMUM_STORAGE_HEADROOM_BYTES,
        "confirmation storage-headroom threshold changed",
    )
    require(
        storage_launch.get("recheck_at_job_start") is True,
        "confirmation storage headroom is not rechecked in the job",
    )
    launch_probes = storage_launch.get("launcher_probes")
    require(
        isinstance(launch_probes, dict) and set(launch_probes) == {"output", "model_cache"},
        "confirmation launcher storage probe inventory changed",
    )
    for label, probe in launch_probes.items():
        require(isinstance(probe, dict), f"confirmation launcher {label} probe is invalid")
        require(
            isinstance(probe.get("path"), str) and probe["path"],
            f"confirmation launcher {label} probe path is missing",
        )
        available = probe.get("available_bytes")
        require(
            isinstance(available, int)
            and not isinstance(available, bool)
            and available >= MINIMUM_STORAGE_HEADROOM_BYTES,
            f"confirmation launcher {label} headroom was insufficient: {available!r}",
        )
    launched_image_inventory = launch.get("image_inventory")
    require(
        isinstance(launched_image_inventory, dict),
        "confirmation image inventory is missing",
    )
    current_image_inventory = compute_image_inventory(
        data / "subsets" / "vqa_fresh_confirm_5k.json",
        data / "images" / "val2014",
    )
    require(
        launched_image_inventory == current_image_inventory,
        "fresh VQA image bytes changed after confirmation launch",
    )
    require(
        launched_image_inventory.get("images") == FRESH_IMAGES,
        "fresh VQA image inventory count changed",
    )

    paths = {
        "development_summary": root / "development_summary.json",
        "development_launch": root / "development_launch_manifest.json",
        "development_summarizer": code_dir / "summarize_recovery_vqa_development.py",
        "artifact_validation": root / "artifact_validation.json",
        "artifact_validator": code_dir / "validate_recovery_vqa_replay.py",
        "training_launch": root / "training_launch_manifest.json",
        "protocol": code_dir / "recovery_vqa_replay_protocol.json",
        "promotion": runs / "promote_gcq_b4.25.json",
        "fresh_manifest": data / "subsets" / "vqa_fresh_confirm_5k.json",
        "fresh_metadata": data / "subsets" / "vqa_fresh_confirm_5k.meta.json",
        "selected_adapter": adapter_dir / "adapter_model.safetensors",
        "selected_adapter_config": adapter_dir / "adapter_config.json",
        "selected_adapter_manifest": adapter_dir / "gcq_recovery_manifest.json",
        "eval_vqa": code_dir / "eval_vqa.py",
        "recovery_utils": code_dir / "recovery_utils.py",
        "quant_utils": code_dir / "quant_utils.py",
        "gcq_patches": code_dir / "gcq_patches.py",
        "environment": code_dir / "env.sh",
        "launcher_script": code_dir / "launch_recovery_vqa_confirmation.sh",
        "batch_script": code_dir / "batch_recovery_vqa_confirmation.sh",
        "confirmation_summarizer": Path(__file__).resolve(),
        "analysis_vqa_helper": code_dir / "summarize_recovery_checkpoint_sweep.py",
        "analysis_pairing_helper": code_dir / "summarize_recovery_selected_eval.py",
        "analysis_recovery_helper": code_dir / "summarize_recovery_pilot.py",
    }
    launched_paths = launch.get("paths")
    launched_hashes = launch.get("hashes")
    require(
        isinstance(launched_paths, dict)
        and launched_paths == {key: str(path) for key, path in paths.items()},
        "confirmation launch path inventory changed",
    )
    require(
        isinstance(launched_hashes, dict) and set(launched_hashes) == set(paths),
        "confirmation launch hash inventory changed",
    )
    for label, path in paths.items():
        require_hash(path, launched_hashes[label], label)
    require(launched_hashes["fresh_manifest"] == FRESH_MANIFEST_SHA256,
            "confirmation launch fresh-manifest hash changed")
    require(launched_hashes["fresh_metadata"] == FRESH_METADATA_SHA256,
            "confirmation launch fresh-metadata hash changed")
    require(launched_hashes["protocol"] == PROTOCOL_SHA256,
            "confirmation launch protocol hash changed")
    require(launched_hashes["promotion"] == PROMOTION_SHA256,
            "confirmation launch promotion hash changed")
    require(candidate["adapter_sha256"] == launched_hashes["selected_adapter"],
            "candidate adapter hash differs from path inventory")
    require(
        candidate["adapter_config_sha256"] == launched_hashes["selected_adapter_config"],
        "candidate adapter-config hash differs from path inventory",
    )
    require(candidate["manifest_sha256"] == launched_hashes["selected_adapter_manifest"],
            "candidate manifest hash differs from path inventory")

    development_summary = load_json(paths["development_summary"])
    require(
        paths["development_summary"].stat().st_mode & 0o777 == 0o444,
        "development summary is not immutable mode 0444",
    )
    require(isinstance(development_summary, dict), "development summary is not an object")
    require(development_summary.get("schema_version") == 1,
            "development summary schema changed")
    require(development_summary.get("recipe_id") == RECIPE_ID,
            "development summary recipe changed")
    require(development_summary.get("base_model") == BASE_MODEL,
            "development summary model changed")
    require(development_summary.get("base_revision") == BASE_REVISION,
            "development summary revision changed")
    require(development_summary.get("selection_succeeded") is True,
            "development selection did not succeed")
    selected = development_summary.get("selected")
    require(isinstance(selected, dict) and selected.get("eligible") is True,
            "development-selected checkpoint is not eligible")
    require(_candidate_projection(selected) == candidate,
            "confirmation candidate differs from development selection")
    require(
        development_summary.get("development_launch_manifest")
        == str(paths["development_launch"]),
        "development summary launch path changed",
    )
    require(
        development_summary.get("development_launch_manifest_sha256")
        == launched_hashes["development_launch"],
        "development summary does not bind the frozen development launch",
    )
    require(development_summary.get("protocol_sha256") == PROTOCOL_SHA256,
            "development summary used a different protocol")

    development_launch = load_json(paths["development_launch"])
    require(development_launch.get("schema_version") == 1,
            "development launch schema changed")
    require(development_launch.get("recipe_id") == RECIPE_ID,
            "development launch recipe changed")
    require(development_launch.get("base_model") == BASE_MODEL,
            "development launch model changed")
    require(development_launch.get("base_revision") == BASE_REVISION,
            "development launch revision changed")
    require(development_launch.get("candidate_steps") == list(CANDIDATE_STEPS),
            "development candidate steps changed")
    development_hashes = development_launch.get("hashes")
    require(isinstance(development_hashes, dict), "development launch hashes are missing")
    cross_launch_hashes = {
        "validation": "artifact_validation",
        "artifact_validator": "artifact_validator",
        "protocol": "protocol",
        "promotion": "promotion",
        "environment": "environment",
        "eval_vqa": "eval_vqa",
        "recovery_utils": "recovery_utils",
        "quant_utils": "quant_utils",
        "gcq_patches": "gcq_patches",
        "development_summarizer": "development_summarizer",
        "summary_pilot_support": "analysis_recovery_helper",
        "summary_checkpoint_support": "analysis_vqa_helper",
        "summary_selected_support": "analysis_pairing_helper",
    }
    for development_key, confirmation_key in cross_launch_hashes.items():
        require(
            development_hashes.get(development_key) == launched_hashes[confirmation_key],
            f"confirmation {confirmation_key} differs from the frozen development launch",
        )
    validated_selected = validate_development_selection(
        development_summary, development_launch, root
    )
    require(
        _candidate_projection(validated_selected) == candidate,
        "recomputed development winner differs from the confirmation candidate",
    )
    checkpoint = development_launch.get("checkpoints", {}).get(str(step))
    expected_checkpoint = {
        "adapter_dir": candidate["adapter_dir"],
        "adapter_sha256": candidate["adapter_sha256"],
        "adapter_config_sha256": candidate["adapter_config_sha256"],
        "manifest_sha256": candidate["manifest_sha256"],
    }
    require(checkpoint == expected_checkpoint,
            "confirmation candidate differs from the development launch")

    validation = load_json(paths["artifact_validation"])
    require(validation.get("schema_version") == 1, "artifact validation schema changed")
    require(validation.get("recipe_id") == RECIPE_ID, "artifact validation recipe changed")
    require(validation.get("base_model") == BASE_MODEL, "artifact validation model changed")
    require(validation.get("base_revision") == BASE_REVISION,
            "artifact validation revision changed")
    validated = validation.get("checkpoints", {}).get(str(step))
    require(isinstance(validated, dict), "selected artifact validation is missing")
    require(validated.get("directory") == candidate["adapter_dir"],
            "artifact validation adapter path changed")
    require(validated.get("artifact", {}).get("sha256") == candidate["adapter_sha256"],
            "artifact validation adapter hash changed")
    require(
        validated.get("adapter_config", {}).get("sha256")
        == candidate["adapter_config_sha256"],
        "artifact validation adapter-config hash changed",
    )
    require(validated.get("manifest_sha256") == candidate["manifest_sha256"],
            "artifact validation manifest hash changed")

    training_launch = load_json(paths["training_launch"])
    require(training_launch.get("schema_version") == 1, "training launch schema changed")
    require(training_launch.get("recipe_id") == RECIPE_ID, "training launch recipe changed")
    require(training_launch.get("checkpoint_steps") == list(CANDIDATE_STEPS),
            "training checkpoint steps changed")
    require(training_launch.get("protocol_sha256") == PROTOCOL_SHA256,
            "training launch protocol changed")
    require(training_launch.get("promotion_sha256") == PROMOTION_SHA256,
            "training launch promotion changed")

    protocol = load_json(paths["protocol"])
    require(protocol.get("schema_version") == 1, "confirmation protocol schema changed")
    require(protocol.get("recipe_id") == RECIPE_ID, "confirmation protocol recipe changed")
    require(protocol.get("base_model") == BASE_MODEL, "confirmation protocol model changed")
    require(protocol.get("base_revision") == BASE_REVISION,
            "confirmation protocol revision changed")
    expected_confirmation = {
        "vqa_manifest": "vqa_fresh_confirm_5k.json",
        "vqa_manifest_sha256": FRESH_MANIFEST_SHA256,
        "questions": FRESH_EXAMPLES,
        "unique_images": FRESH_IMAGES,
        "image_disjoint_from": ["vqa_val_5k.json", "POPE"],
        "allowed_trained_candidates": 1,
        "paired_image_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "vqa_paired_ci95_lower_bound_min": NONINFERIORITY_MARGIN,
        "pope_role": "exposed secondary regression benchmark, not untouched confirmation",
        "on_failure": "do not evaluate a second trained candidate on this confirmation set",
    }
    require(protocol.get("confirmation") == expected_confirmation,
            "frozen confirmation protocol changed")
    return launch, candidate, paths


def _write_new_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o444)
        os.link(temporary_path, path)
        temporary_path.unlink()
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def publish_scientific_summary(path: Path, summary: dict) -> None:
    """Publish either a valid PASS or valid FAIL without treating FAIL as an error."""
    require(summary.get("integrity_validation_pass") is True,
            "refusing to publish a summary that did not pass integrity validation")
    require(isinstance(summary.get("confirmation_pass"), bool),
            "confirmation summary is missing its Boolean scientific outcome")
    expected_outcome = "PASS" if summary["confirmation_pass"] else "FAIL"
    require(summary.get("scientific_outcome") == expected_outcome,
            "confirmation Boolean and named scientific outcomes disagree")
    _write_new_json_atomic(path, summary)


def main() -> None:
    runs = Path(os.environ["GCQ_RUNS"])
    data = Path(os.environ["GCQ_DATA"])
    code_dir = Path(__file__).resolve().parent
    root = runs / "recovery_vqa_replay"
    confirmation = root / "confirmation"
    output = root / "vqa_confirmation_summary.json"
    require(not output.exists(), f"refusing to replace confirmation summary: {output}")

    launch, candidate, paths = verify_launch_and_inputs(runs, data, code_dir)
    launch_path = confirmation / "confirmation_launch_manifest.json"
    expected_uids, image_by_uid, answers_by_uid = validate_fresh_manifest(
        paths["fresh_manifest"], paths["fresh_metadata"]
    )
    candidate_tag = launch["evaluation"]["candidate_tag"]
    runtime_path = confirmation / "runtime_provenance.json"
    provenance = validate_runtime_provenance(
        runtime_path, launch_path, candidate, candidate_tag
    )

    baseline_jsonl = confirmation / f"{BASELINE_TAG}.vqa.jsonl"
    baseline_metrics = confirmation / f"{BASELINE_TAG}.vqa.metrics.json"
    candidate_jsonl = confirmation / f"{candidate_tag}.vqa.jsonl"
    candidate_metrics = confirmation / f"{candidate_tag}.vqa.metrics.json"
    results_csv = confirmation / "results.csv"
    execution_ledger = confirmation / "execution_ledger.json"
    expected_names = {
        "confirmation_launch_manifest.json",
        "runtime_provenance.json",
        baseline_jsonl.name,
        baseline_metrics.name,
        candidate_jsonl.name,
        candidate_metrics.name,
        "results.csv",
        "execution_ledger.json",
    }
    actual_names = {path.name for path in confirmation.iterdir()}
    require(
        actual_names == expected_names,
        "confirmation output inventory changed: "
        f"missing={sorted(expected_names - actual_names)}, "
        f"unexpected={sorted(actual_names - expected_names)}",
    )

    baseline_rows = load_vqa_predictions(
        baseline_jsonl, expected_uids, answers_by_uid
    )
    candidate_rows = load_vqa_predictions(
        candidate_jsonl, expected_uids, answers_by_uid
    )
    validate_vqa_metrics(
        baseline_metrics,
        baseline_rows,
        expected_tag=BASELINE_TAG,
        manifest_path=paths["fresh_manifest"],
    )
    validate_execution_ledger(
        execution_ledger,
        launch_path,
        provenance,
        candidate,
        candidate_tag,
        fresh_manifest=paths["fresh_manifest"],
        evaluator=paths["eval_vqa"],
        promotion=paths["promotion"],
        confirmation=confirmation,
    )
    validate_vqa_metrics(
        candidate_metrics,
        candidate_rows,
        expected_tag=candidate_tag,
        manifest_path=paths["fresh_manifest"],
    )
    evaluation_order = validate_results_csv(
        results_csv,
        candidate_tag,
        baseline_accuracy=mean_vqa(baseline_rows),
        candidate_accuracy=mean_vqa(candidate_rows),
    )
    ordered_outputs = [
        runtime_path,
        baseline_jsonl,
        baseline_metrics,
        candidate_jsonl,
        candidate_metrics,
        results_csv,
        execution_ledger,
    ]
    mtimes = [path.stat().st_mtime_ns for path in ordered_outputs]
    require(
        mtimes == sorted(mtimes),
        "confirmation output timestamps contradict the frozen sequential execution order",
    )
    for path in ordered_outputs:
        os.chmod(path, 0o444)

    result = scientific_result(
        baseline_rows,
        candidate_rows,
        image_by_uid,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        threshold=NONINFERIORITY_MARGIN,
    )
    gate_name = "VQA_fresh_paired_CI95_lower_at_least_minus_1.5pt"
    gate_pass = bool(result.pop("gate_pass"))
    output_hashes = {
        "runtime_provenance": sha256_file(runtime_path),
        "baseline_jsonl": sha256_file(baseline_jsonl),
        "baseline_metrics": sha256_file(baseline_metrics),
        "candidate_jsonl": sha256_file(candidate_jsonl),
        "candidate_metrics": sha256_file(candidate_metrics),
        "results_csv": sha256_file(results_csv),
        "execution_ledger": sha256_file(execution_ledger),
    }
    summary = {
        "schema_version": 1,
        "evaluation_role": (
            "one-time untouched VQAv2 confirmation of the single "
            "development-selected candidate"
        ),
        "recipe_id": RECIPE_ID,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "confirmation_launch_manifest": str(launch_path),
        "confirmation_launch_manifest_sha256": sha256_file(launch_path),
        "development_summary": str(paths["development_summary"]),
        "development_summary_sha256": launch["hashes"]["development_summary"],
        "selected": candidate,
        "fresh_vqa": {
            "manifest": str(paths["fresh_manifest"]),
            "manifest_sha256": FRESH_MANIFEST_SHA256,
            "metadata": str(paths["fresh_metadata"]),
            "metadata_sha256": FRESH_METADATA_SHA256,
            "questions": FRESH_EXAMPLES,
            "unique_images": FRESH_IMAGES,
            "ordered_question_ids_sha256": ORDERED_QUESTION_IDS_SHA256,
            "sorted_unique_image_ids_sha256": SORTED_IMAGE_IDS_SHA256,
            "image_inventory": launch["image_inventory"],
            "untouched_audit": launch["untouched_audit"],
        },
        "execution": {
            "order": evaluation_order,
            "same_scheduler_job_and_visible_gpu": True,
            "runtime_provenance": provenance,
        },
        "bootstrap": launch["bootstrap"],
        "frozen_gate_threshold": NONINFERIORITY_MARGIN,
        "vqa": result,
        "gates": {gate_name: gate_pass},
        "confirmation_pass": gate_pass,
        "recovery_pipeline_pass": gate_pass,
        "scientific_outcome": "PASS" if gate_pass else "FAIL",
        "integrity_validation_pass": True,
        "output_hashes": output_hashes,
        "next_if_pass": (
            "run the already-frozen untouched grounding confirmation for this exact candidate"
        ),
        "next_if_fail": (
            "report VQA noninferiority failure; do not evaluate another trained candidate "
            "on this closed confirmation set"
        ),
    }
    publish_scientific_summary(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"VQA CONFIRMATION SUMMARY: {output}")
    print(
        "SCIENTIFIC OUTCOME: "
        f"{'PASS' if gate_pass else 'FAIL'} "
        f"(CI95 lower={result['paired_delta']['ci95'][0]:.6f}, "
        f"required >= {NONINFERIORITY_MARGIN:.6f})"
    )
    # A valid FAIL is a completed, non-retryable scientific result.  Deliberately
    # return normally; all integrity failures above raise and exit nonzero.


def cli() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-pristine", action="store_true")
    modes.add_argument("--compute-image-inventory", action="store_true")
    modes.add_argument("--validate-development-selection", action="store_true")
    parser.add_argument("--runs")
    parser.add_argument("--fresh-manifest")
    parser.add_argument("--image-dir")
    parser.add_argument("--baseline-tag")
    parser.add_argument("--candidate-tag")
    args = parser.parse_args()

    if args.audit_pristine:
        require(
            all(
                isinstance(value, str) and value
                for value in (
                    args.runs,
                    args.fresh_manifest,
                    args.baseline_tag,
                    args.candidate_tag,
                )
            ),
            "--audit-pristine requires runs, fresh manifest, baseline tag, and candidate tag",
        )
        require(args.image_dir is None, "--image-dir is invalid with --audit-pristine")
        report = audit_prior_fresh_predictions(
            Path(args.runs),
            Path(args.fresh_manifest),
            baseline_tag=args.baseline_tag,
            candidate_tag=args.candidate_tag,
        )
        print(json.dumps(report, sort_keys=True))
        return

    if args.compute_image_inventory:
        require(
            isinstance(args.fresh_manifest, str)
            and args.fresh_manifest
            and isinstance(args.image_dir, str)
            and args.image_dir,
            "--compute-image-inventory requires fresh manifest and image directory",
        )
        require(
            args.runs is None and args.baseline_tag is None and args.candidate_tag is None,
            "unrelated arguments are invalid with --compute-image-inventory",
        )
        print(json.dumps(compute_image_inventory(
            Path(args.fresh_manifest), Path(args.image_dir)
        ), sort_keys=True))
        return

    if args.validate_development_selection:
        require(
            isinstance(args.runs, str) and args.runs,
            "--validate-development-selection requires --runs",
        )
        require(
            args.fresh_manifest is None
            and args.image_dir is None
            and args.baseline_tag is None
            and args.candidate_tag is None,
            "unrelated arguments are invalid with --validate-development-selection",
        )
        root = Path(args.runs) / "recovery_vqa_replay"
        summary = load_json(root / "development_summary.json")
        launch = load_json(root / "development_launch_manifest.json")
        selected = validate_development_selection(summary, launch, root)
        print(json.dumps(_candidate_projection(selected), sort_keys=True))
        return

    require(
        all(
            value is None
            for value in (
                args.runs,
                args.fresh_manifest,
                args.image_dir,
                args.baseline_tag,
                args.candidate_tag,
            )
        ),
        "audit-only arguments require an audit mode",
    )
    main()


if __name__ == "__main__":
    cli()

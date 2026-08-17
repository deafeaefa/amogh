import csv
import hashlib
import json
from pathlib import Path

import pytest
import summarize_recovery_grounding_confirmation as grounding

from summarize_recovery_grounding_confirmation import (
    PARSE_MARGIN,
    audit_prior_grounding_predictions,
    candidate_projection,
    gates_for_split,
    load_prediction_rows,
    paired_image_delta,
    publish_summary,
    validate_completion_marker,
    validate_results_csv,
)


def _prediction(uid, image_id, *, iou, giou, box):
    return {
        "uid": uid,
        "image_id": image_id,
        "task": "rec",
        "source": "refcocoplus",
        "relative_area": 0.25,
        "area_quartile": 1,
        "pred_raw": "prediction",
        "box1000": box,
        "iou": iou,
        "giou": giou,
        "hit": iou >= 0.5,
    }


def _manifest(uid, image_id):
    return {
        "uid": uid,
        "image_id": image_id,
        "task": "rec",
        "source": "refcocoplus",
        "width": 100,
        "height": 100,
        "bbox_xywh": [0.0, 0.0, 50.0, 50.0],
    }


def _comparison(rec_lower=0.001, giou=0.01, precise=0.01, parse=PARSE_MARGIN):
    return {
        "selected_adapter_minus_untrained_gcq4.25": {
            "rec": {"observed": 0.01, "ci95": [rec_lower, 0.02]},
            "giou": {"observed": giou, "ci95": [-0.01, 0.02]},
            "precise_iou": {"observed": precise, "ci95": [-0.01, 0.02]},
            "parse_fail": {"observed": parse, "ci95": [-0.01, 0.02]},
        }
    }


def test_protocol_describes_frozen_manifest_and_unseen_predictions_not_unread_labels():
    protocol_path = Path(__file__).with_name("recovery_grounding_confirmation_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    status = protocol["data"]["scientific_status"]
    assert status["manifest_frozen_before_selection"] is True
    assert status["model_predictions_and_metrics_unseen_before_selection"] is True
    assert status["global_coco_image_unseen_claim"] is False
    assert "labels_unread" not in json.dumps(protocol).lower()


def test_batch_contract_is_two_split_tasks_with_four_sequential_arms():
    code_dir = Path(__file__).parent
    batch = (code_dir / "batch_recovery_grounding_confirmation.sh").read_text()
    assert "#$ -t 1-2" in batch
    positions = [
        batch.index('BF16_TAG="'),
        batch.index('W4_TAG="'),
        batch.index('GCQ_TAG="'),
        batch.index('SELECTED_TAG="'),
    ]
    assert positions == sorted(positions)
    assert 'GCQ_RUNS="$BASE_RUNS" "$PYT" "$SUMMARIZER"' in batch
    launcher = (code_dir / "launch_recovery_grounding_confirmation.sh").read_text()
    assert 'qsub "$BATCH_SCRIPT"' in launcher
    assert '"$PYT" "$SUMMARIZER" --preflight' in launcher


def test_grounding_gates_use_strict_positive_boundaries_and_inclusive_parse_margin():
    assert all(gates_for_split(_comparison()).values())
    assert not gates_for_split(_comparison(rec_lower=0.0))[
        "selected_minus_untrained_GCQ_REC_CI95_lower_strictly_above_zero"
    ]
    assert not gates_for_split(_comparison(giou=0.0))[
        "selected_minus_untrained_GCQ_GIoU_point_strictly_above_zero"
    ]
    assert not gates_for_split(_comparison(precise=0.0))[
        "selected_minus_untrained_GCQ_precise_IoU_point_strictly_above_zero"
    ]
    assert not gates_for_split(_comparison(parse=PARSE_MARGIN + 1e-12))[
        "selected_parse_fail_increase_at_most_0.5pt"
    ]


def test_paired_bootstrap_clusters_repeated_expressions_by_image():
    reference = [
        {"uid": "u1", "image_id": 1, "iou": 0.0, "giou": 0.0, "parse_fail": 0.0},
        {"uid": "u2", "image_id": 1, "iou": 0.0, "giou": 0.0, "parse_fail": 0.0},
        {"uid": "u3", "image_id": 2, "iou": 1.0, "giou": 1.0, "parse_fail": 0.0},
    ]
    candidate = [
        {"uid": "u1", "image_id": 1, "iou": 1.0, "giou": 1.0, "parse_fail": 0.0},
        {"uid": "u2", "image_id": 1, "iou": 1.0, "giou": 1.0, "parse_fail": 0.0},
        {"uid": "u3", "image_id": 2, "iou": 1.0, "giou": 1.0, "parse_fail": 0.0},
    ]
    result = paired_image_delta(reference, candidate, "rec", resamples=200, seed=11)
    repeated = paired_image_delta(reference, candidate, "rec", resamples=200, seed=11)
    assert result == repeated
    assert result["observed"] == pytest.approx(2 / 3)
    assert result["n_expressions"] == 3
    assert result["n_images"] == 2


def test_prediction_loader_requires_exact_manifest_order_and_recomputes_geometry(tmp_path):
    manifest = [_manifest("u1", 1), _manifest("u2", 2)]
    rows = [
        _prediction("u1", 1, iou=1.0, giou=1.0, box=[0, 0, 500, 500]),
        _prediction("u2", 2, iou=0.0, giou=-1.0, box=None),
    ]
    valid = tmp_path / "valid.rec.jsonl"
    valid.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded = load_prediction_rows(valid, manifest)
    assert [row["uid"] for row in loaded] == ["u1", "u2"]
    assert loaded[1]["parse_fail"] == 1.0

    reordered = tmp_path / "reordered.rec.jsonl"
    reordered.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    with pytest.raises(ValueError, match="UID/order mismatch"):
        load_prediction_rows(reordered, manifest)

    tampered = dict(rows[0], iou=0.9)
    invalid = tmp_path / "invalid.rec.jsonl"
    invalid.write_text(json.dumps(tampered) + "\n" + json.dumps(rows[1]) + "\n")
    with pytest.raises(ValueError, match="recomputed IoU"):
        load_prediction_rows(invalid, manifest)


def test_candidate_projection_keeps_exact_artifact_identity():
    candidate = {
        "step": 300,
        "recipe_id": "gcq425_lora_cwce_vqa50_g5_lr5e5_s0",
        "adapter_dir": "/tmp/checkpoint-000300",
        "adapter_sha256": "a" * 64,
        "adapter_config_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "ignored_score": 0.9,
    }
    assert candidate_projection(candidate) == {
        key: candidate[key]
        for key in (
            "step",
            "recipe_id",
            "adapter_dir",
            "adapter_sha256",
            "adapter_config_sha256",
            "manifest_sha256",
        )
    }


def test_valid_scientific_fail_summary_is_published_without_exception(tmp_path):
    output = tmp_path / "grounding_confirmation_summary.json"
    summary = {
        "scientific_outcome": "FAIL",
        "confirmation_pass": False,
        "integrity_validation_pass": True,
        "next_if_fail": "do not substitute another checkpoint",
    }
    # Scientific failure is data, not an exception or retry authorization.
    publish_summary(output, summary)
    assert json.loads(output.read_text()) == summary
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError, match="refusing to overwrite"):
        publish_summary(output, summary)


def test_completion_marker_binds_all_pre_summary_outputs(tmp_path):
    launch = tmp_path / "launch.json"
    provenance = tmp_path / "runtime.json"
    execution = tmp_path / "execution.jsonl"
    results = tmp_path / "results.csv"
    for path, value in (
        (launch, "launch\n"),
        (provenance, "runtime\n"),
        (execution, "execution\n"),
        (results, "results\n"),
    ):
        path.write_text(value)
    import hashlib

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    marker = tmp_path / "complete.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "testA",
                "grounding_launch_manifest_sha256": digest(launch),
                "runtime_provenance_sha256": digest(provenance),
                "arm_execution_sha256": digest(execution),
                "results_csv_sha256": digest(results),
                "all_four_arms_complete": True,
            }
        )
    )
    validate_completion_marker(
        marker,
        split="testA",
        launch_path=launch,
        provenance_path=provenance,
        execution_path=execution,
        results_path=results,
    )
    results.write_text("tampered\n")
    with pytest.raises(ValueError, match="completion marker changed"):
        validate_completion_marker(
            marker,
            split="testA",
            launch_path=launch,
            provenance_path=provenance,
            execution_path=execution,
            results_path=results,
        )


def _audit_fixture(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"uid": "refcocoplus_testA:00000"}]) + "\n")
    spec = {
        "subset": "refcocoplus_testA_confirm_full",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "expressions": 1,
    }
    monkeypatch.setattr(grounding, "SPLITS", {"testA": spec})
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs, manifest


def test_no_peeking_audit_rejects_alternate_tag_uid_log(tmp_path, monkeypatch):
    runs, manifest = _audit_fixture(tmp_path, monkeypatch)
    clean = audit_prior_grounding_predictions(
        runs, {"testA": manifest}, requested_splits=("testA",)
    )
    assert clean["confirmation_uid_prediction_logs_found"] == 0

    prediction = runs / "innocent-looking-alternate-tag.rec.jsonl"
    prediction.write_text(json.dumps({"uid": "refcocoplus_testA:00000"}) + "\n")
    with pytest.raises(ValueError, match="was already evaluated"):
        audit_prior_grounding_predictions(
            runs, {"testA": manifest}, requested_splits=("testA",)
        )


def test_no_peeking_audit_rejects_alternate_tag_subset_metrics(tmp_path, monkeypatch):
    runs, manifest = _audit_fixture(tmp_path, monkeypatch)
    metrics = runs / "renamed.rec.metrics.json"
    metrics.write_text(json.dumps({"subset": "refcocoplus_testA_confirm_full"}) + "\n")
    with pytest.raises(ValueError, match="subset was already evaluated"):
        audit_prior_grounding_predictions(
            runs, {"testA": manifest}, requested_splits=("testA",)
        )


def test_grounding_csv_is_recomputed_from_exact_prediction_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(
        grounding,
        "SPLITS",
        {"testA": {"subset": "fixture", "expressions": 10}},
    )
    path = tmp_path / "results.csv"
    fields = [
        "tag", "model", "task", "subset", "n", "acc", "mean_giou",
        "parse_fail", "acc_small", "acc_medium", "acc_large", "blank_image",
        "seconds",
    ]
    scores = {}
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, arm in enumerate(grounding.ARMS):
            scores[arm] = {"rec": 0.5, "giou": 0.25, "parse_fail": 0.1}
            writer.writerow({
                "tag": grounding.ARM_TAG_TEMPLATES[arm].format(split="testA"),
                "model": grounding.BASE_MODEL,
                "task": "rec",
                "subset": "fixture",
                "n": 10,
                "acc": "0.5000",
                "mean_giou": "0.2500",
                "parse_fail": "0.1000",
                "acc_small": "", "acc_medium": "", "acc_large": "",
                "blank_image": "False", "seconds": index,
            })
    validate_results_csv(path, "testA", scores)
    text = path.read_text().replace("0.5000", "0.4000", 1)
    path.write_text(text)
    with pytest.raises(ValueError, match="disagrees with recomputed"):
        validate_results_csv(path, "testA", scores)


def test_publish_rejects_integrity_failure_or_inconsistent_outcome(tmp_path):
    output = tmp_path / "summary.json"
    with pytest.raises(ValueError, match="integrity"):
        publish_summary(output, {
            "integrity_validation_pass": False,
            "confirmation_pass": False,
            "scientific_outcome": "FAIL",
        })
    with pytest.raises(ValueError, match="disagree"):
        publish_summary(output, {
            "integrity_validation_pass": True,
            "confirmation_pass": False,
            "scientific_outcome": "PASS",
        })
    assert not output.exists()


def test_shell_pipeline_has_storage_recheck_atomic_task_lock_and_safe_summary_race():
    code_dir = Path(__file__).parent
    launcher = (code_dir / "launch_recovery_grounding_confirmation.sh").read_text()
    batch = (code_dir / "batch_recovery_grounding_confirmation.sh").read_text()
    assert "MINIMUM_STORAGE_HEADROOM_BYTES=1073741824" in launcher
    assert "--audit-pristine --audit-split all" in launcher
    assert "pristine launch was rolled back" in launcher
    assert "MINIMUM_STORAGE_HEADROOM_BYTES=1073741824" in batch
    assert 'mkdir "$OUTPUT"' in batch
    assert 'if [[ ! -e "$SUMMARY" ]]; then' in batch
    assert '--audit-pristine --audit-split "$SPLIT"' in batch

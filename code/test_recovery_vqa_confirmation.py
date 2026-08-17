import hashlib
import json
from pathlib import Path

import pytest

from summarize_recovery_vqa_confirmation import (
    BASELINE_TAG,
    MINIMUM_STORAGE_HEADROOM_BYTES,
    NONINFERIORITY_MARGIN,
    audit_prior_fresh_predictions,
    compute_image_inventory,
    confirmation_gate,
    load_vqa_predictions,
    publish_scientific_summary,
    scientific_result,
    validate_development_selection,
    validate_results_csv,
    validate_runtime_provenance,
)


def test_confirmation_gate_is_inclusive_at_the_frozen_ci_boundary():
    assert confirmation_gate({"ci95": [NONINFERIORITY_MARGIN, 0.01]})
    assert not confirmation_gate(
        {"ci95": [NONINFERIORITY_MARGIN - 0.000001, 0.01]}
    )
    with pytest.raises(ValueError, match="invalid CI95"):
        confirmation_gate({"ci95": [float("nan"), 0.01]})
    with pytest.raises(ValueError, match="reversed"):
        confirmation_gate({"ci95": [0.01, -0.01]})


def test_scientific_failure_is_a_valid_result_not_an_exception():
    baseline = [
        {"uid": "vqa:1", "score": 1.0},
        {"uid": "vqa:2", "score": 1.0},
        {"uid": "vqa:3", "score": 1.0},
        {"uid": "vqa:4", "score": 1.0},
    ]
    candidate = [
        {"uid": "vqa:1", "score": 0.0},
        {"uid": "vqa:2", "score": 0.0},
        {"uid": "vqa:3", "score": 0.0},
        {"uid": "vqa:4", "score": 0.0},
    ]
    images = {"vqa:1": "a", "vqa:2": "a", "vqa:3": "b", "vqa:4": "c"}
    result = scientific_result(
        baseline,
        candidate,
        images,
        resamples=500,
        seed=20260850,
    )
    assert result["gate_pass"] is False
    assert result["paired_delta"]["n_examples"] == 4
    assert result["paired_delta"]["n_images"] == 3
    assert result["paired_delta"]["seed"] == 20260850
    assert result["point_delta"] == -1.0


def test_valid_scientific_failure_is_published_normally(tmp_path):
    output = tmp_path / "summary.json"
    summary = {
        "integrity_validation_pass": True,
        "confirmation_pass": False,
        "scientific_outcome": "FAIL",
    }
    assert publish_scientific_summary(output, summary) is None
    assert json.loads(output.read_text()) == summary
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError, match="integrity validation"):
        publish_scientific_summary(
            tmp_path / "invalid.json",
            {
                "integrity_validation_pass": False,
                "confirmation_pass": False,
                "scientific_outcome": "FAIL",
            },
        )


def test_image_clustered_confirmation_bootstrap_is_deterministic():
    baseline = [
        {"uid": "vqa:1", "score": 0.0},
        {"uid": "vqa:2", "score": 1.0},
        {"uid": "vqa:3", "score": 0.0},
    ]
    candidate = [
        {"uid": "vqa:1", "score": 1.0},
        {"uid": "vqa:2", "score": 0.0},
        {"uid": "vqa:3", "score": 1.0},
    ]
    images = {"vqa:1": "shared", "vqa:2": "shared", "vqa:3": "other"}
    first = scientific_result(
        baseline, candidate, images, resamples=1_000, seed=20260850
    )
    second = scientific_result(
        baseline, candidate, images, resamples=1_000, seed=20260850
    )
    assert first["paired_delta"] == second["paired_delta"]
    assert first["paired_delta"]["n_images"] == 2


def test_confirmation_prediction_loader_rejects_reordered_rows(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"uid": "vqa:2", "pred": "no", "score": 0.0})
        + "\n"
        + json.dumps({"uid": "vqa:1", "pred": "yes", "score": 1.0})
        + "\n"
    )
    with pytest.raises(ValueError, match="UID/order mismatch"):
        load_vqa_predictions(path, ["vqa:1", "vqa:2"])


def test_confirmation_prediction_loader_recomputes_soft_score(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"uid": "vqa:1", "pred": "yes", "score": 1.0}) + "\n")
    rows = load_vqa_predictions(path, ["vqa:1"], {"vqa:1": ["yes"] * 10})
    assert rows == [{"uid": "vqa:1", "score": 1.0}]
    path.write_text(json.dumps({"uid": "vqa:1", "pred": "yes", "score": 0.9}) + "\n")
    with pytest.raises(ValueError, match="recomputed VQA score"):
        load_vqa_predictions(path, ["vqa:1"], {"vqa:1": ["yes"] * 10})


def _runtime_fixture(launch_path: Path, *, device_name: str = "NVIDIA L40S") -> dict:
    launch_sha = hashlib.sha256(launch_path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "recipe_id": "gcq425_lora_cwce_vqa50_g5_lr5e5_s0",
        "selected_step": 300,
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "base_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "confirmation_launch_manifest": str(launch_path),
        "confirmation_launch_manifest_sha256": launch_sha,
        "scheduler": {
            "job_id": "123",
            "task_id": "undefined",
            "queue": "l40s.q",
            "hostname": "fixture",
            "batch_shell_pid": 12,
        },
        "python": {"executable": "/fixture/python", "version": "3.12.0"},
        "packages": {
            "torch": "1",
            "transformers": "1",
            "peft": "1",
            "safetensors": "1",
            "numpy": "1",
        },
        "cuda": {
            "available": True,
            "visible_devices": "0",
            "runtime_version": "12.8",
            "cudnn_version": 90000,
            "driver_versions": ["570.0"],
            "device_count": 1,
            "devices": [
                {
                    "index": 0,
                    "name": device_name,
                    "compute_capability": [8, 9],
                    "total_memory_bytes": 40 * 2**30,
                }
            ],
            "nvidia_smi": [],
        },
        "execution": {
            "mode": "single non-array scheduler job",
            "order": ["untrained_gcq_baseline", "single_selected_candidate"],
            "baseline_tag": BASELINE_TAG,
            "candidate_tag": "vqa50_step300_vqa_fresh5k",
            "device_argument_for_both": "cuda:0",
            "same_process_environment_for_both": True,
        },
        "hardware_contract": "exactly one visible NVIDIA L40S",
        "hardware_gate_pass": True,
        "storage_headroom": {
            "minimum_available_bytes": MINIMUM_STORAGE_HEADROOM_BYTES,
            "probes": {
                "output": {
                    "path": "/fixture/output",
                    "available_bytes": MINIMUM_STORAGE_HEADROOM_BYTES,
                },
                "model_cache": {
                    "path": "/fixture/cache",
                    "available_bytes": MINIMUM_STORAGE_HEADROOM_BYTES,
                },
            },
            "gate_pass": True,
        },
    }


def test_runtime_provenance_enforces_one_nonarray_l40s_job(tmp_path):
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({
        "storage_headroom": {
            "launcher_probes": {
                "output": {"path": "/fixture/output"},
                "model_cache": {"path": "/fixture/cache"},
            }
        }
    }) + "\n")
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(_runtime_fixture(launch)) + "\n")
    candidate = {"step": 300}
    validated = validate_runtime_provenance(
        runtime, launch, candidate, "vqa50_step300_vqa_fresh5k"
    )
    assert validated["hardware_gate_pass"] is True

    runtime.write_text(
        json.dumps(_runtime_fixture(launch, device_name="NVIDIA RTX A6000")) + "\n"
    )
    with pytest.raises(ValueError, match="non-L40S"):
        validate_runtime_provenance(
            runtime, launch, candidate, "vqa50_step300_vqa_fresh5k"
        )


def test_batch_contract_is_sequential_and_not_an_array_job():
    code_dir = Path(__file__).resolve().parent
    batch = (code_dir / "batch_recovery_vqa_confirmation.sh").read_text()
    launcher = (code_dir / "launch_recovery_vqa_confirmation.sh").read_text()
    assert "#$ -q l40s" in batch
    assert "#$ -t" not in batch
    baseline = batch.index('--tag "$BASE_TAG" --start 0 --limit 5000')
    candidate = batch.index('--tag "$CANDIDATE_TAG" --start 0 --limit 5000')
    summarize = batch.rindex('GCQ_RUNS="$PROJECT_RUNS" "$PYT" "$SUMMARIZER"')
    assert baseline < candidate < summarize
    assert "--adapter-dir \"$ADAPTER\"" not in batch[baseline:candidate]
    assert "--adapter-dir \"$ADAPTER\"" in batch[candidate:summarize]
    assert launcher.index("scan_for_prior_predictions") < launcher.rindex(
        'qsub "$BATCH_SCRIPT"'
    )


def test_no_peeking_audit_catches_alternate_tag_by_uid_and_input_hash(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "fresh.json"
    rows = [
        {"uid": "vqa:1", "file_name": "one.jpg"},
        {"uid": "vqa:2", "file_name": "two.jpg"},
    ]
    manifest.write_text(json.dumps(rows) + "\n")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "summarize_recovery_vqa_confirmation.FRESH_MANIFEST_SHA256", manifest_hash
    )
    monkeypatch.setattr("summarize_recovery_vqa_confirmation.FRESH_EXAMPLES", 2)
    runs = tmp_path / "runs"
    runs.mkdir()
    clean = audit_prior_fresh_predictions(
        runs,
        manifest,
        baseline_tag="baseline",
        candidate_tag="candidate",
    )
    assert clean["matching_input_metrics_found"] == 0

    alternate = runs / "innocent-looking.vqa.metrics.json"
    alternate.write_text(json.dumps({
        "input_files": [{"path": "/different/path", "sha256": manifest_hash}]
    }) + "\n")
    with pytest.raises(ValueError, match="already evaluated according to metrics"):
        audit_prior_fresh_predictions(
            runs,
            manifest,
            baseline_tag="baseline",
            candidate_tag="candidate",
        )
    alternate.unlink()

    predictions = runs / "alternate-tag.vqa.jsonl"
    predictions.write_text(json.dumps({"uid": "vqa:2", "score": 1.0}) + "\n")
    with pytest.raises(ValueError, match="was already evaluated"):
        audit_prior_fresh_predictions(
            runs,
            manifest,
            baseline_tag="baseline",
            candidate_tag="candidate",
        )


def test_image_inventory_binds_filename_size_and_bytes(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"jpeg-a")
    (images / "b.jpg").write_bytes(b"jpeg-b")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"file_name": "b.jpg"},
        {"file_name": "a.jpg"},
        {"file_name": "b.jpg"},
    ]) + "\n")
    first = compute_image_inventory(manifest, images)
    assert first["images"] == 2
    assert first["total_bytes"] == len(b"jpeg-a") + len(b"jpeg-b")
    (images / "a.jpg").write_bytes(b"jpeg-A")
    second = compute_image_inventory(manifest, images)
    assert second["aggregate_sha256"] != first["aggregate_sha256"]


def test_confirmation_csv_accuracy_and_blank_image_are_cross_checked(tmp_path):
    path = tmp_path / "results.csv"
    header = (
        "tag,model,task,subset,n,acc,mean_giou,parse_fail,acc_small,"
        "acc_medium,acc_large,blank_image,seconds\n"
    )
    candidate_tag = "vqa50_step300_vqa_fresh5k"
    rows = (
        f"{BASELINE_TAG},Qwen/Qwen3-VL-2B-Instruct,vqa,vqa,5000,0.8000,,,,,,False,10\n"
        f"{candidate_tag},Qwen/Qwen3-VL-2B-Instruct,vqa,vqa,5000,0.7900,,,,,,False,11\n"
    )
    path.write_text(header + rows)
    assert validate_results_csv(
        path,
        candidate_tag,
        baseline_accuracy=0.8,
        candidate_accuracy=0.79,
    ) == [BASELINE_TAG, candidate_tag]
    path.write_text((header + rows).replace(",False,11", ",True,11"))
    with pytest.raises(ValueError, match="blank images"):
        validate_results_csv(
            path,
            candidate_tag,
            baseline_accuracy=0.8,
            candidate_accuracy=0.79,
        )


def test_development_selection_is_recomputed_from_all_candidates(tmp_path):
    recipe = "gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
    steps = (200, 300, 400, 500, 600, 750)
    gate_names = (
        "primary_REC_gain_at_least_1pt",
        "primary_GIoU_above_untrained_GCQ",
        "primary_precise_IoU_above_untrained_GCQ",
        "primary_REC_above_W4_weighted_control",
        "primary_parse_fail_within_0.5pt",
        "VQA_dev_point_drop_within_0.5pt",
        "VQA_dev_paired_CI95_lower_at_least_minus_1.5pt",
    )
    checkpoints = {}
    candidates = {}
    for step in steps:
        adapter_dir = tmp_path / "adapters" / str(step)
        checkpoint = {
            "adapter_dir": str(adapter_dir),
            "adapter_sha256": f"{step:064x}",
            "adapter_config_sha256": f"{step + 1:064x}",
            "manifest_sha256": f"{step + 2:064x}",
        }
        checkpoints[str(step)] = checkpoint
        directory = tmp_path / "development" / f"step{step}"
        directory.mkdir(parents=True)
        tag = f"vqa50_step{step}"
        output_paths = {
            "runtime_provenance": directory / "runtime_provenance.json",
            "rec_jsonl": directory / f"{tag}_recoverydev.rec.jsonl",
            "rec_metrics": directory / f"{tag}_recoverydev.rec.metrics.json",
            "vqa_jsonl": directory / f"{tag}_vqa5k_dev.vqa.jsonl",
            "vqa_metrics": directory / f"{tag}_vqa5k_dev.vqa.metrics.json",
        }
        for label, path in output_paths.items():
            path.write_text(f"{step}-{label}\n")
        intended_eligible = step in (200, 300)
        primary = {
            "n": 750,
            "rec": (0.84 if step == 300 else 0.83) if intended_eligible else 0.80,
            "precise_iou": (0.72 if step == 300 else 0.70) if intended_eligible else 0.65,
            "giou": (0.75 if step == 300 else 0.73) if intended_eligible else 0.69,
            "parse_fail": 0.001,
        }
        output_paths["rec_metrics"].write_text(json.dumps({
            "by_task": {
                "rec": {
                    "n": primary["n"],
                    "acc_iou_0.5": primary["rec"],
                    "mean_giou": primary["giou"],
                    "mean_acc_iou_0.50_0.95": primary["precise_iou"],
                    "parse_fail": primary["parse_fail"],
                }
            }
        }) + "\n")
        output_paths["vqa_jsonl"].write_text("".join(
            json.dumps({"uid": f"vqa:{index}", "score": 0.77}) + "\n"
            for index in range(5000)
        ))
        gates = {
            gate_names[0]: primary["rec"] - 0.8053333333333333 >= 0.01,
            gate_names[1]: primary["giou"] > 0.7087078091647582,
            gate_names[2]: primary["precise_iou"] > 0.6674666666666667,
            gate_names[3]: primary["rec"] > 0.8173333333333334,
            gate_names[4]: primary["parse_fail"] <= 0.0013333333333333333 + 0.005,
            gate_names[5]: 0.77 >= 0.7737 - 0.005,
            gate_names[6]: True,
        }
        eligible = all(gates.values())
        candidates[str(step)] = {
            "step": step,
            "recipe_id": recipe,
            **checkpoint,
            "primary_rec": primary,
            "vqa_dev_5k": {
                "untrained_gcq": 0.7737,
                "candidate": 0.77,
                "point_delta": 0.77 - 0.7737,
                "paired_delta": {"ci95": [-0.01, 0.002]},
            },
            "gates": gates,
            "eligible": eligible,
            "output_hashes": {
                label: hashlib.sha256(path.read_bytes()).hexdigest()
                for label, path in output_paths.items()
            },
        }
    winner = candidates["300"]
    selected = {
        "step": 300,
        "recipe_id": recipe,
        "adapter_dir": winner["adapter_dir"],
        "adapter_sha256": winner["adapter_sha256"],
        "adapter_config_sha256": winner["adapter_config_sha256"],
        "manifest_sha256": winner["manifest_sha256"],
        "selection_key": [0.84, 0.72, 0.75, -300, recipe],
        "scores": {
            "primary_rec": winner["primary_rec"],
            "vqa_dev_5k": winner["vqa_dev_5k"],
        },
        "gates": winner["gates"],
        "eligible": True,
    }
    summary = {
        "references": {
            "untrained_gcq": {
                "primary_rec": {
                    "n": 750,
                    "rec": 0.8053333333333333,
                    "giou": 0.7087078091647582,
                    "precise_iou": 0.6674666666666667,
                    "parse_fail": 0.0013333333333333333,
                },
                "vqa_dev_5k": 0.7737,
            },
            "w4_cwce": {"primary_rec": {"rec": 0.8173333333333334}},
        },
        "frozen_gate_thresholds": {
            "primary_rec_gain_over_untrained_gcq_min": 0.01,
            "primary_giou_must_improve": True,
            "primary_precise_iou_must_improve": True,
            "primary_rec_must_exceed_w4_cwce": True,
            "primary_parse_fail_max_increase": 0.005,
            "vqa_point_drop_max": 0.005,
            "vqa_paired_ci95_lower_bound_min": -0.015,
        },
        "candidates": candidates,
        "eligible_steps": [200, 300],
        "selected": selected,
        "selection_succeeded": True,
        "confirmation_authorization": (
            "evaluate exactly this one selected checkpoint once on the frozen fresh VQA set"
        ),
    }
    launch = {"checkpoints": checkpoints}
    assert validate_development_selection(summary, launch, tmp_path)["step"] == 300
    summary["selected"] = {**selected, "step": 200}
    with pytest.raises(ValueError, match="not the frozen winner"):
        validate_development_selection(summary, launch, tmp_path)

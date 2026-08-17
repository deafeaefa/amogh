from __future__ import annotations

import json
from pathlib import Path

import pytest

import summarize_recovery_vqa_development_round2 as round2
import validate_recovery_vqa_development_round2 as validator


CODE = Path(__file__).resolve().parent
ROOT = Path("/projectnb/rise-tower/eric1/GCQ/runs/recovery_vqa_replay")


def test_candidate_set_is_complete_valid_complement() -> None:
    assert validator.ALL_SAVED == tuple(range(50, 751, 50))
    assert validator.VALID_STEPS == tuple(range(50, 701, 50))
    assert tuple(step for step in validator.VALID_STEPS
                 if step not in validator.PARENT_VALID) == round2.CANDIDATE_STEPS
    assert round2.CANDIDATE_STEPS == (50, 100, 150, 250, 350, 450, 550, 650, 700)
    assert 750 not in round2.CANDIDATE_STEPS


def test_protocol_preserves_parent_gates_bootstrap_and_selection() -> None:
    protocol = json.loads((CODE / "recovery_vqa_development_round2_protocol.json").read_text())
    original = json.loads((CODE / "recovery_vqa_replay_protocol.json").read_text())
    assert protocol["round_id"] == round2.ROUND_ID
    assert protocol["candidate_steps"] == list(round2.CANDIDATE_STEPS)
    assert protocol["gates"] == original["development"]["gates"]
    assert protocol["selection_order"] == original["development"]["selection_order"]
    assert protocol["paired_bootstrap"] == {
        "unit": "image-clustered paired candidate-minus-untrained-GCQ",
        "resamples": 10_000,
        "base_seed": 20260850,
        "candidate_seed_stride": 5,
        "metric_offsets": {"rec": 0, "giou": 1, "precise_iou": 2,
                           "parse_fail": 3, "vqa": 4},
        "parse_fail_scope": "primary REC rows only, as fixed by the bound parent amendment",
    }


def test_step750_incident_is_preregistered_and_excluded() -> None:
    protocol = json.loads((CODE / "recovery_vqa_development_round2_protocol.json").read_text())
    incident = protocol["step750_integrity_incident"]
    assert incident["exactly_zero_tensor_count"] == 296
    assert incident["apparent_file_bytes"] == 69_788_264
    assert incident["allocated_file_bytes"] == 16_777_216
    assert protocol["training_checkpoint_inventory"]["invalid_steps"] == [750]
    assert "invalid" in incident["interpretation"]


def test_real_valid_adapter_has_full_nonzero_lora_structure() -> None:
    path = (ROOT / "adapters" / validator.RECIPE / "checkpoint-000050"
            / "adapter_model.safetensors")
    inventory = validator.tensor_inventory(path)
    assert inventory["tensor_count"] == 392
    assert inventory["parameter_count"] == 17_432_576
    assert inventory["lora_a_count"] == inventory["lora_b_count"] == 196
    assert inventory["zero_tensor_count"] == 0
    assert inventory["allocated_bytes"] >= inventory["apparent_bytes"]


def test_real_step750_matches_sparse_corruption_signature() -> None:
    path = ROOT / "adapters" / validator.RECIPE / "adapter_model.safetensors"
    inventory = validator.tensor_inventory(path)
    assert inventory["sha256"] == validator.EXPECTED_STEP750_SHA
    assert inventory["tensor_count"] == 392
    assert inventory["zero_tensor_count"] == 296
    assert inventory["allocated_bytes"] == 16_777_216
    assert inventory["allocated_bytes"] < inventory["apparent_bytes"]


def test_round2_selection_key_keeps_every_parent_tiebreak() -> None:
    def candidate(rec: float, precise: float, giou: float) -> dict:
        return {"primary_rec": {"rec": rec, "precise_iou": precise, "giou": giou}}

    assert round2.selection_key(100, candidate(.83, .71, .74)) > round2.selection_key(
        50, candidate(.82, .99, .99))
    assert round2.selection_key(100, candidate(.83, .72, .70)) > round2.selection_key(
        50, candidate(.83, .71, .99))
    assert round2.selection_key(100, candidate(.83, .72, .75)) > round2.selection_key(
        50, candidate(.83, .72, .74))
    assert round2.selection_key(50, candidate(.83, .72, .75)) > round2.selection_key(
        100, candidate(.83, .72, .75))


def test_round2_gates_keep_both_vqa_noninferiority_checks() -> None:
    gates = {
        "primary_rec_gain_over_untrained_gcq_min": .01,
        "primary_giou_must_improve": True,
        "primary_precise_iou_must_improve": True,
        "primary_rec_must_exceed_w4_cwce": True,
        "primary_parse_fail_max_increase": .005,
        "vqa_point_drop_max": .005,
        "vqa_paired_ci95_lower_bound_min": -.015,
    }
    baseline = {"rec": .80, "giou": .70, "precise_iou": .65, "parse_fail": .001}
    passing = {"rec": .82, "giou": .71, "precise_iou": .66, "parse_fail": .006}
    result = round2.gates_for_candidate(
        passing, baseline, {"rec": .81}, .768, .773,
        {"ci95": [-.015, .001]}, gates,
    )
    assert all(result.values())
    result = round2.gates_for_candidate(
        passing, baseline, {"rec": .81}, .767999, .773,
        {"ci95": [-.015, .001]}, gates,
    )
    assert not result["VQA_dev_point_drop_within_0.5pt"]


def test_round2_code_has_no_confirmation_input_path() -> None:
    paths = round2.expected_paths(
        Path("/projectnb/rise-tower/eric1/GCQ/runs"),
        Path("/projectnb/rise-tower/eric1/GCQ/data"), CODE,
    )
    assert all("confirm" not in key.lower() for key in paths)
    batch = (CODE / "batch_recovery_vqa_development_round2.sh").read_text()
    assert "#$ -t 1-9" in batch
    assert "vqa_val_5k.json" in batch
    assert "vqa_fresh_confirm" not in batch
    assert "refcoco" not in batch.lower()


def test_runtime_provenance_rejects_wrong_task_mapping(tmp_path: Path) -> None:
    launch = tmp_path / "launch.json"
    launch.write_text("{}\n")
    provenance = {
        "schema_version": 1, "round_id": round2.ROUND_ID,
        "recipe_id": round2.RECIPE_ID, "checkpoint_step": 50,
        "base_model": round2.BASE_MODEL, "base_revision": round2.BASE_REVISION,
        "development_launch_manifest": str(launch),
        "development_launch_manifest_sha256": round2.sha256_file(launch),
        "hardware_contract": "exactly one visible NVIDIA L40S",
        "hardware_gate_pass": True,
        "scheduler": {"job_id": "1", "task_id": "2", "hostname": "node"},
    }
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="wrong scheduler task ID"):
        round2.validate_runtime_provenance(path, launch, 50)

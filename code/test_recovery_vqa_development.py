import json

import pytest

from summarize_recovery_vqa_development import (
    gates_for_candidate,
    load_rec_rows,
    selection_key,
)


FROZEN_GATES = {
    "primary_rec_gain_over_untrained_gcq_min": 0.01,
    "primary_giou_must_improve": True,
    "primary_precise_iou_must_improve": True,
    "primary_rec_must_exceed_w4_cwce": True,
    "primary_parse_fail_max_increase": 0.005,
    "vqa_point_drop_max": 0.005,
    "vqa_paired_ci95_lower_bound_min": -0.015,
}


def test_balanced_replay_gates_include_point_and_paired_vqa_noninferiority():
    baseline = {"rec": 0.80, "giou": 0.70, "precise_iou": 0.65, "parse_fail": 0.001}
    w4 = {"rec": 0.81}
    passing = {"rec": 0.82, "giou": 0.71, "precise_iou": 0.67, "parse_fail": 0.006}
    gates = gates_for_candidate(
        passing,
        baseline,
        w4,
        candidate_vqa=0.773 - 0.005,
        baseline_vqa=0.773,
        vqa_pair={"ci95": [-0.015, 0.001]},
        frozen_gates=FROZEN_GATES,
    )
    assert all(gates.values())

    failed_ci = gates_for_candidate(
        passing,
        baseline,
        w4,
        candidate_vqa=0.773,
        baseline_vqa=0.773,
        vqa_pair={"ci95": [-0.015001, 0.001]},
        frozen_gates=FROZEN_GATES,
    )
    assert not failed_ci["VQA_dev_paired_CI95_lower_at_least_minus_1.5pt"]


def test_balanced_replay_selection_key_uses_all_frozen_tiebreaks():
    def candidate(rec, precise, giou):
        return {"primary_rec": {"rec": rec, "precise_iou": precise, "giou": giou}}

    assert selection_key(300, candidate(0.83, 0.71, 0.74)) > selection_key(
        200, candidate(0.82, 0.99, 0.99)
    )
    assert selection_key(300, candidate(0.83, 0.72, 0.70)) > selection_key(
        200, candidate(0.83, 0.71, 0.99)
    )
    assert selection_key(300, candidate(0.83, 0.72, 0.75)) > selection_key(
        200, candidate(0.83, 0.72, 0.74)
    )
    assert selection_key(200, candidate(0.83, 0.72, 0.75)) > selection_key(
        300, candidate(0.83, 0.72, 0.75)
    )
    assert selection_key(200, candidate(0.83, 0.72, 0.75), "recipe_b") > selection_key(
        200, candidate(0.83, 0.72, 0.75), "recipe_a"
    )


def test_rec_loader_rejects_reordered_predictions(tmp_path):
    manifest = [
        {"uid": "u1", "image_id": 1, "task": "rec", "source": "fixture"},
        {"uid": "u2", "image_id": 2, "task": "rec", "source": "fixture"},
    ]
    rows = [
        {
            "uid": "u2", "image_id": 2, "task": "rec", "source": "fixture",
            "box1000": [0, 0, 1, 1], "iou": 1.0, "giou": 1.0, "hit": True,
        },
        {
            "uid": "u1", "image_id": 1, "task": "rec", "source": "fixture",
            "box1000": None, "iou": 0.0, "giou": -1.0, "hit": False,
        },
    ]
    path = tmp_path / "reordered.rec.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="UID/order mismatch"):
        load_rec_rows(path, manifest)

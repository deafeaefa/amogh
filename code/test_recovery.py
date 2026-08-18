import json
import hashlib
import os
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from build_recovery_data import bbox_answer, normalize_box_1000
from eval_odinw import match_boxes, stratified_indices
from eval_rec import area_quartiles, finalize_groups, update_group
from eval_vqa import parse_yes_no, slice_eval_items, vqa_normalize, vqa_soft_score
from recovery_utils import (
    IOU_THRESHOLDS,
    coordinate_number_spans,
    coordinate_weighted_ce,
    find_answer_coordinate_token_groups,
    find_answer_token_positions,
    precise_iou_score,
)
from summarize_recovery_pilot import paired_cluster_contrast
from summarize_recovery_checkpoint_sweep import load_vqa_rows, mean_vqa
from summarize_recovery_selected_eval import paired_image_delta
from summarize_recovery_vqa_development import validate_runtime_provenance


class PieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, token_ids, skip_special_tokens=False):
        assert len(token_ids) == 1
        return self.pieces[token_ids[0]]


def test_normalize_box_and_json_answer():
    assert normalize_box_1000([10, 20, 30, 40], 100, 200) == [100, 100, 400, 300]
    assert json.loads(bbox_answer([10, 20, 30, 40], 100, 200)) == {
        "bbox_2d": [100, 100, 400, 300]
    }


def test_coordinate_spans_exclude_bbox_name_and_punctuation():
    answer = '{"bbox_2d": [10,20,300,400]}'
    spans = coordinate_number_spans(answer)
    assert [answer[a:b] for a, b in spans] == ["10", "20", "300", "400"]
    assert all(not (a <= answer.index("2d") < b) for a, b in spans)


def test_context_token_mask_hits_only_coordinate_numbers():
    pieces = [
        "<|im_start|>", "assistant\n", '{\"bbox_', "2d\": [", "10", ",", "20", ",",
        "300", ",", "400", "]}", "<|im_end|>", "\n",
    ]
    tokenizer = PieceTokenizer(pieces)
    answer = '{"bbox_2d": [10,20,300,400]}'
    answer_positions, coordinate_positions = find_answer_token_positions(
        tokenizer, list(range(len(pieces))), answer
    )
    assert set(coordinate_positions) == {4, 6, 8, 10}
    assert 3 in answer_positions  # contains bbox_2d's non-coordinate "2"
    assert 3 not in coordinate_positions
    assert 5 not in coordinate_positions  # comma


def test_context_token_groups_preserve_four_coordinates_and_multitoken_numbers():
    pieces = [
        "prefix", '{"bbox_2d": [', "1", "00", ",", "20", ",", "3", "00", ",", "1000", "]}"
    ]
    tokenizer = PieceTokenizer(pieces)
    answer = '{"bbox_2d": [100,20,300,1000]}'
    groups = find_answer_coordinate_token_groups(
        tokenizer, list(range(len(pieces))), answer
    )
    assert groups == [[2, 3], [5], [7, 8], [10]]


def test_gamma_one_matches_per_example_masked_ce():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.tensor([
        [-100, -100, 2, 3, 4, -100],
        [-100, 5, 6, -100, -100, -100],
    ])
    coordinates = torch.zeros_like(labels, dtype=torch.float32)
    got = coordinate_weighted_ce(logits, labels, coordinates, gamma=1)

    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    expected_rows = []
    for row in range(2):
        valid = shifted_labels[row] != -100
        expected_rows.append(F.cross_entropy(shifted_logits[row, valid], shifted_labels[row, valid]))
    expected = torch.stack(expected_rows).mean()
    torch.testing.assert_close(got, expected)


def test_coordinate_weighting_is_normalized_per_example():
    logits = torch.zeros(2, 4, 3)
    labels = torch.tensor([[-100, 0, 1, -100], [-100, 1, -100, -100]])
    coordinates = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
    # Uniform logits have identical CE at every position; weighting must not
    # change the loss merely by increasing the total effective weight.
    plain = coordinate_weighted_ce(logits, labels, coordinates, gamma=1)
    weighted = coordinate_weighted_ce(logits, labels, coordinates, gamma=5)
    torch.testing.assert_close(plain, weighted)


def test_odinw_sampler_redistributes_short_datasets_exactly():
    by_set = {"a": [0], "b": [1, 2, 3, 4], "c": [5, 6, 7, 8]}
    selected = stratified_indices(by_set, requested=7, seed=0)
    assert len(selected) == 7
    assert len(set(selected)) == 7
    assert 0 in selected


def test_odinw_matching_maximizes_thresholded_pairs():
    gts = [[0, 0, 10, 10], [8, 0, 18, 10]]
    preds = [[0, 0, 18, 10], [0, 0, 10, 10]]
    matches = match_boxes(preds, gts, threshold=0.5)
    assert len(matches) == 2
    assert {(p, g) for p, g, _ in matches} == {(0, 1), (1, 0)}


def test_eval_slice_has_exact_offset_limit_and_rejects_empty_requests():
    items = list(range(10))
    assert slice_eval_items(items, start=3, limit=4) == [3, 4, 5, 6]
    assert slice_eval_items(items, start=8, limit=0) == [8, 9]
    with pytest.raises(ValueError, match="nonnegative"):
        slice_eval_items(items, start=-1, limit=1)
    with pytest.raises(ValueError, match="nonnegative"):
        slice_eval_items(items, start=0, limit=-1)
    with pytest.raises(ValueError, match="outside"):
        slice_eval_items(items, start=10, limit=0)
    with pytest.raises(ValueError, match="exceeds"):
        slice_eval_items(items, start=8, limit=3)


def test_relative_area_quartiles_are_balanced_and_outcome_free():
    records = [
        {"uid": str(i), "bbox_xywh": [0, 0, i + 1, 1], "width": 10, "height": 10}
        for i in range(8)
    ]
    assigned = area_quartiles(records)
    assert [sum(q == wanted for q in assigned.values()) for wanted in range(1, 5)] == [2, 2, 2, 2]


@pytest.mark.parametrize(("raw", "normalized"), [
    (" The TWO cats!\n", "2 cats"),
    ("Dont", "don't"),
    ("1,000.50", "1000.50"),
    ("2:50 p.m.", "2:50 pm"),
    ("girl's", "girl's"),
    ("The red-blue car", "red blue car"),
    ("none", "0"),
])
def test_official_vqa_normalization(raw, normalized):
    assert vqa_normalize(raw) == normalized


def test_vqa_leave_one_annotator_out_truth_table():
    expected = [0.0, 0.3, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for matches, wanted in enumerate(expected):
        answers = ["target"] * matches + [f"other{index}" for index in range(10 - matches)]
        score, prediction = vqa_soft_score("target", answers)
        assert prediction == "target"
        assert score == pytest.approx(wanted)
    score, _ = vqa_soft_score("The TWO.", ["2", "two", "Two!"] + ["other"] * 7)
    assert score == pytest.approx(0.9)
    with pytest.raises(ValueError, match="exactly 10"):
        vqa_soft_score("target", ["target"] * 9)
    with pytest.raises(ValueError, match="exactly 10"):
        vqa_soft_score("target", ["target"] * 11)


@pytest.mark.parametrize(("raw", "parsed"), [
    ("yes", "yes"), (" Yes.", "yes"), ("YES, there is", "yes"),
    ("no", "no"), (" No.", "no"), ("NO, there isn't", "no"),
    ("maybe", None), ("not sure", None), ("", None),
    ("yesterday", None), ("none", None),
])
def test_pope_parser_requires_explicit_leading_yes_or_no(raw, parsed):
    assert parse_yes_no(raw) == parsed


def test_precise_iou_uses_exact_shared_boundaries():
    assert IOU_THRESHOLDS == (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    for index, threshold in enumerate(IOU_THRESHOLDS, 1):
        assert precise_iou_score(threshold) == pytest.approx(index / 10)
    assert precise_iou_score(np.nextafter(0.50, 0.0)) == 0.0
    assert precise_iou_score(0.49996) == 0.0
    assert precise_iou_score(0.50004) == pytest.approx(0.1)


def test_rec_group_aggregation_separates_primary_and_auxiliary():
    records = [
        ("refcoco", "rec", 0.60, 0.4, False),
        ("refcoco", "rec", 0.00, -1.0, True),
        ("refcocoplus", "rec", 0.40, 0.1, False),
        ("refcocoplus", "rec", 0.49, 0.2, False),
        ("refcocog", "rec", 0.80, 0.7, False),
        ("refcocog", "rec", 0.90, 0.8, False),
        ("coco_detection", "coco_grounding", 0.55, 0.6, False),
        ("coco_detection", "coco_grounding", 0.95, 0.9, False),
    ]
    by_task, by_source, overall = {}, {}, {}
    for source, task, iou, giou, failed in records:
        update_group(by_task, task, iou, giou, failed)
        update_group(by_source, source, iou, giou, failed)
        update_group(overall, "overall", iou, giou, failed)
    tasks = finalize_groups(by_task)
    sources = finalize_groups(by_source)
    total = finalize_groups(overall)["overall"]
    assert tasks["rec"]["n"] == 6
    assert tasks["rec"]["acc_iou_0.5"] == pytest.approx(0.5)
    assert tasks["rec"]["mean_giou"] == pytest.approx(0.2)
    assert tasks["rec"]["parse_fail"] == pytest.approx(1 / 6)
    assert tasks["rec"]["mean_acc_iou_0.50_0.95"] == pytest.approx(1.9 / 6)
    assert tasks["coco_grounding"]["n"] == 2
    assert tasks["coco_grounding"]["acc_iou_0.5"] == 1.0
    assert tasks["coco_grounding"]["mean_giou"] == pytest.approx(0.75)
    assert tasks["coco_grounding"]["mean_acc_iou_0.50_0.95"] == pytest.approx(0.6)
    assert total["n"] == 8
    assert total["acc_iou_0.5"] == pytest.approx(0.625)
    assert total["mean_giou"] == pytest.approx(0.3375)
    assert total["parse_fail"] == pytest.approx(0.125)
    assert total["mean_acc_iou_0.50_0.95"] == pytest.approx(0.3875)
    assert sources["refcoco"]["mean_giou"] == pytest.approx(-0.3)


def test_paired_bootstrap_clusters_by_image_and_is_deterministic():
    baseline, method = {}, {}
    for index, (image_id, baseline_iou, method_iou) in enumerate([
        ("A", 0.0, 1.0), ("A", 0.0, 1.0),
        ("B", 1.0, 0.0), ("B", 1.0, 0.0),
    ]):
        uid = f"u{index}"
        metadata = {"uid": uid, "image_id": image_id, "task": "rec", "source": "fixture"}
        baseline[uid] = {**metadata, "iou": baseline_iou, "giou": baseline_iou}
        method[uid] = {**metadata, "iou": method_iou, "giou": method_iou}
    kwargs = dict(runs={"baseline": baseline, "method": method},
                  coefficients={"method": 1.0, "baseline": -1.0},
                  metric_name="rec", resamples=2_000, seed=7)
    first = paired_cluster_contrast(**kwargs)
    second = paired_cluster_contrast(**kwargs)
    assert first == second
    assert first["observed"] == 0.0
    assert first["n_examples"] == 4 and first["n_images"] == 2
    assert first["ci95"] == [-1.0, 1.0]
    identical = paired_cluster_contrast(
        {"baseline": baseline, "method": baseline},
        {"method": 1.0, "baseline": -1.0}, "rec", resamples=100, seed=0,
    )
    assert identical["observed"] == 0.0
    assert identical["ci95"] == [0.0, 0.0]


def test_checkpoint_vqa_loader_requires_complete_unique_rows(tmp_path):
    path = tmp_path / "vqa.jsonl"
    path.write_text(
        '\n'.join([
            json.dumps({"uid": "vqa:1", "score": 0.3}),
            json.dumps({"uid": "vqa:2", "score": 0.9}),
        ]) + '\n'
    )
    rows = load_vqa_rows(path, expected_count=2)
    assert [row["uid"] for row in rows] == ["vqa:1", "vqa:2"]
    assert mean_vqa(rows) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="exactly 3"):
        load_vqa_rows(path, expected_count=3)

    path.write_text(
        '\n'.join([
            json.dumps({"uid": "vqa:1", "score": 0.3}),
            json.dumps({"uid": "vqa:1", "score": 0.6}),
        ]) + '\n'
    )
    with pytest.raises(ValueError, match="duplicate VQA uid"):
        load_vqa_rows(path, expected_count=2)


def test_general_capability_bootstrap_clusters_by_image():
    baseline = [
        {"uid": "u1", "score": 0.0}, {"uid": "u2", "score": 0.0},
        {"uid": "u3", "score": 1.0}, {"uid": "u4", "score": 1.0},
    ]
    candidate = [
        {"uid": "u1", "score": 1.0}, {"uid": "u2", "score": 1.0},
        {"uid": "u3", "score": 0.0}, {"uid": "u4", "score": 0.0},
    ]
    images = {"u1": "A", "u2": "A", "u3": "B", "u4": "B"}
    first = paired_image_delta(baseline, candidate, images, resamples=2000, seed=9)
    second = paired_image_delta(baseline, candidate, images, resamples=2000, seed=9)
    assert first == second
    assert first["observed"] == 0.0
    assert first["ci95"] == [-1.0, 1.0]
    assert first["n_examples"] == 4 and first["n_images"] == 2


def test_development_runtime_provenance_requires_one_l40s(tmp_path):
    launch = tmp_path / "development_launch_manifest.json"
    launch.write_text("{}\n")
    provenance = {
        "schema_version": 1,
        "recipe_id": "gcq425_lora_cwce_vqa50_g5_lr5e5_s0",
        "checkpoint_step": 200,
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "base_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "development_launch_manifest": str(launch),
        "development_launch_manifest_sha256": hashlib.sha256(
            launch.read_bytes()
        ).hexdigest(),
        "scheduler": {"job_id": "1", "task_id": "1", "hostname": "worker"},
        "python": {"executable": "/python", "version": "3.12.0"},
        "packages": {
            name: "1.0" for name in (
                "torch", "transformers", "peft", "safetensors", "numpy"
            )
        },
        "cuda": {
            "available": True,
            "visible_devices": "0",
            "runtime_version": "12.8",
            "cudnn_version": 90000,
            "driver_versions": ["570.00"],
            "device_count": 1,
            "devices": [{
                "index": 0,
                "name": "NVIDIA L40S",
                "compute_capability": [8, 9],
                "total_memory_bytes": 48 * 2**30,
            }],
            "nvidia_smi": ["NVIDIA L40S, uuid, driver, 46068"],
        },
        "hardware_contract": "exactly one visible NVIDIA L40S",
        "hardware_gate_pass": True,
    }
    path = tmp_path / "runtime_provenance.json"
    path.write_text(json.dumps(provenance))
    assert validate_runtime_provenance(path, launch, 200) == provenance

    provenance["cuda"]["devices"][0]["name"] = "NVIDIA RTX A6000"
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="non-L40S"):
        validate_runtime_provenance(path, launch, 200)


def test_frozen_recovery_manifests_have_declared_composition_and_disjoint_images():
    subset_dir = Path(os.environ["GCQ_DATA"]) / "subsets"
    train = json.load(open(subset_dir / "recovery_train_8k.json"))
    dev = json.load(open(subset_dir / "recovery_dev_1k.json"))
    assert len(train) == 8000 and len(dev) == 1000
    assert Counter(row["task"] for row in train) == {
        "rec": 4500, "coco_grounding": 1500, "caption": 2000,
    }
    assert Counter(row["task"] for row in dev) == {"rec": 750, "coco_grounding": 250}
    assert Counter(row["source"] for row in dev) == {
        "refcoco": 250, "refcocoplus": 250, "refcocog": 250,
        "coco_detection": 250,
    }
    train_images = {row["image_id"] for row in train}
    dev_images = {row["image_id"] for row in dev}
    assert train_images.isdisjoint(dev_images)
    assert len(dev_images) == 1000
    assert len({row["uid"] for row in train}) == 8000
    assert len({row["uid"] for row in dev}) == 1000


def test_balanced_vqa_replay_manifest_preserves_grounding_and_pairs_images():
    subset_dir = Path(os.environ["GCQ_DATA"]) / "subsets"
    parent = json.load(open(subset_dir / "recovery_train_8k.json"))
    replay = json.load(open(subset_dir / "recovery_train_vqa_replay_12k.json"))
    grounding = [row for row in parent if row["task"] != "caption"]
    assert len(replay) == 12000
    assert replay[::2] == grounding
    assert Counter(row["task"] for row in replay) == {
        "rec": 4500, "coco_grounding": 1500, "vqa": 6000,
    }
    assert len({row["uid"] for row in replay}) == 12000
    assert all(
        ground["image_id"] == vqa["image_id"]
        and vqa["task"] == "vqa"
        and vqa["answer"] == vqa["multiple_choice_answer"]
        and vqa["prompt"].endswith(" Answer with a single word or phrase.")
        for ground, vqa in zip(replay[::2], replay[1::2])
    )


def test_fresh_vqa_confirmation_is_hash_frozen_and_image_disjoint():
    subset_dir = Path(os.environ["GCQ_DATA"]) / "subsets"
    fresh_path = subset_dir / "vqa_fresh_confirm_5k.json"
    assert hashlib.sha256(fresh_path.read_bytes()).hexdigest() == (
        "416aea5f9dd6f4a4cfa7061c3b5ca0af88647e49d5f17ed00a8d83885ea21038"
    )
    fresh = json.load(open(fresh_path))
    exposed = json.load(open(subset_dir / "vqa_val_5k.json"))
    assert len(fresh) == 5000 and len({row["question_id"] for row in fresh}) == 5000
    fresh_images = {row["image_id"] for row in fresh}
    assert len(fresh_images) == 4571
    assert fresh_images.isdisjoint({row["image_id"] for row in exposed})
    pope_images = set()
    for variant in ("random", "popular", "adversarial"):
        path = Path(os.environ["GCQ_DATA"]) / "pope" / f"coco_pope_{variant}.json"
        for line in open(path):
            name = json.loads(line)["image"]
            pope_images.add(int(name.removesuffix(".jpg").rsplit("_", 1)[1]))
    assert fresh_images.isdisjoint(pope_images)

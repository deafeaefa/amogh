import json

import pytest

from build_gcq_vqa_control_data import (
    VQAControlDataError,
    build_manifest,
    canonical_sha256,
    exclusion_image_ids,
    validate_manifest,
    write_exclusive,
)


def payloads(order=(0, 1, 2, 3)):
    questions = []
    annotations = []
    for image_id in range(10, 14):
        for local in range(2):
            question_id = image_id * 10 + local
            questions.append({"question_id": question_id, "image_id": image_id, "question": f"q{question_id}"})
            annotations.append(
                {
                    "question_id": question_id,
                    "image_id": image_id,
                    "multiple_choice_answer": f"a{question_id}",
                    "question_type": "what",
                    "answer_type": "other",
                }
            )
    qgroups = [questions[index * 2 : index * 2 + 2] for index in order]
    agroups = [annotations[index * 2 : index * 2 + 2] for index in reversed(order)]
    return {"questions": [row for group in qgroups for row in reversed(group)]}, {
        "annotations": [row for group in agroups for row in group]
    }


def test_selection_is_order_invariant_unique_and_training_only():
    first = build_manifest(*payloads(), exclusions={13}, rows=3, seed=7)
    second = build_manifest(*payloads((3, 1, 0, 2)), exclusions={13}, rows=3, seed=7)
    assert first == second
    records = validate_manifest(first, expected_rows=3)
    assert len({row["image_id"] for row in records}) == 3
    assert all(row["split"] == "train" and row["source"] == "vqav2_train" for row in records)
    assert first["records_sha256"] == canonical_sha256(records)


def test_exclusions_only_take_training_images():
    excluded = exclusion_image_ids(
        [
            [
                {"image_id": 1, "file_name": "COCO_train2014_000000000001.jpg"},
                {"image_id": 2, "file_name": "COCO_val2014_000000000002.jpg", "split": "val"},
                {"image_id": 3, "source": "vqav2_train"},
            ]
        ]
    )
    assert excluded == {1, 3}


def test_join_and_capacity_fail_loudly():
    questions, annotations = payloads()
    annotations["annotations"][0]["image_id"] = 999
    with pytest.raises(VQAControlDataError, match="join mismatch"):
        build_manifest(questions, annotations, exclusions=set(), rows=3)
    questions, annotations = payloads()
    with pytest.raises(VQAControlDataError, match="eligible unique images"):
        build_manifest(questions, annotations, exclusions={10, 11, 12}, rows=2)


def test_write_is_exclusive(tmp_path):
    value = build_manifest(*payloads(), exclusions={13}, rows=3)
    path = tmp_path / "vqa.json"
    assert len(write_exclusive(path, value)) == 64
    assert json.loads(path.read_text()) == value
    with pytest.raises(VQAControlDataError, match="refusing to overwrite"):
        write_exclusive(path, value)

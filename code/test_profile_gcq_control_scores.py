import copy

import pytest
import torch

from profile_gcq_control_scores import (
    ControlProfileError,
    MABA_MERGED_KIND,
    MABA_SLICE_KIND,
    VQA_MERGED_KIND,
    VQA_SLICE_KIND,
    answer_prediction_positions,
    kl_from_teacher_log_probs,
    local_reconstruction_metrics,
    merge_slices,
)


class Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return {1: "prefix ", 2: "blue", 3: " sky"}[ids[0]]


def test_answer_positions_are_causal_and_context_tokenized():
    assert answer_prediction_positions(Tokenizer(), [1, 2, 3], "blue sky") == [0, 1]


def test_full_vocabulary_teacher_student_kl():
    teacher_logits = torch.tensor([[2.0, 0.0, -1.0]])
    student_logits = torch.tensor([[1.0, 1.0, -2.0]])
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    expected = torch.sum(
        torch.softmax(teacher_logits, dim=-1)
        * (teacher_log_probs - torch.log_softmax(student_logits, dim=-1)),
        dim=-1,
    )
    assert torch.allclose(
        kl_from_teacher_log_probs(teacher_log_probs, student_logits), expected
    )


def test_maba_style_score_is_exact_half_image_half_text():
    inputs = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    image = torch.tensor([[True, False]])
    text = torch.tensor([[False, True]])
    teacher = torch.eye(2)
    w4 = torch.zeros(2, 2)
    w8 = torch.tensor([[0.5, 0.0], [0.0, 1.0]])
    result = local_reconstruction_metrics(inputs, image, text, teacher, w4, w8)
    assert result["image_relative_repair"] == pytest.approx(0.75)
    assert result["text_relative_repair"] == pytest.approx(1.0)
    assert result["modality_balanced_score"] == pytest.approx(0.875)


def slice_value(kind, names, offset=0.0):
    return {
        "schema_version": 1,
        "artifact_kind": kind,
        "protocol_context_sha256": "1" * 64,
        "candidate_catalog_hash": "2" * 64,
        "module_slice": "x",
        "profiled_modules": names,
        "details": {name: {"value": index + offset} for index, name in enumerate(names)},
        "scores": {name: index + offset for index, name in enumerate(names)},
    }


@pytest.mark.parametrize(
    "slice_kind,merged_kind",
    [(VQA_SLICE_KIND, VQA_MERGED_KIND), (MABA_SLICE_KIND, MABA_MERGED_KIND)],
)
def test_slice_merge_requires_disjoint_exact_coverage(slice_kind, merged_kind):
    values = [slice_value(slice_kind, ["a"]), slice_value(slice_kind, ["b"], 2.0)]
    merged = merge_slices(
        values, expected_names=["a", "b"], slice_kind=slice_kind, merged_kind=merged_kind
    )
    assert list(merged["scores"]) == ["a", "b"]
    with pytest.raises(ControlProfileError, match="overlap"):
        merge_slices(
            [values[0], values[0]],
            expected_names=["a", "b"],
            slice_kind=slice_kind,
            merged_kind=merged_kind,
        )
    with pytest.raises(ControlProfileError, match="coverage mismatch"):
        merge_slices(
            [values[0]],
            expected_names=["a", "b"],
            slice_kind=slice_kind,
            merged_kind=merged_kind,
        )
    bad = copy.deepcopy(values[1])
    bad["protocol_context_sha256"] = "9" * 64
    with pytest.raises(ControlProfileError, match="provenance"):
        merge_slices(
            [values[0], bad],
            expected_names=["a", "b"],
            slice_kind=slice_kind,
            merged_kind=merged_kind,
        )

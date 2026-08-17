from __future__ import annotations

import pytest

import summarize_recovery_vqa_development_amended as amended


def _rows(*, candidate: bool) -> list[dict]:
    rows: list[dict] = []
    for index in range(750):
        rows.append(
            {
                "uid": f"rec:{index}",
                "task": "rec",
                "parse_fail": float(index == (1 if candidate else 0)),
            }
        )
    auxiliary_failures = {0} if candidate else {0, 1, 2, 3}
    for index in range(250):
        rows.append(
            {
                "uid": f"aux:{index}",
                "task": "coco_grounding",
                "parse_fail": float(index in auxiliary_failures),
            }
        )
    return rows


def test_parse_failure_amendment_uses_only_primary_rec() -> None:
    reference = _rows(candidate=False)
    candidate = _rows(candidate=True)
    images = {row["uid"]: f"image:{index}" for index, row in enumerate(reference)}

    original = amended._frozen_paired_image_delta(
        reference,
        candidate,
        images,
        field="parse_fail",
        resamples=20,
        seed=7,
    )
    repaired = amended.paired_image_delta_primary_rec(
        reference,
        candidate,
        images,
        field="parse_fail",
        resamples=20,
        seed=7,
    )

    assert original["observed"] == pytest.approx(-0.003)
    assert repaired["observed"] == 0.0
    assert repaired["n_examples"] == 750
    assert repaired["n_images"] == 750


def test_non_parse_calls_delegate_without_filtering() -> None:
    reference = [
        {"uid": "a", "score": 0.0},
        {"uid": "b", "score": 1.0},
    ]
    candidate = [
        {"uid": "a", "score": 1.0},
        {"uid": "b", "score": 1.0},
    ]
    result = amended.paired_image_delta_primary_rec(
        reference,
        candidate,
        {"a": "image-a", "b": "image-b"},
        resamples=20,
        seed=7,
    )
    assert result["observed"] == 0.5
    assert result["n_examples"] == 2

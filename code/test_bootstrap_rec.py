import json

import pytest

from bootstrap_rec import load_jsonl, validate_paired_rows


def _rows():
    baseline = {
        "a": {"uid": "a", "image_id": 1, "iou": 0.4, "giou": 0.2},
        "b": {"uid": "b", "image_id": 2, "iou": 0.6, "giou": 0.5},
    }
    method = {
        "a": {"uid": "a", "image_id": 1, "iou": 0.5, "giou": 0.3},
        "b": {"uid": "b", "image_id": 2, "iou": 0.7, "giou": 0.6},
    }
    subset = [{"uid": "a", "image_id": 1}, {"uid": "b", "image_id": 2}]
    return baseline, method, subset


def test_load_jsonl_rejects_duplicate_uid(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    path.write_text("\n".join(json.dumps({"uid": "same"}) for _ in range(2)))
    with pytest.raises(ValueError, match="duplicates UID"):
        load_jsonl(path)


def test_paired_validation_requires_uid_order_and_image_identity():
    baseline, method, subset = _rows()
    assert validate_paired_rows(baseline, method, subset) == {"a": 1, "b": 2}
    reversed_method = dict(reversed(list(method.items())))
    with pytest.raises(ValueError, match="identical order"):
        validate_paired_rows(baseline, reversed_method, subset)
    wrong_image = {name: dict(row) for name, row in method.items()}
    wrong_image["a"]["image_id"] = 99
    with pytest.raises(ValueError, match="image_id mismatch"):
        validate_paired_rows(baseline, wrong_image, subset)


def test_paired_validation_rejects_subset_duplicates_and_reordering():
    baseline, method, subset = _rows()
    with pytest.raises(ValueError, match="subset duplicates"):
        validate_paired_rows(baseline, method, [subset[0], subset[0]])
    with pytest.raises(ValueError, match="identical order"):
        validate_paired_rows(baseline, method, list(reversed(subset)))

import copy
import random

import pytest

from gcq_profile_metrics import (
    aggregate_coordinate_candidate,
    build_shortlist,
    canonical_sha256,
    coordinate_row_kl,
    decoded_macro_summary,
    paired_decoded_summary,
    rerank_shortlist,
)


def _raw_rows():
    rows = []
    for task in ("rec", "coco_grounding"):
        for quartile in range(1, 5):
            rows.append({
                "uid": f"{task}:q{quartile}",
                "task": task,
                "area_quartile": quartile,
                # Coordinate 1 is deliberately split into three tokens.  Its
                # mean, rather than its token count, must control its weight.
                "w4_coordinate_token_kl": [[3.0, 3.0, 3.0], [1.0], [1.0], [1.0]],
                "w8_coordinate_token_kl": [[1.0, 1.0, 1.0], [1.0], [1.0], [1.0]],
            })
    return rows


def test_coordinate_aggregation_equalizes_tokens_coordinates_and_cells():
    assert coordinate_row_kl([[3, 3, 3], [1], [1], [1]]) == pytest.approx(1.5)
    summary = aggregate_coordinate_candidate(_raw_rows())
    assert summary["n_cells"] == 8
    assert summary["kl_w4_macro"] == pytest.approx(1.5)
    assert summary["kl_w8_macro"] == pytest.approx(1.0)
    assert summary["repair_macro"] == pytest.approx(0.5)


def test_coordinate_aggregation_rejects_bad_groups_duplicates_and_missing_cells():
    with pytest.raises(ValueError, match="four"):
        coordinate_row_kl([[1], [2], [3]])
    rows = _raw_rows()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_coordinate_candidate(rows)
    with pytest.raises(ValueError, match="missing"):
        aggregate_coordinate_candidate(_raw_rows()[:-1])


def test_shortlist_is_positive_exact_cost_and_order_independent():
    candidates = [
        {"module_name": f"m{i:02d}", "repair_macro": float(i + 1), "delta_bytes": 10}
        for i in range(30)
    ]
    candidates.append({"module_name": "negative", "repair_macro": -100, "delta_bytes": 1})
    first = build_shortlist(candidates, top_k=24, source_hashes={"bank": "abc"})
    random.Random(9).shuffle(candidates)
    second = build_shortlist(candidates, top_k=24, source_hashes={"bank": "abc"})
    assert first == second
    assert len(first["candidates"]) == 24
    assert first["candidates"][0]["module_name"] == "m29"
    assert "negative" not in {row["module_name"] for row in first["candidates"]}
    unhashed = {key: value for key, value in first.items() if key != "shortlist_sha256"}
    assert first["shortlist_sha256"] == canonical_sha256(unhashed)


def _decoded(delta_by_cell=0.0):
    rows = []
    for task in ("rec", "coco_grounding"):
        for q in range(1, 5):
            rows.append({
                "uid": f"{task}:q{q}",
                "image_id": len(rows),
                "task": task,
                "area_quartile": q,
                "giou": 0.4 + delta_by_cell,
                "precise_iou": 0.5 + delta_by_cell,
                "box1000": [1, 1, 2, 2],
                "manifest_sha256": "manifest",
            })
    return rows


def test_paired_decoded_summary_is_strict_and_macro_balanced():
    baseline = _decoded(0.0)
    promoted = _decoded(0.125)
    summary = paired_decoded_summary(
        baseline, promoted, expected_manifest_sha256="manifest"
    )
    assert summary["giou_delta_macro"] == pytest.approx(0.125)
    assert summary["precise_iou_delta_macro"] == pytest.approx(0.125)

    reordered = promoted[::-1]
    with pytest.raises(ValueError, match="UID/order"):
        paired_decoded_summary(baseline, reordered)
    wrong_image = copy.deepcopy(promoted)
    wrong_image[0]["image_id"] = 999
    with pytest.raises(ValueError, match="image_id"):
        paired_decoded_summary(baseline, wrong_image)


def test_absolute_decoded_macro_binds_manifest_and_allocation_state():
    rows = _decoded(0.05)
    for row in rows:
        row["allocation_state_id"] = "state-one"
    summary = decoded_macro_summary(
        rows,
        expected_manifest_sha256="manifest",
        expected_state_id="state-one",
    )
    assert summary["mean_giou_macro"] == pytest.approx(0.45)
    wrong = copy.deepcopy(rows)
    wrong[0]["allocation_state_id"] = "state-two"
    with pytest.raises(ValueError, match="allocation state"):
        decoded_macro_summary(wrong, expected_state_id="state-one")


def test_rerank_cannot_change_shortlist_membership_and_uses_decoded_giou():
    candidates = [
        {"module_name": "a", "repair_macro": 2.0, "delta_bytes": 10},
        {"module_name": "b", "repair_macro": 1.0, "delta_bytes": 10},
    ]
    shortlist = build_shortlist(candidates, top_k=2)
    result = rerank_shortlist(
        shortlist,
        _decoded(0.0),
        {"a": _decoded(0.01), "b": _decoded(0.05)},
        expected_manifest_sha256="manifest",
    )
    assert [row["module_name"] for row in result["candidates"]] == ["b", "a"]
    with pytest.raises(ValueError, match="frozen shortlist"):
        rerank_shortlist(shortlist, _decoded(), {"a": _decoded()}, expected_manifest_sha256="manifest")

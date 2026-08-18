import copy

import pytest

from score_gcq_plan import score_plan


def _rows(state):
    rows = []
    for task in ("rec", "coco_grounding"):
        for q in range(1, 5):
            rows.append({
                "uid": f"{task}:q{q}",
                "image_id": len(rows),
                "task": task,
                "area_quartile": q,
                "giou": 0.25,
                "precise_iou": 0.5,
                "box1000": [1, 1, 2, 2],
                "manifest_sha256": "m" * 64,
                "allocation_state_id": state,
            })
    return rows


def test_score_plan_requires_exact_states_and_emits_macro_giou():
    context = "c" * 64
    plan = {
        "context_sha256": context,
        "run_fingerprint": "run",
        "round_index": 2,
        "states": [{"state_id": "s1"}, {"state_id": "s2"}],
    }
    result = score_plan(
        plan,
        {"s1": _rows("s1"), "s2": _rows("s2")},
        manifest_sha256="m" * 64,
        context_sha256=context,
    )
    assert result["scores"] == {"s1": pytest.approx(0.25), "s2": pytest.approx(0.25)}
    with pytest.raises(ValueError, match="differs from plan"):
        score_plan(
            plan, {"s1": _rows("s1")},
            manifest_sha256="m" * 64, context_sha256=context,
        )


def test_score_plan_rejects_wrong_context_manifest_and_state():
    context = "c" * 64
    plan = {"context_sha256": context, "states": [{"state_id": "s1"}]}
    with pytest.raises(ValueError, match="protocol context"):
        score_plan(plan, {"s1": _rows("s1")}, manifest_sha256="m" * 64, context_sha256="x" * 64)
    wrong = _rows("s1")
    wrong[0]["manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="manifest hash"):
        score_plan(plan, {"s1": wrong}, manifest_sha256="m" * 64, context_sha256=context)
    wrong_state = copy.deepcopy(_rows("s1"))
    wrong_state[0]["allocation_state_id"] = "s2"
    with pytest.raises(ValueError, match="allocation state"):
        score_plan(plan, {"s1": wrong_state}, manifest_sha256="m" * 64, context_sha256=context)


def test_score_plan_can_bind_exact_manifest_identity_and_order():
    context = "c" * 64
    plan = {"context_sha256": context, "states": [{"state_id": "s1"}]}
    rows = _rows("s1")
    manifest = [
        {field: row[field] for field in ("uid", "image_id", "task", "area_quartile")}
        for row in rows
    ]
    score_plan(
        plan,
        {"s1": rows},
        manifest_sha256="m" * 64,
        context_sha256=context,
        expected_manifest_records=manifest,
    )
    reversed_rows = list(reversed(rows))
    with pytest.raises(ValueError, match="mismatch at row 0"):
        score_plan(
            plan,
            {"s1": reversed_rows},
            manifest_sha256="m" * 64,
            context_sha256=context,
            expected_manifest_records=manifest,
        )

import copy

import pytest

from build_gcq_comparison_plan import (
    ComparisonPlanError,
    build_comparison_plan,
    canonical_sha256,
    state_id_for_members,
    write_plan_exclusive,
)


CONTEXT = "1" * 64
CATALOG = "2" * 64
RUN = "3" * 64


def selection(members, cost, cap):
    value = {
        "schema_version": 1,
        "artifact_kind": "gcq_frozen_beam_selection",
        "run_fingerprint": RUN,
        "catalog_hash": CATALOG,
        "context_sha256": CONTEXT,
        "primary_budget_bytes": 9,
        "selection_cap_bytes": cap,
        "selection_rule": "higher score; fewer bytes; lexicographic members",
        "state": {
            "state_id": state_id_for_members(members),
            "members": sorted(members),
            "cost_bytes": cost,
            "score": 0.5,
        },
    }
    value["selection_sha256"] = canonical_sha256(value)
    return value


def controls():
    def row(policy, members):
        return {
            "policy": policy,
            "members": sorted(members),
            "cost_bytes": 3,
            "matches_target": True,
        }

    return {
        "schema_version": 1,
        "artifact_kind": "gcq_exact_cost_matched_controls",
        "protocol_context_sha256": CONTEXT,
        "projection_catalog_sha256": CATALOG,
        "target_delta_bytes": 3,
        "score_driven_controls": {
            "additive_gcq": row("additive_gcq", ["a", "b"]),
            "vqa_driven": row("vqa_driven", ["b", "c"]),
            "maba_style_additive": row("maba_style_additive", ["a", "b"]),
        },
        "random_controls": {
            "no_best_seed_selection": True,
            "frozen_seeds": [11, 12],
            "samples": [
                {"seed": 11, "members": ["a", "b"], "cost_bytes": 3, "matches_target": True},
                {"seed": 12, "members": ["a", "c"], "cost_bytes": 3, "matches_target": True},
            ],
        },
        "coarse_historical_control": {
            "members": ["c"],
            "actual_cost_bytes": 1,
        },
    }


def test_build_deduplicates_member_sets_but_preserves_every_label():
    plan = build_comparison_plan(
        selection(["a", "b"], 3, 3),
        controls(),
        context_sha256=CONTEXT,
        secondary_selection=selection(["c"], 1, 1),
    )
    assert plan["artifact_kind"] == "gcq_frozen_comparison_plan"
    assert len(plan["label_to_state_id"]) == 9
    assert len(plan["states"]) == 5
    primary_id = state_id_for_members(["a", "b"])
    assert plan["label_to_state_id"]["gcq_primary"] == primary_id
    labels = next(row["labels"] for row in plan["states"] if row["state_id"] == primary_id)
    assert labels == ["additive_gcq", "gcq_primary", "maba_style_additive", "random_seed_11"]
    assert [row["members"] for row in plan["states"]] == sorted(
        [row["members"] for row in plan["states"]]
    )


def test_rejects_cross_catalog_and_cross_context_inputs():
    bad_controls = controls()
    bad_controls["projection_catalog_sha256"] = "9" * 64
    with pytest.raises(ComparisonPlanError, match="different catalogs"):
        build_comparison_plan(selection(["a", "b"], 3, 3), bad_controls, context_sha256=CONTEXT)
    bad_selection = selection(["a", "b"], 3, 3)
    bad_selection["context_sha256"] = "8" * 64
    bad_selection["selection_sha256"] = canonical_sha256(
        {key: value for key, value in bad_selection.items() if key != "selection_sha256"}
    )
    with pytest.raises(ComparisonPlanError, match="another launch protocol"):
        build_comparison_plan(bad_selection, controls(), context_sha256=CONTEXT)


def test_rejects_tampered_selection_and_mismatched_target():
    tampered = selection(["a", "b"], 3, 3)
    tampered["state"]["score"] = 1.0
    with pytest.raises(ComparisonPlanError, match="content hash"):
        build_comparison_plan(tampered, controls(), context_sha256=CONTEXT)
    bad_controls = copy.deepcopy(controls())
    bad_controls["target_delta_bytes"] = 2
    with pytest.raises(ComparisonPlanError, match="target differs"):
        build_comparison_plan(selection(["a", "b"], 3, 3), bad_controls, context_sha256=CONTEXT)


def test_exclusive_writer_refuses_overwrite(tmp_path):
    path = tmp_path / "comparison.json"
    digest = write_plan_exclusive(path, {"a": 1})
    assert len(digest) == 64
    with pytest.raises(ComparisonPlanError, match="refusing to overwrite"):
        write_plan_exclusive(path, {"a": 2})

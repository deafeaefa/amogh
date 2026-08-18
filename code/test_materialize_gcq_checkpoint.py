import copy

import pytest

from build_gcq_comparison_plan import build_comparison_plan, canonical_sha256
from materialize_gcq_checkpoint import MaterializationError, build_metadata, resolve_state


CONTEXT = "1" * 64
CATALOG = "2" * 64
RUN = "3" * 64


def selection(members, cost):
    value = {
        "schema_version": 1,
        "artifact_kind": "gcq_frozen_beam_selection",
        "run_fingerprint": RUN,
        "catalog_hash": CATALOG,
        "context_sha256": CONTEXT,
        "primary_budget_bytes": 3,
        "selection_cap_bytes": 3,
        "selection_rule": "higher score; fewer bytes; lexicographic members",
        "state": {
            "state_id": "state-" + canonical_sha256(sorted(members)),
            "members": sorted(members),
            "cost_bytes": cost,
            "score": 0.7,
        },
    }
    value["selection_sha256"] = canonical_sha256(value)
    return value


def controls():
    def row(policy, members):
        return {"policy": policy, "members": members, "cost_bytes": 3, "matches_target": True}
    return {
        "schema_version": 1,
        "artifact_kind": "gcq_exact_cost_matched_controls",
        "protocol_context_sha256": CONTEXT,
        "projection_catalog_sha256": CATALOG,
        "target_delta_bytes": 3,
        "score_driven_controls": {
            "additive_gcq": row("additive_gcq", ["a", "b"]),
            "vqa_driven": row("vqa_driven", ["b", "c"]),
            "maba_style_additive": row("maba_style_additive", ["b", "c"]),
        },
        "random_controls": {
            "no_best_seed_selection": True,
            "frozen_seeds": [7],
            "samples": [{"seed": 7, "members": ["a", "b"], "cost_bytes": 3, "matches_target": True}],
        },
    }


def test_resolve_frozen_selection_and_comparison_label():
    costs = {"a": 1, "b": 2, "c": 1}
    state, label = resolve_state(
        selection(["a", "b"], 3),
        label=None,
        context_sha256=CONTEXT,
        catalog_hash=CATALOG,
        cache_costs=costs,
    )
    assert label == "gcq_selection"
    assert state["members"] == ["a", "b"]
    plan = build_comparison_plan(selection(["a", "b"], 3), controls(), context_sha256=CONTEXT)
    state, label = resolve_state(
        plan,
        label="vqa_driven",
        context_sha256=CONTEXT,
        catalog_hash=CATALOG,
        cache_costs=costs,
    )
    assert label == "vqa_driven"
    assert state["members"] == ["b", "c"]


def test_resolve_rejects_tampering_and_cost_disagreement():
    costs = {"a": 1, "b": 2, "c": 1}
    plan = build_comparison_plan(selection(["a", "b"], 3), controls(), context_sha256=CONTEXT)
    bad = copy.deepcopy(plan)
    bad["label_to_state_id"]["vqa_driven"] = bad["label_to_state_id"]["gcq_primary"]
    with pytest.raises(MaterializationError, match="fingerprint"):
        resolve_state(
            bad,
            label="vqa_driven",
            context_sha256=CONTEXT,
            catalog_hash=CATALOG,
            cache_costs=costs,
        )
    with pytest.raises(MaterializationError, match="byte cost"):
        resolve_state(
            selection(["a", "b"], 3),
            label=None,
            context_sha256=CONTEXT,
            catalog_hash=CATALOG,
            cache_costs={"a": 1, "b": 1},
        )


class FakeCache:
    def composition_manifest(self, members):
        return {
            "added_code_bytes_over_uniform_w4": 3,
            "dense_qdq_is_compressed_checkpoint": False,
        }


def test_metadata_never_labels_dense_qdq_as_compressed():
    metadata = build_metadata(
        state={"state_id": "state-x", "members": ["a"], "cost_bytes": 3},
        label="gcq_primary",
        cache=FakeCache(),
        source_sha256={"protocol_context": CONTEXT},
    )
    assert metadata["dense_model"]["is_compressed_checkpoint"] is False
    assert metadata["packed_decoder"]["contains_actual_selected_payload"] is True
    assert metadata["packed_decoder"]["packed_inference_kernel_included"] is False

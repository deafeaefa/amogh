import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval_gcq_plan import (
    EvaluationProvenance,
    EVAL_IMPLEMENTATION_FILES,
    GenerationSettings,
    GroundingHelpers,
    PlanEvaluationError,
    artifact_canonical_sha256,
    build_state_rows,
    canonical_sha256,
    evaluate_plan_states,
    incremental_state_order,
    score_prediction,
    sha256_file,
    state_id_for_members,
    state_result_path,
    validate_candidate_catalog,
    validate_decode_manifest,
    validate_plan,
    validate_runtime_bindings,
    validate_state_rows,
)
from gptq_candidates import EXPECTED_DECODER_WEIGHTS, EXPECTED_PROJECTIONS
from recovery_utils import BASE_MODEL, BASE_REVISION


MODULE_A = "model.language_model.layers.0.self_attn.k_proj"
MODULE_B = "model.language_model.layers.0.self_attn.q_proj"


def _parse_box(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    box = value.get("bbox_2d") if isinstance(value, dict) else None
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(type(item) is not int for item in box)
        or not box[0] < box[2]
        or not box[1] < box[3]
    ):
        return None
    return box


def _to_pixels(box, width, height):
    return [
        box[0] * width / 1000.0,
        box[1] * height / 1000.0,
        box[2] * width / 1000.0,
        box[3] * height / 1000.0,
    ]


def _iou_giou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0.0
    cx1, cy1 = min(ax1, bx1), min(ay1, by1)
    cx2, cy2 = max(ax2, bx2), max(ay2, by2)
    hull = (cx2 - cx1) * (cy2 - cy1)
    return iou, iou - (hull - union) / hull if hull > 0 else iou


HELPERS = GroundingHelpers(
    parse_box=_parse_box,
    to_pixels=_to_pixels,
    iou_giou=_iou_giou,
)


def _manifest():
    rows = []
    for task in ("rec", "coco_grounding"):
        for quartile in range(1, 5):
            for _ in range(64):
                index = len(rows)
                image_id = 100_000 + index
                rows.append(
                    {
                        "uid": f"gcq_profile_decode_train_512:{index:05d}",
                        "candidate_id": f"candidate:{index}",
                        "task": task,
                        "source": "refcoco" if task == "rec" else "coco_detection",
                        "split": "train",
                        "image_id": image_id,
                        "file_name": f"COCO_train2014_{image_id:012d}.jpg",
                        "width": 100,
                        "height": 100,
                        "bbox_xywh": [10.0, 10.0, 20.0, 20.0],
                        "relative_area": 0.04,
                        "area_quartile": quartile,
                        "prompt": "Locate the object, output its bbox_2d in JSON.",
                        "answer": '{"bbox_2d": [100, 100, 300, 300]}',
                    }
                )
    return rows


def _state(members, costs=None):
    costs = costs or {MODULE_A: 1, MODULE_B: 1}
    members = sorted(members)
    return {
        "state_id": state_id_for_members(members),
        "members": members,
        "cost_bytes": sum(costs[name] for name in members),
        "parents": [],
    }


def _provenance():
    return EvaluationProvenance(
        plan_sha256="11" * 32,
        context_sha256="22" * 32,
        decode_manifest_sha256="33" * 32,
        candidate_bank_manifest_file_sha256="44" * 32,
        candidate_bank_manifest_content_sha256="55" * 32,
        generation_settings_sha256=GenerationSettings().sha256,
        run_fingerprint="66" * 32,
        round_index=2,
    )


class FakeCache:
    def __init__(self):
        self.manifest = {
            "candidates": [
                {"module_name": MODULE_A, "delta_bytes": 1},
                {"module_name": MODULE_B, "delta_bytes": 1},
            ]
        }
        self.manifest_sha256 = "55" * 32
        self.compose_calls = []

    def compose(self, model, promotions=(), *, previous_promotions=None, **kwargs):
        selected = frozenset(promotions)
        previous = None if previous_promotions is None else frozenset(previous_promotions)
        self.compose_calls.append((selected, previous))
        return selected


def test_manifest_is_exact_training_only_and_ordered():
    rows = _manifest()
    assert validate_decode_manifest(rows) == rows
    swapped = list(rows)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(PlanEvaluationError, match="UID/order"):
        validate_decode_manifest(swapped)
    holdout = [dict(row) for row in rows]
    holdout[0]["split"] = "testA"
    with pytest.raises(PlanEvaluationError, match="training-only"):
        validate_decode_manifest(holdout)
    holdout = [dict(row) for row in rows]
    holdout[0]["file_name"] = "COCO_val2014_000000100000.jpg"
    with pytest.raises(PlanEvaluationError, match="COCO train"):
        validate_decode_manifest(holdout)


def test_plan_validation_and_incremental_order_are_canonical():
    cache = FakeCache()
    states = [_state(()), _state((MODULE_A,)), _state((MODULE_A, MODULE_B))]
    context = {"allocation": {"primary_cap_added_payload_bytes": 2, "beam_width": 4}}
    plan = {
        "schema_version": 1,
        "context_sha256": "22" * 32,
        "objective": "maximize_score",
        "strict_positive_conditional_marginal": True,
        "pareto_prune": False,
        "budget_bytes": 2,
        "beam_width": 4,
        "kind": "expansion",
        "run_fingerprint": "66" * 32,
        "catalog_hash": "77" * 32,
        "round_index": 2,
        "states": states,
    }
    assert validate_plan(
        plan, cache=cache, context=context, context_sha256="22" * 32
    ) == states
    unordered = [
        _state((MODULE_A, MODULE_B)),
        _state((MODULE_B,)),
        _state(()),
        _state((MODULE_A,)),
    ]
    assert [row["members"] for row in incremental_state_order(unordered)] == [
        [],
        [MODULE_A],
        [MODULE_A, MODULE_B],
        [MODULE_B],
    ]
    bad = json.loads(json.dumps(plan))
    bad["states"][1]["cost_bytes"] = 2
    with pytest.raises(PlanEvaluationError, match="byte cost"):
        validate_plan(bad, cache=cache, context=context, context_sha256="22" * 32)


def test_candidate_catalog_is_cost_checked_and_hash_bound(tmp_path):
    cache = FakeCache()
    catalog = tmp_path / "shortlist.json"
    rows = [
        {"name": MODULE_B, "delta_bytes": 1},
        {"name": MODULE_A, "delta_bytes": 1},
    ]
    catalog.write_text(json.dumps({"candidates": rows}))
    digest, normalized = validate_candidate_catalog(catalog, cache)
    assert normalized == sorted(rows, key=lambda row: row["name"])
    assert digest == canonical_sha256(normalized)
    rows[0]["delta_bytes"] = 2
    catalog.write_text(json.dumps({"candidates": rows}))
    with pytest.raises(PlanEvaluationError, match="catalog/cache mismatch"):
        validate_candidate_catalog(catalog, cache)


def test_prediction_parsing_and_scoring_fields_match_score_bridge():
    record = _manifest()[0]
    exact = score_prediction(record, record["answer"], helpers=HELPERS)
    assert exact["box1000"] == [100, 100, 300, 300]
    assert exact["target_box1000"] == [100, 100, 300, 300]
    assert exact["iou"] == pytest.approx(1.0)
    assert exact["giou"] == pytest.approx(1.0)
    assert exact["precise_iou"] == pytest.approx(1.0)
    assert exact["parse_failed"] is False
    failed = score_prediction(record, "not JSON", helpers=HELPERS)
    assert failed["box1000"] is None
    assert failed["parse_failed"] is True
    assert failed["iou"] == 0.0
    assert failed["giou"] == -1.0
    assert failed["precise_iou"] == 0.0


def test_rows_preserve_manifest_identity_and_validate_provenance():
    manifest = _manifest()
    state = _state((MODULE_A,))
    predictions = [record["answer"] for record in manifest]
    rows = build_state_rows(
        state,
        manifest,
        [{"text": value, "truncated": index == 0} for index, value in enumerate(predictions)],
        provenance=_provenance(),
        helpers=HELPERS,
    )
    assert [row["uid"] for row in rows] == [row["uid"] for row in manifest]
    assert all(row["box1000"] == [100, 100, 300, 300] for row in rows)
    assert rows[0]["generation_truncated"] is True
    assert all(not row["generation_truncated"] for row in rows[1:])
    summary = validate_state_rows(
        rows, state, manifest, provenance=_provenance(), helpers=HELPERS
    )
    assert summary["n"] == 512
    assert summary["mean_giou_macro"] == pytest.approx(1.0)
    rows[17]["context_sha256"] = "ff" * 32
    with pytest.raises(PlanEvaluationError, match="provenance mismatch"):
        validate_state_rows(
            rows, state, manifest, provenance=_provenance(), helpers=HELPERS
        )


def test_write_once_resume_and_incremental_composition(tmp_path):
    cache = FakeCache()
    manifest = _manifest()
    states = [_state(()), _state((MODULE_A,)), _state((MODULE_A, MODULE_B))]
    decode_calls = []

    def decoder(model, records):
        decode_calls.append(len(records))
        return [row["answer"] for row in records]

    first = evaluate_plan_states(
        cache=cache,
        model=object(),
        states=states,
        manifest=manifest,
        provenance=_provenance(),
        output_dir=tmp_path,
        decoder=decoder,
        helpers=HELPERS,
    )
    assert len(first) == 3
    assert decode_calls == [512, 512, 512]
    assert cache.compose_calls == [
        (frozenset(), None),
        (frozenset(), frozenset()),
        (frozenset({MODULE_A}), frozenset()),
        (frozenset({MODULE_A, MODULE_B}), frozenset({MODULE_A})),
    ]
    cache.compose_calls.clear()
    decode_calls.clear()
    resumed = evaluate_plan_states(
        cache=cache,
        model=object(),
        states=states,
        manifest=manifest,
        provenance=_provenance(),
        output_dir=tmp_path,
        decoder=decoder,
        helpers=HELPERS,
    )
    assert all(row["resumed"] for row in resumed.values())
    assert not cache.compose_calls
    assert not decode_calls

    existing_path = state_result_path(tmp_path, states[1]["state_id"])
    existing_hash = sha256_file(existing_path)
    from eval_gcq_plan import write_state_rows_exclusive
    with pytest.raises(FileExistsError):
        write_state_rows_exclusive(existing_path, [{"different": True}])
    assert sha256_file(existing_path) == existing_hash

    corrupt_path = state_result_path(tmp_path, states[0]["state_id"])
    lines = corrupt_path.read_text().splitlines()
    row = json.loads(lines[0])
    row["uid"] = "wrong"
    lines[0] = json.dumps(row)
    corrupt_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(PlanEvaluationError, match="identity mismatch"):
        evaluate_plan_states(
            cache=cache,
            model=object(),
            states=states,
            manifest=manifest,
            provenance=_provenance(),
            output_dir=tmp_path,
            decoder=decoder,
            helpers=HELPERS,
        )


def _full_persisted_manifest():
    rows = []
    for index in range(EXPECTED_PROJECTIONS):
        rows.append(
            {
                "module_name": f"model.language_model.layers.{index // 7}.projection_{index}",
                "delta_bytes": 1,
                "w4": {"logical_payload_bytes": 3},
                "w8": {"logical_payload_bytes": 4},
            }
        )
    return {
        "recipe": {
            "base_model": BASE_MODEL,
            "revision": BASE_REVISION,
            "bits": [4, 8],
            "prefix_policy": "earlier_decoder_layers_cached_w4",
        },
        "architecture": {
            "projections": EXPECTED_PROJECTIONS,
            "weights": EXPECTED_DECODER_WEIGHTS,
        },
        "candidates": rows,
    }


def test_launch_context_binds_model_cache_manifest_and_worker(tmp_path):
    decode = tmp_path / "gcq_profile_decode_train_512.json"
    decode.write_text(json.dumps(_manifest()))
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    bank_manifest = cache_root / "manifest.json"
    bank = _full_persisted_manifest()
    bank_manifest.write_text(json.dumps(bank))
    cache = SimpleNamespace(
        root=cache_root,
        manifest=bank,
        manifest_sha256="55" * 32,
    )
    implementation_files = {
        name: sha256_file(Path(__file__).with_name(name).resolve())
        for name in EVAL_IMPLEMENTATION_FILES
    }
    context = {
        "schema_version": 1,
        "status": "launch_frozen",
        "model": {
            "id": BASE_MODEL,
            "revision": BASE_REVISION,
            "expected_projection_count": EXPECTED_PROJECTIONS,
            "expected_decoder_weight_count": EXPECTED_DECODER_WEIGHTS,
        },
        "profiling_data": {"decode_manifest": decode.name},
        "allocation": {
            "primary_cap_added_payload_bytes": 2,
            "beam_width": 4,
        },
        "bound_hashes": {
            "decode_manifest_sha256": sha256_file(decode),
            "candidate_bank_manifest_sha256": sha256_file(bank_manifest),
        },
        "bound_inputs": {
            "decode_manifest": {
                "file_name": decode.name,
                "sha256": sha256_file(decode),
            },
            "candidate_bank_manifest": {
                "file_name": bank_manifest.name,
                "sha256": sha256_file(bank_manifest),
            },
        },
        "implementation_files": implementation_files,
    }
    context["protocol_sha256"] = artifact_canonical_sha256(context)
    context_path = tmp_path / "frozen_protocol.json"
    context_path.write_text(json.dumps(context))
    validated, context_hash, decode_hash, bank_hash = validate_runtime_bindings(
        context_path, decode, cache
    )
    assert validated["status"] == "launch_frozen"
    assert context_hash == sha256_file(context_path)
    assert decode_hash == sha256_file(decode)
    assert bank_hash == sha256_file(bank_manifest)

    decode.write_text(json.dumps(_manifest()) + "\n")
    with pytest.raises(PlanEvaluationError, match="protocol-bound"):
        validate_runtime_bindings(context_path, decode, cache)

import json
from pathlib import Path

import pytest

from freeze_gcq_upgrade import (
    bind_protocol,
    canonical_sha256,
    validate_candidate_bank,
    validate_profile_pair,
)


def _profile(start_image):
    rows = []
    for task in ("rec", "coco_grounding"):
        for quartile in range(1, 5):
            for cell_index in range(64):
                image_id = start_image + len(rows)
                row = {
                    "uid": f"{start_image}:{task}:q{quartile}:{cell_index}",
                    "image_id": image_id,
                    "task": task,
                    "area_quartile": quartile,
                    "relative_area": (quartile + cell_index / 1000) / 10,
                    "answer": '{"bbox_2d": [1, 2, 3, 4]}',
                }
                if task == "coco_grounding":
                    row["absolute_size"] = ("small", "medium", "large")[cell_index % 3]
                rows.append(row)
    return rows


def _bank():
    rows = []
    roles = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    )
    for layer in range(28):
        for role in roles:
            rows.append({
                "module_name": f"language_model.layers.{layer}.{role}",
                "delta_bytes": 100,
                "w4_sha256": "04" * 32,
                "w8_sha256": "08" * 32,
            })
    return {"candidates": rows}


def _vqa_control():
    records = [
        {
            "uid": f"vqa:{index}",
            "source": "vqav2_train",
            "split": "train",
            "task": "vqa",
            "image_id": 20_000 + index,
            "question_id": 30_000 + index,
            "prompt": "What is shown? Answer with a single word or phrase.",
            "answer": "object",
        }
        for index in range(512)
    ]
    from build_gcq_vqa_control_data import canonical_sha256

    return {
        "schema_version": 1,
        "artifact_kind": "gcq_vqa_control_training_manifest",
        "rows": 512,
        "unique_images": 512,
        "records": records,
        "records_sha256": canonical_sha256(records),
    }


def test_profile_pair_and_candidate_bank_validation():
    summary = validate_profile_pair(_profile(0), _profile(10_000))
    assert summary["proxy"]["examples"] == 512
    assert validate_candidate_bank(_bank())["projections"] == 196
    overlap = _profile(0)
    with pytest.raises(ValueError, match="overlap"):
        validate_profile_pair(overlap, overlap)
    malformed = _bank()
    malformed["candidates"].pop()
    with pytest.raises(ValueError, match="expected 196"):
        validate_candidate_bank(malformed)


def test_candidate_bank_strictly_binds_design_and_calibration_sources():
    design = {
        "model": {
            "id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": "r" * 40,
            "expected_projection_count": 196,
            "expected_decoder_weight_count": 123,
        },
        "quantization": {
            "candidate_bits": [4, 8],
            "group_size": 128,
            "block_size": 128,
            "percdamp": 0.01,
            "scale_dtype": "float16",
        },
    }
    bank = _bank()
    bank.update({
        "schema_version": 1,
        "artifact_kind": "gcq_packed_gptq_projection_bank",
        "recipe": {
            "base_model": design["model"]["id"],
            "revision": design["model"]["revision"],
            "bits": [4, 8],
            "group_size": 128,
            "block_size": 128,
            "percdamp": 0.01,
            "scale_dtype": "float16",
            "prefix_policy": "earlier_decoder_layers_cached_w4",
        },
        "architecture": {"projections": 196, "weights": 123},
        "provenance": {"source_files": {
            "protocol_file_sha256": "d" * 64,
            "calibration_manifest_file_sha256": "c" * 64,
        }},
    })
    bank["manifest_content_sha256"] = canonical_sha256(bank)
    validate_candidate_bank(
        bank,
        design=design,
        design_file_sha256="d" * 64,
        calibration_file_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="calibration_manifest_file_sha256 mismatch"):
        validate_candidate_bank(
            bank,
            design=design,
            design_file_sha256="d" * 64,
            calibration_file_sha256="x" * 64,
        )


def test_bind_protocol_hashes_every_input_without_scores(tmp_path):
    design = {
        "status": "design_frozen_inputs_unbound",
        "bound_hashes": None,
        "required_bound_hashes": [
            "implementation_tree_sha256",
            "calibration_manifest_sha256",
            "proxy_manifest_sha256",
            "decode_manifest_sha256",
            "vqa_control_manifest_sha256",
            "candidate_bank_manifest_sha256",
            "packing_spec_sha256",
        ],
    }
    values = {
        "design": design,
        "calibration": {"role": "calibration"},
        "proxy": _profile(0),
        "decode": _profile(10_000),
        "vqa_control": _vqa_control(),
        "bank": _bank(),
        "packing": {"format": "test"},
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        paths[name] = path
    implementation = tmp_path / "implementation.py"
    implementation.write_text("pass\n")
    result = bind_protocol(
        paths["design"],
        calibration_manifest=paths["calibration"],
        proxy_manifest=paths["proxy"],
        decode_manifest=paths["decode"],
        vqa_control_manifest=paths["vqa_control"],
        candidate_bank_manifest=paths["bank"],
        packing_spec=paths["packing"],
        implementation_files=[implementation],
        bound_at_utc="2026-08-18T00:00:00Z",
    )
    assert result["status"] == "launch_frozen"
    digest = result.pop("protocol_sha256")
    assert digest == canonical_sha256(result)
    assert set(result["bound_hashes"]) == set(design["required_bound_hashes"])

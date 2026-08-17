"""Preflight the complete valid checkpoint inventory for development round 2.

This validator is score-blind.  It checks training artifacts and frozen
development inputs only, including the known corrupt step-750 root adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from safetensors import safe_open


RECIPE = "gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
ALL_SAVED = tuple(range(50, 751, 50))
VALID_STEPS = tuple(range(50, 701, 50))
PARENT_VALID = (200, 300, 400, 500, 600)
CANDIDATES = (50, 100, 150, 250, 350, 450, 550, 650, 700)
INVALID_STEPS = (750,)
EXPECTED_TENSORS = 392
EXPECTED_PARAMETERS = 17_432_576
EXPECTED_FILE_BYTES = 69_788_264
EXPECTED_STEP750_ALLOCATED_BYTES = 16_777_216
EXPECTED_STEP750_ZERO_TENSORS = 296
EXPECTED_STEP750_SHA = "513d8902b2b751059b5fbde617423f90306995376826d476c19c40ebb1ae22e0"
DATA_SHA = "8bf3b6a1589527f5847ea28a7c5f0daeb89f6e0d7fa220451db87c52314aec4a"
PROMOTION_SHA = "78b3138da1c70f2f944e3c63d5cfc754f59bf6062c27564d8cc6b2792ba0c1e6"
PARENT_SUMMARY_SHA = "4b923471b3af3244b7874995e871ae38619c61298bdf07851ba93fac5cc35698"
AMENDMENT_SHA = "3261c3cfa13401d5577d831f9d52a500b7a74d186c8314289e51395b4e78d3f6"
AMENDED_WRAPPER_SHA = "63ccce3388abd53ea46a793d6b2fbde923639eb5bb9460acf8bb9c30e673206d"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON input: {path}")
    with open(path) as stream:
        return json.load(stream)


def tensor_inventory(path: Path) -> dict[str, Any]:
    stat = path.stat()
    tensors: list[dict[str, Any]] = []
    zero_names: list[str] = []
    norm_squared = 0.0
    parameters = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for name in keys:
            tensor = handle.get_tensor(name)
            norm = float(tensor.float().norm().item())
            require(math.isfinite(norm), f"non-finite tensor norm in {path}: {name}")
            if norm == 0.0:
                zero_names.append(name)
            parameters += tensor.numel()
            norm_squared += norm * norm
            tensors.append({
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            })
    structure_bytes = json.dumps(
        tensors, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "apparent_bytes": stat.st_size,
        "allocated_bytes": stat.st_blocks * 512,
        "tensor_count": len(tensors),
        "parameter_count": parameters,
        "lora_a_count": sum(".lora_A.weight" in item["name"] for item in tensors),
        "lora_b_count": sum(".lora_B.weight" in item["name"] for item in tensors),
        "zero_tensor_count": len(zero_names),
        "zero_tensor_names": zero_names,
        "aggregate_l2_norm": math.sqrt(norm_squared),
        "structure_sha256": hashlib.sha256(structure_bytes).hexdigest(),
        "tensors": tensors,
    }


def validate_manifest(path: Path, step: int, artifact: dict[str, Any]) -> dict[str, Any]:
    manifest = load_json(path)
    require(manifest.get("schema_version") == 1, f"bad manifest schema at step {step}")
    require(manifest.get("checkpoint_step") == step, f"bad checkpoint step at {path}")
    require(manifest.get("base_model") == MODEL, f"bad model at {path}")
    require(manifest.get("base_revision") == REVISION, f"bad revision at {path}")
    data = manifest.get("data", {})
    require(data.get("sha256") == DATA_SHA and data.get("examples") == 12_000,
            f"bad training data binding at {path}")
    require(manifest.get("objective") == {"name": "cwce", "coordinate_weight": 5.0},
            f"bad objective at {path}")
    optimization = manifest.get("optimization", {})
    for key, expected in {
        "batch_size": 2, "effective_batch_size": 16, "epochs": 1,
        "gradient_accumulation": 8, "learning_rate": 0.00005,
        "seed": 0, "total_optimizer_steps": 750,
    }.items():
        require(optimization.get(key) == expected,
                f"bad optimization {key} at {path}")
    lora = manifest.get("lora", {})
    for key, expected in {
        "rank": 16, "alpha": 32, "dropout": 0.05,
        "trainable_parameters": EXPECTED_PARAMETERS, "targeted_linears": 196,
    }.items():
        require(lora.get(key) == expected, f"bad LoRA {key} at {path}")
    quantization = manifest.get("quantization", {})
    require(quantization.get("method") == "rtn_quantize_dequantize"
            and quantization.get("bits") == 4
            and quantization.get("group_size") == 128
            and quantization.get("promote_sha256") == PROMOTION_SHA,
            f"bad quantization binding at {path}")
    recorded = manifest.get("artifact", {})
    require(recorded.get("file_bytes") == artifact["apparent_bytes"],
            f"artifact byte count mismatch at {path}")
    require(recorded.get("sha256") == artifact["sha256"],
            f"artifact hash mismatch at {path}")
    return manifest


def validate_adapter_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    require(config.get("base_model_name_or_path") == MODEL, f"bad adapter base model: {path}")
    require(config.get("peft_type") == "LORA" and config.get("task_type") == "CAUSAL_LM",
            f"bad adapter type: {path}")
    require(config.get("r") == 16 and config.get("lora_alpha") == 32
            and config.get("lora_dropout") == 0.05 and config.get("bias") == "none",
            f"bad adapter hyperparameters: {path}")
    require(set(config.get("target_modules", [])) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }, f"bad adapter target modules: {path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--code-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    protocol_path = Path(args.protocol).resolve()
    runs = Path(args.runs).resolve()
    code_dir = Path(args.code_dir).resolve()
    root = runs / "recovery_vqa_replay"
    adapter_root = root / "adapters" / RECIPE

    protocol = load_json(protocol_path)
    require(protocol.get("round_id") == "balanced-replay-remaining-checkpoints-development-round-2",
            "unexpected round-2 protocol ID")
    require(tuple(protocol.get("candidate_steps", [])) == CANDIDATES,
            "round-2 candidates changed")
    inventory = protocol.get("training_checkpoint_inventory", {})
    require(tuple(inventory.get("systematic_saved_steps", [])) == ALL_SAVED,
            "saved checkpoint inventory changed")
    require(tuple(inventory.get("structurally_valid_steps", [])) == VALID_STEPS,
            "valid checkpoint inventory changed")
    require(tuple(inventory.get("parent_valid_evaluated_steps", [])) == PARENT_VALID,
            "parent valid checkpoint inventory changed")
    require(tuple(inventory.get("invalid_steps", [])) == INVALID_STEPS,
            "invalid checkpoint inventory changed")
    require(tuple(step for step in VALID_STEPS if step not in PARENT_VALID) == CANDIDATES,
            "candidate complement is incomplete")

    actual_checkpoint_dirs = sorted(
        int(path.name.split("-")[-1])
        for path in adapter_root.glob("checkpoint-*") if path.is_dir()
    )
    require(actual_checkpoint_dirs == list(VALID_STEPS),
            f"unexpected saved checkpoint directory inventory: {actual_checkpoint_dirs}")

    parent_paths = {
        "summary": root / "development_summary.json",
        "amendment": code_dir / "development_summarizer_amendment.json",
        "amended_wrapper": code_dir / "summarize_recovery_vqa_development_amended.py",
    }
    for label, expected in {
        "summary": PARENT_SUMMARY_SHA,
        "amendment": AMENDMENT_SHA,
        "amended_wrapper": AMENDED_WRAPPER_SHA,
    }.items():
        require(sha256_file(parent_paths[label]) == expected,
                f"parent {label} hash changed")
    parent_summary = load_json(parent_paths["summary"])
    require(parent_summary.get("selection_succeeded") is False
            and parent_summary.get("eligible_steps") == []
            and sorted(map(int, parent_summary.get("candidates", {})))
            == [200, 300, 400, 500, 600, 750],
            "parent development failure contract changed")

    checkpoints: dict[str, Any] = {}
    reference_structure: str | None = None
    for step in VALID_STEPS:
        directory = adapter_root / f"checkpoint-{step:06d}"
        artifact = tensor_inventory(directory / "adapter_model.safetensors")
        require(artifact["tensor_count"] == EXPECTED_TENSORS
                and artifact["parameter_count"] == EXPECTED_PARAMETERS,
                f"bad tensor inventory at step {step}")
        require(artifact["lora_a_count"] == 196 and artifact["lora_b_count"] == 196,
                f"incomplete LoRA A/B inventory at step {step}")
        require(artifact["zero_tensor_count"] == 0,
                f"zero LoRA tensor at trained checkpoint {step}")
        require(artifact["apparent_bytes"] == EXPECTED_FILE_BYTES,
                f"bad apparent adapter size at step {step}")
        require(artifact["allocated_bytes"] >= artifact["apparent_bytes"],
                f"sparse or incompletely allocated adapter at step {step}")
        if reference_structure is None:
            reference_structure = artifact["structure_sha256"]
        require(artifact["structure_sha256"] == reference_structure,
                f"tensor key/shape/dtype structure changed at step {step}")
        config_path = directory / "adapter_config.json"
        manifest_path = directory / "gcq_recovery_manifest.json"
        validate_adapter_config(config_path)
        validate_manifest(manifest_path, step, artifact)
        artifact.pop("tensors")
        artifact.pop("zero_tensor_names")
        checkpoints[str(step)] = {
            "directory": str(directory),
            "artifact": artifact,
            "adapter_config_sha256": sha256_file(config_path),
            "manifest_sha256": sha256_file(manifest_path),
        }

    final_artifact = tensor_inventory(adapter_root / "adapter_model.safetensors")
    require(final_artifact["sha256"] == EXPECTED_STEP750_SHA,
            "known step-750 artifact bytes changed")
    require(final_artifact["tensor_count"] == EXPECTED_TENSORS
            and final_artifact["parameter_count"] == EXPECTED_PARAMETERS
            and final_artifact["zero_tensor_count"] == EXPECTED_STEP750_ZERO_TENSORS,
            "step-750 corruption signature changed")
    require(final_artifact["apparent_bytes"] == EXPECTED_FILE_BYTES
            and final_artifact["allocated_bytes"] == EXPECTED_STEP750_ALLOCATED_BYTES
            and final_artifact["allocated_bytes"] < final_artifact["apparent_bytes"],
            "step-750 sparse-allocation signature changed")
    tokenizer = adapter_root / "processor" / "tokenizer.json"
    require(tokenizer.is_file() and tokenizer.stat().st_size == 0,
            "step-750 zero-byte tokenizer incident signature changed")

    report = {
        "schema_version": 1,
        "role": "score-blind development-round-2 artifact integrity preflight",
        "round_id": protocol["round_id"],
        "recipe_id": RECIPE,
        "base_model": MODEL,
        "base_revision": REVISION,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "candidate_steps": list(CANDIDATES),
        "valid_saved_steps": list(VALID_STEPS),
        "parent_valid_evaluated_steps": list(PARENT_VALID),
        "checkpoints": {str(step): checkpoints[str(step)] for step in CANDIDATES},
        "all_valid_checkpoint_integrity": {
            "steps": list(VALID_STEPS),
            "all_tensors_nonzero": True,
            "all_files_fully_allocated": True,
            "common_structure_sha256": reference_structure,
        },
        "step750_integrity_incident": {
            "status": "invalid_excluded_before_round2_launch",
            "artifact": {key: value for key, value in final_artifact.items()
                         if key not in {"tensors", "zero_tensor_names"}},
            "zero_tensor_names_sha256": hashlib.sha256(
                "\n".join(final_artifact["zero_tensor_names"]).encode()
            ).hexdigest(),
            "processor_tokenizer_bytes": 0,
        },
        "parent_bindings": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in parent_paths.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({
        "output": str(output),
        "candidate_steps": list(CANDIDATES),
        "valid_steps": list(VALID_STEPS),
        "step750_status": "invalid_excluded_before_round2_launch",
    }, sort_keys=True))


if __name__ == "__main__":
    main()

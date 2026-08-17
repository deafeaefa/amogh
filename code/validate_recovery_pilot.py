"""Fail-closed validation for the four recovery-pilot adapter artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open

from recovery_utils import BASE_MODEL, BASE_REVISION, sha256_file


ARMS = {
    "w4rtn_lora_ce_s0": {"objective": "ce", "gamma": 1.0, "gcq": False},
    "w4rtn_lora_cwce_g5_s0": {"objective": "cwce", "gamma": 5.0, "gcq": False},
    "gcq425_lora_ce_s0": {"objective": "ce", "gamma": 1.0, "gcq": True},
    "gcq425_lora_cwce_g5_s0": {"objective": "cwce", "gamma": 5.0, "gcq": True},
}
EXPECTED_DATA_SHA256 = "b89f3d391a9bce0553d5213babc8eeadc55b3dab367d6398088b5ad58fba4f62"
EXPECTED_PROMOTE_SHA256 = "78b3138da1c70f2f944e3c63d5cfc754f59bf6062c27564d8cc6b2792ba0c1e6"
EXPECTED_LORA_PARAMETERS = 17_432_576


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def inspect_safetensors(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size > 0, f"missing adapter tensor file: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        parameter_count = 0
        dtypes = set()
        for key in keys:
            tensor = tensors.get_tensor(key)
            parameter_count += tensor.numel()
            dtypes.add(str(tensor.dtype))
    require(parameter_count == EXPECTED_LORA_PARAMETERS,
            f"{path}: {parameter_count} parameters != {EXPECTED_LORA_PARAMETERS}")
    require(len(keys) == 392, f"{path}: expected 392 LoRA tensors, found {len(keys)}")
    return {
        "file_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "parameter_count": parameter_count,
        "tensor_count": len(keys),
        "dtypes": sorted(dtypes),
        "bf16_deployment_parameter_bytes": parameter_count * 2,
    }


def validate_arm(root: Path, tag: str, expected: dict, data_sha: str, promote_sha: str) -> dict:
    adapter_dir = root / "adapters" / tag
    manifest_path = adapter_dir / "gcq_recovery_manifest.json"
    require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    prefix = f"{tag}: "
    require(manifest.get("base_model") == BASE_MODEL, prefix + "wrong base model")
    require(manifest.get("base_revision") == BASE_REVISION, prefix + "wrong base revision")
    require(manifest.get("data", {}).get("sha256") == data_sha, prefix + "wrong train-data hash")
    require(manifest.get("data", {}).get("examples") == 8000, prefix + "expected 8000 examples")
    require(manifest.get("data", {}).get("max_samples") == 0, prefix + "max_samples is not zero")

    objective = manifest.get("objective", {})
    require(objective.get("name") == expected["objective"], prefix + "wrong objective")
    require(float(objective.get("coordinate_weight", -1)) == expected["gamma"], prefix + "wrong gamma")

    quantization = manifest.get("quantization", {})
    require(quantization.get("method") == "rtn_quantize_dequantize", prefix + "wrong quantizer")
    require(quantization.get("bits") == 4 and quantization.get("group_size") == 128,
            prefix + "wrong W4/group configuration")
    require(quantization.get("quantized_linears") == 196, prefix + "wrong quantized-linear count")
    if expected["gcq"]:
        require(quantization.get("promote_sha256") == promote_sha, prefix + "wrong promotion hash")
        require(quantization.get("promoted_linears") == 28, prefix + "wrong promoted-linear count")
    else:
        require(quantization.get("promote_sha256") is None, prefix + "unexpected promotions")
        require(quantization.get("promoted_linears") == 0, prefix + "unexpected promoted linears")

    lora = manifest.get("lora", {})
    require(lora.get("rank") == 16 and lora.get("alpha") == 32, prefix + "wrong LoRA rank/alpha")
    require(float(lora.get("dropout", -1)) == 0.05, prefix + "wrong LoRA dropout")
    require(lora.get("targeted_linears") == 196, prefix + "wrong LoRA target count")
    require(lora.get("trainable_parameters") == EXPECTED_LORA_PARAMETERS,
            prefix + "wrong trainable-parameter count")

    optimization = manifest.get("optimization", {})
    require(optimization.get("seed") == 0, prefix + "wrong seed")
    require(optimization.get("effective_batch_size") == 16, prefix + "wrong effective batch")
    require(optimization.get("total_optimizer_steps") == 500, prefix + "wrong planned step count")
    require(manifest.get("completed", {}).get("optimizer_steps") == 500, prefix + "training incomplete")
    verification = manifest.get("verification", {})
    require(verification.get("zero_init_max_logit_diff") == 0.0, prefix + "zero-init parity failed")
    require(verification.get("first_step_nonzero_lora_grad") is True, prefix + "gradient check failed")
    require(verification.get("base_parameters_received_grad") is False, prefix + "base received gradient")

    artifact = inspect_safetensors(adapter_dir / "adapter_model.safetensors")
    return {
        "manifest": str(manifest_path),
        "objective": objective,
        "quantization": {
            "promote_sha256": quantization.get("promote_sha256"),
            "promoted_linears": quantization.get("promoted_linears"),
        },
        "training_seconds": manifest["completed"]["seconds"],
        "artifact": artifact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-empty-eval", action="store_true")
    args = parser.parse_args()

    runs = Path(os.environ["GCQ_RUNS"])
    data = Path(os.environ["GCQ_DATA"])
    pilot = runs / "recovery_pilot"
    train_path = data / "subsets" / "recovery_train_8k.json"
    promote_path = runs / "promote_gcq_b4.25.json"
    data_sha = sha256_file(train_path)
    promote_sha = sha256_file(promote_path)
    require(data_sha == EXPECTED_DATA_SHA256, "frozen training manifest hash changed")
    require(promote_sha == EXPECTED_PROMOTE_SHA256, "frozen GCQ promotion map hash changed")

    if args.require_empty_eval:
        for tag in ARMS:
            eval_dir = pilot / "eval" / tag
            require(not eval_dir.exists() or not any(eval_dir.iterdir()),
                    f"refusing to mix stale evaluation outputs: {eval_dir}")

    report = {
        "schema_version": 1,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "train_manifest_sha256": data_sha,
        "promote_sha256": promote_sha,
        "arms": {
            tag: validate_arm(pilot, tag, expected, data_sha, promote_sha)
            for tag, expected in ARMS.items()
        },
    }
    output = pilot / "pilot_artifact_validation.json"
    with open(output, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"RECOVERY PILOT ARTIFACTS VALID: {output}")


if __name__ == "__main__":
    main()

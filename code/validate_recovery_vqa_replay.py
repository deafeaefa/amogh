"""Fail-closed validation for the balanced VQAv2-replay recovery run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open

from recovery_utils import BASE_MODEL, BASE_REVISION, sha256_file


RECIPE_ID = "gcq425_lora_cwce_vqa50_g5_lr5e5_s0"
DATA_SHA256 = "8bf3b6a1589527f5847ea28a7c5f0daeb89f6e0d7fa220451db87c52314aec4a"
METADATA_SHA256 = "5e11f95ae6baebe0d92ea384e80f78bf6e0929003769e8e4d08767d85346b375"
PROMOTION_SHA256 = "78b3138da1c70f2f944e3c63d5cfc754f59bf6062c27564d8cc6b2792ba0c1e6"
PROTOCOL_SHA256 = "fd938d0d39116b989ffdcde4dd5ce64bbb419a1292e4ab9f32864416953e5e6d"
CHECKPOINT_STEPS = (200, 300, 400, 500, 600, 750)
EXPECTED_LORA_PARAMETERS = 17_432_576
EXPECTED_LORA_TARGETS = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON file: {path}")
    with open(path) as f:
        return json.load(f)


def inspect_adapter(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size > 0, f"missing adapter tensor: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        parameters = 0
        dtypes = set()
        for key in keys:
            tensor = tensors.get_tensor(key)
            parameters += tensor.numel()
            dtypes.add(str(tensor.dtype))
    require(len(keys) == 392, f"{path}: expected 392 tensors, found {len(keys)}")
    require(parameters == EXPECTED_LORA_PARAMETERS,
            f"{path}: expected {EXPECTED_LORA_PARAMETERS} parameters, found {parameters}")
    require(dtypes == {"torch.float32"}, f"{path}: expected FP32 tensors, found {dtypes}")
    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensor_count": len(keys),
        "parameter_count": parameters,
        "dtypes": sorted(dtypes),
    }


def inspect_adapter_config(path: Path) -> dict:
    config = load_json(path)
    require(config.get("base_model_name_or_path") == BASE_MODEL,
            f"{path}: wrong base model")
    require(config.get("peft_type") == "LORA", f"{path}: wrong PEFT type")
    require(config.get("task_type") == "CAUSAL_LM", f"{path}: wrong task type")
    require(config.get("inference_mode") is True, f"{path}: adapter is not inference-only")
    require(config.get("r") == 16 and config.get("lora_alpha") == 32,
            f"{path}: wrong LoRA rank/alpha")
    require(float(config.get("lora_dropout", -1)) == 0.05,
            f"{path}: wrong LoRA dropout")
    require(config.get("bias") == "none", f"{path}: unexpected LoRA bias")
    require(config.get("modules_to_save") is None, f"{path}: unexpected modules_to_save")
    require(config.get("use_dora") is False, f"{path}: unexpected DoRA adapter")
    require(config.get("use_rslora") is False, f"{path}: unexpected rank-stabilized LoRA")
    targets = config.get("target_modules")
    require(isinstance(targets, list) and set(targets) == EXPECTED_LORA_TARGETS
            and len(targets) == len(EXPECTED_LORA_TARGETS),
            f"{path}: wrong LoRA target modules")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
    }


def validate_manifest(manifest: dict, *, step: int, is_final: bool) -> None:
    prefix = f"step {step}: "
    require(manifest.get("base_model") == BASE_MODEL, prefix + "wrong base model")
    require(manifest.get("base_revision") == BASE_REVISION, prefix + "wrong base revision")
    data = manifest.get("data", {})
    require(data.get("sha256") == DATA_SHA256, prefix + "wrong data hash")
    require(data.get("examples") == 12000, prefix + "wrong data count")
    require(data.get("max_samples") == 0, prefix + "max_samples is nonzero")
    require(Path(data.get("path", "")).name == "recovery_train_vqa_replay_12k.json",
            prefix + "wrong data path")
    objective = manifest.get("objective", {})
    require(objective.get("name") == "cwce", prefix + "wrong objective")
    require(float(objective.get("coordinate_weight", -1)) == 5.0, prefix + "wrong gamma")
    quantization = manifest.get("quantization", {})
    require(quantization.get("method") == "rtn_quantize_dequantize", prefix + "wrong quantizer")
    require(quantization.get("bits") == 4, prefix + "wrong quantization bits")
    require(quantization.get("group_size") == 128, prefix + "wrong group size")
    require(quantization.get("promote_sha256") == PROMOTION_SHA256,
            prefix + "wrong promotion hash")
    require(quantization.get("quantized_linears") == 196, prefix + "wrong quantized count")
    require(quantization.get("promoted_linears") == 28, prefix + "wrong promoted count")
    lora = manifest.get("lora", {})
    require(lora.get("rank") == 16 and lora.get("alpha") == 32,
            prefix + "wrong LoRA rank/alpha")
    require(float(lora.get("dropout", -1)) == 0.05, prefix + "wrong dropout")
    require(lora.get("targeted_linears") == 196, prefix + "wrong target count")
    require(set(lora.get("target_modules", [])) == EXPECTED_LORA_TARGETS
            and len(lora.get("target_modules", [])) == len(EXPECTED_LORA_TARGETS),
            prefix + "wrong target modules")
    require(lora.get("trainable_parameters") == EXPECTED_LORA_PARAMETERS,
            prefix + "wrong trainable count")
    require(lora.get("runtime_training_dtypes") == ["torch.float32"],
            prefix + "wrong training dtype")
    require(lora.get("runtime_training_parameter_bytes") == EXPECTED_LORA_PARAMETERS * 4,
            prefix + "wrong training parameter bytes")
    optimization = manifest.get("optimization", {})
    require(optimization.get("seed") == 0, prefix + "wrong seed")
    require(optimization.get("batch_size") == 2, prefix + "wrong microbatch")
    require(optimization.get("gradient_accumulation") == 8,
            prefix + "wrong gradient accumulation")
    require(optimization.get("effective_batch_size") == 16, prefix + "wrong batch")
    require(float(optimization.get("learning_rate", -1)) == 5e-5, prefix + "wrong LR")
    require(float(optimization.get("weight_decay", -1)) == 0.0,
            prefix + "wrong weight decay")
    require(float(optimization.get("warmup_ratio", -1)) == 0.03,
            prefix + "wrong warmup ratio")
    require(float(optimization.get("max_grad_norm", -1)) == 1.0,
            prefix + "wrong max grad norm")
    require(optimization.get("epochs") == 1, prefix + "wrong epochs")
    require(optimization.get("total_optimizer_steps") == 750, prefix + "wrong total steps")
    processor = manifest.get("processor", {})
    require(processor.get("model") == BASE_MODEL, prefix + "wrong processor model")
    require(processor.get("revision") == BASE_REVISION, prefix + "wrong processor revision")
    require(processor.get("max_pixels") == 1_003_520, prefix + "wrong max pixels")
    verification = manifest.get("verification", {})
    require(verification.get("zero_init_max_logit_diff") == 0.0,
            prefix + "zero-init parity failed")
    if is_final:
        require(manifest.get("completed", {}).get("optimizer_steps") == 750,
                prefix + "training incomplete")
        require(verification.get("first_step_nonzero_lora_grad") is True,
                prefix + "gradient check failed")
        require(verification.get("base_parameters_received_grad") is False,
                prefix + "base received gradients")
    else:
        require(manifest.get("checkpoint_step") == step, prefix + "wrong checkpoint step")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-empty-development", action="store_true")
    args = parser.parse_args()

    runs = Path(os.environ["GCQ_RUNS"])
    data = Path(os.environ["GCQ_DATA"])
    root = runs / "recovery_vqa_replay"
    adapter_root = root / "adapters" / RECIPE_ID
    development = root / "development"
    if args.require_empty_development and development.exists():
        require(development.is_dir(), f"development output is not a directory: {development}")
        require(not any(development.iterdir()),
                f"refusing to mix stale development outputs: {development}")

    train_data = data / "subsets" / "recovery_train_vqa_replay_12k.json"
    require(sha256_file(data / "subsets" / "recovery_train_vqa_replay_12k.json") == DATA_SHA256,
            "balanced replay data hash changed")
    require(sha256_file(data / "subsets" / "recovery_train_vqa_replay_12k.meta.json") == METADATA_SHA256,
            "balanced replay metadata hash changed")
    require(sha256_file(runs / "promote_gcq_b4.25.json") == PROMOTION_SHA256,
            "promotion map hash changed")
    protocol = Path(__file__).with_name("recovery_vqa_replay_protocol.json")
    require(sha256_file(protocol) == PROTOCOL_SHA256, "frozen protocol hash changed")

    launch_path = root / "training_launch_manifest.json"
    launch = load_json(launch_path)
    require(launch.get("recipe_id") == RECIPE_ID, "wrong launched recipe")
    require(launch.get("data_sha256") == DATA_SHA256, "launch data hash mismatch")
    require(launch.get("metadata_sha256") == METADATA_SHA256, "launch metadata hash mismatch")
    require(launch.get("promotion_sha256") == PROMOTION_SHA256,
            "launch promotion hash mismatch")
    require(launch.get("protocol_sha256") == PROTOCOL_SHA256, "launch protocol hash mismatch")
    require(launch.get("checkpoint_steps") == list(CHECKPOINT_STEPS),
            "launch checkpoint list mismatch")
    require(launch.get("schema_version") == 1, "wrong training-launch schema")
    require(launch.get("objective") == "cwce", "wrong launched objective")
    require(float(launch.get("coordinate_weight", -1)) == 5.0,
            "wrong launched coordinate weight")
    require(float(launch.get("learning_rate", -1)) == 5e-5,
            "wrong launched learning rate")
    require(launch.get("effective_batch_size") == 16, "wrong launched batch size")
    require(launch.get("epochs") == 1, "wrong launched epoch count")
    require(launch.get("planned_optimizer_steps") == 750,
            "wrong launched optimizer-step count")
    require(launch.get("seed") == 0, "wrong launched seed")
    code_dir = Path(__file__).resolve().parent
    require(launch.get("trainer_sha256") == sha256_file(code_dir / "train_recovery.py"),
            "trainer changed after training launch")
    require(launch.get("batch_script_sha256")
            == sha256_file(code_dir / "batch_recovery_vqa_replay.sh"),
            "training batch script changed after training launch")

    artifacts = {}
    for step in CHECKPOINT_STEPS:
        directory = adapter_root if step == 750 else adapter_root / f"checkpoint-{step:06d}"
        manifest_path = directory / "gcq_recovery_manifest.json"
        manifest = load_json(manifest_path)
        validate_manifest(manifest, step=step, is_final=step == 750)
        require(Path(manifest["data"]["path"]).resolve() == train_data.resolve(),
                f"step {step}: noncanonical training-data path")
        artifact = inspect_adapter(directory / "adapter_model.safetensors")
        adapter_config = inspect_adapter_config(directory / "adapter_config.json")
        recorded = manifest.get("artifact", {})
        require(recorded.get("sha256") == artifact["sha256"],
                f"step {step}: recorded artifact hash mismatch")
        require(recorded.get("file_bytes") == artifact["file_bytes"],
                f"step {step}: recorded artifact size mismatch")
        require(recorded.get("runtime_training_dtypes") == ["torch.float32"],
                f"step {step}: recorded artifact dtype mismatch")
        artifacts[str(step)] = {
            "directory": str(directory),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "artifact": artifact,
            "adapter_config": adapter_config,
        }

    report = {
        "schema_version": 1,
        "recipe_id": RECIPE_ID,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "data_sha256": DATA_SHA256,
        "metadata_sha256": METADATA_SHA256,
        "promotion_sha256": PROMOTION_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "training_launch_manifest": str(launch_path),
        "training_launch_manifest_sha256": sha256_file(launch_path),
        "checkpoints": artifacts,
    }
    output = root / "artifact_validation.json"
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    with open(temporary, "x") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"BALANCED VQA-REPLAY ARTIFACTS VALID: {output}")


if __name__ == "__main__":
    main()

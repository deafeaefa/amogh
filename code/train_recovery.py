"""Train a LoRA recovery adapter on a frozen Qwen3-VL quantize-dequantize base.

This is deliberately a small custom loop: no TRL, no learned quantizer, and no
base-weight updates. Quantization is applied before PEFT injection so the
underlying W4/GCQ numerics stay fixed. PEFT's explicit stable-training default
keeps the trainable LoRA tensors in FP32; deployment dtype is audited separately.

Examples:
  # ordinary post-quantization LoRA
  python train_recovery.py --objective ce --rtn-bits 4 --output-dir ...

  # coordinate-weighted recovery on GCQ B=4.25
  python train_recovery.py --objective cwce --rtn-bits 4 --promote-file ... \
      --gamma 5 --output-dir ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

import gcq_patches
from quant_utils import apply_rtn
from recovery_utils import (
    BASE_MODEL,
    BASE_REVISION,
    RecoveryCollator,
    coordinate_weighted_ce,
    read_promotions,
    sha256_file,
    write_adapter_manifest,
)


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EXPECTED_LORA_PARAMETERS_R16 = 17_432_576


class ManifestDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def move_batch(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor, torch.Tensor]:
    labels = batch.pop("labels").to(device)
    coordinate_mask = batch.pop("coordinate_mask").to(device)
    inputs = {key: value.to(device) for key, value in batch.items()}
    return inputs, labels, coordinate_mask


@torch.no_grad()
def selected_logits(model, inputs: dict, labels: torch.Tensor) -> torch.Tensor:
    logits = model(**inputs, use_cache=False, return_dict=True).logits[:, :-1, :]
    valid = labels[:, 1:].ne(-100)
    return logits[valid].float().cpu()


def save_checkpoint(model, optimizer, scheduler, output_dir: Path, step: int, manifest: dict) -> None:
    checkpoint = output_dir / f"checkpoint-{step:06d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint, safe_serialization=True)
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }, checkpoint / "training_state.pt")
    checkpoint_manifest = dict(manifest)
    checkpoint_manifest["checkpoint_step"] = step
    tensor_path = checkpoint / "adapter_model.safetensors"
    checkpoint_manifest["artifact"] = {
        "file_bytes": tensor_path.stat().st_size,
        "sha256": sha256_file(tensor_path),
        "runtime_training_dtypes": manifest["lora"]["runtime_training_dtypes"],
    }
    write_adapter_manifest(checkpoint, checkpoint_manifest)


def parse_args() -> argparse.Namespace:
    data_dir = os.environ.get("GCQ_DATA", "data")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--revision", default=BASE_REVISION)
    ap.add_argument("--train-file", default=os.path.join(data_dir, "subsets", "recovery_train_8k.json"))
    ap.add_argument("--image-dir", default=os.path.join(data_dir, "images", "train2014"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--objective", choices=("ce", "cwce"), required=True)
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--rtn-bits", type=int, default=4)
    ap.add_argument("--rtn-group", type=int, default=128)
    ap.add_argument("--promote-file", default="")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=1_003_520)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--max-samples", type=int, default=0, help="deterministic prefix for smoke tests only")
    ap.add_argument("--dry-run-collator", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.objective == "ce":
        gamma = 1.0
    else:
        if args.gamma <= 1:
            raise SystemExit("--objective cwce requires --gamma > 1")
        gamma = args.gamma
    if args.rtn_bits <= 0:
        raise SystemExit("the recovery pilot requires --rtn-bits > 0")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    gcq_patches.apply_fast_patch_embed()

    with open(args.train_file) as f:
        records = json.load(f)
    if args.max_samples:
        records = records[: args.max_samples]
    if not records:
        raise SystemExit("empty training manifest")
    missing = [x["file_name"] for x in records if not (Path(args.image_dir) / x["file_name"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} training images are missing; first: {missing[:3]}")

    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, max_pixels=args.max_pixels
    )
    collator = RecoveryCollator(processor, args.image_dir)
    if args.dry_run_collator:
        batch = collator(records[: min(args.batch_size, len(records))])
        valid = batch["labels"].ne(-100)
        print(json.dumps({
            "batch_shape": list(batch["input_ids"].shape),
            "assistant_tokens": int(valid.sum()),
            "coordinate_tokens": int(batch["coordinate_mask"].sum()),
            "examples": min(args.batch_size, len(records)),
        }, indent=2))
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for recovery training")
    device = torch.device(args.device)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    promote, promote_spec = read_promotions(args.promote_file)
    applied = apply_rtn(model, args.rtn_bits, args.rtn_group, promote=promote)
    promoted_linears = sum(1 for _, bits in applied if bits != args.rtn_bits)
    model.requires_grad_(False)
    model.eval()

    dataset = ManifestDataset(records)
    verify_loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False, collate_fn=collator)
    verify_raw = next(iter(verify_loader))
    verify_inputs, verify_labels, _ = move_batch(verify_raw, device)
    base_logits = selected_logits(model, verify_inputs, verify_labels)

    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=list(LORA_TARGETS),
    )
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    model.train()
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for _, p in trainable)
    trainable_dtypes = sorted({str(parameter.dtype) for _, parameter in trainable})
    trainable_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for _, parameter in trainable
    )
    lora_modules = sum(1 for name, _ in model.named_modules() if name.endswith("lora_A"))
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("non-LoRA parameters are trainable")
    if trainable_dtypes != ["torch.float32"]:
        raise RuntimeError(f"expected explicit FP32 LoRA training tensors, found {trainable_dtypes}")
    if args.rank == 16 and trainable_count != EXPECTED_LORA_PARAMETERS_R16:
        raise RuntimeError(
            f"unexpected rank-16 trainable count: {trainable_count} != {EXPECTED_LORA_PARAMETERS_R16}"
        )
    if lora_modules != 196:
        raise RuntimeError(f"expected 196 LoRA-targeted decoder linears, found {lora_modules}")

    model.eval()
    adapter_logits = selected_logits(model, verify_inputs, verify_labels)
    zero_init_max_diff = float((base_logits - adapter_logits).abs().max().item())
    if zero_init_max_diff != 0.0:
        raise RuntimeError(f"zero-initialized adapter changed base logits: max diff={zero_init_max_diff}")
    del base_logits, adapter_logits, verify_inputs, verify_labels, verify_raw
    torch.cuda.empty_cache()
    model.train()

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
    )
    optimizer = AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    promote_hash = sha256_file(args.promote_file) if args.promote_file else None
    manifest = {
        "schema_version": 1,
        "base_model": args.model,
        "base_revision": args.revision,
        "git_commit": git_commit(),
        "quantization": {
            "method": "rtn_quantize_dequantize",
            "bits": args.rtn_bits,
            "group_size": args.rtn_group,
            "promote_sha256": promote_hash,
            "promote_spec": promote_spec,
            "quantized_linears": len(applied),
            "promoted_linears": promoted_linears,
        },
        "processor": {"model": args.model, "revision": args.revision, "max_pixels": args.max_pixels},
        "data": {
            "path": str(Path(args.train_file).resolve()),
            "sha256": sha256_file(args.train_file),
            "examples": len(records),
            "max_samples": args.max_samples,
        },
        "objective": {"name": args.objective, "coordinate_weight": gamma},
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "bias": "none",
            "target_modules": list(LORA_TARGETS),
            "targeted_linears": lora_modules,
            "trainable_parameters": trainable_count,
            "runtime_training_dtypes": trainable_dtypes,
            "runtime_training_parameter_bytes": trainable_parameter_bytes,
            "hypothetical_bf16_parameter_bytes": trainable_count * 2,
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "total_optimizer_steps": total_steps,
            "seed": args.seed,
        },
        "verification": {"zero_init_max_logit_diff": zero_init_max_diff},
    }
    write_adapter_manifest(output_dir, manifest)

    metrics_path = output_dir / "train_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    accumulated_microbatches = 0
    running_loss = 0.0
    running_microbatches = 0
    first_step_nonzero_lora_grad = False
    start = time.time()
    for epoch in range(args.epochs):
        for batch_index, raw_batch in enumerate(loader):
            inputs, labels, coordinate_mask = move_batch(raw_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs, use_cache=False, return_dict=True)
                loss = coordinate_weighted_ce(outputs.logits, labels, coordinate_mask, gamma)
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            running_loss += float(loss.detach())
            running_microbatches += 1
            accumulated_microbatches += 1
            final_microbatch = batch_index + 1 == len(loader)
            if accumulated_microbatches < args.grad_accum and not final_microbatch:
                continue

            if accumulated_microbatches < args.grad_accum:
                correction = args.grad_accum / accumulated_microbatches
                for _, parameter in trainable:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            if global_step == 0:
                first_step_nonzero_lora_grad = any(
                    parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
                    for _, parameter in trainable
                )
                if not first_step_nonzero_lora_grad:
                    raise RuntimeError("first backward pass produced no nonzero LoRA gradients")
                if any(parameter.grad is not None for name, parameter in model.named_parameters()
                       if "lora_" not in name):
                    raise RuntimeError("a frozen base parameter received a gradient")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable], args.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            accumulated_microbatches = 0

            if global_step % args.log_every == 0 or global_step == 1 or global_step == total_steps:
                record = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": running_loss / running_microbatches,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "seconds": round(time.time() - start, 1),
                    "max_cuda_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                running_loss = 0.0
                running_microbatches = 0

            if args.save_every and global_step % args.save_every == 0 and global_step != total_steps:
                save_checkpoint(model, optimizer, scheduler, output_dir, global_step, manifest)

    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir / "processor")
    tensor_path = output_dir / "adapter_model.safetensors"
    manifest["artifact"] = {
        "file_bytes": tensor_path.stat().st_size,
        "sha256": sha256_file(tensor_path),
        "runtime_training_dtypes": trainable_dtypes,
    }
    manifest["completed"] = {
        "optimizer_steps": global_step,
        "seconds": round(time.time() - start, 1),
        "max_cuda_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
    }
    manifest["verification"]["first_step_nonzero_lora_grad"] = first_step_nonzero_lora_grad
    manifest["verification"]["base_parameters_received_grad"] = False
    write_adapter_manifest(output_dir, manifest)
    print(f"saved adapter to {output_dir}")


if __name__ == "__main__":
    main()

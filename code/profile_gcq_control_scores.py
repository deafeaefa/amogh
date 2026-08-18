#!/usr/bin/env python3
"""Profile frozen VQA-driven and MABA-style additive control scores.

VQA scores are the mean reduction in full-vocabulary BF16-teacher KL at the
causal logits predicting official training-answer tokens.  MABA-style scores
are a clearly labeled local-reconstruction reimplementation: for each
projection, W4-to-W8 relative reconstruction-error repair is measured on the
all-W4 activation stream and averaged with exactly 0.5 image-token and 0.5
text-token weight.  Both scores use the same pristine cached W4/W8 candidates
as the primary method and never read a development or confirmation outcome.

``profile`` writes immutable slice artifacts. ``merge`` requires disjoint,
complete coverage of the exact frozen shortlist before publishing score maps
accepted by ``build_gcq_controls.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from allocate_gcq_beam import load_candidates
from build_gcq_vqa_control_data import canonical_bytes, validate_manifest
from gptq_candidates import GPTQCandidateCache, canonical_sha256 as artifact_sha256
from recovery_utils import BASE_MODEL, BASE_REVISION, find_answer_token_positions


SCHEMA_VERSION = 1
IMPLEMENTATION_FILES = (
    "profile_gcq_control_scores.py",
    "build_gcq_vqa_control_data.py",
    "gptq_candidates.py",
    "allocate_gcq_beam.py",
    "recovery_utils.py",
    "gcq_patches.py",
)
VQA_SLICE_KIND = "gcq_vqa_answer_token_kl_repair_scores_slice"
MABA_SLICE_KIND = "gcq_maba_style_local_reconstruction_scores_slice"
VQA_MERGED_KIND = "gcq_vqa_answer_token_kl_repair_scores"
MABA_MERGED_KIND = "gcq_maba_style_local_reconstruction_scores"


class ControlProfileError(ValueError):
    """Raised when control scoring would violate the frozen comparison."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocator_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlProfileError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ControlProfileError(f"{label} must be finite")
    return result


def answer_prediction_positions(
    tokenizer: Any, input_ids: list[int], answer: str, max_tail_tokens: int = 128
) -> list[int]:
    answer_positions, _ = find_answer_token_positions(
        tokenizer, input_ids, answer, max_tail_tokens
    )
    if any(position <= 0 for position in answer_positions):
        raise ControlProfileError("an answer token has no preceding causal logit")
    result = [position - 1 for position in answer_positions]
    if len(result) != len(set(result)):
        raise ControlProfileError("answer causal logit positions are duplicated")
    return result


def kl_from_teacher_log_probs(
    teacher_log_probs: torch.Tensor, student_logits: torch.Tensor
) -> torch.Tensor:
    if teacher_log_probs.shape != student_logits.shape or teacher_log_probs.ndim != 2:
        raise ControlProfileError("teacher/student answer logits must be matching 2D tensors")
    teacher = teacher_log_probs.float()
    student = F.log_softmax(student_logits.float(), dim=-1)
    return torch.sum(torch.exp(teacher) * (teacher - student), dim=-1)


def local_reconstruction_metrics(
    inputs: torch.Tensor,
    image_mask: torch.Tensor,
    text_mask: torch.Tensor,
    teacher_weight: torch.Tensor,
    w4_weight: torch.Tensor,
    w8_weight: torch.Tensor,
    *,
    chunk_tokens: int = 256,
) -> dict[str, float]:
    """Return exact 0.5/0.5 relative local-reconstruction repair."""
    if inputs.ndim != 3 or image_mask.shape != inputs.shape[:2] or text_mask.shape != inputs.shape[:2]:
        raise ControlProfileError("local reconstruction inputs/masks have incompatible shapes")
    if image_mask.dtype != torch.bool or text_mask.dtype != torch.bool:
        raise ControlProfileError("local reconstruction masks must be boolean")
    if torch.any(image_mask & text_mask):
        raise ControlProfileError("image and text token masks overlap")
    if chunk_tokens <= 0:
        raise ControlProfileError("chunk_tokens must be positive")
    result: dict[str, float] = {}
    repairs = []
    for modality, mask in (("image", image_mask), ("text", text_mask)):
        selected = inputs[mask]
        if selected.numel() == 0:
            raise ControlProfileError(f"batch has no {modality} tokens")
        teacher_energy = 0.0
        w4_error = 0.0
        w8_error = 0.0
        count = 0
        for start in range(0, selected.shape[0], chunk_tokens):
            values = selected[start : start + chunk_tokens]
            teacher_output = F.linear(values, teacher_weight)
            w4_output = F.linear(values, w4_weight)
            w8_output = F.linear(values, w8_weight)
            teacher_energy += float(teacher_output.float().square().sum().item())
            w4_error += float((w4_output - teacher_output).float().square().sum().item())
            w8_error += float((w8_output - teacher_output).float().square().sum().item())
            count += teacher_output.numel()
        if count <= 0 or teacher_energy <= 0 or not all(
            math.isfinite(value) for value in (teacher_energy, w4_error, w8_error)
        ):
            raise ControlProfileError(f"invalid {modality} reconstruction totals")
        repair = (w4_error - w8_error) / teacher_energy
        result[f"{modality}_teacher_energy"] = teacher_energy
        result[f"{modality}_w4_error"] = w4_error
        result[f"{modality}_w8_error"] = w8_error
        result[f"{modality}_relative_repair"] = repair
        result[f"{modality}_output_values"] = float(count)
        repairs.append(repair)
    result["modality_balanced_score"] = 0.5 * repairs[0] + 0.5 * repairs[1]
    return result


class LocalAccumulator:
    """Forward-pre-hook accumulator for a frozen subset of projections."""

    def __init__(self, image_token_id: int, chunk_tokens: int):
        self.image_token_id = image_token_id
        self.chunk_tokens = chunk_tokens
        self.input_ids: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.totals: dict[str, dict[str, float]] = {}

    def set_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def hook(
        self,
        name: str,
        teacher_weight: torch.Tensor,
        w8_weight: torch.Tensor,
    ) -> Callable[[torch.nn.Module, tuple[Any, ...]], None]:
        def capture(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            if self.input_ids is None or self.attention_mask is None:
                raise ControlProfileError("local hook ran without active token masks")
            if not args or not isinstance(args[0], torch.Tensor):
                raise ControlProfileError(f"projection {name} has no tensor input")
            inputs = args[0]
            if tuple(inputs.shape[:2]) != tuple(self.input_ids.shape):
                raise ControlProfileError(f"projection {name} sequence shape differs from token masks")
            real = self.attention_mask.bool()
            image = real & self.input_ids.eq(self.image_token_id)
            text = real & ~image
            metrics = local_reconstruction_metrics(
                inputs,
                image,
                text,
                teacher_weight,
                module.weight,
                w8_weight,
                chunk_tokens=self.chunk_tokens,
            )
            totals = self.totals.setdefault(name, {})
            for key, value in metrics.items():
                if key.endswith("relative_repair") or key == "modality_balanced_score":
                    continue
                totals[key] = totals.get(key, 0.0) + value

        return capture

    def summaries(self, expected_names: Sequence[str]) -> dict[str, dict[str, float]]:
        if set(self.totals) != set(expected_names):
            raise ControlProfileError("local hooks did not cover the exact requested modules")
        output = {}
        for name in expected_names:
            totals = dict(self.totals[name])
            repairs = []
            for modality in ("image", "text"):
                energy = totals[f"{modality}_teacher_energy"]
                repair = (
                    totals[f"{modality}_w4_error"] - totals[f"{modality}_w8_error"]
                ) / energy
                totals[f"{modality}_relative_repair"] = repair
                repairs.append(repair)
            totals["modality_balanced_score"] = 0.5 * repairs[0] + 0.5 * repairs[1]
            output[name] = totals
        return output


def _parse_slice(value: str, names: Sequence[str]) -> list[str]:
    try:
        start_text, end_text = value.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else len(names)
    except (TypeError, ValueError) as error:
        raise ControlProfileError("--modules must be a slice such as 0:24") from error
    if not 0 <= start < end <= len(names):
        raise ControlProfileError(f"module slice {value!r} is outside 0:{len(names)}")
    return list(names[start:end])


def validate_catalog(path: str | Path, cache: GPTQCandidateCache) -> tuple[list[str], str]:
    candidates, _ = load_candidates(path)
    rows = [candidate.as_dict() for candidate in candidates]
    bank_costs = {
        row["module_name"]: row["delta_bytes"] for row in cache.manifest["candidates"]
    }
    for row in rows:
        if bank_costs.get(row["name"]) != row["delta_bytes"]:
            raise ControlProfileError(f"catalog/cache mismatch for {row['name']}")
    return [row["name"] for row in rows], allocator_sha256(rows)


def validate_protocol(
    protocol_path: str | Path,
    *,
    manifest_sha256: str,
    bank_file_sha256: str,
) -> tuple[dict[str, Any], str]:
    with Path(protocol_path).open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if not isinstance(protocol, dict) or protocol.get("status") != "launch_frozen":
        raise ControlProfileError("protocol is not launch-frozen")
    stored = protocol.get("protocol_sha256")
    unhashed = dict(protocol)
    unhashed.pop("protocol_sha256", None)
    if not isinstance(stored, str) or artifact_sha256(unhashed) != stored:
        raise ControlProfileError("protocol content hash mismatch")
    model = protocol.get("model")
    if not isinstance(model, dict) or model.get("id") != BASE_MODEL or model.get("revision") != BASE_REVISION:
        raise ControlProfileError("protocol model/revision mismatch")
    bound = protocol.get("bound_hashes")
    expected = {
        "vqa_control_manifest_sha256": manifest_sha256,
        "candidate_bank_manifest_sha256": bank_file_sha256,
    }
    if not isinstance(bound, dict) or any(bound.get(key) != value for key, value in expected.items()):
        raise ControlProfileError("protocol does not bind the VQA manifest/candidate bank")
    implementations = protocol.get("implementation_files")
    code_dir = Path(__file__).resolve().parent
    if not isinstance(implementations, dict):
        raise ControlProfileError("protocol lacks implementation hashes")
    for file_name in IMPLEMENTATION_FILES:
        if implementations.get(file_name) != sha256_file(code_dir / file_name):
            raise ControlProfileError(f"implementation hash mismatch for {file_name}")
    return protocol, sha256_file(protocol_path)


def _resolve_image_token_id(model: Any, processor: Any) -> int:
    candidates = [
        getattr(model.config, "image_token_id", None),
        getattr(getattr(model.config, "text_config", None), "image_token_id", None),
    ]
    token = getattr(processor, "image_token", None)
    if isinstance(token, str):
        candidates.append(processor.tokenizer.convert_tokens_to_ids(token))
    values = {value for value in candidates if type(value) is int and value >= 0}
    if len(values) != 1:
        raise ControlProfileError(f"could not resolve one image token ID: {sorted(values)}")
    return values.pop()


def _move_inputs(inputs: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _prepare_batches(
    records: Sequence[Mapping[str, Any]],
    processor: Any,
    *,
    image_root: Path,
    batch_size: int,
    max_tail_tokens: int,
) -> tuple[list[dict[str, Any]], str]:
    from PIL import Image

    processor.tokenizer.padding_side = "right"
    batches = []
    token_rows = []
    for start in range(0, len(records), batch_size):
        rows = list(records[start : start + batch_size])
        messages = []
        images = []
        try:
            for row in rows:
                with Image.open(image_root / str(row["file_name"])) as source:
                    image = source.convert("RGB")
                    image.load()
                images.append(image)
                messages.append(
                    [
                        {"role": "user", "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": row["prompt"]},
                        ]},
                        {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
                    ]
                )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        finally:
            for image in images:
                image.close()
        positions = []
        for row_index, row in enumerate(rows):
            n_real = int(inputs["attention_mask"][row_index].sum().item())
            ids = inputs["input_ids"][row_index, :n_real].tolist()
            causal = answer_prediction_positions(
                processor.tokenizer, ids, str(row["answer"]), max_tail_tokens
            )
            positions.append(causal)
            token_rows.append(
                {"uid": row["uid"], "input_ids_sha256": allocator_sha256(ids), "causal_positions": causal}
            )
        batches.append({"inputs": inputs, "rows": rows, "positions": positions})
    return batches, allocator_sha256(token_rows)


@torch.inference_mode()
def _teacher_cache(
    teacher: torch.nn.Module, batches: Sequence[Mapping[str, Any]], *, device: str
) -> list[list[torch.Tensor]]:
    result = []
    for batch in batches:
        inputs = _move_inputs(batch["inputs"], device)
        logits = teacher(**inputs).logits
        values = []
        for row_index, positions in enumerate(batch["positions"]):
            index = torch.tensor(positions, device=logits.device, dtype=torch.long)
            values.append(
                F.log_softmax(logits[row_index].index_select(0, index).float(), dim=-1)
                .detach()
                .cpu()
            )
        result.append(values)
        del logits
    return result


@torch.inference_mode()
def _student_answer_kl(
    student: torch.nn.Module,
    batches: Sequence[Mapping[str, Any]],
    teacher_cache: Sequence[Sequence[torch.Tensor]],
    *,
    device: str,
    before_forward: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[float]:
    result = []
    for batch_index, batch in enumerate(batches):
        inputs = _move_inputs(batch["inputs"], device)
        if before_forward is not None:
            before_forward(inputs)
        logits = student(**inputs).logits
        for row_index, positions in enumerate(batch["positions"]):
            index = torch.tensor(positions, device=logits.device, dtype=torch.long)
            student_logits = logits[row_index].index_select(0, index)
            values = kl_from_teacher_log_probs(
                teacher_cache[batch_index][row_index].to(logits.device), student_logits
            )
            result.append(float(values.mean().item()))
        del logits
    return result


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ControlProfileError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    import gcq_patches
    from transformers import AutoModelForImageTextToText, AutoProcessor

    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = validate_manifest(manifest)
    manifest_hash = sha256_file(args.manifest)
    cache = GPTQCandidateCache.load(args.candidate_cache, verify_hashes=True)
    bank_file_hash = sha256_file(args.candidate_cache / "manifest.json")
    _, protocol_hash = validate_protocol(
        args.protocol_context,
        manifest_sha256=manifest_hash,
        bank_file_sha256=bank_file_hash,
    )
    names, catalog_hash = validate_catalog(args.candidate_catalog, cache)
    selected_names = _parse_slice(args.modules, names)

    gcq_patches.apply_fast_patch_embed()
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, max_pixels=args.max_pixels
    )
    teacher = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16, device_map=args.device
    ).eval()
    student = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16, device_map=args.device
    ).eval()
    teacher_modules = cache.validate_model(teacher)
    student_modules = cache.validate_model(student)
    cache.compose(student, (), verify_installs=True)
    batches, tokenized_hash = _prepare_batches(
        records,
        processor,
        image_root=args.image_root,
        batch_size=args.batch_size,
        max_tail_tokens=args.max_tail_tokens,
    )
    teacher_cache = _teacher_cache(teacher, batches, device=args.device)

    image_token_id = _resolve_image_token_id(student, processor)
    local = LocalAccumulator(image_token_id, args.local_chunk_tokens)
    handles = []
    held_w8 = {}
    for name in selected_names:
        module = student_modules[name]
        w8 = cache.candidate(
            name, 8, device=module.weight.device, dtype=module.weight.dtype
        ).qdq
        held_w8[name] = w8
        handles.append(
            module.register_forward_pre_hook(
                local.hook(name, teacher_modules[name].weight, w8)
            )
        )
    try:
        baseline = _student_answer_kl(
            student,
            batches,
            teacher_cache,
            device=args.device,
            before_forward=lambda inputs: local.set_batch(
                inputs["input_ids"], inputs["attention_mask"]
            ),
        )
    finally:
        for handle in handles:
            handle.remove()
    local_details = local.summaries(selected_names)

    vqa_details = {}
    for name in selected_names:
        module = student_modules[name]
        original = module.weight.detach().clone()
        try:
            with torch.no_grad():
                module.weight.copy_(held_w8[name])
            promoted = _student_answer_kl(
                student, batches, teacher_cache, device=args.device
            )
        finally:
            with torch.no_grad():
                module.weight.copy_(original)
        if len(promoted) != len(baseline):
            raise ControlProfileError(f"VQA row count changed for {name}")
        w4_mean = sum(baseline) / len(baseline)
        w8_mean = sum(promoted) / len(promoted)
        vqa_details[name] = {
            "rows": len(baseline),
            "w4_answer_token_kl": w4_mean,
            "w8_answer_token_kl": w8_mean,
            "repair": w4_mean - w8_mean,
        }

    common = {
        "schema_version": SCHEMA_VERSION,
        "protocol_context_sha256": protocol_hash,
        "vqa_control_manifest_sha256": manifest_hash,
        "candidate_bank_manifest_file_sha256": bank_file_hash,
        "candidate_bank_manifest_content_sha256": cache.manifest_sha256,
        "candidate_catalog_file_sha256": sha256_file(args.candidate_catalog),
        "candidate_catalog_hash": catalog_hash,
        "candidate_catalog_names": names,
        "module_slice": args.modules,
        "profiled_modules": selected_names,
        "tokenized_context_sha256": tokenized_hash,
    }
    vqa_artifact = {
        **common,
        "artifact_kind": VQA_SLICE_KIND,
        "definition": "mean W4 minus W8 BF16-teacher full-vocabulary KL at causal answer-token logits",
        "details": vqa_details,
        "scores": {name: vqa_details[name]["repair"] for name in selected_names},
    }
    maba_artifact = {
        **common,
        "artifact_kind": MABA_SLICE_KIND,
        "definition": "0.5 image-token plus 0.5 text-token relative local reconstruction repair",
        "modality_weights": {"image": 0.5, "text": 0.5},
        "activation_stream": "all-W4 student",
        "details": local_details,
        "scores": {
            name: local_details[name]["modality_balanced_score"]
            for name in selected_names
        },
    }
    vqa_path = args.out_dir / f"vqa_control_scores.{args.tag}.json"
    maba_path = args.out_dir / f"maba_control_scores.{args.tag}.json"
    return {
        "vqa": {"path": str(vqa_path), "sha256": _write_exclusive(vqa_path, vqa_artifact)},
        "maba": {"path": str(maba_path), "sha256": _write_exclusive(maba_path, maba_artifact)},
        "modules": selected_names,
    }


def merge_slices(
    slices: Sequence[Mapping[str, Any]],
    *,
    expected_names: Sequence[str],
    slice_kind: str,
    merged_kind: str,
) -> dict[str, Any]:
    if not slices:
        raise ControlProfileError("at least one score slice is required")
    ignored = {"artifact_kind", "module_slice", "profiled_modules", "details", "scores"}
    reference = {key: value for key, value in slices[0].items() if key not in ignored}
    details: dict[str, Any] = {}
    scores: dict[str, float] = {}
    for index, value in enumerate(slices):
        if value.get("artifact_kind") != slice_kind:
            raise ControlProfileError(f"slice {index} has wrong artifact kind")
        common = {key: item for key, item in value.items() if key not in ignored}
        if common != reference:
            raise ControlProfileError(f"slice {index} provenance differs")
        modules = value.get("profiled_modules")
        slice_details = value.get("details")
        slice_scores = value.get("scores")
        if not isinstance(modules, list) or not isinstance(slice_details, dict) or not isinstance(slice_scores, dict):
            raise ControlProfileError(f"slice {index} payload is invalid")
        if set(modules) != set(slice_details) or set(modules) != set(slice_scores):
            raise ControlProfileError(f"slice {index} module fields disagree")
        overlap = set(modules) & set(scores)
        if overlap:
            raise ControlProfileError(f"score slices overlap: {sorted(overlap)}")
        for name in modules:
            details[name] = slice_details[name]
            scores[name] = _finite(slice_scores[name], f"score {name}")
    if set(scores) != set(expected_names):
        raise ControlProfileError(
            f"score-slice coverage mismatch; missing={sorted(set(expected_names)-set(scores))}, "
            f"extra={sorted(set(scores)-set(expected_names))}"
        )
    return {
        **reference,
        "artifact_kind": merged_kind,
        "profiled_modules": list(expected_names),
        "details": {name: details[name] for name in expected_names},
        "scores": {name: scores[name] for name in expected_names},
        "merged_slice_count": len(slices),
    }


def run_merge(args: argparse.Namespace) -> dict[str, Any]:
    candidates, _ = load_candidates(args.candidate_catalog)
    names = [candidate.name for candidate in candidates]
    def load_many(paths: Sequence[Path]) -> list[dict[str, Any]]:
        values = []
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ControlProfileError(f"slice {path} is not an object")
            values.append(value)
        return values
    vqa = merge_slices(
        load_many(args.vqa_slice),
        expected_names=names,
        slice_kind=VQA_SLICE_KIND,
        merged_kind=VQA_MERGED_KIND,
    )
    maba = merge_slices(
        load_many(args.maba_slice),
        expected_names=names,
        slice_kind=MABA_SLICE_KIND,
        merged_kind=MABA_MERGED_KIND,
    )
    if {
        key: value for key, value in vqa.items() if key not in {"artifact_kind", "definition", "details", "scores", "merged_slice_count"}
    } != {
        key: value for key, value in maba.items() if key not in {"artifact_kind", "definition", "modality_weights", "activation_stream", "details", "scores", "merged_slice_count"}
    }:
        raise ControlProfileError("merged VQA/MABA provenance differs")
    vqa_path = args.out_dir / "vqa_control_scores.json"
    maba_path = args.out_dir / "maba_control_scores.json"
    return {
        "vqa": {"path": str(vqa_path), "sha256": _write_exclusive(vqa_path, vqa)},
        "maba": {"path": str(maba_path), "sha256": _write_exclusive(maba_path, maba)},
        "modules": names,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    profile = sub.add_parser("profile")
    profile.add_argument("--manifest", type=Path, required=True)
    profile.add_argument("--candidate-cache", type=Path, required=True)
    profile.add_argument("--candidate-catalog", type=Path, required=True)
    profile.add_argument("--protocol-context", type=Path, required=True)
    profile.add_argument("--image-root", type=Path, required=True)
    profile.add_argument("--out-dir", type=Path, required=True)
    profile.add_argument("--device", default="cuda:0")
    profile.add_argument("--modules", default=":")
    profile.add_argument("--tag", default="full")
    profile.add_argument("--batch-size", type=int, default=8)
    profile.add_argument("--max-pixels", type=int, default=1_003_520)
    profile.add_argument("--max-tail-tokens", type=int, default=128)
    profile.add_argument("--local-chunk-tokens", type=int, default=256)
    merge = sub.add_parser("merge")
    merge.add_argument("--candidate-catalog", type=Path, required=True)
    merge.add_argument("--vqa-slice", type=Path, action="append", required=True)
    merge.add_argument("--maba-slice", type=Path, action="append", required=True)
    merge.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_profile(args) if args.command == "profile" else run_merge(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

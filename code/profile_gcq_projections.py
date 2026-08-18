#!/usr/bin/env python3
"""Exact projection-level coordinate-KL profiler for upgraded GCQ.

The runtime loads the immutable BF16 Qwen teacher and a second copy of the same
model as the student, verifies a persisted :class:`GPTQCandidateCache`, composes
the student entirely from its W4 payloads, and measures every projection by
swapping only that exact module to its persisted W8 candidate.  The W4 tensor is
restored in a ``finally`` block and verified before the next intervention.

Only logits that predict the four numeric ``bbox_2d`` coordinate spans are
scored.  ``find_answer_coordinate_token_groups`` returns coordinate *input-token
indices*; causal next-token logits are therefore read explicitly at ``index-1``.
Raw output records keep token indices, token IDs, and prediction-logit positions
as separate fields.  Punctuation, brackets, the ``2`` in ``bbox_2d``, and label
text never create groups.

Example::

    python profile_gcq_projections.py \
      --manifest "$GCQ_DATA/subsets/gcq_profile_proxy_train_512.json" \
      --candidate-cache "$GCQ_STORE/gptq_projection_candidates" \
      --protocol-context "$GCQ_RUNS/gcq_upgrade.launch_frozen.json" \
      --data-dir "$GCQ_DATA" --out-dir "$GCQ_RUNS/gcq_projection_profile" \
      --device cuda:0 --batch 16 --tag full

The three outputs (raw JSONL, hierarchical summaries, and provenance metadata)
are canonical and write-once.  Transformer, PIL, repository patch, and packed
cache imports are lazy so the numerical/swap contract can be tested on CPU.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from gcq_profile_metrics import (
    TASKS,
    QUARTILES,
    aggregate_coordinate_candidate,
    canonical_json_bytes,
    canonical_sha256,
    coordinate_row_kl,
)


PROFILE_SCHEMA_VERSION = 1
RAW_ARTIFACT_KIND = "gcq_projection_coordinate_kl_record"
SUMMARY_ARTIFACT_KIND = "gcq_projection_coordinate_kl_summary"
METADATA_ARTIFACT_KIND = "gcq_projection_coordinate_kl_run"
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
BASE_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
DEFAULT_MAX_PIXELS = 1_003_520
EXPECTED_ROWS_PER_CELL = 64
PROFILE_IMPLEMENTATION_FILES = (
    "profile_gcq_projections.py",
    "gptq_candidates.py",
    "gcq_profile_metrics.py",
    "recovery_utils.py",
    "gcq_patches.py",
)
_TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ProjectionProfileError(ValueError):
    """Raised when profiling inputs, state, or provenance violate the protocol."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProjectionProfileError(f"{label} must be finite")
    return result


def full_vocabulary_kl(
    teacher_logits: torch.Tensor, student_logits: torch.Tensor
) -> torch.Tensor:
    """Return ``KL(teacher || student)`` over the complete final vocabulary.

    All softmax, multiplication, and reduction operations are explicitly FP32,
    regardless of the model's BF16 evaluation dtype.
    """
    if teacher_logits.shape != student_logits.shape:
        raise ProjectionProfileError(
            f"teacher/student logit shapes differ: {teacher_logits.shape} != "
            f"{student_logits.shape}"
        )
    if teacher_logits.ndim < 1 or teacher_logits.shape[-1] < 2:
        raise ProjectionProfileError("logits must have a nontrivial vocabulary axis")
    teacher = teacher_logits.float()
    student = student_logits.float()
    teacher_log_probs = F.log_softmax(teacher, dim=-1)
    student_log_probs = F.log_softmax(student, dim=-1)
    return (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(
        dim=-1
    )


def _kl_from_teacher_log_probs(
    teacher_log_probs: torch.Tensor, student_logits: torch.Tensor
) -> torch.Tensor:
    if teacher_log_probs.shape != student_logits.shape:
        raise ProjectionProfileError("cached teacher/student vocabulary shapes differ")
    teacher = teacher_log_probs.float()
    student = F.log_softmax(student_logits.float(), dim=-1)
    return (teacher.exp() * (teacher - student)).sum(dim=-1)


def prediction_position_groups(
    coordinate_token_groups: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Convert coordinate input-token indices to causal prediction positions."""
    if len(coordinate_token_groups) != 4:
        raise ProjectionProfileError(
            f"expected four coordinate token groups, got {len(coordinate_token_groups)}"
        )
    result = []
    for coordinate_index, group in enumerate(coordinate_token_groups):
        if not group:
            raise ProjectionProfileError(
                f"coordinate token group {coordinate_index} is empty"
            )
        normalized = []
        for token_index in group:
            if type(token_index) is not int:
                raise ProjectionProfileError("coordinate token indices must be integers")
            if token_index == 0:
                raise ProjectionProfileError(
                    "coordinate token index 0 has no causal prediction position"
                )
            if token_index < 0:
                raise ProjectionProfileError("coordinate token indices must be nonnegative")
            normalized.append(token_index - 1)
        result.append(normalized)
    return result


def locate_coordinate_layout(
    tokenizer: Any,
    input_ids: Sequence[int],
    answer: str,
    *,
    max_tail_tokens: int = 128,
) -> dict[str, list[list[int]]]:
    """Locate numeric coordinate token IDs and their next-token logit positions."""
    from recovery_utils import find_answer_coordinate_token_groups

    ids = [int(value) for value in input_ids]
    token_groups = find_answer_coordinate_token_groups(
        tokenizer, ids, answer, max_tail_tokens=max_tail_tokens
    )
    prediction_groups = prediction_position_groups(token_groups)
    token_id_groups = [[ids[index] for index in group] for group in token_groups]
    return {
        "coordinate_token_indices": [list(group) for group in token_groups],
        "coordinate_token_ids": token_id_groups,
        "coordinate_prediction_positions": prediction_groups,
    }


def coordinate_kl_from_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    coordinate_prediction_positions: Sequence[Sequence[int]],
) -> list[list[float]]:
    """Compute full-vocabulary token KL grouped by the four coordinates."""
    if teacher_logits.ndim != 2 or student_logits.ndim != 2:
        raise ProjectionProfileError("per-row logits must have shape [sequence, vocabulary]")
    if teacher_logits.shape != student_logits.shape:
        raise ProjectionProfileError("teacher/student per-row logit shapes differ")
    if len(coordinate_prediction_positions) != 4:
        raise ProjectionProfileError("expected four coordinate prediction groups")
    output = []
    for coordinate_index, positions in enumerate(coordinate_prediction_positions):
        if not positions:
            raise ProjectionProfileError(f"coordinate {coordinate_index} has no logits")
        if any(position < 0 or position >= teacher_logits.shape[0] for position in positions):
            raise ProjectionProfileError(
                f"coordinate {coordinate_index} prediction position is out of range"
            )
        index = torch.tensor(positions, device=teacher_logits.device, dtype=torch.long)
        values = full_vocabulary_kl(
            teacher_logits.index_select(0, index),
            student_logits.index_select(0, index),
        )
        output.append([float(value) for value in values.detach().cpu().tolist()])
    return output


@contextmanager
def exact_projection_swap(
    module: torch.nn.Module,
    *,
    w4: torch.Tensor,
    w8: torch.Tensor,
):
    """Install one exact W8 tensor and restore the persisted W4 tensor exactly."""
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise ProjectionProfileError("projection module has no tensor weight")
    if weight.shape != w4.shape or weight.shape != w8.shape:
        raise ProjectionProfileError("projection W4/W8/module shapes differ")
    w4_local = w4.detach().to(device=weight.device, dtype=weight.dtype)
    w8_local = w8.detach().to(device=weight.device, dtype=weight.dtype)
    if not torch.equal(weight.detach(), w4_local):
        raise ProjectionProfileError("projection is not at its persisted W4 baseline")
    with torch.no_grad():
        weight.copy_(w8_local)
    if not torch.equal(weight.detach(), w8_local):
        raise ProjectionProfileError("exact W8 projection installation failed")
    try:
        yield module
    finally:
        with torch.no_grad():
            weight.copy_(w4_local)
        if not torch.equal(weight.detach(), w4_local):
            raise ProjectionProfileError("exact W4 projection restoration failed")


def validate_candidate_cache_contract(
    manifest: Mapping[str, Any],
    *,
    verified_payloads: bool,
    expected_model: str = BASE_MODEL,
    expected_revision: str = BASE_REVISION,
) -> dict[str, Any]:
    """Validate the persisted cache schema and immutable model provenance."""
    if not verified_payloads:
        raise ProjectionProfileError("candidate cache payload hashes were not verified")
    if manifest.get("schema_version") != 1:
        raise ProjectionProfileError("unsupported candidate-cache schema_version")
    if manifest.get("artifact_kind") != "gcq_packed_gptq_projection_bank":
        raise ProjectionProfileError("candidate cache has the wrong artifact_kind")
    content_hash = manifest.get("manifest_content_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_content_sha256", None)
    if not isinstance(content_hash, str) or canonical_sha256(unhashed) != content_hash:
        raise ProjectionProfileError("candidate-cache manifest content hash mismatch")
    recipe = manifest.get("recipe")
    if not isinstance(recipe, Mapping):
        raise ProjectionProfileError("candidate cache is missing its recipe")
    if recipe.get("base_model") != expected_model:
        raise ProjectionProfileError("candidate cache base_model differs from pinned teacher")
    if recipe.get("revision") != expected_revision:
        raise ProjectionProfileError("candidate cache revision differs from pinned teacher")
    if recipe.get("bits") != [4, 8]:
        raise ProjectionProfileError("candidate cache does not contain exact W4/W8 arms")
    rows = manifest.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ProjectionProfileError("candidate cache has no projections")
    names = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProjectionProfileError(f"candidate row {index} is not an object")
        name = row.get("module_name")
        if not isinstance(name, str) or not name:
            raise ProjectionProfileError(f"candidate row {index} has no module_name")
        names.append(name)
        if int(row.get("delta_bytes", 0)) <= 0:
            raise ProjectionProfileError(f"{name} has non-positive delta_bytes")
        for bits in (4, 8):
            arm = row.get(f"w{bits}")
            if not isinstance(arm, Mapping):
                raise ProjectionProfileError(f"{name} lacks persisted W{bits} metadata")
            if not isinstance(arm.get("qdq_sha256"), str):
                raise ProjectionProfileError(f"{name} W{bits} lacks a QDQ hash")
    if len(set(names)) != len(names):
        raise ProjectionProfileError("candidate cache contains duplicate module names")
    return {
        "manifest_sha256": content_hash,
        "modules": len(names),
        "module_names": tuple(names),
        "recipe_sha256": manifest.get("recipe_sha256"),
    }


def validate_bound_protocol(
    protocol: Mapping[str, Any],
    *,
    proxy_manifest_file_sha256: str,
    candidate_bank_file_sha256: str,
) -> None:
    """Require the score-blind launch protocol that binds this GPU run."""
    if protocol.get("status") != "launch_frozen":
        raise ProjectionProfileError("protocol context is not launch_frozen")
    stored_digest = protocol.get("protocol_sha256")
    unhashed = dict(protocol)
    unhashed.pop("protocol_sha256", None)
    if not isinstance(stored_digest, str) or canonical_sha256(unhashed) != stored_digest:
        raise ProjectionProfileError("bound protocol content hash mismatch")
    model = protocol.get("model")
    if not isinstance(model, Mapping) or model.get("id") != BASE_MODEL:
        raise ProjectionProfileError("bound protocol model differs from pinned model")
    if model.get("revision") != BASE_REVISION:
        raise ProjectionProfileError("bound protocol revision differs from pinned revision")
    hashes = protocol.get("bound_hashes")
    if not isinstance(hashes, Mapping):
        raise ProjectionProfileError("bound protocol has no bound_hashes")
    expected = {
        "proxy_manifest_sha256": proxy_manifest_file_sha256,
        "candidate_bank_manifest_sha256": candidate_bank_file_sha256,
    }
    for field, digest in expected.items():
        if hashes.get(field) != digest:
            raise ProjectionProfileError(f"bound protocol {field} mismatch")
    implementations = protocol.get("implementation_files")
    if not isinstance(implementations, Mapping):
        raise ProjectionProfileError("bound protocol has no implementation_files")
    code_dir = Path(__file__).resolve().parent
    for file_name in PROFILE_IMPLEMENTATION_FILES:
        path = code_dir / file_name
        if implementations.get(file_name) != sha256_file(path):
            raise ProjectionProfileError(
                f"bound protocol implementation hash mismatch for {file_name}"
            )


def validate_profile_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_rows_per_cell: int = EXPECTED_ROWS_PER_CELL,
) -> dict[str, Any]:
    if expected_rows_per_cell <= 0:
        raise ProjectionProfileError("expected_rows_per_cell must be positive")
    expected_rows = len(TASKS) * len(QUARTILES) * expected_rows_per_cell
    if len(records) != expected_rows:
        raise ProjectionProfileError(
            f"profile manifest has {len(records)} rows; expected {expected_rows}"
        )
    uids = []
    images = []
    cells = Counter()
    for index, record in enumerate(records):
        uid = record.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ProjectionProfileError(f"manifest row {index} has no uid")
        uids.append(uid)
        image_id = record.get("image_id")
        if type(image_id) is not int:
            raise ProjectionProfileError(f"manifest row {index} has invalid image_id")
        images.append(image_id)
        task = record.get("task")
        quartile = record.get("area_quartile")
        if task not in TASKS or quartile not in QUARTILES:
            raise ProjectionProfileError(f"manifest row {index} has an invalid cell")
        cells[(task, quartile)] += 1
        if record.get("split") != "train":
            raise ProjectionProfileError(f"manifest row {index} is not training-only")
        if not isinstance(record.get("prompt"), str) or not record["prompt"]:
            raise ProjectionProfileError(f"manifest row {index} lacks a prompt")
        if not isinstance(record.get("answer"), str) or not record["answer"]:
            raise ProjectionProfileError(f"manifest row {index} lacks an answer")
    if len(set(uids)) != len(uids):
        raise ProjectionProfileError("profile manifest contains duplicate UIDs")
    if len(set(images)) != len(images):
        raise ProjectionProfileError("profile manifest must contain one row per image")
    expected_cells = Counter(
        {
            (task, quartile): expected_rows_per_cell
            for task in TASKS
            for quartile in QUARTILES
        }
    )
    if cells != expected_cells:
        raise ProjectionProfileError(f"profile manifest cell counts differ: {cells}")
    return {
        "rows": len(records),
        "unique_images": len(set(images)),
        "cells": {f"{task}:q{quartile}": cells[(task, quartile)] for task in TASKS for quartile in QUARTILES},
        "canonical_sha256": canonical_sha256(records),
    }


def validate_raw_profile_record(
    record: Mapping[str, Any],
    *,
    manifest_sha256: str,
    cache_manifest_sha256: str,
    context_sha256: str,
) -> None:
    if record.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ProjectionProfileError("raw record schema_version mismatch")
    if record.get("artifact_kind") != RAW_ARTIFACT_KIND:
        raise ProjectionProfileError("raw record artifact_kind mismatch")
    expected = {
        "manifest_sha256": manifest_sha256,
        "cache_manifest_sha256": cache_manifest_sha256,
        "context_sha256": context_sha256,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ProjectionProfileError(f"raw record {key} mismatch")
    for key in (
        "coordinate_token_indices",
        "coordinate_token_ids",
        "coordinate_prediction_positions",
        "w4_coordinate_token_kl",
        "w8_coordinate_token_kl",
    ):
        groups = record.get(key)
        if not isinstance(groups, list) or len(groups) != 4 or any(
            not isinstance(group, list) or not group for group in groups
        ):
            raise ProjectionProfileError(f"raw record {key} must contain four groups")
    token_groups = record["coordinate_token_indices"]
    id_groups = record["coordinate_token_ids"]
    prediction_groups = record["coordinate_prediction_positions"]
    w4_groups = record["w4_coordinate_token_kl"]
    w8_groups = record["w8_coordinate_token_kl"]
    for coordinate in range(4):
        length = len(token_groups[coordinate])
        if not all(
            len(groups[coordinate]) == length
            for groups in (id_groups, prediction_groups, w4_groups, w8_groups)
        ):
            raise ProjectionProfileError("raw coordinate group lengths differ")
        if prediction_groups[coordinate] != [
            index - 1 for index in token_groups[coordinate]
        ]:
            raise ProjectionProfileError("raw token/logit positions are not causal index-1")
        for value in (*w4_groups[coordinate], *w8_groups[coordinate]):
            _finite(value, "coordinate KL")


def build_projection_summary(
    module_name: str,
    raw_records: Sequence[Mapping[str, Any]],
    *,
    candidate_metadata: Mapping[str, Any],
    manifest_sha256: str,
    cache_manifest_sha256: str,
    context_sha256: str,
) -> dict[str, Any]:
    for record in raw_records:
        validate_raw_profile_record(
            record,
            manifest_sha256=manifest_sha256,
            cache_manifest_sha256=cache_manifest_sha256,
            context_sha256=context_sha256,
        )
        if record.get("module_name") != module_name:
            raise ProjectionProfileError("raw record belongs to another projection")
    aggregate = aggregate_coordinate_candidate(raw_records)
    summary = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "artifact_kind": SUMMARY_ARTIFACT_KIND,
        "module_name": module_name,
        "manifest_sha256": manifest_sha256,
        "cache_manifest_sha256": cache_manifest_sha256,
        "context_sha256": context_sha256,
        "delta_bytes": int(candidate_metadata["delta_bytes"]),
        "w4_sha256": candidate_metadata["w4"]["qdq_sha256"],
        "w8_sha256": candidate_metadata["w8"]["qdq_sha256"],
        **aggregate,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def write_profile_outputs(
    output_dir: Path,
    *,
    tag: str,
    raw_records: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    if not _TAG_RE.fullmatch(tag):
        raise ProjectionProfileError("tag may contain only letters, digits, '_', '-', and '.'")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw": output_dir / f"{tag}.coordinate_kl.jsonl",
        "summaries": output_dir / f"{tag}.coordinate_summaries.json",
        "metadata": output_dir / f"{tag}.profile.meta.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite profiling output(s): " + ", ".join(existing))
    raw_bytes = b"".join(canonical_json_bytes(record) for record in raw_records)
    summary_bytes = canonical_json_bytes(list(summaries))
    final_metadata = dict(metadata)
    final_metadata["outputs"] = {
        "raw": {
            "file_name": paths["raw"].name,
            "records": len(raw_records),
            "sha256": sha256_bytes(raw_bytes),
        },
        "summaries": {
            "file_name": paths["summaries"].name,
            "projections": len(summaries),
            "sha256": sha256_bytes(summary_bytes),
        },
    }
    metadata_bytes = canonical_json_bytes(final_metadata)
    payloads = {
        "raw": raw_bytes,
        "summaries": summary_bytes,
        "metadata": metadata_bytes,
    }
    for key in ("raw", "summaries", "metadata"):
        with paths[key].open("xb") as handle:
            handle.write(payloads[key])
            handle.flush()
            os.fsync(handle.fileno())
    return paths, final_metadata


def _prepare_batches(
    records: Sequence[Mapping[str, Any]],
    processor: Any,
    *,
    data_dir: Path,
    batch_size: int,
    max_tail_tokens: int,
) -> tuple[list[dict[str, Any]], str]:
    from PIL import Image

    processor.tokenizer.padding_side = "right"
    batches = []
    context_rows = []
    for start in range(0, len(records), batch_size):
        rows = list(records[start : start + batch_size])
        messages = []
        opened = []
        try:
            for row in rows:
                path = data_dir / "images" / "train2014" / str(row["file_name"])
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    image.load()
                opened.append(image)
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": row["prompt"]},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": row["answer"]}],
                        },
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
            for image in opened:
                image.close()
        layouts = []
        for row_index, row in enumerate(rows):
            n_real = int(inputs["attention_mask"][row_index].sum().item())
            ids = inputs["input_ids"][row_index, :n_real].tolist()
            layout = locate_coordinate_layout(
                processor.tokenizer,
                ids,
                str(row["answer"]),
                max_tail_tokens=max_tail_tokens,
            )
            layouts.append(layout)
            context_rows.append(
                {
                    "uid": row["uid"],
                    "input_ids_sha256": canonical_sha256(ids),
                    **layout,
                }
            )
        batches.append({"inputs": inputs, "rows": rows, "layouts": layouts})
    return batches, canonical_sha256(context_rows)


def _move_inputs(inputs: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


@torch.inference_mode()
def _cache_teacher_coordinate_log_probs(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, Any]],
    *,
    device: str,
) -> list[list[list[torch.Tensor]]]:
    cached = []
    for batch in batches:
        inputs = _move_inputs(batch["inputs"], device)
        logits = model(**inputs).logits
        batch_cache = []
        for row_index, layout in enumerate(batch["layouts"]):
            row_cache = []
            for positions in layout["coordinate_prediction_positions"]:
                index = torch.tensor(positions, device=logits.device, dtype=torch.long)
                selected = logits[row_index].index_select(0, index).float()
                row_cache.append(F.log_softmax(selected, dim=-1).detach())
            batch_cache.append(row_cache)
        cached.append(batch_cache)
        del logits
    return cached


@torch.inference_mode()
def _score_student(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, Any]],
    teacher_cache: Sequence[Sequence[Sequence[torch.Tensor]]],
    *,
    device: str,
) -> list[list[list[float]]]:
    output = []
    for batch_index, batch in enumerate(batches):
        inputs = _move_inputs(batch["inputs"], device)
        logits = model(**inputs).logits
        for row_index, layout in enumerate(batch["layouts"]):
            row_values = []
            for coordinate_index, positions in enumerate(
                layout["coordinate_prediction_positions"]
            ):
                index = torch.tensor(positions, device=logits.device, dtype=torch.long)
                selected = logits[row_index].index_select(0, index)
                values = _kl_from_teacher_log_probs(
                    teacher_cache[batch_index][row_index][coordinate_index], selected
                )
                row_values.append(
                    [float(value) for value in values.detach().cpu().tolist()]
                )
            output.append(row_values)
        del logits
    return output


def _parse_module_slice(value: str, names: Sequence[str]) -> list[str]:
    try:
        start_text, end_text = value.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else len(names)
    except (ValueError, TypeError) as error:
        raise ProjectionProfileError("--modules must be a slice such as 0:196") from error
    if not (0 <= start < end <= len(names)):
        raise ProjectionProfileError(
            f"module slice {value!r} is outside 0:{len(names)}"
        )
    return list(names[start:end])


def run_profile(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, Any]]:
    """Heavyweight Qwen execution path; imports remain local to this function."""
    import gcq_patches
    from gptq_candidates import GPTQCandidateCache
    from transformers import AutoModelForImageTextToText, AutoProcessor

    gcq_patches.apply_fast_patch_embed()

    manifest_bytes = args.manifest.read_bytes()
    try:
        records = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ProjectionProfileError("profile manifest is invalid JSON") from error
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ProjectionProfileError("profile manifest must be a JSON list of objects")
    if manifest_bytes != canonical_json_bytes(records):
        raise ProjectionProfileError("profile manifest is not canonical JSON")
    manifest_contract = validate_profile_manifest(records)
    manifest_sha256 = sha256_bytes(manifest_bytes)
    if manifest_sha256 != manifest_contract["canonical_sha256"]:
        raise ProjectionProfileError("profile manifest canonical/file hash mismatch")

    cache = GPTQCandidateCache.load(args.candidate_cache, verify_hashes=True)
    cache_contract = validate_candidate_cache_contract(
        cache.manifest, verified_payloads=cache.verified_payloads
    )
    candidate_rows = {
        row["module_name"]: row for row in cache.manifest["candidates"]
    }
    names = list(cache.names)
    selected_names = _parse_module_slice(args.modules, names)

    protocol_bytes = args.protocol_context.read_bytes()
    try:
        protocol = json.loads(protocol_bytes)
    except json.JSONDecodeError as error:
        raise ProjectionProfileError("protocol context is invalid JSON") from error
    if not isinstance(protocol, dict):
        raise ProjectionProfileError("protocol context must be a JSON object")
    candidate_bank_file_sha256 = sha256_file(
        args.candidate_cache / "manifest.json"
    )
    validate_bound_protocol(
        protocol,
        proxy_manifest_file_sha256=manifest_sha256,
        candidate_bank_file_sha256=candidate_bank_file_sha256,
    )
    protocol_context_sha256 = sha256_bytes(protocol_bytes)

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, max_pixels=args.max_pixels
    )
    teacher = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        dtype=torch.bfloat16,
        device_map=args.device,
    ).eval()
    student = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        dtype=torch.bfloat16,
        device_map=args.device,
    ).eval()
    modules = cache.validate_model(student)
    cache.compose(student, promotions=(), verify_installs=True)

    batches, tokenized_context_sha256 = _prepare_batches(
        records,
        processor,
        data_dir=args.data_dir,
        batch_size=args.batch,
        max_tail_tokens=args.max_tail_tokens,
    )
    chat_template = getattr(processor, "chat_template", None) or getattr(
        processor.tokenizer, "chat_template", None
    )
    context = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "manifest_sha256": manifest_sha256,
        "cache_manifest_sha256": cache.manifest_sha256,
        "max_pixels": args.max_pixels,
        "padding_side": "right",
        "max_tail_tokens": args.max_tail_tokens,
        "coordinate_locator": "recovery_utils.find_answer_coordinate_token_groups",
        "causal_logit_position": "coordinate_input_token_index_minus_one",
        "kl": "FP32 full-vocabulary KL(BF16 teacher || candidate student)",
        "aggregation": "token-within-coordinate, four-coordinate row, eight-cell macro",
        "chat_template_sha256": sha256_bytes(str(chat_template).encode("utf-8")),
        "tokenized_context_sha256": tokenized_context_sha256,
    }
    profile_configuration_sha256 = canonical_sha256(context)
    context_sha256 = protocol_context_sha256

    teacher_cache = _cache_teacher_coordinate_log_probs(
        teacher, batches, device=args.device
    )
    baseline_values = _score_student(
        student, batches, teacher_cache, device=args.device
    )
    if len(baseline_values) != len(records):
        raise ProjectionProfileError("W4 baseline row count differs from manifest")

    raw_records = []
    summaries = []
    for module_name in selected_names:
        module = modules[module_name]
        w4 = cache.candidate(
            module_name, 4, device=module.weight.device, dtype=module.weight.dtype
        ).qdq
        w8 = cache.candidate(
            module_name, 8, device=module.weight.device, dtype=module.weight.dtype
        ).qdq
        with exact_projection_swap(module, w4=w4, w8=w8):
            promoted_values = _score_student(
                student, batches, teacher_cache, device=args.device
            )
        if len(promoted_values) != len(records):
            raise ProjectionProfileError(
                f"{module_name} promoted row count differs from manifest"
            )

        module_records = []
        flat_layouts = [
            layout for batch in batches for layout in batch["layouts"]
        ]
        for record, layout, w4_values, w8_values in zip(
            records, flat_layouts, baseline_values, promoted_values
        ):
            raw = {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "artifact_kind": RAW_ARTIFACT_KIND,
                "module_name": module_name,
                "uid": record["uid"],
                "image_id": record["image_id"],
                "task": record["task"],
                "area_quartile": record["area_quartile"],
                "manifest_sha256": manifest_sha256,
                "cache_manifest_sha256": cache.manifest_sha256,
                "context_sha256": context_sha256,
                **layout,
                "w4_coordinate_token_kl": w4_values,
                "w8_coordinate_token_kl": w8_values,
                "w4_row_kl": coordinate_row_kl(w4_values),
                "w8_row_kl": coordinate_row_kl(w8_values),
                "repair": coordinate_row_kl(w4_values)
                - coordinate_row_kl(w8_values),
            }
            validate_raw_profile_record(
                raw,
                manifest_sha256=manifest_sha256,
                cache_manifest_sha256=cache.manifest_sha256,
                context_sha256=context_sha256,
            )
            module_records.append(raw)
        raw_records.extend(module_records)
        summaries.append(
            build_projection_summary(
                module_name,
                module_records,
                candidate_metadata=candidate_rows[module_name],
                manifest_sha256=manifest_sha256,
                cache_manifest_sha256=cache.manifest_sha256,
                context_sha256=context_sha256,
            )
        )

    metadata = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "artifact_kind": METADATA_ARTIFACT_KIND,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "manifest": {
            "path": str(args.manifest),
            "sha256": manifest_sha256,
            **manifest_contract,
        },
        "candidate_cache": {
            "path": str(args.candidate_cache),
            **cache_contract,
        },
        "context": context,
        "profile_configuration_sha256": profile_configuration_sha256,
        "protocol_context": {
            "path": str(args.protocol_context),
            "sha256": protocol_context_sha256,
            "embedded_protocol_sha256": protocol["protocol_sha256"],
        },
        "context_sha256": protocol_context_sha256,
        "all_w4_baseline_composition_sha256": canonical_sha256(
            cache.composition_manifest(())
        ),
        "module_slice": args.modules,
        "profiled_modules": selected_names,
        "profiled_module_count": len(selected_names),
        "raw_record_order": "module_name cache order, then manifest row order",
        "summary_order": "candidate cache module order",
    }
    return write_profile_outputs(
        args.out_dir,
        tag=args.tag,
        raw_records=raw_records,
        summaries=summaries,
        metadata=metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument(
        "--protocol-context", type=Path, required=True,
        help="launch-frozen protocol; its raw file SHA-256 is the shared run context",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max-tail-tokens", type=int, default=128)
    parser.add_argument("--modules", default="0:196")
    parser.add_argument("--tag", default="full")
    args = parser.parse_args()
    if args.batch <= 0 or args.max_pixels <= 0 or args.max_tail_tokens <= 0:
        parser.error("batch, max-pixels, and max-tail-tokens must be positive")
    return args


def main() -> None:
    paths, metadata = run_profile(parse_args())
    print(
        json.dumps(
            {
                "paths": {key: str(value) for key, value in paths.items()},
                "raw_sha256": metadata["outputs"]["raw"]["sha256"],
                "summaries_sha256": metadata["outputs"]["summaries"]["sha256"],
                "metadata_sha256": canonical_sha256(metadata),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

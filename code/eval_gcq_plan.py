#!/usr/bin/env python3
"""Decode one immutable GCQ beam plan on its frozen training-only manifest.

This is the GPU-side counterpart of :mod:`allocate_gcq_beam` and
:mod:`score_gcq_plan`.  It deliberately accepts only the launch-frozen
``gcq_profile_decode_train_512`` manifest.  It verifies the protocol context,
candidate bank, plan, and every input hash before loading a model; composes
candidate-bank W4/W8 tensors by exact module name; and writes one exclusive
JSONL file per allocation state.

Existing state files are treated as immutable cache entries.  A resume first
revalidates every row, its provenance, manifest identity/order, and recomputed
grounding metrics.  Invalid or partial files are never overwritten.

Transformers, PIL, and the model are imported/loaded only inside the real GPU
runtime.  The validation, ordering, scoring, resume, and orchestration APIs are
therefore testable on CPU with injected helpers and decoders.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

from allocate_gcq_beam import load_candidates
from gcq_profile_metrics import decoded_macro_summary
from gptq_candidates import (
    EXPECTED_DECODER_WEIGHTS,
    EXPECTED_PROJECTIONS,
    GPTQCandidateCache,
)
from recovery_utils import BASE_MODEL, BASE_REVISION, IOU_THRESHOLDS


SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
EXPECTED_ROWS = 512
EXPECTED_CELL_ROWS = 64
TASKS = ("rec", "coco_grounding")
QUARTILES = (1, 2, 3, 4)
DEFAULT_MAX_PIXELS = 1_003_520
DEFAULT_MAX_NEW_TOKENS = 64
EVAL_IMPLEMENTATION_FILES = (
    "eval_gcq_plan.py",
    "allocate_gcq_beam.py",
    "gptq_candidates.py",
    "gcq_profile_metrics.py",
    "recovery_utils.py",
    "gcq_patches.py",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STATE_ID_RE = re.compile(r"state-[0-9a-f]{64}")
_TRAIN_IMAGE_RE = re.compile(r"COCO_train2014_(\d{12})\.jpg")


class PlanEvaluationError(ValueError):
    """Raised when an evaluation input or cached result violates the protocol."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Allocator-compatible canonical hash (no trailing newline)."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_canonical_sha256(value: object) -> str:
    """Artifact-writer canonical hash (one trailing newline)."""
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.lower()) is None:
        raise PlanEvaluationError(f"{label} must be a 64-character SHA-256 digest")
    return value.lower()


def _load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise PlanEvaluationError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise PlanEvaluationError(f"JSONL row {path}:{line_number} is not an object")
            rows.append(value)
    return rows


def state_id_for_members(members: Iterable[str]) -> str:
    """Return the allocator's canonical state ID for a projection set."""
    normalized = sorted(members)
    if len(normalized) != len(set(normalized)):
        raise PlanEvaluationError("allocation members contain duplicates")
    return "state-" + canonical_sha256(normalized)


@dataclass(frozen=True)
class GroundingHelpers:
    """The exact parse/geometry functions shared with the REC evaluator."""

    parse_box: Callable[[str], list[int] | None]
    to_pixels: Callable[[Sequence[float], int, int], list[float]]
    iou_giou: Callable[[Sequence[float], Sequence[float]], tuple[float, float]]


def load_grounding_helpers() -> GroundingHelpers:
    """Lazily import the existing REC parsing and GIoU implementation."""
    from eval_rec import iou_giou, parse_box, to_pixels

    return GroundingHelpers(parse_box=parse_box, to_pixels=to_pixels, iou_giou=iou_giou)


@dataclass(frozen=True)
class GenerationSettings:
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    do_sample: bool = False
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.max_pixels <= 0 or self.max_new_tokens <= 0:
            raise PlanEvaluationError("generation pixel/token limits must be positive")
        if self.do_sample or self.dtype != "bfloat16":
            raise PlanEvaluationError("canonical plan decoding is greedy BF16 only")

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class EvaluationProvenance:
    plan_sha256: str
    context_sha256: str
    decode_manifest_sha256: str
    candidate_bank_manifest_file_sha256: str
    candidate_bank_manifest_content_sha256: str
    generation_settings_sha256: str
    run_fingerprint: str
    round_index: int

    def __post_init__(self) -> None:
        for field in (
            "plan_sha256",
            "context_sha256",
            "decode_manifest_sha256",
            "candidate_bank_manifest_file_sha256",
            "candidate_bank_manifest_content_sha256",
            "generation_settings_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if not isinstance(self.run_fingerprint, str) or not self.run_fingerprint:
            raise PlanEvaluationError("run_fingerprint must be non-empty")
        if type(self.round_index) is not int or self.round_index < 0:
            raise PlanEvaluationError("round_index must be a nonnegative integer")

    def row_fields(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "context_sha256": self.context_sha256,
            "manifest_sha256": self.decode_manifest_sha256,
            "candidate_bank_manifest_file_sha256": (
                self.candidate_bank_manifest_file_sha256
            ),
            "candidate_bank_manifest_content_sha256": (
                self.candidate_bank_manifest_content_sha256
            ),
            "generation_settings_sha256": self.generation_settings_sha256,
            "run_fingerprint": self.run_fingerprint,
            "round_index": self.round_index,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
        }


def _target_box_1000(record: Mapping[str, Any]) -> list[int]:
    try:
        answer = json.loads(str(record["answer"]))
        box = answer["bbox_2d"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise PlanEvaluationError(f"{record.get('uid')} has an invalid JSON answer") from error
    if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
        raise PlanEvaluationError(f"{record.get('uid')} answer needs four integer coordinates")
    if not box[0] < box[2] or not box[1] < box[3]:
        raise PlanEvaluationError(f"{record.get('uid')} answer box is degenerate")
    return box


def validate_decode_manifest(value: object) -> list[dict[str, Any]]:
    """Validate the exact training-only 2 x 4 x 64 decoded manifest."""
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise PlanEvaluationError("decode manifest must be a JSON list of row objects")
    if len(value) != EXPECTED_ROWS:
        raise PlanEvaluationError(
            f"decode manifest has {len(value)} rows; expected {EXPECTED_ROWS}"
        )
    rows = list(value)
    cells: Counter[tuple[str, int]] = Counter()
    seen_images: set[int] = set()
    for index, row in enumerate(rows):
        expected_uid = f"gcq_profile_decode_train_512:{index:05d}"
        if row.get("uid") != expected_uid:
            raise PlanEvaluationError(
                f"decode manifest row {index} UID/order mismatch: {row.get('uid')!r}"
            )
        if row.get("split") != "train":
            raise PlanEvaluationError(f"{expected_uid} is not training-only")
        image_id = row.get("image_id")
        if type(image_id) is not int or image_id < 0 or image_id in seen_images:
            raise PlanEvaluationError(f"{expected_uid} has invalid/duplicate image_id")
        seen_images.add(image_id)
        file_name = row.get("file_name")
        match = _TRAIN_IMAGE_RE.fullmatch(str(file_name))
        if match is None or int(match.group(1)) != image_id or Path(str(file_name)).name != file_name:
            raise PlanEvaluationError(f"{expected_uid} is not a canonical COCO train image")
        task = row.get("task")
        quartile = row.get("area_quartile")
        if task not in TASKS or type(quartile) is not int or quartile not in QUARTILES:
            raise PlanEvaluationError(f"{expected_uid} has an invalid task/quartile")
        cells[(task, quartile)] += 1
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise PlanEvaluationError(f"{expected_uid} has no prompt")
        _target_box_1000(row)
        width, height = row.get("width"), row.get("height")
        box = row.get("bbox_xywh")
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise PlanEvaluationError(f"{expected_uid} has invalid image dimensions")
        if not isinstance(box, list) or len(box) != 4:
            raise PlanEvaluationError(f"{expected_uid} has invalid bbox_xywh")
        try:
            x, y, box_width, box_height = (float(item) for item in box)
        except (TypeError, ValueError) as error:
            raise PlanEvaluationError(f"{expected_uid} has nonnumeric bbox_xywh") from error
        if not all(math.isfinite(item) for item in (x, y, box_width, box_height)):
            raise PlanEvaluationError(f"{expected_uid} has nonfinite bbox_xywh")
        if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
            raise PlanEvaluationError(f"{expected_uid} has invalid bbox geometry")
        if x + box_width > width + 1e-4 or y + box_height > height + 1e-4:
            raise PlanEvaluationError(f"{expected_uid} bbox exceeds image bounds")
    expected = Counter(
        {(task, quartile): EXPECTED_CELL_ROWS for task in TASKS for quartile in QUARTILES}
    )
    if cells != expected:
        raise PlanEvaluationError(f"decode manifest cells differ from frozen design: {dict(cells)}")
    return rows


def validate_candidate_cache_contract(cache: GPTQCandidateCache) -> dict[str, int]:
    """Check the packed bank is the pinned, complete Qwen3-VL-2B bank."""
    manifest = cache.manifest
    recipe = manifest.get("recipe", {})
    if recipe.get("base_model") != BASE_MODEL or recipe.get("revision") != BASE_REVISION:
        raise PlanEvaluationError("candidate bank model/revision is not pinned")
    if recipe.get("bits") != [4, 8] or recipe.get("prefix_policy") != (
        "earlier_decoder_layers_cached_w4"
    ):
        raise PlanEvaluationError("candidate bank recipe is not the canonical W4-prefix recipe")
    architecture = manifest.get("architecture")
    if not isinstance(architecture, dict):
        raise PlanEvaluationError("candidate bank lacks strict architecture validation")
    if architecture.get("projections") != EXPECTED_PROJECTIONS:
        raise PlanEvaluationError("candidate bank does not contain 196 projections")
    if architecture.get("weights") != EXPECTED_DECODER_WEIGHTS:
        raise PlanEvaluationError("candidate bank decoder weight count is wrong")
    rows = manifest.get("candidates")
    if not isinstance(rows, list) or len(rows) != EXPECTED_PROJECTIONS:
        raise PlanEvaluationError("candidate bank manifest candidate list is incomplete")
    names: set[str] = set()
    delta_total = 0
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("module_name"), str):
            raise PlanEvaluationError("candidate bank contains an invalid row")
        name = row["module_name"]
        if name in names:
            raise PlanEvaluationError(f"candidate bank duplicates {name}")
        names.add(name)
        delta = row.get("delta_bytes")
        if type(delta) is not int or delta <= 0:
            raise PlanEvaluationError(f"candidate bank has invalid delta for {name}")
        for bits in (4, 8):
            arm = row.get(f"w{bits}")
            if not isinstance(arm, dict) or type(arm.get("logical_payload_bytes")) is not int:
                raise PlanEvaluationError(f"candidate bank lacks persisted W{bits} arm for {name}")
        expected_delta = row["w8"]["logical_payload_bytes"] - row["w4"]["logical_payload_bytes"]
        if delta != expected_delta:
            raise PlanEvaluationError(f"candidate bank delta mismatch for {name}")
        delta_total += delta
    return {"projections": len(names), "sum_delta_bytes": delta_total}


def validate_runtime_bindings(
    context_path: str | Path,
    decode_manifest_path: str | Path,
    cache: GPTQCandidateCache,
) -> tuple[dict[str, Any], str, str, str]:
    """Validate launch-frozen context and return its/file input hashes.

    Returns ``(context, context_file_sha256, decode_file_sha256,
    candidate_manifest_file_sha256)``.
    """
    context_path = Path(context_path)
    decode_manifest_path = Path(decode_manifest_path)
    if cache.root is None:
        raise PlanEvaluationError("GPU plan evaluation requires a persisted candidate bank")
    candidate_manifest_path = cache.root / "manifest.json"
    context = _load_json(context_path)
    if not isinstance(context, dict) or context.get("schema_version") != SCHEMA_VERSION:
        raise PlanEvaluationError("protocol context schema is invalid")
    if context.get("status") != "launch_frozen":
        raise PlanEvaluationError("protocol context is not launch-frozen")
    recorded_protocol_hash = _require_sha256(
        context.get("protocol_sha256"), "protocol_sha256"
    )
    unhashed = dict(context)
    unhashed.pop("protocol_sha256", None)
    if artifact_canonical_sha256(unhashed) != recorded_protocol_hash:
        raise PlanEvaluationError("protocol context content hash is invalid")
    model = context.get("model", {})
    if model.get("id") != BASE_MODEL or model.get("revision") != BASE_REVISION:
        raise PlanEvaluationError("protocol context model/revision is not pinned")
    if model.get("expected_projection_count") != EXPECTED_PROJECTIONS or model.get(
        "expected_decoder_weight_count"
    ) != EXPECTED_DECODER_WEIGHTS:
        raise PlanEvaluationError("protocol context architecture is not Qwen3-VL-2B")
    validate_candidate_cache_contract(cache)

    context_hash = sha256_file(context_path)
    decode_hash = sha256_file(decode_manifest_path)
    bank_file_hash = sha256_file(candidate_manifest_path)
    bound = context.get("bound_hashes")
    if not isinstance(bound, dict):
        raise PlanEvaluationError("protocol context has no bound hashes")
    required = {
        "decode_manifest_sha256": decode_hash,
        "candidate_bank_manifest_sha256": bank_file_hash,
    }
    for key, actual in required.items():
        if bound.get(key) != actual:
            raise PlanEvaluationError(f"protocol-bound {key} does not match launch input")
    bound_inputs = context.get("bound_inputs", {})
    expected_paths = {
        "decode_manifest": decode_manifest_path,
        "candidate_bank_manifest": candidate_manifest_path,
    }
    for key, path in expected_paths.items():
        row = bound_inputs.get(key)
        if not isinstance(row, dict) or row.get("file_name") != path.name:
            raise PlanEvaluationError(f"protocol bound input filename mismatch for {key}")
        if row.get("sha256") != sha256_file(path):
            raise PlanEvaluationError(f"protocol bound input hash mismatch for {key}")
    configured_manifest = context.get("profiling_data", {}).get("decode_manifest")
    if configured_manifest != decode_manifest_path.name:
        raise PlanEvaluationError("worker was not given the protocol's decode manifest")
    implementation = context.get("implementation_files", {})
    code_dir = Path(__file__).resolve().parent
    for file_name in EVAL_IMPLEMENTATION_FILES:
        path = code_dir / file_name
        if implementation.get(file_name) != sha256_file(path):
            raise PlanEvaluationError(
                f"{file_name} is not hash-bound in the launch protocol"
            )
    return context, context_hash, decode_hash, bank_file_hash


def _cache_delta_bytes(cache: GPTQCandidateCache) -> dict[str, int]:
    return {
        str(row["module_name"]): int(row["delta_bytes"])
        for row in cache.manifest["candidates"]
    }


def validate_candidate_catalog(
    path: str | Path, cache: GPTQCandidateCache
) -> tuple[str, list[dict[str, Any]]]:
    """Bind a plan to the exact frozen shortlist used by the allocator."""
    try:
        candidates, _ = load_candidates(path)
    except ValueError as error:
        raise PlanEvaluationError(f"invalid frozen candidate catalog: {error}") from error
    rows = [candidate.as_dict() for candidate in candidates]
    cache_costs = _cache_delta_bytes(cache)
    for row in rows:
        if cache_costs.get(row["name"]) != row["delta_bytes"]:
            raise PlanEvaluationError(
                f"candidate catalog/cache mismatch for {row['name']}"
            )
    return canonical_sha256(rows), rows


def validate_plan(
    value: object,
    *,
    cache: GPTQCandidateCache,
    context: Mapping[str, Any],
    context_sha256: str,
    candidate_catalog_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Validate allocator state IDs, exact names/costs, context, and ordering."""
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise PlanEvaluationError("beam plan schema is invalid")
    if value.get("context_sha256") != context_sha256:
        raise PlanEvaluationError("beam plan is bound to a different protocol context")
    is_comparison = value.get("artifact_kind") == "gcq_frozen_comparison_plan"
    if is_comparison:
        if value.get("kind") != "comparison" or value.get("objective") != "compare_frozen_states":
            raise PlanEvaluationError("comparison plan objective/kind is invalid")
        if value.get("strict_positive_conditional_marginal") is not False:
            raise PlanEvaluationError("comparison plan must not apply a marginal filter")
        labels = value.get("label_to_state_id")
        if not isinstance(labels, dict) or not labels or "all_w4" not in labels:
            raise PlanEvaluationError("comparison plan has no frozen label mapping")
        fingerprint = value.get("comparison_fingerprint")
        _require_sha256(fingerprint, "comparison_fingerprint")
    elif value.get("objective") != "maximize_score" or value.get(
        "strict_positive_conditional_marginal"
    ) is not True:
        raise PlanEvaluationError("beam plan objective/marginal policy is invalid")
    if value.get("pareto_prune") is not False:
        raise PlanEvaluationError("primary conditional beam must not Pareto-prune")
    budget = value.get("budget_bytes")
    expected_budget = context.get("allocation", {}).get("primary_cap_added_payload_bytes")
    if type(budget) is not int or budget != expected_budget:
        raise PlanEvaluationError("beam plan byte budget differs from frozen protocol")
    if value.get("beam_width") != context.get("allocation", {}).get("beam_width"):
        raise PlanEvaluationError("beam plan width differs from frozen protocol")
    if value.get("kind") not in ({"comparison"} if is_comparison else {"baseline", "expansion"}):
        raise PlanEvaluationError("beam plan kind is invalid")
    run_fingerprint = value.get("run_fingerprint")
    catalog_hash = value.get("catalog_hash")
    _require_sha256(run_fingerprint, "run_fingerprint")
    _require_sha256(catalog_hash, "catalog_hash")
    if candidate_catalog_hash is not None and catalog_hash != candidate_catalog_hash:
        raise PlanEvaluationError("plan does not match the frozen candidate catalog")
    if type(value.get("round_index")) is not int or value["round_index"] < 0:
        raise PlanEvaluationError("beam round_index is invalid")
    rows = value.get("states")
    if not isinstance(rows, list) or not rows:
        raise PlanEvaluationError("beam plan has no states")
    costs = _cache_delta_bytes(cache)
    normalized = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PlanEvaluationError(f"beam state {index} is not an object")
        state_id = row.get("state_id")
        members = row.get("members")
        if not isinstance(state_id, str) or _STATE_ID_RE.fullmatch(state_id) is None:
            raise PlanEvaluationError(f"beam state {index} has invalid state_id")
        if state_id in seen_ids:
            raise PlanEvaluationError(f"beam plan duplicates {state_id}")
        seen_ids.add(state_id)
        if not isinstance(members, list) or not all(isinstance(name, str) for name in members):
            raise PlanEvaluationError(f"beam state {state_id} has invalid members")
        if members != sorted(members) or len(members) != len(set(members)):
            raise PlanEvaluationError(f"beam state {state_id} members are not canonical")
        unknown = sorted(set(members) - set(costs))
        if unknown:
            raise PlanEvaluationError(f"beam state {state_id} has unknown candidates: {unknown}")
        if state_id_for_members(members) != state_id:
            raise PlanEvaluationError(f"beam state ID does not match members: {state_id}")
        cost = sum(costs[name] for name in members)
        if row.get("cost_bytes") != cost or cost > budget:
            raise PlanEvaluationError(f"beam state {state_id} has invalid byte cost")
        if is_comparison:
            labels = row.get("labels")
            if (
                not isinstance(labels, list)
                or not labels
                or labels != sorted(labels)
                or len(labels) != len(set(labels))
                or not all(isinstance(label, str) and label for label in labels)
            ):
                raise PlanEvaluationError(f"comparison state {state_id} has invalid labels")
        normalized.append({**row, "members": list(members)})
    if [tuple(row["members"]) for row in normalized] != sorted(
        tuple(row["members"]) for row in normalized
    ):
        raise PlanEvaluationError("beam plan states are not in canonical member order")
    if is_comparison:
        state_ids = {row["state_id"] for row in normalized}
        labels = value["label_to_state_id"]
        if any(not isinstance(label, str) or not label for label in labels):
            raise PlanEvaluationError("comparison plan contains an invalid label")
        if set(labels.values()) - state_ids:
            raise PlanEvaluationError("comparison label maps to an unknown state")
        reverse = {
            label: row["state_id"] for row in normalized for label in row["labels"]
        }
        if reverse != labels:
            raise PlanEvaluationError("comparison state labels disagree with label mapping")
    return normalized


def incremental_state_order(
    states: Sequence[Mapping[str, Any]],
    *,
    start_members: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Deterministic nearest-neighbor order minimizing candidate swaps greedily."""
    remaining = [dict(row) for row in states]
    current = frozenset(start_members)
    ordered: list[dict[str, Any]] = []
    while remaining:
        chosen = min(
            remaining,
            key=lambda row: (
                len(current.symmetric_difference(row["members"])),
                tuple(row["members"]),
                row["state_id"],
            ),
        )
        remaining.remove(chosen)
        ordered.append(chosen)
        current = frozenset(chosen["members"])
    return ordered


def score_prediction(
    record: Mapping[str, Any],
    prediction: str,
    *,
    helpers: GroundingHelpers | None = None,
) -> dict[str, Any]:
    """Parse one decoded prediction and score it exactly like ``eval_rec``."""
    helpers = helpers or load_grounding_helpers()
    target_box1000 = _target_box_1000(record)
    parsed_target = helpers.parse_box(str(record["answer"]))
    if parsed_target != target_box1000:
        raise PlanEvaluationError(f"REC parser disagrees with target for {record['uid']}")
    predicted = helpers.parse_box(prediction)
    width, height = int(record["width"]), int(record["height"])
    x, y, box_width, box_height = (float(value) for value in record["bbox_xywh"])
    target_pixels = [x, y, x + box_width, y + box_height]
    if predicted is None:
        iou, giou = 0.0, -1.0
        predicted_pixels = None
    else:
        predicted_pixels = helpers.to_pixels(predicted, width, height)
        iou, giou = helpers.iou_giou(predicted_pixels, target_pixels)
    iou, giou = float(iou), float(giou)
    if not math.isfinite(iou) or not math.isfinite(giou):
        raise PlanEvaluationError(f"nonfinite grounding metric for {record['uid']}")
    precise_iou = sum(float(iou >= threshold) for threshold in IOU_THRESHOLDS) / len(
        IOU_THRESHOLDS
    )
    return {
        "pred_raw": prediction.strip(),
        "box1000": predicted,
        "pred_box1000": predicted,
        "pred_box_xyxy_pixels": predicted_pixels,
        "target_box1000": target_box1000,
        "target_box_xyxy_pixels": target_pixels,
        "iou": iou,
        "giou": giou,
        "precise_iou": precise_iou,
        "parse_failed": predicted is None,
    }


def _prediction_text_and_truncation(value: object) -> tuple[str, bool]:
    """Normalize injected strings or measured runtime prediction objects."""
    if isinstance(value, str):
        # String injection is for CPU tests and deliberately means
        # "known non-truncated".  The HF runtime always returns an object with
        # a flag measured from generated token IDs.
        return value, False
    if not isinstance(value, Mapping):
        raise PlanEvaluationError("decoder prediction must be text or {text,truncated}")
    text = value.get("text")
    truncated = value.get("truncated")
    if not isinstance(text, str) or type(truncated) is not bool:
        raise PlanEvaluationError("decoder prediction object needs text:str and truncated:bool")
    return text, truncated


def build_state_rows(
    state: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    predictions: Sequence[object],
    *,
    provenance: EvaluationProvenance,
    helpers: GroundingHelpers | None = None,
) -> list[dict[str, Any]]:
    if len(predictions) != len(manifest):
        raise PlanEvaluationError(
            f"decoder returned {len(predictions)} rows; expected {len(manifest)}"
        )
    state_id = str(state["state_id"])
    members = list(state["members"])
    members_hash = canonical_sha256(members)
    common = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "allocation_state_id": state_id,
        "allocation_members": members,
        "allocation_members_sha256": members_hash,
        "allocation_cost_bytes": int(state["cost_bytes"]),
        **provenance.row_fields(),
    }
    rows = []
    for index, (record, prediction_value) in enumerate(zip(manifest, predictions)):
        prediction, truncated = _prediction_text_and_truncation(prediction_value)
        rows.append({
            **common,
            "manifest_row_index": index,
            "uid": record["uid"],
            "image_id": record["image_id"],
            "task": record["task"],
            "area_quartile": record["area_quartile"],
            "source": record.get("source"),
            "generation_truncated": truncated,
            **score_prediction(record, prediction, helpers=helpers),
        })
    return rows


def _same_number(actual: object, expected: object, label: str) -> None:
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError) as error:
        raise PlanEvaluationError(f"{label} is not numeric") from error
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        raise PlanEvaluationError(f"{label} mismatch: {left} != {right}")


def validate_state_rows(
    rows: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    *,
    provenance: EvaluationProvenance,
    helpers: GroundingHelpers | None = None,
) -> dict[str, Any]:
    """Strictly validate a cached state file and recompute all row metrics."""
    if len(rows) != len(manifest):
        raise PlanEvaluationError(
            f"state {state['state_id']} has {len(rows)} rows; expected {len(manifest)}"
        )
    expected_common = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "allocation_state_id": state["state_id"],
        "allocation_members": list(state["members"]),
        "allocation_members_sha256": canonical_sha256(list(state["members"])),
        "allocation_cost_bytes": int(state["cost_bytes"]),
        **provenance.row_fields(),
    }
    identity = ("uid", "image_id", "task", "area_quartile")
    for index, (row, record) in enumerate(zip(rows, manifest)):
        for key, expected in expected_common.items():
            if row.get(key) != expected:
                raise PlanEvaluationError(
                    f"state {state['state_id']} row {index} provenance mismatch: {key}"
                )
        if row.get("manifest_row_index") != index:
            raise PlanEvaluationError(
                f"state {state['state_id']} row {index} manifest index mismatch"
            )
        for key in identity:
            if row.get(key) != record.get(key):
                raise PlanEvaluationError(
                    f"state {state['state_id']} row {index} identity mismatch: {key}"
                )
        prediction = row.get("pred_raw")
        if not isinstance(prediction, str):
            raise PlanEvaluationError(f"state row {index} has no raw prediction")
        if type(row.get("generation_truncated")) is not bool:
            raise PlanEvaluationError(
                f"state {state['state_id']} row {index} has no truncation flag"
            )
        recomputed = score_prediction(record, prediction, helpers=helpers)
        for key in (
            "box1000",
            "pred_box1000",
            "pred_box_xyxy_pixels",
            "target_box1000",
            "target_box_xyxy_pixels",
            "parse_failed",
        ):
            if row.get(key) != recomputed[key]:
                raise PlanEvaluationError(
                    f"state {state['state_id']} row {index} recomputed {key} mismatch"
                )
        for key in ("iou", "giou", "precise_iou"):
            _same_number(
                row.get(key), recomputed[key],
                f"state {state['state_id']} row {index} {key}",
            )
    try:
        return decoded_macro_summary(
            rows,
            expected_manifest_sha256=provenance.decode_manifest_sha256,
            expected_state_id=str(state["state_id"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PlanEvaluationError(
            f"state {state['state_id']} is incompatible with score_gcq_plan: {error}"
        ) from error


def state_result_path(output_dir: str | Path, state_id: str) -> Path:
    if _STATE_ID_RE.fullmatch(state_id) is None:
        raise PlanEvaluationError(f"unsafe allocation state ID {state_id!r}")
    return Path(output_dir) / f"{state_id}.decoded.jsonl"


def write_state_rows_exclusive(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Fsync a temporary file, then exclusively publish one immutable state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link creation is atomic and fails if destination already exists;
        # unlike os.replace it cannot overwrite another completed worker.
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def inspect_existing_results(
    states: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    provenance: EvaluationProvenance,
    helpers: GroundingHelpers | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate and return immutable result cache entries already on disk."""
    completed = {}
    for state in states:
        path = state_result_path(output_dir, str(state["state_id"]))
        if not path.exists():
            continue
        rows = _load_jsonl(path)
        summary = validate_state_rows(
            rows, state, manifest, provenance=provenance, helpers=helpers
        )
        completed[str(state["state_id"])] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "resumed": True,
            "summary": summary,
        }
    return completed


def evaluate_plan_states(
    *,
    cache: GPTQCandidateCache,
    model: Any,
    states: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    provenance: EvaluationProvenance,
    output_dir: str | Path,
    decoder: Callable[[Any, Sequence[Mapping[str, Any]]], Sequence[object]],
    helpers: GroundingHelpers | None = None,
) -> dict[str, dict[str, Any]]:
    """Resume/evaluate states, using incremental exact-cache compositions."""
    results = inspect_existing_results(
        states,
        manifest,
        output_dir=output_dir,
        provenance=provenance,
        helpers=helpers,
    )
    pending = [row for row in states if row["state_id"] not in results]
    if not pending:
        return results

    # Establish a known all-W4 state once, then swap only the symmetric
    # difference between successive promotion sets.
    previous = cache.compose(model, (), previous_promotions=None)
    for state in incremental_state_order(pending, start_members=previous):
        selected = cache.compose(
            model,
            state["members"],
            previous_promotions=previous,
        )
        predictions = list(decoder(model, manifest))
        rows = build_state_rows(
            state,
            manifest,
            predictions,
            provenance=provenance,
            helpers=helpers,
        )
        summary = validate_state_rows(
            rows, state, manifest, provenance=provenance, helpers=helpers
        )
        path = write_state_rows_exclusive(
            state_result_path(output_dir, str(state["state_id"])), rows
        )
        results[str(state["state_id"])] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "resumed": False,
            "summary": summary,
        }
        previous = selected
    return results


def load_hf_runtime(
    cache: GPTQCandidateCache,
    *,
    device: str,
    settings: GenerationSettings,
) -> tuple[Any, Any]:
    """Lazily load the pinned BF16 model/processor and validate exact modules."""
    import torch
    import gcq_patches
    from transformers import AutoModelForImageTextToText, AutoProcessor

    gcq_patches.apply_fast_patch_embed()
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        max_pixels=settings.max_pixels,
    )
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    cache.validate_model(model)
    return model, processor


def make_hf_decoder(
    processor: Any,
    *,
    image_root: str | Path,
    device: str,
    batch_size: int,
    settings: GenerationSettings,
) -> Callable[[Any, Sequence[Mapping[str, Any]]], Sequence[object]]:
    """Build the deterministic manifest decoder; only manifest images are opened."""
    if batch_size <= 0:
        raise PlanEvaluationError("batch_size must be positive")
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise PlanEvaluationError(f"training image root does not exist: {root}")

    def decode(model: Any, records: Sequence[Mapping[str, Any]]) -> Sequence[object]:
        import torch
        from PIL import Image

        predictions: list[object] = []
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            messages = []
            images = []
            try:
                for record in chunk:
                    file_name = str(record["file_name"])
                    if _TRAIN_IMAGE_RE.fullmatch(file_name) is None:
                        raise PlanEvaluationError("refusing to open a non-training image")
                    path = (root / file_name).resolve()
                    if path.parent != root or not path.is_file():
                        raise PlanEvaluationError(f"missing/unsafe manifest image: {path}")
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                    images.append(image)
                    messages.append([
                        {"role": "user", "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": record["prompt"]},
                        ]}
                    ])
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                ).to(device)
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=settings.max_new_tokens,
                        do_sample=settings.do_sample,
                    )
                prompt_width = inputs["input_ids"].shape[1]
                generated_tail = generated[:, prompt_width:]
                eos_value = getattr(model.generation_config, "eos_token_id", None)
                if eos_value is None:
                    eos_value = processor.tokenizer.eos_token_id
                eos_ids = (
                    {int(eos_value)}
                    if isinstance(eos_value, int)
                    else {int(item) for item in (eos_value or [])}
                )
                pad_id = getattr(model.generation_config, "pad_token_id", None)
                if pad_id is None:
                    pad_id = processor.tokenizer.pad_token_id
                for token_row in generated_tail:
                    token_ids = token_row.detach().cpu().tolist()
                    eos_position = next(
                        (index for index, token_id in enumerate(token_ids) if token_id in eos_ids),
                        None,
                    )
                    if eos_position is not None:
                        effective = token_ids[: eos_position + 1]
                        truncated = False
                    else:
                        effective = list(token_ids)
                        if pad_id is not None:
                            while effective and effective[-1] == pad_id:
                                effective.pop()
                        truncated = len(effective) >= settings.max_new_tokens
                    predictions.append({
                        "text": processor.decode(effective, skip_special_tokens=True),
                        "truncated": truncated,
                    })
            finally:
                for image in images:
                    image.close()
        if len(predictions) != len(records):
            raise PlanEvaluationError("decoder output count changed")
        return predictions

    return decode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument(
        "--candidate-catalog",
        type=Path,
        required=True,
        help="exact frozen shortlist JSON used to initialize the allocator",
    )
    parser.add_argument("--decode-manifest", type=Path, required=True)
    parser.add_argument("--protocol-context", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="number of deterministic state shards for parallel GPU workers",
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="zero-based shard index; states are assigned by canonical plan order",
    )
    args = parser.parse_args(argv)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num-shards > 0 and 0 <= shard-index < num-shards")

    settings = GenerationSettings()
    cache = GPTQCandidateCache.load(args.candidate_cache, verify_hashes=True)
    context, context_hash, manifest_hash, bank_file_hash = validate_runtime_bindings(
        args.protocol_context, args.decode_manifest, cache
    )
    candidate_catalog_hash, _ = validate_candidate_catalog(
        args.candidate_catalog, cache
    )
    manifest = validate_decode_manifest(_load_json(args.decode_manifest))
    plan = _load_json(args.plan)
    states = validate_plan(
        plan,
        cache=cache,
        context=context,
        context_sha256=context_hash,
        candidate_catalog_hash=candidate_catalog_hash,
    )
    states = states[args.shard_index :: args.num_shards]
    if not states:
        parser.error(
            f"shard {args.shard_index}/{args.num_shards} has no states in this plan"
        )
    provenance = EvaluationProvenance(
        plan_sha256=sha256_file(args.plan),
        context_sha256=context_hash,
        decode_manifest_sha256=manifest_hash,
        candidate_bank_manifest_file_sha256=bank_file_hash,
        candidate_bank_manifest_content_sha256=cache.manifest_sha256,
        generation_settings_sha256=settings.sha256,
        run_fingerprint=plan["run_fingerprint"],
        round_index=plan["round_index"],
    )
    helpers = load_grounding_helpers()
    resumed = inspect_existing_results(
        states,
        manifest,
        output_dir=args.out_dir,
        provenance=provenance,
        helpers=helpers,
    )
    if len(resumed) == len(states):
        results = resumed
    else:
        model, processor = load_hf_runtime(
            cache, device=args.device, settings=settings
        )
        decoder = make_hf_decoder(
            processor,
            image_root=args.image_root,
            device=args.device,
            batch_size=args.batch_size,
            settings=settings,
        )
        results = evaluate_plan_states(
            cache=cache,
            model=model,
            states=states,
            manifest=manifest,
            provenance=provenance,
            output_dir=args.out_dir,
            decoder=decoder,
            helpers=helpers,
        )
    ordered = {state["state_id"]: results[state["state_id"]] for state in states}
    print(json.dumps({
        "plan": str(args.plan),
        "plan_sha256": provenance.plan_sha256,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "states": ordered,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

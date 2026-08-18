"""Build the frozen, packed W4/W8 GPTQ projection bank for GCQ.

The expensive path is deliberately separated from the validation helpers.  In
particular, importing this module does not import Transformers or access the
network.  A build replays the same 128 x 512 padding-free calibration tokens
for each decoder layer.  Earlier layers are installed from their cached W4
candidates before the next layer is captured; both candidates for the current
projection are made from one pristine weight and one immutable Hessian.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from build_gptq_calibration import (
    canonical_sha256 as calibration_sha256,
    validate_calibration_manifest,
)
from gptq_candidates import (
    EXPECTED_DECODER_WEIGHTS,
    EXPECTED_LAYERS,
    EXPECTED_PROJECTIONS,
    PROJECTION_ROLES,
    GPTQCandidateCache,
    GPTQCandidateError,
    GPTQRecipe,
    QuantizedCandidate,
    enumerate_qwen_projections,
    projection_signature,
    quantize_candidate_pair,
    sha256_file,
    tensor_sha256,
    validate_qwen_projections,
)
from recovery_utils import BASE_MODEL, BASE_REVISION


CALIBRATION_SAMPLES = 128
SEQUENCE_LENGTH = 512
EXPECTED_CAPTURE_TOKENS = CALIBRATION_SAMPLES * SEQUENCE_LENGTH
PROTOCOL_ID = "gcq-projection-gptq-beam-v2"
PROTOCOL_STATUS = "design_frozen_inputs_unbound"
CANONICAL_ROLE_SHAPES: dict[str, tuple[int, int]] = {
    "self_attn.q_proj": (2048, 2048),
    "self_attn.k_proj": (1024, 2048),
    "self_attn.v_proj": (1024, 2048),
    "self_attn.o_proj": (2048, 2048),
    "mlp.gate_proj": (6144, 2048),
    "mlp.up_proj": (6144, 2048),
    "mlp.down_proj": (2048, 6144),
}


class ProjectionBankBuildError(ValueError):
    """Raised when frozen inputs or the layerwise build violate the protocol."""


@dataclass(frozen=True)
class ValidatedProtocol:
    protocol_id: str
    model_id: str
    revision: str
    decoder_layers: int
    expected_projection_count: int
    expected_decoder_weight_count: int
    calibration_seed: int
    recipe: GPTQRecipe


def load_json_object(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ProjectionBankBuildError(f"{path} must contain one JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectionBankBuildError(f"{label} must be an object")
    return value


def _plain_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectionBankBuildError(f"{label} must be an integer")
    return value


def _immutable_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ProjectionBankBuildError(f"{label} must be a full 40-character commit SHA")
    try:
        int(value, 16)
    except ValueError as error:
        raise ProjectionBankBuildError(f"{label} must be a hexadecimal commit SHA") from error
    return value.lower()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProjectionBankBuildError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ProjectionBankBuildError(f"{label} must be hexadecimal") from error
    return value.lower()


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    requested_model: str = BASE_MODEL,
    requested_revision: str = BASE_REVISION,
) -> ValidatedProtocol:
    """Validate all bank-building choices frozen in the design protocol."""
    if requested_model != BASE_MODEL:
        raise ProjectionBankBuildError(f"model must remain pinned to {BASE_MODEL}")
    requested_revision = _immutable_commit(requested_revision, "requested model revision")
    if requested_revision != BASE_REVISION:
        raise ProjectionBankBuildError(f"revision must remain pinned to {BASE_REVISION}")
    if protocol.get("schema_version") != 1 or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ProjectionBankBuildError("wrong GCQ upgrade protocol/schema")
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("bound_hashes") is not None:
        raise ProjectionBankBuildError("candidate bank requires the frozen, pre-score unbound protocol")

    model = _mapping(protocol.get("model"), "protocol model")
    model_id = model.get("id")
    revision = _immutable_commit(model.get("revision"), "protocol model revision")
    if model_id != requested_model or revision != requested_revision:
        raise ProjectionBankBuildError("protocol model/revision differs from the requested pinned model")
    if _plain_int(model.get("decoder_layers"), "decoder_layers") != EXPECTED_LAYERS:
        raise ProjectionBankBuildError("protocol must require exactly 28 decoder layers")
    if list(model.get("projection_suffixes", [])) != list(PROJECTION_ROLES):
        raise ProjectionBankBuildError("protocol projection roles/order differ from the canonical seven")
    if _plain_int(model.get("expected_projection_count"), "expected_projection_count") != EXPECTED_PROJECTIONS:
        raise ProjectionBankBuildError("protocol must require exactly 28 x 7 projections")
    if _plain_int(model.get("expected_decoder_weight_count"), "expected_decoder_weight_count") != EXPECTED_DECODER_WEIGHTS:
        raise ProjectionBankBuildError("protocol decoder weight count is not canonical")

    quantization = _mapping(protocol.get("quantization"), "protocol quantization")
    expected_literals = {
        "algorithm": "GPTQ second-order error compensation",
        "scheme": "symmetric signed absmax, qmax=2^(bits-1)-1, no zero point",
        "scale_dtype": "float16",
        "prefix_policy": "all earlier decoder layers installed from their cached W4 candidates",
        "candidate_pair_rule": (
            "W4 and W8 for one projection use identical pristine BF16 weight "
            "and independent clones of one immutable Hessian"
        ),
    }
    for key, expected in expected_literals.items():
        if quantization.get(key) != expected:
            raise ProjectionBankBuildError(f"protocol quantization.{key} differs from the frozen contract")
    if quantization.get("candidate_bits") != [4, 8]:
        raise ProjectionBankBuildError("candidate_bits must be exactly [4, 8]")
    group_size = _plain_int(quantization.get("group_size"), "group_size")
    block_size = _plain_int(quantization.get("block_size"), "block_size")
    percdamp = quantization.get("percdamp")
    if isinstance(percdamp, bool) or not isinstance(percdamp, (int, float)) or float(percdamp) != 0.01:
        raise ProjectionBankBuildError("percdamp must be exactly 0.01")

    calibration = _mapping(quantization.get("calibration"), "protocol calibration")
    expected_calibration = {
        "source": "Salesforce/wikitext wikitext-2-raw-v1 train",
        "role": "standard text-only quantizer calibration",
        "examples": CALIBRATION_SAMPLES,
        "sequence_length": SEQUENCE_LENGTH,
        "selection_seed": 20260817,
        "padding_allowed": False,
        "revision_and_token_ids_must_be_hashed": True,
    }
    if dict(calibration) != expected_calibration:
        raise ProjectionBankBuildError("protocol calibration contract is not the frozen 128 x 512 design")

    recipe = GPTQRecipe(
        base_model=model_id,
        revision=revision,
        bits=(4, 8),
        group_size=group_size,
        block_size=block_size,
        percdamp=float(percdamp),
        quant_scheme="symmetric_signed_absmax_v1",
        scale_dtype="float16",
        prefix_policy="earlier_decoder_layers_cached_w4",
    )
    return ValidatedProtocol(
        protocol_id=PROTOCOL_ID,
        model_id=model_id,
        revision=revision,
        decoder_layers=EXPECTED_LAYERS,
        expected_projection_count=EXPECTED_PROJECTIONS,
        expected_decoder_weight_count=EXPECTED_DECODER_WEIGHTS,
        calibration_seed=20260817,
        recipe=recipe,
    )


def validate_calibration_provenance(
    calibration: Mapping[str, Any],
    protocol: ValidatedProtocol,
) -> dict[str, Any]:
    """Reject validly hashed calibration that is not the exact frozen source."""
    try:
        validate_calibration_manifest(calibration)
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectionBankBuildError(f"invalid calibration manifest: {error}") from error
    if calibration.get("schema_version") != 1 or calibration.get("role") != "standard_text_gptq_calibration":
        raise ProjectionBankBuildError("wrong calibration schema/role")
    if calibration.get("base_model") != protocol.model_id:
        raise ProjectionBankBuildError("calibration base model differs from protocol")
    revision = _immutable_commit(calibration.get("base_revision"), "calibration base revision")
    if revision != protocol.revision:
        raise ProjectionBankBuildError("calibration model revision differs from protocol")
    if _plain_int(calibration.get("samples"), "calibration samples") != CALIBRATION_SAMPLES:
        raise ProjectionBankBuildError("calibration must contain exactly 128 samples")
    if _plain_int(calibration.get("sequence_length"), "calibration sequence length") != SEQUENCE_LENGTH:
        raise ProjectionBankBuildError("calibration sequences must contain exactly 512 tokens")
    if calibration.get("padding") is not False or calibration.get("attention_mask") != "all ones":
        raise ProjectionBankBuildError("calibration must be padding-free with all-one masks")

    input_ids = calibration.get("input_ids")
    if not isinstance(input_ids, list) or len(input_ids) != CALIBRATION_SAMPLES:
        raise ProjectionBankBuildError("calibration input_ids are not exactly 128 rows")
    for row_index, row in enumerate(input_ids):
        if not isinstance(row, list) or len(row) != SEQUENCE_LENGTH:
            raise ProjectionBankBuildError(f"calibration row {row_index} is not length 512")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in row):
            raise ProjectionBankBuildError(f"calibration row {row_index} contains an invalid token ID")
    input_hash = _sha256(calibration.get("input_ids_sha256"), "calibration input-ID hash")
    if input_hash != calibration_sha256(input_ids):
        raise ProjectionBankBuildError("calibration input-ID hash mismatch")

    dataset = _mapping(calibration.get("dataset"), "calibration dataset")
    if (
        dataset.get("id") != "Salesforce/wikitext"
        or dataset.get("config") != "wikitext-2-raw-v1"
        or dataset.get("split") != "train"
    ):
        raise ProjectionBankBuildError("calibration dataset identity/config/split is not frozen WikiText")
    dataset_revision = _immutable_commit(dataset.get("revision"), "calibration dataset revision")

    selection = _mapping(calibration.get("selection"), "calibration selection")
    if selection.get("namespace") != "gcq-gptq-calibration-v1":
        raise ProjectionBankBuildError("wrong calibration selection namespace")
    if _plain_int(selection.get("seed"), "calibration selection seed") != protocol.calibration_seed:
        raise ProjectionBankBuildError("calibration seed differs from protocol")
    source_rows = selection.get("source_rows")
    if not isinstance(source_rows, list) or not source_rows:
        raise ProjectionBankBuildError("calibration provenance must list its source rows")
    used_tokens = 0
    for index, raw_row in enumerate(source_rows):
        row = _mapping(raw_row, f"calibration source row {index}")
        if not isinstance(row.get("row_id"), str) or not row["row_id"]:
            raise ProjectionBankBuildError("calibration source row has no immutable row ID")
        _sha256(row.get("text_sha256"), "calibration source text hash")
        encoded = _plain_int(row.get("encoded_tokens_including_separator"), "encoded source tokens")
        used = _plain_int(row.get("tokens_used_before_cutoff"), "used source tokens")
        if encoded <= 0 or used < 0 or used > encoded:
            raise ProjectionBankBuildError("calibration source token accounting is invalid")
        used_tokens += used
    if used_tokens != EXPECTED_CAPTURE_TOKENS:
        raise ProjectionBankBuildError("calibration source rows do not account for exactly 128 x 512 tokens")

    tokenizer = _mapping(calibration.get("tokenizer"), "calibration tokenizer")
    if tokenizer.get("model") != protocol.model_id:
        raise ProjectionBankBuildError("calibration tokenizer model differs from protocol")
    if _immutable_commit(tokenizer.get("revision"), "tokenizer revision") != protocol.revision:
        raise ProjectionBankBuildError("calibration tokenizer revision differs from protocol")
    if tokenizer.get("add_special_tokens") is not False:
        raise ProjectionBankBuildError("calibration tokenizer must disable special-token insertion")
    if not isinstance(tokenizer.get("class"), str) or not tokenizer["class"]:
        raise ProjectionBankBuildError("calibration tokenizer class is missing")
    eos_token_id = _plain_int(tokenizer.get("eos_token_id"), "tokenizer eos_token_id")
    if eos_token_id < 0:
        raise ProjectionBankBuildError("tokenizer eos_token_id must be nonnegative")

    return {
        "manifest_content_sha256": _sha256(
            calibration.get("manifest_content_sha256"), "calibration manifest content hash"
        ),
        "input_ids_sha256": input_hash,
        "samples": CALIBRATION_SAMPLES,
        "sequence_length": SEQUENCE_LENGTH,
        "padding": False,
        "attention_mask": "all ones",
        "dataset": {
            "id": dataset["id"],
            "config": dataset["config"],
            "split": dataset["split"],
            "revision": dataset_revision,
            "fingerprint": dataset.get("fingerprint"),
        },
        "selection_namespace": selection["namespace"],
        "selection_seed": protocol.calibration_seed,
        "source_row_count": len(source_rows),
        "tokenizer": {
            "model": tokenizer["model"],
            "revision": protocol.revision,
            "class": tokenizer["class"],
            "add_special_tokens": False,
            "eos_token_id": eos_token_id,
        },
    }


def validate_and_group_projections(
    projections: Mapping[str, Any],
    *,
    expected_layers: int = EXPECTED_LAYERS,
    expected_weight_count: int | None = EXPECTED_DECODER_WEIGHTS,
    expected_shapes: Mapping[str, tuple[int, int]] = CANONICAL_ROLE_SHAPES,
    expected_dtype: torch.dtype | None = torch.bfloat16,
) -> dict[int, tuple[tuple[str, Any], ...]]:
    """Validate exact layer/role coverage and per-role Qwen3-VL shapes."""
    if set(expected_shapes) != set(PROJECTION_ROLES):
        raise ProjectionBankBuildError("expected_shapes must describe the canonical seven roles")
    try:
        validate_qwen_projections(
            projections,
            expected_layers=expected_layers,
            expected_weight_count=expected_weight_count,
        )
    except GPTQCandidateError as error:
        raise ProjectionBankBuildError(str(error)) from error
    grouped: dict[int, dict[str, tuple[str, Any]]] = {layer: {} for layer in range(expected_layers)}
    for name, module in projections.items():
        signature = projection_signature(name)
        if signature is None:  # Covered above; keeps type narrowing explicit.
            raise ProjectionBankBuildError(f"invalid projection name {name!r}")
        layer, role = signature
        weight = module.weight if hasattr(module, "weight") else module
        actual_shape = tuple(int(size) for size in weight.shape)
        if actual_shape != tuple(expected_shapes[role]):
            raise ProjectionBankBuildError(
                f"{name} has shape {actual_shape}; expected {tuple(expected_shapes[role])}"
            )
        if expected_dtype is not None and weight.dtype != expected_dtype:
            raise ProjectionBankBuildError(
                f"{name} has dtype {weight.dtype}; expected pristine {expected_dtype}"
            )
        grouped[layer][role] = (name, module)
    return {
        layer: tuple(grouped[layer][role] for role in PROJECTION_ROLES)
        for layer in range(expected_layers)
    }


def build_calibration_batches(
    input_ids: object,
    *,
    batch_size: int,
    device: torch.device | str,
) -> list[dict[str, torch.Tensor]]:
    batch_size = _plain_int(batch_size, "batch_size")
    if batch_size <= 0:
        raise ProjectionBankBuildError("batch_size must be positive")
    if not isinstance(input_ids, list) or len(input_ids) != CALIBRATION_SAMPLES:
        raise ProjectionBankBuildError("runtime calibration must have exactly 128 rows")
    if any(not isinstance(row, list) or len(row) != SEQUENCE_LENGTH for row in input_ids):
        raise ProjectionBankBuildError("runtime calibration rows must all have length 512")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for row in input_ids
        for token in row
    ):
        raise ProjectionBankBuildError("runtime calibration contains invalid token IDs")
    batches = []
    for start in range(0, CALIBRATION_SAMPLES, batch_size):
        ids = torch.tensor(input_ids[start:start + batch_size], dtype=torch.long, device=device)
        batches.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    if sum(batch["input_ids"].numel() for batch in batches) != EXPECTED_CAPTURE_TOKENS:
        raise AssertionError("runtime calibration token-count invariant failed")
    return batches


def validate_hessian_capture(
    layer_modules: Sequence[tuple[str, Any]],
    hessians: Mapping[str, torch.Tensor],
    counts: Mapping[str, int],
    *,
    expected_tokens: int = EXPECTED_CAPTURE_TOKENS,
) -> dict[str, int]:
    expected_names = {name for name, _ in layer_modules}
    if set(hessians) != expected_names or set(counts) != expected_names:
        raise ProjectionBankBuildError("Hessian capture keys differ from the requested layer projections")
    for name, module in layer_modules:
        width = int(module.weight.shape[1])
        hessian = hessians[name]
        if not hessian.is_floating_point() or tuple(hessian.shape) != (width, width):
            raise ProjectionBankBuildError(f"Hessian shape/dtype mismatch for {name}")
        if hessian.device != module.weight.device:
            raise ProjectionBankBuildError(f"Hessian device mismatch for {name}")
        if not torch.isfinite(hessian).all() or torch.any(torch.diag(hessian) < 0):
            raise ProjectionBankBuildError(f"Hessian is non-finite or has a negative diagonal for {name}")
        if counts[name] != expected_tokens:
            raise ProjectionBankBuildError(
                f"{name} captured {counts[name]} activation rows; expected exactly {expected_tokens}"
            )
    return dict(counts)


@torch.no_grad()
def capture_layer_hessians(
    model: torch.nn.Module,
    layer_modules: Sequence[tuple[str, Any]],
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    expected_tokens: int = EXPECTED_CAPTURE_TOKENS,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Replay calibration once and capture sum(X.T @ X) for one layer."""
    names = [name for name, _ in layer_modules]
    if len(names) != len(set(names)) or not names:
        raise ProjectionBankBuildError("layer projection names must be nonempty and unique")
    hessians = {
        name: torch.zeros(
            int(module.weight.shape[1]),
            int(module.weight.shape[1]),
            dtype=torch.float32,
            device=module.weight.device,
        )
        for name, module in layer_modules
    }
    counts = {name: 0 for name in names}
    hooks = []

    def make_hook(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ProjectionBankBuildError(f"projection {name} did not receive a tensor input")
            activation = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            if activation.shape[1] != hessians[name].shape[0]:
                raise ProjectionBankBuildError(f"activation width mismatch for {name}")
            hessians[name].add_(activation.transpose(0, 1) @ activation)
            counts[name] += int(activation.shape[0])
        return hook

    try:
        for name, module in layer_modules:
            hooks.append(module.register_forward_hook(make_hook(name)))
        for batch in batches:
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
    finally:
        for hook in hooks:
            hook.remove()
    validate_hessian_capture(
        layer_modules,
        hessians,
        counts,
        expected_tokens=expected_tokens,
    )
    return hessians, counts


def move_candidate_pair_to_cpu(
    pair: Mapping[int, QuantizedCandidate],
) -> dict[int, QuantizedCandidate]:
    if set(pair) != {4, 8}:
        raise ProjectionBankBuildError("candidate pair must contain exact W4 and W8 arms")
    moved: dict[int, QuantizedCandidate] = {}
    for bits in (4, 8):
        source = pair[bits]
        source.validate()
        candidate = QuantizedCandidate(
            bits=source.bits,
            group_size=source.group_size,
            codes=source.codes.detach().contiguous().cpu(),
            scales=source.scales.detach().contiguous().cpu(),
            qdq=source.qdq.detach().contiguous().cpu(),
            source_weight_sha256=source.source_weight_sha256,
            hessian_sha256=source.hessian_sha256,
        )
        candidate.validate()
        moved[bits] = candidate
    if moved[4].source_weight_sha256 != moved[8].source_weight_sha256:
        raise ProjectionBankBuildError("W4/W8 candidates did not use one pristine weight")
    if moved[4].hessian_sha256 != moved[8].hessian_sha256:
        raise ProjectionBankBuildError("W4/W8 candidates did not use one immutable Hessian")
    return moved


def assert_candidate_bank_cpu(
    candidates: Mapping[str, Mapping[int, QuantizedCandidate]],
) -> None:
    for name, pair in candidates.items():
        if set(pair) != {4, 8}:
            raise ProjectionBankBuildError(f"{name} does not have exact W4/W8 candidates")
        for bits, candidate in pair.items():
            if any(tensor.device.type != "cpu" for tensor in (candidate.codes, candidate.scales, candidate.qdq)):
                raise ProjectionBankBuildError(f"stored {name} W{bits} candidate is not on CPU")


@torch.no_grad()
def install_layer_w4(
    layer_modules: Sequence[tuple[str, Any]],
    candidates: Mapping[str, Mapping[int, QuantizedCandidate]],
) -> list[dict[str, Any]]:
    """Install the just-built W4 layer, checking that its source stayed pristine."""
    if set(candidates) != {name for name, _ in layer_modules}:
        raise ProjectionBankBuildError("W4 install candidates do not exactly match the layer")
    trace = []
    for name, module in layer_modules:
        pair = candidates[name]
        if set(pair) != {4, 8}:
            raise ProjectionBankBuildError(f"{name} is missing a matched W4/W8 pair")
        source_hash = tensor_sha256(module.weight.detach())
        if source_hash != pair[4].source_weight_sha256 or source_hash != pair[8].source_weight_sha256:
            raise ProjectionBankBuildError(f"{name} changed before its W4 prefix installation")
        w4 = pair[4]
        expected = w4.qdq.to(device=module.weight.device, dtype=module.weight.dtype)
        if tuple(expected.shape) != tuple(module.weight.shape):
            raise ProjectionBankBuildError(f"W4 shape mismatch while installing {name}")
        module.weight.copy_(expected)
        if not torch.equal(module.weight, expected):
            raise ProjectionBankBuildError(f"W4 installation verification failed for {name}")
        trace.append({
            "module_name": name,
            "source_weight_sha256": source_hash,
            "hessian_sha256": w4.hessian_sha256,
            "installed_w4_sha256": w4.qdq_sha256,
        })
    return trace


def validate_loaded_model_identity(model: torch.nn.Module, protocol: ValidatedProtocol) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        raise ProjectionBankBuildError("loaded model has no Transformers config")
    commit_hash = getattr(config, "_commit_hash", None)
    if not isinstance(commit_hash, str) or commit_hash.lower() != protocol.revision:
        raise ProjectionBankBuildError("loaded model config does not attest the pinned revision")
    if getattr(config, "model_type", None) != "qwen3_vl":
        raise ProjectionBankBuildError("loaded model is not Qwen3-VL")
    text_config = getattr(config, "text_config", None)
    layers = getattr(text_config, "num_hidden_layers", None)
    if layers != protocol.decoder_layers:
        raise ProjectionBankBuildError("loaded model does not have exactly 28 text decoder layers")
    vocab_size = getattr(text_config, "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ProjectionBankBuildError("loaded model has no valid text vocabulary size")
    return {
        "id": protocol.model_id,
        "revision": protocol.revision,
        "model_type": config.model_type,
        "decoder_layers": layers,
        "vocab_size": vocab_size,
        "config_name_or_path": getattr(config, "_name_or_path", None),
    }


def validate_token_range(input_ids: Sequence[Sequence[int]], vocab_size: int) -> None:
    if any(token >= vocab_size for row in input_ids for token in row):
        raise ProjectionBankBuildError("calibration contains a token outside the pinned model vocabulary")


@contextlib.contextmanager
def exclusive_output_lock(destination: str | Path) -> Iterator[None]:
    """Serialize cooperating write-once builders without reserving the output."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate bank: {path}")
    lock = path.with_name(f".{path.name}.build.lock")
    try:
        with open(lock, "x") as handle:
            json.dump({"pid": os.getpid(), "destination": str(path)}, handle, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"candidate-bank build lock already exists: {lock}") from error
    try:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite candidate bank: {path}")
        yield
    finally:
        lock.unlink(missing_ok=True)


def persist_candidate_cache(
    cache: GPTQCandidateCache,
    destination: str | Path,
    *,
    expected_candidates: int | None = EXPECTED_PROJECTIONS,
) -> GPTQCandidateCache:
    """Atomically save once, reload, and verify every packed payload hash."""
    destination = Path(destination)
    with exclusive_output_lock(destination):
        cache.save(destination)
    loaded = GPTQCandidateCache.load(destination, verify_hashes=True)
    if expected_candidates is not None and len(loaded.names) != expected_candidates:
        raise ProjectionBankBuildError(
            f"persisted bank contains {len(loaded.names)} candidates; expected {expected_candidates}"
        )
    if not loaded.verified_payloads:
        raise AssertionError("persisted candidate payload verification was not retained")
    return loaded


def build_gptq_projection_bank(
    model: torch.nn.Module,
    calibration: Mapping[str, Any],
    protocol: ValidatedProtocol,
    destination: str | Path,
    *,
    batch_size: int = 8,
    device: torch.device | str = "cuda:0",
    source_provenance: Mapping[str, Any] | None = None,
) -> GPTQCandidateCache:
    """Capture, quantize, prefix-install, pack, and verify the canonical bank."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite candidate bank: {destination}")
    calibration_provenance = validate_calibration_provenance(calibration, protocol)
    model_provenance = validate_loaded_model_identity(model, protocol)
    input_ids = calibration["input_ids"]
    validate_token_range(input_ids, model_provenance["vocab_size"])
    model.eval()

    projections = enumerate_qwen_projections(
        model,
        expected_weight_count=protocol.expected_decoder_weight_count,
    )
    grouped = validate_and_group_projections(projections)
    if len(projections) != protocol.expected_projection_count or len(grouped) != protocol.decoder_layers:
        raise ProjectionBankBuildError("loaded model projection layout differs from the protocol")
    batches = build_calibration_batches(input_ids, batch_size=batch_size, device=device)

    candidates: dict[str, dict[int, QuantizedCandidate]] = {}
    layer_trace = []
    installed_prefix: list[int] = []
    for layer in range(protocol.decoder_layers):
        if installed_prefix != list(range(layer)):
            raise AssertionError("W4 prefix ordering invariant failed")
        modules = grouped[layer]
        hessians, counts = capture_layer_hessians(model, modules, batches)
        layer_candidates: dict[str, dict[int, QuantizedCandidate]] = {}
        for name, module in modules:
            pristine_weight = module.weight.detach().clone()
            pair = quantize_candidate_pair(
                pristine_weight,
                hessians[name],
                recipe=protocol.recipe,
            )
            cpu_pair = move_candidate_pair_to_cpu(pair)
            layer_candidates[name] = cpu_pair
            candidates[name] = cpu_pair
        install_trace = install_layer_w4(modules, layer_candidates)
        installed_prefix.append(layer)
        layer_trace.append({
            "layer": layer,
            "prefix_layers_before_capture": list(range(layer)),
            "captured_activation_rows": dict(sorted(counts.items())),
            "installed_w4_before_advancing": install_trace,
        })
        del hessians, layer_candidates
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if installed_prefix != list(range(EXPECTED_LAYERS)) or len(candidates) != EXPECTED_PROJECTIONS:
        raise ProjectionBankBuildError("build did not complete the exact 28 x 7 W4 prefix")
    assert_candidate_bank_cpu(candidates)
    provenance = {
        "builder": {
            "script": Path(__file__).name,
            "script_sha256": sha256_file(__file__),
        },
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "model_id": protocol.model_id,
            "revision": protocol.revision,
            "recipe_sha256": protocol.recipe.recipe_sha256,
        },
        "model": model_provenance,
        "calibration": calibration_provenance,
        "hessian_capture": {
            "method": "layerwise_sum_x_transpose_x",
            "samples": CALIBRATION_SAMPLES,
            "sequence_length": SEQUENCE_LENGTH,
            "activation_rows_per_projection": EXPECTED_CAPTURE_TOKENS,
            "padding": False,
            "attention_mask": "all ones",
            "prefix_policy": protocol.recipe.prefix_policy,
            "layer_trace": layer_trace,
        },
        "source_files": dict(source_provenance or {}),
    }
    cache = GPTQCandidateCache.build(
        candidates,
        recipe=protocol.recipe,
        strict_qwen=True,
        provenance=provenance,
    )
    architecture = cache.manifest.get("architecture")
    if not isinstance(architecture, dict) or architecture.get("projections") != EXPECTED_PROJECTIONS:
        raise ProjectionBankBuildError("candidate cache did not retain exact Qwen architecture validation")
    return persist_candidate_cache(cache, destination)


def load_pinned_model(protocol: ValidatedProtocol, *, device: str) -> torch.nn.Module:
    """Lazy GPU/model imports keep CPU test collection network-free."""
    import gcq_patches
    gcq_patches.apply_fast_patch_embed()
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        protocol.model_id,
        revision=protocol.revision,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("gcq_upgrade_protocol.json"),
        help="frozen pre-score GCQ upgrade protocol JSON",
    )
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new packed bank directory (must not exist)")
    parser.add_argument("--model", default=BASE_MODEL, help="must equal the pinned protocol model")
    parser.add_argument("--revision", default=BASE_REVISION, help="must equal the pinned immutable commit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.out.exists():
        parser.error(f"--out already exists; refusing overwrite: {args.out}")
    protocol_value = load_json_object(args.protocol)
    protocol = validate_protocol(
        protocol_value,
        requested_model=args.model,
        requested_revision=args.revision,
    )
    calibration = load_json_object(args.calibration_manifest)
    calibration_summary = validate_calibration_provenance(calibration, protocol)
    source_provenance = {
        "protocol_path": str(args.protocol),
        "protocol_file_sha256": sha256_file(args.protocol),
        "calibration_manifest_path": str(args.calibration_manifest),
        "calibration_manifest_file_sha256": sha256_file(args.calibration_manifest),
        "calibration_manifest_content_sha256": calibration_summary["manifest_content_sha256"],
    }
    model = load_pinned_model(protocol, device=args.device)
    bank = build_gptq_projection_bank(
        model,
        calibration,
        protocol,
        args.out,
        batch_size=args.batch_size,
        device=args.device,
        source_provenance=source_provenance,
    )
    print(json.dumps({
        "out": str(args.out),
        "manifest_sha256": bank.manifest_sha256,
        "model": protocol.model_id,
        "revision": protocol.revision,
        "candidates": len(bank.names),
        "calibration_samples": CALIBRATION_SAMPLES,
        "sequence_length": SEQUENCE_LENGTH,
        "padding": False,
        "verified_payloads": bank.verified_payloads,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

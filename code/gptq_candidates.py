"""Canonical packed GPTQ projection candidates for the GCQ upgrade.

The W4 and W8 alternatives for a projection are generated from the same
pristine weight and independent clones of one Hessian. Candidates persist as
raw signed code payloads plus row/group FP16 scales; dense QDQ tensors are
reconstructed only for evaluation. Consequently ``logical_payload_bytes`` is
also the exact size of the raw selected decoder payload, while container/model
overhead remains explicitly separate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from recovery_utils import BASE_MODEL, BASE_REVISION


PROJECTION_ROLES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
EXPECTED_LAYERS = 28
EXPECTED_PROJECTIONS = EXPECTED_LAYERS * len(PROJECTION_ROLES)
EXPECTED_DECODER_WEIGHTS = 1_409_286_144
SCHEMA_VERSION = 1
CANONICAL_PACKING = {
    "code_order": "row_major",
    "w4": "signed_twos_complement_nibbles_low_nibble_first_zero_pad",
    "w8": "signed_int8",
    "scales": "row_group_float16_little_endian",
    "zero_point": None,
    "g_idx": None,
}
_PROJECTION_RE = re.compile(
    r"(?:^|.*\.)language_model\.layers\.(\d+)\."
    r"(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
)


class GPTQCandidateError(ValueError):
    """Raised when a quantized candidate or cache violates its contract."""


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    raw = value.view(torch.uint8).numpy().tobytes()
    header = canonical_json_bytes({"dtype": _dtype_name(value.dtype), "shape": list(value.shape)})
    return sha256_bytes(header + raw)


@dataclass(frozen=True)
class GPTQRecipe:
    base_model: str = BASE_MODEL
    revision: str = BASE_REVISION
    bits: tuple[int, ...] = (4, 8)
    group_size: int = 128
    block_size: int = 128
    percdamp: float = 0.01
    quant_scheme: str = "symmetric_signed_absmax_v1"
    scale_dtype: str = "float16"
    prefix_policy: str = "earlier_decoder_layers_cached_w4"

    def __post_init__(self) -> None:
        if not self.base_model or not self.revision:
            raise GPTQCandidateError("base model and immutable revision are required")
        if tuple(self.bits) != (4, 8):
            raise GPTQCandidateError("the canonical candidate bank must contain bits=(4, 8)")
        if self.group_size <= 0 or self.block_size <= 0:
            raise GPTQCandidateError("group_size and block_size must be positive")
        if not math.isfinite(self.percdamp) or self.percdamp <= 0:
            raise GPTQCandidateError("percdamp must be finite and positive")
        if self.scale_dtype != "float16":
            raise GPTQCandidateError("canonical packed scales must be float16")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["bits"] = list(self.bits)
        return value

    @property
    def recipe_sha256(self) -> str:
        return canonical_sha256(self.as_dict())


def projection_signature(name: str) -> tuple[int, str] | None:
    match = _PROJECTION_RE.fullmatch(name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def validate_qwen_projections(
    projections: Mapping[str, Any],
    *,
    expected_layers: int = EXPECTED_LAYERS,
    expected_weight_count: int | None = EXPECTED_DECODER_WEIGHTS,
) -> dict[str, Any]:
    if len(projections) != expected_layers * len(PROJECTION_ROLES):
        raise GPTQCandidateError(
            f"found {len(projections)} projections; expected {expected_layers * len(PROJECTION_ROLES)}"
        )
    counts: dict[tuple[int, str], int] = {}
    total = 0
    for name, value in projections.items():
        signature = projection_signature(name)
        if signature is None:
            raise GPTQCandidateError(f"invalid projection module name {name!r}")
        layer, role = signature
        if not 0 <= layer < expected_layers:
            raise GPTQCandidateError(f"projection layer outside 0..{expected_layers - 1}: {name}")
        counts[signature] = counts.get(signature, 0) + 1
        weight = value.weight if hasattr(value, "weight") else value
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise GPTQCandidateError(f"projection {name!r} has no 2D weight")
        total += weight.numel()
    expected = {(layer, role) for layer in range(expected_layers) for role in PROJECTION_ROLES}
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        raise GPTQCandidateError("projection bank does not contain exactly one of seven roles per layer")
    if expected_weight_count is not None and total != expected_weight_count:
        raise GPTQCandidateError(
            f"decoder projection weight count is {total:,}; expected {expected_weight_count:,}"
        )
    return {
        "layers": expected_layers,
        "projections": len(projections),
        "weights": total,
        "role_counts": {role: sum(layer_role[1] == role for layer_role in counts) for role in PROJECTION_ROLES},
    }


def enumerate_qwen_projections(
    model: torch.nn.Module,
    *,
    expected_weight_count: int | None = EXPECTED_DECODER_WEIGHTS,
) -> dict[str, torch.nn.Linear]:
    projections = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and projection_signature(name) is not None
    }
    validate_qwen_projections(
        projections,
        expected_layers=EXPECTED_LAYERS,
        expected_weight_count=expected_weight_count,
    )
    return dict(sorted(projections.items()))


def pack_signed_codes(codes: torch.Tensor, bits: int) -> bytes:
    flat = codes.detach().contiguous().cpu().to(torch.int8).reshape(-1)
    qmax = 2 ** (bits - 1) - 1
    if bits not in (4, 8) or torch.any(flat < -qmax) or torch.any(flat > qmax):
        raise GPTQCandidateError(f"codes are outside canonical signed W{bits} range")
    if bits == 8:
        return flat.view(torch.uint8).numpy().tobytes()
    nibble = torch.bitwise_and(flat.to(torch.int16), 0x0F).to(torch.uint8)
    if nibble.numel() % 2:
        nibble = torch.cat((nibble, torch.zeros(1, dtype=torch.uint8)))
    packed = nibble[0::2] | (nibble[1::2] << 4)
    return packed.numpy().tobytes()


def unpack_signed_codes(payload: bytes, bits: int, shape: Sequence[int]) -> torch.Tensor:
    count = math.prod(int(size) for size in shape)
    if bits == 8:
        if len(payload) != count:
            raise GPTQCandidateError(f"W8 payload has {len(payload)} bytes; expected {count}")
        raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
        return raw.view(torch.int8).reshape(tuple(shape))
    if bits != 4:
        raise GPTQCandidateError("only W4 and W8 signed payloads are supported")
    expected_bytes = (count + 1) // 2
    if len(payload) != expected_bytes:
        raise GPTQCandidateError(f"W4 payload has {len(payload)} bytes; expected {expected_bytes}")
    packed = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    values = torch.empty(packed.numel() * 2, dtype=torch.int16)
    values[0::2] = packed & 0x0F
    values[1::2] = packed >> 4
    values = values[:count]
    values = torch.where(values >= 8, values - 16, values)
    return values.to(torch.int8).reshape(tuple(shape))


def pack_fp16_scales(scales: torch.Tensor) -> bytes:
    if sys.byteorder != "little":
        raise GPTQCandidateError("canonical raw scale payload currently requires little-endian host")
    value = scales.detach().contiguous().cpu().to(torch.float16)
    return value.view(torch.uint8).numpy().tobytes()


def unpack_fp16_scales(payload: bytes, shape: Sequence[int]) -> torch.Tensor:
    count = math.prod(int(size) for size in shape)
    if len(payload) != count * 2:
        raise GPTQCandidateError(f"scale payload has {len(payload)} bytes; expected {count * 2}")
    return torch.frombuffer(bytearray(payload), dtype=torch.float16).clone().reshape(tuple(shape))


@dataclass
class QuantizedCandidate:
    bits: int
    group_size: int
    codes: torch.Tensor
    scales: torch.Tensor
    qdq: torch.Tensor
    source_weight_sha256: str
    hessian_sha256: str

    def validate(self) -> None:
        if self.bits not in (4, 8) or self.group_size <= 0:
            raise GPTQCandidateError("invalid candidate bits/group size")
        if self.codes.dtype != torch.int8 or self.codes.ndim != 2:
            raise GPTQCandidateError("candidate codes must be a 2D int8 tensor")
        expected_groups = math.ceil(self.codes.shape[1] / self.group_size)
        if self.scales.dtype != torch.float16 or tuple(self.scales.shape) != (
            self.codes.shape[0], expected_groups
        ):
            raise GPTQCandidateError("candidate scale shape/dtype mismatch")
        if tuple(self.qdq.shape) != tuple(self.codes.shape) or not self.qdq.is_floating_point():
            raise GPTQCandidateError("candidate QDQ shape/dtype mismatch")
        if not torch.isfinite(self.scales).all() or torch.any(self.scales <= 0):
            raise GPTQCandidateError("candidate scales must be finite and positive")
        expected = self.dequantize(dtype=self.qdq.dtype, device=self.qdq.device)
        if not torch.equal(expected, self.qdq):
            raise GPTQCandidateError("candidate QDQ does not match packed codes/scales")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.codes.shape)  # type: ignore[return-value]

    @property
    def numel(self) -> int:
        return self.codes.numel()

    @property
    def logical_code_bytes(self) -> int:
        return (self.numel * self.bits + 7) // 8

    @property
    def logical_scale_bytes(self) -> int:
        return self.scales.numel() * 2

    @property
    def logical_payload_bytes(self) -> int:
        return self.logical_code_bytes + self.logical_scale_bytes

    @property
    def codes_sha256(self) -> str:
        return sha256_bytes(pack_signed_codes(self.codes, self.bits))

    @property
    def scales_sha256(self) -> str:
        return sha256_bytes(pack_fp16_scales(self.scales))

    @property
    def qdq_sha256(self) -> str:
        return tensor_sha256(self.qdq)

    def dequantize(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        codes = self.codes.to(device=device, dtype=torch.float32)
        scales = self.scales.to(device=device, dtype=torch.float32)
        columns = torch.arange(self.codes.shape[1], device=device) // self.group_size
        return (codes * scales[:, columns]).to(dtype)


@torch.no_grad()
def gptq_quantize_linear(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    bits: int,
    recipe: GPTQRecipe | None = None,
) -> QuantizedCandidate:
    """Return a packed candidate without mutating ``weight`` or ``hessian``."""
    recipe = recipe or GPTQRecipe()
    if bits not in recipe.bits:
        raise GPTQCandidateError(f"bits must be one of {recipe.bits}")
    if weight.ndim != 2 or not weight.is_floating_point():
        raise GPTQCandidateError("weight must be a floating 2D tensor")
    out_features, in_features = weight.shape
    if tuple(hessian.shape) != (in_features, in_features) or not hessian.is_floating_point():
        raise GPTQCandidateError("hessian shape/dtype does not match weight input width")
    if weight.device != hessian.device:
        raise GPTQCandidateError("weight and hessian must be on the same device")

    source_hash = tensor_sha256(weight)
    hessian_hash = tensor_sha256(hessian)
    work = weight.detach().clone().float()
    h_work = hessian.detach().clone().float()
    qmax = 2 ** (bits - 1) - 1
    diagonal = torch.diag(h_work)
    if torch.any(diagonal < 0) or not torch.isfinite(h_work).all():
        raise GPTQCandidateError("hessian must be finite with nonnegative diagonal")
    dead = diagonal == 0
    if dead.any():
        indices = torch.arange(in_features, device=h_work.device)[dead]
        h_work[indices, indices] = 1.0
        work[:, dead] = 0.0
    damp = recipe.percdamp * torch.mean(torch.diag(h_work))
    if not torch.isfinite(damp) or damp <= 0:
        raise GPTQCandidateError("hessian damping is not finite and positive")
    h_work.diagonal().add_(damp)
    try:
        chol = torch.linalg.cholesky(h_work)
    except RuntimeError as error:
        raise GPTQCandidateError("damped hessian is not positive definite") from error
    inverse = torch.cholesky_inverse(chol)
    inverse_factor = torch.linalg.cholesky(inverse, upper=True)

    groups = math.ceil(in_features / recipe.group_size)
    scales32 = torch.zeros(out_features, groups, device=work.device, dtype=torch.float32)
    codes = torch.zeros_like(work, dtype=torch.int8)
    for block_start in range(0, in_features, recipe.block_size):
        block_end = min(block_start + recipe.block_size, in_features)
        block = work[:, block_start:block_end].clone()
        errors = torch.zeros_like(block)
        block_inverse = inverse_factor[block_start:block_end, block_start:block_end]
        for local_column in range(block_end - block_start):
            column = block_start + local_column
            group_index = column // recipe.group_size
            if column % recipe.group_size == 0:
                group_end = min(column + recipe.group_size, in_features)
                # The serialized quantizer stores FP16 scales.  Round the scale
                # before both error propagation and code selection so the GPTQ
                # optimization sees exactly the quantizer that is later loaded,
                # rather than an unreportable FP32-scale surrogate.
                scale = work[:, column:group_end].abs().amax(dim=1) / qmax
                scale = scale.to(torch.float16).clamp_min(torch.finfo(torch.float16).tiny).float()
                scales32[:, group_index] = scale
            scale = scales32[:, group_index]
            current = block[:, local_column]
            integer = torch.round(current / scale).clamp(-qmax, qmax).to(torch.int8)
            quantized = integer.float() * scale
            denominator = block_inverse[local_column, local_column]
            if not torch.isfinite(denominator) or denominator == 0:
                raise GPTQCandidateError("invalid inverse-Hessian diagonal")
            error = (current - quantized) / denominator
            block[:, local_column] = quantized
            codes[:, column] = integer
            if local_column + 1 < block_end - block_start:
                block[:, local_column + 1:] -= (
                    error.unsqueeze(1)
                    * block_inverse[local_column, local_column + 1:].unsqueeze(0)
                )
            errors[:, local_column] = error
        work[:, block_start:block_end] = block
        if block_end < in_features:
            work[:, block_end:] -= errors @ inverse_factor[block_start:block_end, block_end:]

    scales = scales32.to(torch.float16)
    columns = torch.arange(in_features, device=work.device) // recipe.group_size
    qdq = (codes.float() * scales.float()[:, columns]).to(weight.dtype)
    candidate = QuantizedCandidate(
        bits=bits,
        group_size=recipe.group_size,
        codes=codes,
        scales=scales,
        qdq=qdq,
        source_weight_sha256=source_hash,
        hessian_sha256=hessian_hash,
    )
    candidate.validate()
    return candidate


def quantize_candidate_pair(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    recipe: GPTQRecipe | None = None,
) -> dict[int, QuantizedCandidate]:
    recipe = recipe or GPTQRecipe()
    pair = {
        bits: gptq_quantize_linear(weight, hessian, bits=bits, recipe=recipe)
        for bits in recipe.bits
    }
    if pair[4].source_weight_sha256 != pair[8].source_weight_sha256:
        raise AssertionError("W4/W8 source weight hash mismatch")
    if pair[4].hessian_sha256 != pair[8].hessian_sha256:
        raise AssertionError("W4/W8 Hessian hash mismatch")
    return pair


class GPTQCandidateCache:
    """Packed candidate bank with exact-name model composition."""

    def __init__(
        self,
        manifest: dict[str, Any],
        *,
        root: Path | None = None,
        candidates: Mapping[str, Mapping[int, QuantizedCandidate]] | None = None,
        verified_payloads: bool = False,
    ):
        self.manifest = manifest
        self.root = root
        self._candidates = {
            name: dict(pair) for name, pair in (candidates or {}).items()
        }
        self.verified_payloads = verified_payloads

    @classmethod
    def build(
        cls,
        candidates: Mapping[str, Mapping[int, QuantizedCandidate]],
        *,
        recipe: GPTQRecipe | None = None,
        strict_qwen: bool = True,
        provenance: Mapping[str, Any] | None = None,
    ) -> "GPTQCandidateCache":
        recipe = recipe or GPTQRecipe()
        if not candidates:
            raise GPTQCandidateError("candidate bank is empty")
        projection_values = {}
        rows = []
        normalized: dict[str, dict[int, QuantizedCandidate]] = {}
        for name in sorted(candidates):
            if projection_signature(name) is None:
                raise GPTQCandidateError(f"invalid projection name {name!r}")
            pair = dict(candidates[name])
            if set(pair) != {4, 8}:
                raise GPTQCandidateError(f"{name} must contain exact W4 and W8 candidates")
            for candidate in pair.values():
                candidate.validate()
                if candidate.group_size != recipe.group_size:
                    raise GPTQCandidateError(f"{name} candidate group size differs from recipe")
            if pair[4].shape != pair[8].shape:
                raise GPTQCandidateError(f"{name} W4/W8 shape mismatch")
            if pair[4].source_weight_sha256 != pair[8].source_weight_sha256:
                raise GPTQCandidateError(f"{name} W4/W8 source weight mismatch")
            if pair[4].hessian_sha256 != pair[8].hessian_sha256:
                raise GPTQCandidateError(f"{name} W4/W8 Hessian mismatch")
            projection_values[name] = torch.empty(pair[4].shape)
            normalized[name] = pair
            rows.append({
                "module_name": name,
                "shape": list(pair[4].shape),
                "numel": pair[4].numel,
                "source_weight_sha256": pair[4].source_weight_sha256,
                "hessian_sha256": pair[4].hessian_sha256,
                "delta_bytes": pair[8].logical_payload_bytes - pair[4].logical_payload_bytes,
                "w4_sha256": pair[4].qdq_sha256,
                "w8_sha256": pair[8].qdq_sha256,
            })
        architecture = None
        if strict_qwen:
            architecture = validate_qwen_projections(projection_values)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "gcq_packed_gptq_projection_bank",
            "recipe": recipe.as_dict(),
            "recipe_sha256": recipe.recipe_sha256,
            "packing": dict(CANONICAL_PACKING),
            "architecture": architecture,
            "provenance": dict(provenance or {}),
            "candidates": rows,
        }
        manifest["manifest_content_sha256"] = canonical_sha256(manifest)
        return cls(manifest, candidates=normalized, verified_payloads=True)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(row["module_name"] for row in self.manifest["candidates"])

    @property
    def manifest_sha256(self) -> str:
        return self.manifest["manifest_content_sha256"]

    def _row(self, name: str) -> dict[str, Any]:
        for row in self.manifest["candidates"]:
            if row["module_name"] == name:
                return row
        raise GPTQCandidateError(f"unknown projection candidate {name!r}")

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite candidate bank: {destination}")
        if not self._candidates:
            raise GPTQCandidateError("cache has no in-memory candidates to save")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        temporary.mkdir()
        tensor_dir = temporary / "tensors"
        tensor_dir.mkdir()
        try:
            rows = []
            for base_row in self.manifest["candidates"]:
                name = base_row["module_name"]
                stem = hashlib.sha256(name.encode()).hexdigest()[:24]
                row = dict(base_row)
                for bits in (4, 8):
                    candidate = self._candidates[name][bits]
                    codes = pack_signed_codes(candidate.codes, bits)
                    scales = pack_fp16_scales(candidate.scales)
                    code_rel = f"tensors/{stem}.w{bits}.codes.bin"
                    scale_rel = f"tensors/{stem}.w{bits}.scales.f16"
                    (temporary / code_rel).write_bytes(codes)
                    (temporary / scale_rel).write_bytes(scales)
                    row[f"w{bits}"] = {
                        "bits": bits,
                        "group_size": candidate.group_size,
                        "codes_file": code_rel,
                        "scales_file": scale_rel,
                        "codes_sha256": sha256_bytes(codes),
                        "scales_sha256": sha256_bytes(scales),
                        "qdq_sha256": candidate.qdq_sha256,
                        "code_bytes": len(codes),
                        "scale_bytes": len(scales),
                        "logical_payload_bytes": len(codes) + len(scales),
                    }
                row["delta_bytes"] = row["w8"]["logical_payload_bytes"] - row["w4"]["logical_payload_bytes"]
                rows.append(row)
            saved_manifest = dict(self.manifest)
            saved_manifest["candidates"] = rows
            saved_manifest.pop("manifest_content_sha256", None)
            saved_manifest["manifest_content_sha256"] = canonical_sha256(saved_manifest)
            (temporary / "manifest.json").write_bytes(canonical_json_bytes(saved_manifest))
            os.replace(temporary, destination)
            self.manifest = saved_manifest
            self.root = destination
            return destination
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path: str | Path, *, verify_hashes: bool = True) -> "GPTQCandidateCache":
        root = Path(path)
        with open(root / "manifest.json") as handle:
            manifest = json.load(handle)
        content_hash = manifest.get("manifest_content_sha256")
        unhashed = dict(manifest)
        unhashed.pop("manifest_content_sha256", None)
        if canonical_sha256(unhashed) != content_hash:
            raise GPTQCandidateError("candidate-bank manifest content hash mismatch")
        cls._validate_saved_manifest(manifest)
        cache = cls(manifest, root=root, verified_payloads=False)
        if verify_hashes:
            for row in manifest["candidates"]:
                for bits in (4, 8):
                    arm = row[f"w{bits}"]
                    code_path = root / arm["codes_file"]
                    scale_path = root / arm["scales_file"]
                    if code_path.stat().st_size != arm["code_bytes"] or sha256_file(code_path) != arm["codes_sha256"]:
                        raise GPTQCandidateError(f"corrupt W{bits} code payload for {row['module_name']}")
                    if scale_path.stat().st_size != arm["scale_bytes"] or sha256_file(scale_path) != arm["scales_sha256"]:
                        raise GPTQCandidateError(f"corrupt W{bits} scale payload for {row['module_name']}")
        cache.verified_payloads = verify_hashes
        return cache

    @staticmethod
    def _validate_saved_manifest(manifest: Mapping[str, Any]) -> None:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise GPTQCandidateError("unsupported candidate-bank schema version")
        if manifest.get("artifact_kind") != "gcq_packed_gptq_projection_bank":
            raise GPTQCandidateError("wrong candidate-bank artifact kind")
        recipe_value = manifest.get("recipe")
        if not isinstance(recipe_value, dict):
            raise GPTQCandidateError("candidate-bank recipe is missing")
        try:
            recipe_args = dict(recipe_value)
            recipe_args["bits"] = tuple(recipe_args.get("bits", ()))
            recipe = GPTQRecipe(**recipe_args)
        except (TypeError, GPTQCandidateError) as error:
            raise GPTQCandidateError("invalid candidate-bank recipe") from error
        if manifest.get("recipe_sha256") != recipe.recipe_sha256:
            raise GPTQCandidateError("candidate-bank recipe hash mismatch")
        if manifest.get("packing") != CANONICAL_PACKING:
            raise GPTQCandidateError("candidate-bank packing contract mismatch")
        rows = manifest.get("candidates")
        if not isinstance(rows, list) or not rows:
            raise GPTQCandidateError("candidate-bank manifest has no candidates")
        names = [row.get("module_name") if isinstance(row, dict) else None for row in rows]
        if not all(isinstance(name, str) and name for name in names) or len(names) != len(set(names)):
            raise GPTQCandidateError("candidate-bank manifest names are empty or duplicated")
        payload_files: set[str] = set()
        for row in rows:
            name = row["module_name"]
            if projection_signature(name) is None:
                raise GPTQCandidateError(f"invalid persisted projection name {name!r}")
            shape = row.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
            ):
                raise GPTQCandidateError(f"invalid persisted shape for {name}")
            numel = shape[0] * shape[1]
            if row.get("numel") != numel:
                raise GPTQCandidateError(f"persisted numel mismatch for {name}")
            for field in ("source_weight_sha256", "hessian_sha256", "w4_sha256", "w8_sha256"):
                if not _is_sha256(row.get(field)):
                    raise GPTQCandidateError(f"invalid {field} for {name}")
            arms = {}
            for bits in (4, 8):
                arm = row.get(f"w{bits}")
                if not isinstance(arm, dict) or arm.get("bits") != bits:
                    raise GPTQCandidateError(f"invalid W{bits} metadata for {name}")
                if arm.get("group_size") != recipe.group_size:
                    raise GPTQCandidateError(f"W{bits} group size mismatch for {name}")
                expected_code_bytes = (numel * bits + 7) // 8
                expected_scale_bytes = 2 * shape[0] * math.ceil(shape[1] / recipe.group_size)
                if arm.get("code_bytes") != expected_code_bytes:
                    raise GPTQCandidateError(f"W{bits} code-byte mismatch for {name}")
                if arm.get("scale_bytes") != expected_scale_bytes:
                    raise GPTQCandidateError(f"W{bits} scale-byte mismatch for {name}")
                if arm.get("logical_payload_bytes") != expected_code_bytes + expected_scale_bytes:
                    raise GPTQCandidateError(f"W{bits} payload-byte mismatch for {name}")
                if arm.get("qdq_sha256") != row[f"w{bits}_sha256"]:
                    raise GPTQCandidateError(f"W{bits} QDQ hash mismatch for {name}")
                for file_field, hash_field in (
                    ("codes_file", "codes_sha256"),
                    ("scales_file", "scales_sha256"),
                ):
                    relative = arm.get(file_field)
                    if (
                        not isinstance(relative, str)
                        or not relative
                        or Path(relative).is_absolute()
                        or ".." in Path(relative).parts
                        or relative in payload_files
                    ):
                        raise GPTQCandidateError(f"unsafe or duplicate W{bits} payload path for {name}")
                    payload_files.add(relative)
                    if not _is_sha256(arm.get(hash_field)):
                        raise GPTQCandidateError(f"invalid W{bits} payload hash for {name}")
                arms[bits] = arm
            expected_delta = arms[8]["logical_payload_bytes"] - arms[4]["logical_payload_bytes"]
            if row.get("delta_bytes") != expected_delta or expected_delta <= 0:
                raise GPTQCandidateError(f"promotion-byte mismatch for {name}")

    def candidate(
        self,
        name: str,
        bits: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> QuantizedCandidate:
        if bits not in (4, 8):
            raise GPTQCandidateError("candidate bits must be W4 or W8")
        if name in self._candidates:
            source = self._candidates[name][bits]
            qdq = source.dequantize(dtype=dtype, device=device)
            return QuantizedCandidate(
                bits=bits,
                group_size=source.group_size,
                codes=source.codes.to(device),
                scales=source.scales.to(device),
                qdq=qdq,
                source_weight_sha256=source.source_weight_sha256,
                hessian_sha256=source.hessian_sha256,
            )
        if self.root is None:
            raise GPTQCandidateError("cache has neither in-memory nor persisted candidates")
        if not self.verified_payloads:
            raise GPTQCandidateError(
                "persisted payloads were loaded without hash verification"
            )
        row = self._row(name)
        arm = row[f"w{bits}"]
        codes = unpack_signed_codes((self.root / arm["codes_file"]).read_bytes(), bits, row["shape"])
        scales_shape = (row["shape"][0], math.ceil(row["shape"][1] / arm["group_size"]))
        scales = unpack_fp16_scales((self.root / arm["scales_file"]).read_bytes(), scales_shape)
        temporary = QuantizedCandidate(
            bits=bits,
            group_size=arm["group_size"],
            codes=codes.to(device),
            scales=scales.to(device),
            qdq=torch.empty(tuple(row["shape"]), device=device, dtype=dtype),
            source_weight_sha256=row["source_weight_sha256"],
            hessian_sha256=row["hessian_sha256"],
        )
        temporary.qdq = temporary.dequantize(dtype=dtype, device=device)
        return temporary

    def validate_model(self, model: torch.nn.Module) -> dict[str, torch.nn.Linear]:
        modules = dict(model.named_modules())
        selected = {}
        for row in self.manifest["candidates"]:
            name = row["module_name"]
            module = modules.get(name)
            if not isinstance(module, torch.nn.Linear):
                raise GPTQCandidateError(f"model is missing exact linear module {name!r}")
            if list(module.weight.shape) != row["shape"]:
                raise GPTQCandidateError(f"model shape mismatch for {name!r}")
            selected[name] = module
        return selected

    @torch.no_grad()
    def _install_module(
        self,
        module: torch.nn.Linear,
        name: str,
        bits: int,
        *,
        verify: bool,
    ) -> None:
        candidate = self.candidate(name, bits, device=module.weight.device, dtype=module.weight.dtype)
        module.weight.copy_(candidate.qdq)
        arm = self._row(name).get(f"w{bits}")
        if verify and arm and tensor_sha256(module.weight) != arm["qdq_sha256"]:
            raise GPTQCandidateError(f"installed QDQ hash mismatch for {name} W{bits}")

    @torch.no_grad()
    def install(
        self,
        model: torch.nn.Module,
        name: str,
        bits: int,
        *,
        verify: bool = False,
    ) -> None:
        module = self.validate_model(model)[name]
        self._install_module(module, name, bits, verify=verify)

    @torch.no_grad()
    def compose(
        self,
        model: torch.nn.Module,
        promotions: Iterable[str] = (),
        *,
        previous_promotions: Iterable[str] | None = None,
        verify_installs: bool = False,
    ) -> frozenset[str]:
        selected = frozenset(promotions)
        unknown = sorted(selected - set(self.names))
        if unknown:
            raise GPTQCandidateError(f"composition contains unknown projections: {unknown}")
        modules = self.validate_model(model)
        if previous_promotions is None:
            for name in self.names:
                self._install_module(modules[name], name, 4, verify=verify_installs)
            for name in sorted(selected):
                self._install_module(modules[name], name, 8, verify=verify_installs)
        else:
            previous = frozenset(previous_promotions)
            unknown_previous = sorted(previous - set(self.names))
            if unknown_previous:
                raise GPTQCandidateError(f"previous composition contains unknown projections: {unknown_previous}")
            for name in sorted(previous - selected):
                self._install_module(modules[name], name, 4, verify=verify_installs)
            for name in sorted(selected - previous):
                self._install_module(modules[name], name, 8, verify=verify_installs)
        return selected

    def composition_manifest(self, promotions: Iterable[str]) -> dict[str, Any]:
        selected = frozenset(promotions)
        unknown = sorted(selected - set(self.names))
        if unknown:
            raise GPTQCandidateError(f"composition contains unknown projections: {unknown}")
        payload = 0
        code_bytes = 0
        scale_bytes = 0
        weight_count = 0
        rows = []
        for row in self.manifest["candidates"]:
            bits = 8 if row["module_name"] in selected else 4
            arm = row[f"w{bits}"]
            payload += arm["logical_payload_bytes"]
            code_bytes += arm["code_bytes"]
            scale_bytes += arm["scale_bytes"]
            weight_count += row["numel"]
            rows.append({"module_name": row["module_name"], "bits": bits})
        uniform_w4_code_bytes = sum(row["w4"]["code_bytes"] for row in self.manifest["candidates"])
        added_code_bytes = code_bytes - uniform_w4_code_bytes
        return {
            "schema_version": 1,
            "artifact_kind": "gcq_decoder_payload_composition",
            "candidate_bank_manifest_sha256": self.manifest_sha256,
            "promotions": sorted(selected),
            "projection_bits": rows,
            "decoder_weights": weight_count,
            "code_bytes": code_bytes,
            "scale_bytes": scale_bytes,
            "logical_packed_decoder_payload_bytes": payload,
            "added_code_bytes_over_uniform_w4": added_code_bytes,
            "average_decoder_code_bits": 4.0 + 8.0 * added_code_bytes / weight_count,
            "average_decoder_payload_bits": 8.0 * payload / weight_count,
            "dense_qdq_is_compressed_checkpoint": False,
            "fixed_nondecoder_components_included": False,
        }

    def write_composition_manifest(self, path: str | Path, promotions: Iterable[str]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "x") as handle:
            json.dump(self.composition_manifest(promotions), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return destination

    def export_packed_composition(
        self, path: str | Path, promotions: Iterable[str]
    ) -> Path:
        """Materialize only the selected packed arm for every projection.

        This artifact makes the reported decoder payload a set of actual raw
        files rather than a counterfactual sum over the two-arm candidate bank.
        It is archival/evaluation metadata, not a claim that a packed inference
        kernel is implemented.
        """
        if self.root is None or not self.verified_payloads:
            raise GPTQCandidateError(
                "packed composition export requires a verified persisted bank"
            )
        selected = frozenset(promotions)
        unknown = sorted(selected - set(self.names))
        if unknown:
            raise GPTQCandidateError(f"composition contains unknown projections: {unknown}")
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite packed composition: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        (temporary / "tensors").mkdir(parents=True)
        try:
            exported_rows = []
            payload_bytes = 0
            for row in self.manifest["candidates"]:
                name = row["module_name"]
                bits = 8 if name in selected else 4
                arm = row[f"w{bits}"]
                stem = hashlib.sha256(name.encode()).hexdigest()[:24]
                code_relative = f"tensors/{stem}.w{bits}.codes.bin"
                scale_relative = f"tensors/{stem}.w{bits}.scales.f16"
                shutil.copyfile(self.root / arm["codes_file"], temporary / code_relative)
                shutil.copyfile(self.root / arm["scales_file"], temporary / scale_relative)
                if sha256_file(temporary / code_relative) != arm["codes_sha256"]:
                    raise GPTQCandidateError(f"exported code hash mismatch for {name}")
                if sha256_file(temporary / scale_relative) != arm["scales_sha256"]:
                    raise GPTQCandidateError(f"exported scale hash mismatch for {name}")
                payload_bytes += arm["logical_payload_bytes"]
                exported_rows.append({
                    "module_name": name,
                    "bits": bits,
                    "shape": row["shape"],
                    "numel": row["numel"],
                    "codes_file": code_relative,
                    "scales_file": scale_relative,
                    "codes_sha256": arm["codes_sha256"],
                    "scales_sha256": arm["scales_sha256"],
                    "qdq_sha256": arm["qdq_sha256"],
                    "code_bytes": arm["code_bytes"],
                    "scale_bytes": arm["scale_bytes"],
                    "logical_payload_bytes": arm["logical_payload_bytes"],
                })
            composition = self.composition_manifest(selected)
            if payload_bytes != composition["logical_packed_decoder_payload_bytes"]:
                raise AssertionError("packed composition payload accounting mismatch")
            manifest = {
                "schema_version": 1,
                "artifact_kind": "gcq_materialized_packed_decoder_composition",
                "candidate_bank_manifest_file_sha256": sha256_file(self.root / "manifest.json"),
                "candidate_bank_manifest_content_sha256": self.manifest_sha256,
                "packing": self.manifest["packing"],
                "composition": composition,
                "payload_file_bytes": payload_bytes,
                "payload_files": len(exported_rows) * 2,
                "packed_inference_kernel_included": False,
                "candidates": exported_rows,
            }
            manifest["manifest_content_sha256"] = canonical_sha256(manifest)
            (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            os.replace(temporary, destination)
            return destination
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

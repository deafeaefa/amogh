#!/usr/bin/env python3
"""Materialize one frozen GCQ state as dense BF16 QDQ plus packed payload.

The dense ``save_pretrained`` directory exists only so the repository's current
evaluation harnesses can load the selected quantize/dequantize weights.  It is
explicitly not a compressed checkpoint.  The sibling packed composition stores
the actual selected W4/W8 code and FP16-scale payload used for byte accounting;
it does not contain a packed inference kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from allocate_gcq_beam import load_candidates
from build_gcq_comparison_plan import canonical_sha256, validate_selection
from gptq_candidates import GPTQCandidateCache, canonical_sha256 as artifact_sha256
from recovery_utils import BASE_MODEL, BASE_REVISION


SCHEMA_VERSION = 1
IMPLEMENTATION_FILES = (
    "materialize_gcq_checkpoint.py",
    "gptq_candidates.py",
    "allocate_gcq_beam.py",
    "build_gcq_comparison_plan.py",
    "recovery_utils.py",
    "gcq_patches.py",
)


class MaterializationError(ValueError):
    """Raised when a final state is not completely frozen and hash-bound."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise MaterializationError(f"{label} must be a JSON object")
    return value


def candidate_catalog_hash(
    path: str | Path, cache: GPTQCandidateCache
) -> tuple[str, dict[str, int]]:
    try:
        candidates, _ = load_candidates(path)
    except ValueError as error:
        raise MaterializationError(f"invalid frozen candidate catalog: {error}") from error
    rows = [candidate.as_dict() for candidate in candidates]
    cache_costs = {
        str(row["module_name"]): int(row["delta_bytes"])
        for row in cache.manifest["candidates"]
    }
    for row in rows:
        if cache_costs.get(row["name"]) != row["delta_bytes"]:
            raise MaterializationError(
                f"candidate catalog/cache mismatch for {row['name']}"
            )
    return canonical_sha256(rows), cache_costs


def validate_context(
    path: str | Path, cache: GPTQCandidateCache
) -> tuple[dict[str, Any], str, str]:
    if cache.root is None or not cache.verified_payloads:
        raise MaterializationError("materialization requires a verified persisted candidate bank")
    context = _load_object(path, "protocol context")
    if context.get("schema_version") != SCHEMA_VERSION or context.get("status") != "launch_frozen":
        raise MaterializationError("protocol context is not launch-frozen")
    stored = context.get("protocol_sha256")
    unhashed = dict(context)
    unhashed.pop("protocol_sha256", None)
    if not isinstance(stored, str) or artifact_sha256(unhashed) != stored:
        raise MaterializationError("protocol context content hash mismatch")
    model = context.get("model")
    if not isinstance(model, dict) or model.get("id") != BASE_MODEL or model.get("revision") != BASE_REVISION:
        raise MaterializationError("protocol context does not pin the expected model revision")
    bank_path = cache.root / "manifest.json"
    bank_file_hash = sha256_file(bank_path)
    bound = context.get("bound_hashes")
    if not isinstance(bound, dict) or bound.get("candidate_bank_manifest_sha256") != bank_file_hash:
        raise MaterializationError("candidate bank is not bound by the launch protocol")
    implementations = context.get("implementation_files")
    if not isinstance(implementations, dict):
        raise MaterializationError("protocol context has no implementation hash map")
    code_dir = Path(__file__).resolve().parent
    for file_name in IMPLEMENTATION_FILES:
        if implementations.get(file_name) != sha256_file(code_dir / file_name):
            raise MaterializationError(f"implementation hash mismatch for {file_name}")
    return context, sha256_file(path), bank_file_hash


def _normalize_state(
    state: Mapping[str, Any], cache_costs: Mapping[str, int]
) -> dict[str, Any]:
    members = state.get("members")
    if (
        not isinstance(members, list)
        or not all(isinstance(name, str) and name for name in members)
        or members != sorted(members)
        or len(members) != len(set(members))
    ):
        raise MaterializationError("selected state members are not sorted unique names")
    unknown = sorted(set(members) - set(cache_costs))
    if unknown:
        raise MaterializationError(f"selected state contains unknown projections: {unknown}")
    state_id = "state-" + canonical_sha256(members)
    if state.get("state_id") != state_id:
        raise MaterializationError("selected state ID does not match its members")
    cost = sum(cache_costs[name] for name in members)
    if state.get("cost_bytes") != cost:
        raise MaterializationError("selected state byte cost differs from candidate bank")
    return {"state_id": state_id, "members": list(members), "cost_bytes": cost}


def resolve_state(
    artifact: Mapping[str, Any],
    *,
    label: str | None,
    context_sha256: str,
    catalog_hash: str,
    cache_costs: Mapping[str, int],
) -> tuple[dict[str, Any], str]:
    """Resolve and verify one selection or one label in a comparison plan."""
    kind = artifact.get("artifact_kind")
    if kind == "gcq_frozen_beam_selection":
        if label not in (None, ""):
            raise MaterializationError("--label is only valid with a comparison plan")
        try:
            state, selected_catalog, _ = validate_selection(
                artifact,
                label="selection",
                context_sha256=context_sha256,
                catalog_hash=catalog_hash,
            )
        except ValueError as error:
            raise MaterializationError(str(error)) from error
        if selected_catalog != catalog_hash:
            raise MaterializationError("selection uses a different candidate catalog")
        return _normalize_state(state, cache_costs), "gcq_selection"

    if kind != "gcq_frozen_comparison_plan" or artifact.get("schema_version") != SCHEMA_VERSION:
        raise MaterializationError("state artifact is neither a frozen selection nor comparison plan")
    if not label:
        raise MaterializationError("a comparison plan requires --label")
    if artifact.get("context_sha256") != context_sha256:
        raise MaterializationError("comparison plan is bound to another launch protocol")
    if artifact.get("catalog_hash") != catalog_hash:
        raise MaterializationError("comparison plan uses a different candidate catalog")
    label_map = artifact.get("label_to_state_id")
    states = artifact.get("states")
    if not isinstance(label_map, dict) or label not in label_map or not isinstance(states, list):
        raise MaterializationError(f"unknown comparison label: {label}")
    fingerprint_payload = {
        "run_fingerprint": artifact.get("run_fingerprint"),
        "catalog_hash": artifact.get("catalog_hash"),
        "context_sha256": artifact.get("context_sha256"),
        "label_to_state_id": dict(sorted(label_map.items())),
        "states": states,
    }
    if artifact.get("comparison_fingerprint") != canonical_sha256(fingerprint_payload):
        raise MaterializationError("comparison plan fingerprint mismatch")
    state_id = label_map[label]
    matches = [row for row in states if isinstance(row, dict) and row.get("state_id") == state_id]
    if len(matches) != 1:
        raise MaterializationError("comparison label does not resolve to exactly one state")
    labels = matches[0].get("labels")
    if not isinstance(labels, list) or label not in labels:
        raise MaterializationError("comparison state labels disagree with label mapping")
    return _normalize_state(matches[0], cache_costs), label


def build_metadata(
    *,
    state: Mapping[str, Any],
    label: str,
    cache: GPTQCandidateCache,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    composition = cache.composition_manifest(state["members"])
    if composition["added_code_bytes_over_uniform_w4"] != state["cost_bytes"]:
        raise MaterializationError("composition accounting differs from selected state cost")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "gcq_dense_qdq_evaluation_checkpoint",
        "label": label,
        "state": dict(state),
        "composition": composition,
        "source_sha256": dict(sorted(source_sha256.items())),
        "dense_model": {
            "directory": "dense_qdq_model",
            "dtype": "bfloat16",
            "purpose": "compatibility with existing evaluation harnesses",
            "is_compressed_checkpoint": False,
        },
        "packed_decoder": {
            "directory": "packed_decoder",
            "contains_actual_selected_payload": True,
            "packed_inference_kernel_included": False,
        },
    }


def materialize(
    *,
    cache: GPTQCandidateCache,
    state: Mapping[str, Any],
    label: str,
    output: str | Path,
    device: str,
    source_sha256: Mapping[str, str],
) -> Path:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite materialization: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        import torch
        import gcq_patches
        from transformers import AutoModelForImageTextToText, AutoProcessor

        gcq_patches.apply_fast_patch_embed()
        model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL,
            revision=BASE_REVISION,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        processor = AutoProcessor.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        cache.validate_model(model)
        cache.compose(model, state["members"], verify_installs=True)
        dense_dir = temporary / "dense_qdq_model"
        model.save_pretrained(
            dense_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        processor.save_pretrained(dense_dir)
        cache.export_packed_composition(temporary / "packed_decoder", state["members"])
        metadata = build_metadata(
            state=state,
            label=label,
            cache=cache,
            source_sha256=source_sha256,
        )
        metadata["metadata_content_sha256"] = artifact_sha256(metadata)
        (temporary / "materialization.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-artifact", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--candidate-catalog", type=Path, required=True)
    parser.add_argument("--protocol-context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    cache = GPTQCandidateCache.load(args.candidate_cache, verify_hashes=True)
    _, context_hash, bank_hash = validate_context(args.protocol_context, cache)
    catalog_hash, cache_costs = candidate_catalog_hash(args.candidate_catalog, cache)
    artifact = _load_object(args.state_artifact, "state artifact")
    state, label = resolve_state(
        artifact,
        label=args.label or None,
        context_sha256=context_hash,
        catalog_hash=catalog_hash,
        cache_costs=cache_costs,
    )
    sources = {
        "candidate_bank_manifest": bank_hash,
        "candidate_catalog": sha256_file(args.candidate_catalog),
        "protocol_context": context_hash,
        "state_artifact": sha256_file(args.state_artifact),
    }
    path = materialize(
        cache=cache,
        state=state,
        label=label,
        output=args.out,
        device=args.device,
        source_sha256=sources,
    )
    print(json.dumps({"out": str(path), "label": label, "state": state}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

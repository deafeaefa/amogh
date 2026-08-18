"""Freeze equal-length, padding-free standard-text GPTQ calibration tokens.

The output is a self-contained JSON manifest containing exactly the input IDs
used by the candidate-bank builder plus immutable model/dataset provenance and
hashes. Dataset and tokenizer imports are lazy so the deterministic selection
and validation layer remains unit-testable without network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from recovery_utils import BASE_MODEL, BASE_REVISION


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rank(seed: int, row_id: str, text_sha256: str) -> str:
    return hashlib.sha256(f"gcq-gptq-calibration-v1\0{seed}\0{row_id}\0{text_sha256}".encode()).hexdigest()


def build_calibration_manifest(
    rows: Iterable[Mapping[str, object]],
    encode: Callable[[str], Sequence[int]],
    *,
    eos_token_id: int,
    samples: int = 128,
    sequence_length: int = 512,
    seed: int = 20260817,
    base_model: str = BASE_MODEL,
    base_revision: str = BASE_REVISION,
    dataset_id: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    dataset_revision: str,
    dataset_fingerprint: str | None = None,
    tokenizer_class: str | None = None,
) -> dict:
    if samples <= 0 or sequence_length <= 0:
        raise ValueError("samples and sequence_length must be positive")
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError("eos_token_id must be a nonnegative integer")
    if not dataset_revision or dataset_revision in {"main", "master"}:
        raise ValueError("dataset_revision must be an immutable revision, not main/master")

    normalized = []
    seen_ids = set()
    for index, row in enumerate(rows):
        row_id = str(row.get("row_id", ""))
        text = str(row.get("text", "")).strip()
        if not row_id or not text:
            continue
        if row_id in seen_ids:
            raise ValueError(f"duplicate calibration row_id {row_id!r}")
        seen_ids.add(row_id)
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        normalized.append((_rank(seed, row_id, text_hash), row_id, text, text_hash))
    normalized.sort()
    if not normalized:
        raise ValueError("no non-empty calibration text rows")

    needed = samples * sequence_length
    stream: list[int] = []
    sources = []
    for _, row_id, text, text_hash in normalized:
        raw_ids = list(encode(text))
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in raw_ids):
            raise ValueError(f"tokenizer returned invalid IDs for row {row_id!r}")
        if not raw_ids:
            continue
        before = len(stream)
        stream.extend(raw_ids)
        stream.append(eos_token_id)
        used = min(len(stream), needed) - before
        sources.append({
            "row_id": row_id,
            "text_sha256": text_hash,
            "encoded_tokens_including_separator": len(raw_ids) + 1,
            "tokens_used_before_cutoff": max(0, used),
        })
        if len(stream) >= needed:
            break
    if len(stream) < needed:
        raise ValueError(f"only encoded {len(stream)} tokens; need {needed}")
    stream = stream[:needed]
    input_ids = [
        stream[offset:offset + sequence_length]
        for offset in range(0, needed, sequence_length)
    ]
    if len(input_ids) != samples or any(len(sample) != sequence_length for sample in input_ids):
        raise AssertionError("calibration chunking invariant failed")
    manifest = {
        "schema_version": 1,
        "role": "standard_text_gptq_calibration",
        "base_model": base_model,
        "base_revision": base_revision,
        "dataset": {
            "id": dataset_id,
            "config": dataset_config,
            "split": "train",
            "revision": dataset_revision,
            "fingerprint": dataset_fingerprint,
        },
        "selection": {
            "namespace": "gcq-gptq-calibration-v1",
            "seed": seed,
            "rule": "SHA-256 rank of immutable row ID and text hash; concatenate with EOS; cut exact prefix",
            "source_rows": sources,
        },
        "tokenizer": {
            "model": base_model,
            "revision": base_revision,
            "class": tokenizer_class,
            "add_special_tokens": False,
            "eos_token_id": eos_token_id,
        },
        "samples": samples,
        "sequence_length": sequence_length,
        "padding": False,
        "attention_mask": "all ones",
        "input_ids": input_ids,
        "input_ids_sha256": canonical_sha256(input_ids),
    }
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_calibration_manifest(value: Mapping[str, object]) -> None:
    samples = int(value["samples"])
    sequence_length = int(value["sequence_length"])
    input_ids = value["input_ids"]
    if value.get("padding") is not False or value.get("attention_mask") != "all ones":
        raise ValueError("calibration must be padding-free with an all-ones attention mask")
    if not isinstance(input_ids, list) or len(input_ids) != samples:
        raise ValueError("calibration sample count mismatch")
    if any(not isinstance(row, list) or len(row) != sequence_length for row in input_ids):
        raise ValueError("calibration sequence length mismatch")
    if canonical_sha256(input_ids) != value.get("input_ids_sha256"):
        raise ValueError("calibration input-ID hash mismatch")
    without_hash = dict(value)
    content_hash = without_hash.pop("manifest_content_sha256", None)
    if canonical_sha256(without_hash) != content_hash:
        raise ValueError("calibration manifest content hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-revision", required=True, help="immutable Hugging Face dataset commit")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
        revision=args.dataset_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    rows = ({"row_id": str(index), "text": row["text"]} for index, row in enumerate(dataset))
    manifest = build_calibration_manifest(
        rows,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        eos_token_id=int(tokenizer.eos_token_id),
        samples=args.samples,
        sequence_length=args.sequence_length,
        seed=args.seed,
        dataset_revision=args.dataset_revision,
        dataset_fingerprint=getattr(dataset, "_fingerprint", None),
        tokenizer_class=type(tokenizer).__name__,
    )
    validate_calibration_manifest(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "out": str(args.out),
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "input_ids_sha256": manifest["input_ids_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()

"""Shared data, loss, manifest, and adapter-loading helpers for GCQ recovery."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
BASE_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
ADAPTER_MANIFEST = "gcq_recovery_manifest.json"
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
ASSISTANT_PREFIX = "<|im_start|>assistant\n"
COORD_ARRAY_RE = re.compile(r'"bbox_2d"\s*:\s*\[([^\]]+)\]')
NUMBER_RE = re.compile(r"-?\d+")


def precise_iou_score(iou: float) -> float:
    return sum(float(iou >= threshold) for threshold in IOU_THRESHOLDS) / len(IOU_THRESHOLDS)


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_promotions(path: str) -> tuple[dict[str, int] | None, dict | None]:
    if not path:
        return None, None
    with open(path) as f:
        spec = json.load(f)
    promote = {name: int(spec["bits"]) for name in spec["substrings"]}
    return promote, spec


def coordinate_number_spans(answer: str) -> list[tuple[int, int]]:
    match = COORD_ARRAY_RE.search(answer)
    if not match:
        return []
    spans = []
    offset = match.start(1)
    for number in NUMBER_RE.finditer(match.group(1)):
        spans.append((offset + number.start(), offset + number.end()))
    if len(spans) != 4:
        raise ValueError(f"expected four bbox coordinates, found {len(spans)} in {answer!r}")
    return spans


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and a[1] > b[0]


def _answer_token_layout(tokenizer, ids: list[int], answer: str, max_tail_tokens: int):
    """Return context-tokenized tail pieces and answer-relative character spans."""
    k = min(max_tail_tokens, len(ids))
    pieces = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in ids[-k:]]
    tail = "".join(pieces)
    answer_start = tail.rfind(answer)
    if answer_start < 0:
        raise ValueError(f"assistant answer not found in decoded token tail: {answer!r} vs {tail[-300:]!r}")
    answer_span = (answer_start, answer_start + len(answer))
    number_spans = [(answer_start + a, answer_start + b) for a, b in coordinate_number_spans(answer)]

    token_spans = []
    offset = 0
    for local_index, piece in enumerate(pieces):
        token_spans.append((len(ids) - k + local_index, (offset, offset + len(piece))))
        offset += len(piece)
    return answer_span, number_spans, token_spans


def find_answer_coordinate_token_groups(
    tokenizer, ids: list[int], answer: str, max_tail_tokens: int = 128
) -> list[list[int]]:
    """Locate one context-token-position group for each numeric coordinate.

    The returned outer list always has four entries. A coordinate may occupy
    multiple BPE tokens; callers can therefore average within each inner list
    before averaging across coordinates.
    """
    _, number_spans, token_spans = _answer_token_layout(
        tokenizer, ids, answer, max_tail_tokens
    )
    groups = [
        [index for index, token_span in token_spans if spans_overlap(token_span, number_span)]
        for number_span in number_spans
    ]
    if len(groups) != 4 or any(not group for group in groups):
        raise ValueError(f"failed to locate all four coordinate token groups: {groups}")
    return groups


def find_answer_token_positions(tokenizer, ids: list[int], answer: str, max_tail_tokens: int = 128):
    """Locate assistant-answer tokens and numeric coordinate token positions.

    We decode the actual context-tokenized tail one token at a time.  This is
    intentionally not based on separately tokenizing ``answer`` because BPE
    boundaries can differ in context.
    """
    answer_span, number_spans, token_spans = _answer_token_layout(
        tokenizer, ids, answer, max_tail_tokens
    )

    answer_positions = []
    coord_positions = []
    for absolute_index, token_span in token_spans:
        if spans_overlap(token_span, answer_span):
            answer_positions.append(absolute_index)
        if any(spans_overlap(token_span, number_span) for number_span in number_spans):
            coord_positions.append(absolute_index)

    if not answer_positions:
        raise ValueError("no assistant-answer tokens found")
    if number_spans:
        groups = [
            [index for index, token_span in token_spans if spans_overlap(token_span, number_span)]
            for number_span in number_spans
        ]
        if any(not group for group in groups):
            raise ValueError(f"coordinate token grouping failed: {groups}")
        if len(coord_positions) < 4:
            raise ValueError(f"too few coordinate tokens: {coord_positions}")
    return answer_positions, coord_positions


class RecoveryCollator:
    """Build a right-padded Qwen3-VL teacher-forced batch and loss masks."""

    def __init__(self, processor, image_dir: str | os.PathLike[str], max_tail_tokens: int = 128):
        self.processor = processor
        self.image_dir = Path(image_dir)
        self.max_tail_tokens = max_tail_tokens
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, records: list[dict]) -> dict[str, torch.Tensor]:
        messages = []
        answers = []
        for record in records:
            path = self.image_dir / record["file_name"]
            with Image.open(path) as source:
                image = source.convert("RGB")
            messages.append([
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": record["prompt"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": record["answer"]},
                ]},
            ])
            answers.append(record["answer"])

        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        labels = torch.full_like(batch["input_ids"], -100)
        coordinate_mask = torch.zeros_like(batch["input_ids"], dtype=torch.float32)
        for row, answer in enumerate(answers):
            n_real = int(batch["attention_mask"][row].sum().item())
            ids = batch["input_ids"][row, :n_real].tolist()
            answer_positions, coord_positions = find_answer_token_positions(
                self.processor.tokenizer, ids, answer, self.max_tail_tokens
            )
            labels[row, answer_positions] = batch["input_ids"][row, answer_positions]
            coordinate_mask[row, coord_positions] = 1.0
            if COORD_ARRAY_RE.search(answer) and len(coord_positions) < 4:
                raise ValueError(f"coordinate mask failed for row {row}")

        batch["labels"] = labels
        batch["coordinate_mask"] = coordinate_mask
        return batch


def coordinate_weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coordinate_mask: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Per-example masked CE, with numeric coordinate tokens weighted by gamma."""
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    shift_coordinates = coordinate_mask[:, 1:].to(shift_logits.device)
    valid = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_ce = F.cross_entropy(
        shift_logits.transpose(1, 2), safe_labels, reduction="none"
    )
    weights = 1.0 + (float(gamma) - 1.0) * shift_coordinates
    weighted_valid = weights * valid
    denominator = weighted_valid.sum(dim=1)
    if torch.any(denominator == 0):
        raise ValueError("batch contains an example without assistant target tokens")
    return ((token_ce * weighted_valid).sum(dim=1) / denominator).mean()


def write_adapter_manifest(output_dir: str | os.PathLike[str], manifest: dict) -> Path:
    path = Path(output_dir) / ADAPTER_MANIFEST
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def read_adapter_manifest(adapter_dir: str | os.PathLike[str]) -> dict:
    path = Path(adapter_dir) / ADAPTER_MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"adapter is missing {ADAPTER_MANIFEST}: {adapter_dir}")
    with open(path) as f:
        return json.load(f)


def validate_adapter_quantization(
    manifest: dict,
    model: str,
    rtn_bits: int,
    rtn_group: int,
    promote_file: str,
    max_pixels: int,
    base_revision: str | None = None,
) -> None:
    expected = manifest["quantization"]
    actual_hash = sha256_file(promote_file) if promote_file else None
    mismatches = []
    checks = {
        "base_model": (manifest["base_model"], model),
        "rtn_bits": (int(expected["bits"]), int(rtn_bits)),
        "rtn_group": (int(expected["group_size"]), int(rtn_group)),
        "promote_sha256": (expected.get("promote_sha256"), actual_hash),
        "max_pixels": (int(manifest["processor"]["max_pixels"]), int(max_pixels)),
    }
    if base_revision is not None:
        checks["base_revision"] = (manifest["base_revision"], base_revision)
    for key, (wanted, got) in checks.items():
        if wanted != got:
            mismatches.append(f"{key}: adapter expects {wanted!r}, evaluator has {got!r}")
    if mismatches:
        raise ValueError("adapter/base mismatch:\n  " + "\n  ".join(mismatches))


def attach_adapter(model, adapter_dir: str):
    """Attach, without merging, an adapter to an already-quantized model."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)

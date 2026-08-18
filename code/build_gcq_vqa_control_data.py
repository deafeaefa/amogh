#!/usr/bin/env python3
"""Build the frozen, training-only 512-image VQAv2 control manifest.

Official VQAv2 train questions and annotations are joined by question ID.  One
question per eligible image is selected by SHA-256 rank, then 512 unique images
are selected by a second SHA-256 rank.  Exclusion manifests are part of the
output provenance and remove all earlier recovery/profile/confirmation images.
The result is canonical, order-invariant, and write-once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import uuid

from build_vqa_replay_data import (
    ANNOTATIONS_ARCHIVE,
    QUESTIONS_ARCHIVE,
    load_verified_zip_json,
    sha256_file,
)


SCHEMA_VERSION = 1
ROWS = 512
SEED = 20260817
NAMESPACE = "gcq-vqa-control-v1"
PROMPT_SUFFIX = " Answer with a single word or phrase."


class VQAControlDataError(ValueError):
    """Raised when the control manifest cannot be derived unambiguously."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _rank(seed: int, kind: str, image_id: int, question_id: int) -> str:
    value = f"{NAMESPACE}\0{seed}\0{kind}\0{image_id}\0{question_id}".encode()
    return hashlib.sha256(value).hexdigest()


def manifest_rows(value: object, label: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("records", value.get("examples"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise VQAControlDataError(f"{label} has no JSON record list")
    return list(value)


def exclusion_image_ids(values: Iterable[object]) -> set[int]:
    excluded: set[int] = set()
    for index, value in enumerate(values):
        for row in manifest_rows(value, f"exclusion {index}"):
            file_name = row.get("file_name")
            split = row.get("split")
            source = row.get("source")
            is_train = (
                isinstance(file_name, str) and "train2014" in file_name
            ) or split == "train" or source == "vqav2_train"
            if not is_train:
                continue
            image_id = row.get("image_id")
            if type(image_id) is int:
                excluded.add(image_id)
    return excluded


def build_manifest(
    questions_payload: Mapping[str, Any],
    annotations_payload: Mapping[str, Any],
    *,
    exclusions: set[int],
    seed: int = SEED,
    rows: int = ROWS,
    source_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if type(seed) is not int or seed < 0 or type(rows) is not int or rows <= 0:
        raise VQAControlDataError("seed/rows must be nonnegative/positive integers")
    questions = questions_payload.get("questions")
    annotations = annotations_payload.get("annotations")
    if not isinstance(questions, list) or not isinstance(annotations, list):
        raise VQAControlDataError("official payloads lack question/annotation lists")
    question_by_id: dict[int, dict[str, Any]] = {}
    for value in questions:
        if not isinstance(value, dict):
            raise VQAControlDataError("question row is not an object")
        question_id = value.get("question_id")
        image_id = value.get("image_id")
        question = value.get("question")
        if type(question_id) is not int or type(image_id) is not int:
            raise VQAControlDataError("question IDs must be integers")
        if not isinstance(question, str) or not question.strip():
            raise VQAControlDataError(f"question {question_id} has empty text")
        if question_id in question_by_id:
            raise VQAControlDataError(f"duplicate question_id {question_id}")
        question_by_id[question_id] = value

    candidates_by_image: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    annotation_ids: set[int] = set()
    for value in annotations:
        if not isinstance(value, dict):
            raise VQAControlDataError("annotation row is not an object")
        question_id = value.get("question_id")
        image_id = value.get("image_id")
        answer = value.get("multiple_choice_answer")
        if type(question_id) is not int or type(image_id) is not int:
            raise VQAControlDataError("annotation IDs must be integers")
        if question_id in annotation_ids:
            raise VQAControlDataError(f"duplicate annotation question_id {question_id}")
        annotation_ids.add(question_id)
        question = question_by_id.get(question_id)
        if question is None or question.get("image_id") != image_id:
            raise VQAControlDataError(f"question/annotation join mismatch for {question_id}")
        if not isinstance(answer, str) or not answer.strip():
            raise VQAControlDataError(f"annotation {question_id} has empty answer")
        if image_id not in exclusions:
            candidates_by_image.setdefault(image_id, []).append((question, value))
    if set(question_by_id) != annotation_ids:
        raise VQAControlDataError("official question and annotation ID sets differ")
    if len(candidates_by_image) < rows:
        raise VQAControlDataError(
            f"only {len(candidates_by_image)} eligible unique images remain; need {rows}"
        )

    selected_per_image = []
    for image_id, candidates in candidates_by_image.items():
        question, annotation = min(
            candidates,
            key=lambda pair: (
                _rank(seed, "question", image_id, int(pair[0]["question_id"])),
                int(pair[0]["question_id"]),
            ),
        )
        selected_per_image.append((image_id, question, annotation))
    selected = sorted(
        selected_per_image,
        key=lambda triple: (
            _rank(seed, "image", triple[0], int(triple[1]["question_id"])),
            triple[0],
            int(triple[1]["question_id"]),
        ),
    )[:rows]

    records = []
    for index, (image_id, question, annotation) in enumerate(selected):
        question_id = int(question["question_id"])
        question_text = str(question["question"]).strip()
        answer = str(annotation["multiple_choice_answer"]).strip()
        records.append(
            {
                "uid": f"gcq_vqa_control_train_512:{index:05d}",
                "source": "vqav2_train",
                "split": "train",
                "task": "vqa",
                "image_id": image_id,
                "file_name": f"COCO_train2014_{image_id:012d}.jpg",
                "question_id": question_id,
                "question": question_text,
                "question_type": annotation.get("question_type"),
                "answer_type": annotation.get("answer_type"),
                "prompt": question_text + PROMPT_SUFFIX,
                "answer": answer,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "gcq_vqa_control_training_manifest",
        "selection_namespace": NAMESPACE,
        "selection_seed": seed,
        "selection_rule": "one hash-ranked question per image, then hash-ranked unique images",
        "rows": rows,
        "unique_images": rows,
        "excluded_training_image_count": len(exclusions),
        "records": records,
    }
    if source_sha256 is not None:
        manifest["source_sha256"] = dict(sorted(source_sha256.items()))
    manifest["records_sha256"] = canonical_sha256(records)
    return manifest


def validate_manifest(value: object, *, expected_rows: int = ROWS) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise VQAControlDataError("VQA control manifest schema is invalid")
    if value.get("artifact_kind") != "gcq_vqa_control_training_manifest":
        raise VQAControlDataError("VQA control manifest artifact kind is invalid")
    records = manifest_rows(value, "VQA control manifest")
    if len(records) != expected_rows or value.get("rows") != expected_rows:
        raise VQAControlDataError(f"VQA control manifest must contain {expected_rows} rows")
    images = []
    uids = []
    for index, row in enumerate(records):
        if row.get("source") != "vqav2_train" or row.get("split") != "train" or row.get("task") != "vqa":
            raise VQAControlDataError(f"VQA control row {index} is not training-only VQAv2")
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("answer"), str):
            raise VQAControlDataError(f"VQA control row {index} lacks prompt/answer")
        if type(row.get("image_id")) is not int or type(row.get("question_id")) is not int:
            raise VQAControlDataError(f"VQA control row {index} has invalid IDs")
        uids.append(row.get("uid"))
        images.append(row["image_id"])
    if len(set(uids)) != expected_rows or len(set(images)) != expected_rows:
        raise VQAControlDataError("VQA control UIDs/images must be unique")
    if value.get("unique_images") != expected_rows or value.get("records_sha256") != canonical_sha256(records):
        raise VQAControlDataError("VQA control count/content hash mismatch")
    return records


def write_exclusive(path: str | Path, value: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise VQAControlDataError(f"refusing to overwrite {destination}") from error
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-zip", type=Path, required=True)
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    questions = load_verified_zip_json(args.questions_zip, QUESTIONS_ARCHIVE)
    annotations = load_verified_zip_json(args.annotations_zip, ANNOTATIONS_ARCHIVE)
    exclusion_values = []
    source_hashes = {
        "questions_zip": sha256_file(args.questions_zip),
        "annotations_zip": sha256_file(args.annotations_zip),
    }
    for index, path in enumerate(args.exclude):
        with path.open(encoding="utf-8") as handle:
            exclusion_values.append(json.load(handle))
        source_hashes[f"exclusion_{index:02d}_{path.name}"] = sha256_file(path)
    manifest = build_manifest(
        questions,
        annotations,
        exclusions=exclusion_image_ids(exclusion_values),
        seed=args.seed,
        source_sha256=source_hashes,
    )
    records = validate_manifest(manifest)
    missing = [row["file_name"] for row in records if not (args.image_root / row["file_name"]).is_file()]
    if missing:
        raise VQAControlDataError(
            f"{len(missing)} selected images are missing under {args.image_root}; first={missing[:3]}"
        )
    digest = write_exclusive(args.out, manifest)
    print(json.dumps({"out": str(args.out), "sha256": digest, "rows": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

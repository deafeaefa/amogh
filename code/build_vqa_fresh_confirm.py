#!/usr/bin/env python3
"""Build the frozen, unexposed 5k VQAv2 confirmation set.

The input is the pair of official VQAv2 validation ZIP archives already stored
under ``$GCQ_DATA/vqa``.  Selection is deterministic and image-disjoint from
both the exposed ``vqa_val_5k`` development set and all three POPE variants.
The resulting JSON is directly consumable by ``eval_vqa.py``.

This builder deliberately refuses to overwrite either output.  The expected
hashes below were computed in an independent protocol audit before the frozen
manifest was generated.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable
import zipfile


SELECTION_SALT = b"gcq-vqa-fresh-confirm-v1\0"
EXPECTED_QUESTION_COUNT = 214_354
EXPECTED_ANNOTATION_COUNT = 214_354
EXPECTED_EXPOSED_VQA_COUNT = 5_000
EXPECTED_POPE_ROW_COUNT = 9_000
EXPECTED_POPE_IMAGE_COUNT = 500
EXPECTED_FORBIDDEN_IMAGE_COUNT = 4_994
EXPECTED_ELIGIBLE_QUESTION_COUNT = 172_900
EXPECTED_SELECTED_COUNT = 5_000
EXPECTED_SELECTED_IMAGE_COUNT = 4_571

EXPECTED_QID_SHA256 = (
    "238e349350af36cd22a3c251d7c71ceda0152d20399ebf361c4373823ce2e383"
)
EXPECTED_IMAGE_SHA256 = (
    "1c2b9c1b25e358d3c3741cac975d512971d9a5dd2d6c449e5cc8d0ef07de7618"
)
EXPECTED_MANIFEST_SHA256 = (
    "416aea5f9dd6f4a4cfa7061c3b5ca0af88647e49d5f17ed00a8d83885ea21038"
)

QUESTION_ZIP_NAME = "v2_Questions_Val_mscoco.zip"
QUESTION_MEMBER = "v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATION_ZIP_NAME = "v2_Annotations_Val_mscoco.zip"
ANNOTATION_MEMBER = "v2_mscoco_val2014_annotations.json"
EXPECTED_QUESTION_ZIP_SHA256 = (
    "e71f6c5c3e97a51d050f28243e262b28cd0c48d11a6b4632d769d30d3f93222a"
)
EXPECTED_ANNOTATION_ZIP_SHA256 = (
    "0caae7c8d1dafd852727f5ac046bc1efca9b72026bd6ffa34fc489f3a7b3291e"
)
OUTPUT_NAME = "vqa_fresh_confirm_5k.json"
METADATA_NAME = "vqa_fresh_confirm_5k.meta.json"
POPE_VARIANTS = ("random", "popular", "adversarial")
COCO_VAL_RE = re.compile(r"^COCO_val2014_(\d{12})\.jpg$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def newline_int_bytes(values: Iterable[int]) -> bytes:
    return "".join(f"{value}\n" for value in values).encode("utf-8")


def require_int(value: Any, label: str) -> int:
    if type(value) is not int:  # Reject booleans as well as non-integers.
        raise TypeError(f"{label} must be an integer, found {type(value).__name__}")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, found {type(value).__name__}")
    return value


def load_zip_json(path: Path, member: str, expected_sha256: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing official VQAv2 archive: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"official VQAv2 archive hash mismatch for {path}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"CRC failure in {path}: {bad_member}")
        if member not in archive.namelist():
            raise ValueError(f"{path} does not contain expected member {member!r}")
        with archive.open(member) as raw:
            return json.load(io.TextIOWrapper(raw, encoding="utf-8"))


def load_exposed_vqa(path: Path) -> tuple[set[int], set[int]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or len(rows) != EXPECTED_EXPOSED_VQA_COUNT:
        raise ValueError(
            f"{path} must contain exactly {EXPECTED_EXPOSED_VQA_COUNT} rows"
        )
    qids: set[int] = set()
    images: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"{path}[{index}] must be an object")
        qid = require_int(row.get("question_id"), f"{path}[{index}].question_id")
        image_id = require_int(row.get("image_id"), f"{path}[{index}].image_id")
        if qid in qids:
            raise ValueError(f"duplicate exposed VQA question_id {qid}")
        qids.add(qid)
        images.add(image_id)
    return qids, images


def image_id_from_filename(filename: Any, label: str) -> int:
    filename = require_str(filename, label)
    match = COCO_VAL_RE.fullmatch(filename)
    if match is None:
        raise ValueError(f"{label} is not a canonical COCO val2014 filename: {filename!r}")
    return int(match.group(1))


def load_pope_images(data_dir: Path) -> tuple[set[int], dict[str, dict[str, Any]]]:
    images: set[int] = set()
    provenance: dict[str, dict[str, Any]] = {}
    total = 0
    for variant in POPE_VARIANTS:
        path = data_dir / "pope" / f"coco_pope_{variant}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen POPE input: {path}")
        rows = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"blank line in {path}:{line_number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"{path}:{line_number} must contain an object")
                images.add(
                    image_id_from_filename(row.get("image"), f"{path}:{line_number}.image")
                )
                rows += 1
        provenance[variant] = {
            "path": f"pope/{path.name}",
            "rows": rows,
            "sha256": sha256_file(path),
        }
        total += rows
    if total != EXPECTED_POPE_ROW_COUNT:
        raise ValueError(f"expected {EXPECTED_POPE_ROW_COUNT} POPE rows, found {total}")
    if len(images) != EXPECTED_POPE_IMAGE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_POPE_IMAGE_COUNT} unique POPE images, found {len(images)}"
        )
    return images, provenance


def validate_official_inputs(
    questions_payload: Any, annotations_payload: Any
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    if not isinstance(questions_payload, dict) or not isinstance(
        questions_payload.get("questions"), list
    ):
        raise TypeError("official question JSON must contain a questions list")
    if not isinstance(annotations_payload, dict) or not isinstance(
        annotations_payload.get("annotations"), list
    ):
        raise TypeError("official annotation JSON must contain an annotations list")

    questions = questions_payload["questions"]
    annotations = annotations_payload["annotations"]
    if len(questions) != EXPECTED_QUESTION_COUNT:
        raise ValueError(
            f"expected {EXPECTED_QUESTION_COUNT} official questions, found {len(questions)}"
        )
    if len(annotations) != EXPECTED_ANNOTATION_COUNT:
        raise ValueError(
            f"expected {EXPECTED_ANNOTATION_COUNT} annotations, found {len(annotations)}"
        )

    question_ids: set[int] = set()
    for index, row in enumerate(questions):
        if not isinstance(row, dict):
            raise TypeError(f"questions[{index}] must be an object")
        qid = require_int(row.get("question_id"), f"questions[{index}].question_id")
        require_int(row.get("image_id"), f"questions[{index}].image_id")
        require_str(row.get("question"), f"questions[{index}].question")
        if qid in question_ids:
            raise ValueError(f"duplicate official question_id {qid}")
        question_ids.add(qid)

    annotations_by_qid: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(annotations):
        if not isinstance(row, dict):
            raise TypeError(f"annotations[{index}] must be an object")
        qid = require_int(row.get("question_id"), f"annotations[{index}].question_id")
        require_int(row.get("image_id"), f"annotations[{index}].image_id")
        require_str(
            row.get("multiple_choice_answer"),
            f"annotations[{index}].multiple_choice_answer",
        )
        answers = row.get("answers")
        if not isinstance(answers, list) or len(answers) != 10:
            raise ValueError(f"annotations[{index}].answers must contain exactly 10 rows")
        for answer_index, answer in enumerate(answers):
            if not isinstance(answer, dict):
                raise TypeError(f"annotations[{index}].answers[{answer_index}] must be an object")
            require_str(
                answer.get("answer"),
                f"annotations[{index}].answers[{answer_index}].answer",
            )
        if qid in annotations_by_qid:
            raise ValueError(f"duplicate official annotation question_id {qid}")
        annotations_by_qid[qid] = row

    if question_ids != set(annotations_by_qid):
        raise ValueError("official question and annotation question_id sets differ")
    return questions, annotations_by_qid


def build_records(
    questions: list[dict[str, Any]],
    annotations_by_qid: dict[int, dict[str, Any]],
    forbidden_images: set[int],
) -> tuple[list[dict[str, Any]], bytes, bytes]:
    eligible = [row for row in questions if row["image_id"] not in forbidden_images]
    if len(eligible) != EXPECTED_ELIGIBLE_QUESTION_COUNT:
        raise ValueError(
            f"expected {EXPECTED_ELIGIBLE_QUESTION_COUNT} eligible questions, found {len(eligible)}"
        )

    def rank_key(row: dict[str, Any]) -> tuple[bytes, int]:
        qid = row["question_id"]
        rank = hashlib.sha256(SELECTION_SALT + str(qid).encode("ascii")).digest()
        return rank, qid

    selected = sorted(eligible, key=rank_key)[:EXPECTED_SELECTED_COUNT]
    records: list[dict[str, Any]] = []
    for row in selected:
        qid = row["question_id"]
        image_id = row["image_id"]
        annotation = annotations_by_qid[qid]
        if annotation["image_id"] != image_id:
            raise ValueError(f"question/annotation image mismatch for question_id {qid}")
        records.append(
            {
                "uid": f"vqa:{qid}",
                "question_id": qid,
                "image_id": image_id,
                "file_name": f"COCO_val2014_{image_id:012d}.jpg",
                "question": row["question"],
                "answers": [answer["answer"] for answer in annotation["answers"]],
                "multiple_choice_answer": annotation["multiple_choice_answer"],
            }
        )

    qids = [row["question_id"] for row in records]
    image_ids = sorted({row["image_id"] for row in records})
    if len(qids) != EXPECTED_SELECTED_COUNT or len(set(qids)) != len(qids):
        raise ValueError("selected manifest must contain 5,000 unique question IDs")
    if len(image_ids) != EXPECTED_SELECTED_IMAGE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_SELECTED_IMAGE_COUNT} selected images, found {len(image_ids)}"
        )
    if set(image_ids) & forbidden_images:
        raise ValueError("fresh confirmation manifest overlaps an excluded image")

    qid_bytes = newline_int_bytes(qids)
    image_bytes = newline_int_bytes(image_ids)
    if sha256_bytes(qid_bytes) != EXPECTED_QID_SHA256:
        raise ValueError("ordered selected-question hash differs from the protocol audit")
    if sha256_bytes(image_bytes) != EXPECTED_IMAGE_SHA256:
        raise ValueError("sorted selected-image hash differs from the protocol audit")
    return records, qid_bytes, image_bytes


def write_new_atomic(path: Path, data: bytes) -> None:
    """Atomically publish bytes while refusing to replace an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publish is atomic and, unlike os.replace/os.rename, cannot
        # overwrite a path created between the initial check and this operation.
        os.link(temporary_path, path)
        temporary_path.unlink()
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = os.environ.get("GCQ_DATA")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(default_data) if default_data else None,
        help="GCQ data root (default: $GCQ_DATA)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: DATA_DIR/subsets)",
    )
    args = parser.parse_args()
    if args.data_dir is None:
        parser.error("--data-dir is required when GCQ_DATA is unset")
    return args


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "subsets").resolve()
    manifest_path = output_dir / OUTPUT_NAME
    metadata_path = output_dir / METADATA_NAME
    existing = [str(path) for path in (manifest_path, metadata_path) if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing output(s): " + ", ".join(existing))

    question_zip = data_dir / "vqa" / QUESTION_ZIP_NAME
    annotation_zip = data_dir / "vqa" / ANNOTATION_ZIP_NAME
    exposed_path = data_dir / "subsets" / "vqa_val_5k.json"
    if not exposed_path.is_file():
        raise FileNotFoundError(f"missing exposed VQA subset: {exposed_path}")

    question_payload = load_zip_json(
        question_zip, QUESTION_MEMBER, EXPECTED_QUESTION_ZIP_SHA256
    )
    annotation_payload = load_zip_json(
        annotation_zip, ANNOTATION_MEMBER, EXPECTED_ANNOTATION_ZIP_SHA256
    )
    questions, annotations_by_qid = validate_official_inputs(
        question_payload, annotation_payload
    )
    exposed_qids, exposed_images = load_exposed_vqa(exposed_path)
    pope_images, pope_provenance = load_pope_images(data_dir)
    forbidden_images = exposed_images | pope_images
    if len(forbidden_images) != EXPECTED_FORBIDDEN_IMAGE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_FORBIDDEN_IMAGE_COUNT} forbidden images, "
            f"found {len(forbidden_images)}"
        )

    records, qid_bytes, image_bytes = build_records(
        questions, annotations_by_qid, forbidden_images
    )
    if exposed_qids & {row["question_id"] for row in records}:
        raise ValueError("fresh confirmation manifest overlaps exposed VQA question IDs")

    manifest_bytes = canonical_json_bytes(records)
    manifest_sha = sha256_bytes(manifest_bytes)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "canonical manifest hash differs from the protocol audit: " + manifest_sha
        )

    metadata = {
        "schema_version": 1,
        "dataset": "VQAv2 val2014 fresh confirmation subset",
        "output": {
            "path": f"subsets/{OUTPUT_NAME}",
            "format": "UTF-8 canonical JSON plus trailing newline",
            "questions": len(records),
            "unique_images": len({row["image_id"] for row in records}),
            "manifest_sha256": manifest_sha,
            "ordered_question_ids_sha256": sha256_bytes(qid_bytes),
            "sorted_unique_image_ids_sha256": sha256_bytes(image_bytes),
        },
        "selection": {
            "eligible_questions": EXPECTED_ELIGIBLE_QUESTION_COUNT,
            "rank": "SHA256(salt || ASCII(question_id)), ascending digest; question_id tie-break",
            "salt_hex": SELECTION_SALT.hex(),
            "selected_questions": EXPECTED_SELECTED_COUNT,
        },
        "exclusions": {
            "rule": "exclude every image in exposed vqa_val_5k and every image in POPE",
            "exposed_vqa": {
                "path": "subsets/vqa_val_5k.json",
                "rows": EXPECTED_EXPOSED_VQA_COUNT,
                "unique_images": len(exposed_images),
                "sha256": sha256_file(exposed_path),
            },
            "pope": pope_provenance,
            "unique_forbidden_images": len(forbidden_images),
        },
        "sources": {
            "questions": {
                "archive": f"vqa/{QUESTION_ZIP_NAME}",
                "member": QUESTION_MEMBER,
                "rows": EXPECTED_QUESTION_COUNT,
                "sha256": sha256_file(question_zip),
            },
            "annotations": {
                "archive": f"vqa/{ANNOTATION_ZIP_NAME}",
                "member": ANNOTATION_MEMBER,
                "rows": EXPECTED_ANNOTATION_COUNT,
                "sha256": sha256_file(annotation_zip),
            },
        },
    }

    write_new_atomic(manifest_path, manifest_bytes)
    try:
        write_new_atomic(metadata_path, canonical_json_bytes(metadata))
    except Exception:
        # Preserve no-overwrite semantics.  A successfully published manifest
        # remains useful and is never silently removed or replaced.
        raise

    print(f"wrote {manifest_path} ({len(records)} questions, {EXPECTED_SELECTED_IMAGE_COUNT} images)")
    print(f"wrote {metadata_path}")
    print(f"ordered question IDs sha256: {EXPECTED_QID_SHA256}")
    print(f"sorted image IDs sha256:     {EXPECTED_IMAGE_SHA256}")
    print(f"canonical manifest sha256:   {EXPECTED_MANIFEST_SHA256}")


if __name__ == "__main__":
    main()

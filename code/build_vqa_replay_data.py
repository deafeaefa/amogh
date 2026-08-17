"""Build the frozen 12k grounding + VQAv2 replay training manifest.

This builder derives a new training set from ``recovery_train_8k.json`` while
leaving that frozen parent untouched.  It retains the parent's 6,000 grounding
rows verbatim, drops all caption replay, and adds exactly one official VQAv2
train question/``multiple_choice_answer`` target for every grounding image.
Because the parent has 1,500 unique grounding images from each source, this
produces 1,500 VQA examples per replay group and 12,000 examples in total.

Question selection is deterministic and independent of archive ordering.  For
each image, candidates are ranked by::

    SHA256("gcq-vqa-replay-v1\\0{seed}\\0{image_id}\\0{question_id}")

with ``question_id`` as an explicit (effectively unreachable) tie-breaker.  The
minimum-ranked question is selected.  Output order follows the frozen parent
grounding order, with each grounding row immediately followed by its VQA row.

The two output files are created exclusively; this script never overwrites an
existing manifest or sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_PARENT_SHA256 = "b89f3d391a9bce0553d5213babc8eeadc55b3dab367d6398088b5ad58fba4f62"
EXPECTED_PARENT_EXAMPLES = 8_000
EXPECTED_GROUNDING_EXAMPLES = 6_000
EXPECTED_REPLAY_PER_GROUP = 1_500
EXPECTED_VQA_EXAMPLES = 6_000
EXPECTED_OUTPUT_EXAMPLES = 12_000
EXPECTED_VQA_TRAIN_QUESTIONS = 443_757
REPLAY_GROUPS = ("refcoco", "refcocoplus", "refcocog", "coco_detection")
VQA_PROMPT_SUFFIX = " Answer with a single word or phrase."
SELECTION_NAMESPACE = "gcq-vqa-replay-v1"

QUESTIONS_ARCHIVE = {
    "file_name": "v2_Questions_Train_mscoco.zip",
    "member": "v2_OpenEnded_mscoco_train2014_questions.json",
    "sha256": "05a64b6e2582d06d7585f5429674a9a33851878be1bff9f8668cdcf792df611e",
    "url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
}
ANNOTATIONS_ARCHIVE = {
    "file_name": "v2_Annotations_Train_mscoco.zip",
    "member": "v2_mscoco_train2014_annotations.json",
    "sha256": "fb101bcefe91422c543c2bb6d70af11eb3119d0ff745ae283d09acdf66250853",
    "url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
}
OUTPUT_NAME = "recovery_train_vqa_replay_12k.json"
METADATA_NAME = "recovery_train_vqa_replay_12k.meta.json"


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def canonical_rows_sha256(rows: list[dict]) -> str:
    encoded = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def load_verified_zip_json(path: Path, specification: dict[str, str]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing official VQAv2 archive: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != specification["sha256"]:
        raise ValueError(
            f"unexpected SHA-256 for {path}: {actual_hash}; "
            f"expected {specification['sha256']}"
        )
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if members != [specification["member"]]:
            raise ValueError(
                f"unexpected members in {path}: {members}; "
                f"expected only {specification['member']!r}"
            )
        with archive.open(specification["member"]) as handle:
            return json.load(io.TextIOWrapper(handle, encoding="utf-8"))


def selection_key(seed: int, image_id: int, question_id: int) -> tuple[str, int]:
    payload = f"{SELECTION_NAMESPACE}\0{seed}\0{image_id}\0{question_id}".encode()
    return hashlib.sha256(payload).hexdigest(), question_id


def select_vqa_rows(
    grounding: list[dict], questions_payload: dict, annotations_payload: dict, seed: int
) -> tuple[list[dict], dict[str, int]]:
    questions = questions_payload.get("questions")
    annotations = annotations_payload.get("annotations")
    if not isinstance(questions, list) or not isinstance(annotations, list):
        raise ValueError("official VQAv2 JSON is missing questions/annotations lists")
    if len(questions) != EXPECTED_VQA_TRAIN_QUESTIONS:
        raise ValueError(f"unexpected VQAv2 train question count: {len(questions)}")
    if len(annotations) != EXPECTED_VQA_TRAIN_QUESTIONS:
        raise ValueError(f"unexpected VQAv2 train annotation count: {len(annotations)}")

    grounding_images = {int(row["image_id"]) for row in grounding}
    annotations_by_question: dict[int, dict] = {}
    annotation_question_ids: set[int] = set()
    for annotation in annotations:
        question_id = int(annotation["question_id"])
        if question_id in annotation_question_ids:
            raise ValueError(f"duplicate VQAv2 annotation question_id: {question_id}")
        annotation_question_ids.add(question_id)
        if int(annotation["image_id"]) in grounding_images:
            annotations_by_question[question_id] = annotation

    candidates_by_image: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    question_ids: set[int] = set()
    for question in questions:
        question_id = int(question["question_id"])
        if question_id in question_ids:
            raise ValueError(f"duplicate VQAv2 question question_id: {question_id}")
        question_ids.add(question_id)
        image_id = int(question["image_id"])
        if image_id not in grounding_images:
            continue
        annotation = annotations_by_question.get(question_id)
        if annotation is None:
            raise ValueError(f"question {question_id} has no joined official annotation")
        if int(annotation["image_id"]) != image_id:
            raise ValueError(f"question/annotation image mismatch for {question_id}")
        candidates_by_image[image_id].append((question, annotation))

    if question_ids != annotation_question_ids:
        missing_annotations = sorted(question_ids - annotation_question_ids)[:3]
        missing_questions = sorted(annotation_question_ids - question_ids)[:3]
        raise ValueError(
            "official VQAv2 question/annotation IDs differ; "
            f"questions without annotations={missing_annotations}, "
            f"annotations without questions={missing_questions}"
        )
    missing_images = sorted(grounding_images - candidates_by_image.keys())
    if missing_images:
        raise ValueError(
            f"{len(missing_images)} grounding images lack a VQAv2 train question; "
            f"first={missing_images[:3]}"
        )

    replay = []
    candidate_count = 0
    for index, base in enumerate(grounding):
        image_id = int(base["image_id"])
        candidates = candidates_by_image[image_id]
        candidate_count += len(candidates)
        question, annotation = min(
            candidates,
            key=lambda pair: selection_key(seed, image_id, int(pair[0]["question_id"])),
        )
        question_id = int(question["question_id"])
        question_text = question["question"]
        answer = annotation["multiple_choice_answer"]
        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError(f"empty/non-string official question for {question_id}")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"empty/non-string official target for {question_id}")
        replay.append({
            "source": "vqav2_train",
            "replay_group": base["source"],
            "task": "vqa",
            "image_id": image_id,
            "file_name": base["file_name"],
            "width": int(base["width"]),
            "height": int(base["height"]),
            "question_id": question_id,
            "question": question_text,
            "question_type": annotation["question_type"],
            "answer_type": annotation["answer_type"],
            "multiple_choice_answer": answer,
            "prompt": question_text + VQA_PROMPT_SUFFIX,
            "answer": answer,
            "uid": f"recovery_vqa_replay:{index:05d}",
        })
    return replay, {
        "joined_candidate_questions": candidate_count,
        "covered_grounding_images": len(candidates_by_image),
    }


def parse_args() -> argparse.Namespace:
    data_dir = Path(os.environ.get("GCQ_DATA", "data"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=data_dir / "subsets")
    parser.add_argument(
        "--parent", type=Path, default=data_dir / "subsets" / "recovery_train_8k.json"
    )
    parser.add_argument(
        "--dev", type=Path, default=data_dir / "subsets" / "recovery_dev_1k.json"
    )
    parser.add_argument(
        "--questions-zip", type=Path, default=data_dir / "vqa" / QUESTIONS_ARCHIVE["file_name"]
    )
    parser.add_argument(
        "--annotations-zip",
        type=Path,
        default=data_dir / "vqa" / ANNOTATIONS_ARCHIVE["file_name"],
    )
    parser.add_argument("--image-dir", type=Path, default=data_dir / "images" / "train2014")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.out_dir / OUTPUT_NAME
    metadata_path = args.out_dir / METADATA_NAME
    existing = [str(path) for path in (output_path, metadata_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output(s): {existing}")

    parent_hash = sha256_file(args.parent)
    if parent_hash != EXPECTED_PARENT_SHA256:
        raise ValueError(
            f"frozen parent SHA-256 mismatch: {parent_hash}; expected {EXPECTED_PARENT_SHA256}"
        )
    with open(args.parent, encoding="utf-8") as handle:
        parent = json.load(handle)
    with open(args.dev, encoding="utf-8") as handle:
        dev = json.load(handle)
    if len(parent) != EXPECTED_PARENT_EXAMPLES:
        raise ValueError(f"unexpected frozen parent row count: {len(parent)}")

    grounding = [row for row in parent if row.get("task") != "caption"]
    caption_rows = [row for row in parent if row.get("task") == "caption"]
    if len(grounding) != EXPECTED_GROUNDING_EXAMPLES or len(caption_rows) != 2_000:
        raise ValueError(
            f"unexpected parent composition: grounding={len(grounding)}, captions={len(caption_rows)}"
        )
    grounding_snapshot_hash = canonical_rows_sha256(grounding)
    grounding_images = {int(row["image_id"]) for row in grounding}
    if len(grounding_images) != EXPECTED_GROUNDING_EXAMPLES:
        raise ValueError("the frozen grounding parent does not contain 6,000 unique images")
    expected_groups = Counter({group: EXPECTED_REPLAY_PER_GROUP for group in REPLAY_GROUPS})
    if Counter(row["source"] for row in grounding) != expected_groups:
        raise ValueError("the frozen grounding parent is not balanced 1,500 per source")

    dev_images = {int(row["image_id"]) for row in dev}
    overlap = grounding_images & dev_images
    if overlap:
        raise ValueError(f"training/dev image leakage: {len(overlap)} images; first={sorted(overlap)[:3]}")
    missing_files = sorted({
        row["file_name"] for row in grounding
        if not (args.image_dir / row["file_name"]).is_file()
    })
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} grounding/replay image files are missing; first={missing_files[:3]}"
        )

    questions_payload = load_verified_zip_json(args.questions_zip, QUESTIONS_ARCHIVE)
    annotations_payload = load_verified_zip_json(args.annotations_zip, ANNOTATIONS_ARCHIVE)
    replay, join_stats = select_vqa_rows(
        grounding, questions_payload, annotations_payload, args.seed
    )

    # Exact target, uniqueness, balance, and leakage invariants.
    if len(replay) != EXPECTED_VQA_EXAMPLES:
        raise ValueError(f"unexpected VQA replay count: {len(replay)}")
    if len({row["image_id"] for row in replay}) != EXPECTED_VQA_EXAMPLES:
        raise ValueError("VQA replay image IDs are not unique")
    if {row["image_id"] for row in replay} != grounding_images:
        raise ValueError("VQA replay images do not exactly match the grounding images")
    if len({row["question_id"] for row in replay}) != EXPECTED_VQA_EXAMPLES:
        raise ValueError("VQA replay question IDs are not unique")
    if Counter(row["replay_group"] for row in replay) != expected_groups:
        raise ValueError("VQA replay is not balanced 1,500 per original source")
    if any(row["prompt"] != row["question"] + VQA_PROMPT_SUFFIX for row in replay):
        raise ValueError("a VQA replay prompt differs from the evaluation prompt format")
    if any(row["answer"] != row["multiple_choice_answer"] for row in replay):
        raise ValueError("a VQA training target differs from its official multiple_choice_answer")
    if {row["image_id"] for row in replay} & dev_images:
        raise ValueError("VQA replay leaks a recovery-dev image")

    output = []
    for base, vqa_row in zip(grounding, replay):
        if int(base["image_id"]) != int(vqa_row["image_id"]):
            raise AssertionError("internal grounding/replay pairing mismatch")
        output.extend((base, vqa_row))

    output_grounding = [row for row in output if row["task"] != "vqa"]
    if output_grounding != grounding:
        raise ValueError("a retained parent grounding row changed")
    if canonical_rows_sha256(output_grounding) != grounding_snapshot_hash:
        raise ValueError("retained parent grounding content hash changed")
    if len(output) != EXPECTED_OUTPUT_EXAMPLES:
        raise ValueError(f"unexpected output row count: {len(output)}")
    if any(row["task"] == "caption" for row in output):
        raise ValueError("caption replay unexpectedly remains in the 12k manifest")
    if Counter(row["task"] for row in output) != Counter({
        "rec": 4_500, "coco_grounding": 1_500, "vqa": 6_000
    }):
        raise ValueError("unexpected final task composition")
    if len({row["uid"] for row in output}) != EXPECTED_OUTPUT_EXAMPLES:
        raise ValueError("output UIDs are not unique")
    if {int(row["image_id"]) for row in output} & dev_images:
        raise ValueError("final manifest leaks a recovery-dev image")
    if any(not isinstance(row.get("answer"), str) or not row["answer"].strip() for row in output):
        raise ValueError("final manifest contains an empty/non-string target")

    output_bytes = json_bytes(output)
    output_hash = sha256_bytes(output_bytes)
    metadata = {
        "schema_version": 1,
        "dataset": "GCQ balanced grounding + VQAv2 train replay",
        "seed": args.seed,
        "selection_algorithm": (
            f"minimum SHA256({SELECTION_NAMESPACE}\\0seed\\0image_id\\0question_id), "
            "question_id tie-break; parent grounding order; paired grounding then VQA"
        ),
        "prompt_template": "{official_question}" + VQA_PROMPT_SUFFIX,
        "parent": {
            "file_name": args.parent.name,
            "sha256": parent_hash,
            "examples": len(parent),
            "retained_grounding_examples": len(grounding),
            "retained_grounding_canonical_sha256": grounding_snapshot_hash,
            "dropped_caption_examples": len(caption_rows),
        },
        "recovery_dev": {
            "file_name": args.dev.name,
            "sha256": sha256_file(args.dev),
            "examples": len(dev),
            "image_overlap": 0,
        },
        "vqav2_train": {
            "questions_archive": dict(QUESTIONS_ARCHIVE),
            "annotations_archive": dict(ANNOTATIONS_ARCHIVE),
            "official_questions": len(questions_payload["questions"]),
            "official_annotations": len(annotations_payload["annotations"]),
            **join_stats,
            "selected_questions": len(replay),
            "unique_selected_question_ids": len({row["question_id"] for row in replay}),
            "unique_selected_images": len({row["image_id"] for row in replay}),
            "question_type_counts": dict(sorted(Counter(
                row["question_type"] for row in replay
            ).items())),
            "answer_type_counts": dict(sorted(Counter(
                row["answer_type"] for row in replay
            ).items())),
        },
        "output": {
            "file_name": output_path.name,
            "sha256": output_hash,
            "examples": len(output),
            "grounding_examples": len(grounding),
            "vqa_examples": len(replay),
            "caption_examples": 0,
            "unique_images": len({row["image_id"] for row in output}),
            "task_counts": dict(sorted(Counter(row["task"] for row in output).items())),
            "grounding_source_counts": dict(sorted(Counter(
                row["source"] for row in grounding
            ).items())),
            "vqa_replay_group_counts": dict(sorted(Counter(
                row["replay_group"] for row in replay
            ).items())),
            "verified_local_image_files": len(grounding_images),
            "parent_grounding_rows_unchanged": True,
            "recovery_dev_image_overlap": 0,
        },
        "provenance": {
            "vqav2": "VQA v2.0 official train2014 open-ended questions and annotations",
            "target": "official annotation multiple_choice_answer, byte-for-byte",
            "images": "MS COCO train2014 files named by the frozen grounding parent",
        },
    }
    metadata_bytes = json_bytes(metadata)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is deliberate: frozen data products are never replaced.
    with open(output_path, "xb") as handle:
        handle.write(output_bytes)
    with open(metadata_path, "xb") as handle:
        handle.write(metadata_bytes)

    print(json.dumps({
        "manifest": str(output_path),
        "manifest_sha256": output_hash,
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_bytes(metadata_bytes),
        "examples": len(output),
        "grounding": len(grounding),
        "vqa": len(replay),
        "vqa_replay_groups": metadata["output"]["vqa_replay_group_counts"],
        "joined_candidate_questions": join_stats["joined_candidate_questions"],
    }, indent=2))


if __name__ == "__main__":
    main()

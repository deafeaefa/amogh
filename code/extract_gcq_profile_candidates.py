#!/usr/bin/env python3
"""Extract normalized, training-only candidates for ``build_gcq_profile_data``.

The pure functions in this module adapt jxu124 RefCOCO-family rows and official
COCO ``instances_train2014`` annotations into the normalized JSON schema
documented by :mod:`build_gcq_profile_data`.  Dataset loading is confined to the
CLI, so unit tests and protocol audits require no network access.

Typical use::

    python extract_gcq_profile_candidates.py \
      --coco-instances "$GCQ_DATA/coco_ann/ann2014.zip" \
      --ref-revision refcoco=<immutable-commit> \
      --ref-revision refcocoplus=<immutable-commit> \
      --ref-revision refcocog=<immutable-commit> \
      --exclude-manifest "$GCQ_DATA/subsets/dprobe_refcoco_train_512.json" \
      --exclude-manifest "$GCQ_DATA/subsets/recovery_train_vqa_replay_12k.json" \
      --exclude-manifest "$GCQ_DATA/subsets/recovery_dev_1k.json" \
      --exclude-manifest "$GCQ_DATA/subsets/refcocoplus_testA_confirm_full.json" \
      --exclude-manifest "$GCQ_DATA/subsets/refcocoplus_testB_confirm_full.json" \
      --out-dir "$GCQ_DATA/profile_candidates"

The CLI lazily loads ``jxu124/refcoco``, ``jxu124/refcocoplus``, and
``jxu124/refcocog``.  It writes canonical, write-once candidate lists, the union
of cross-variant nontrain and supplied-manifest image exclusions, and a metadata
sidecar containing logical output hashes plus source-file/fingerprint hashes.

Important source conventions are checked rather than assumed:

* jxu124's top-level ``bbox`` is XYXY; ``raw_anns['bbox']`` is COCO XYWH.
  When both exist they must agree geometrically.
* one expression is chosen per reference by a namespaced SHA-256 rank over the
  seed, source, reference, sentence ID, and text; input order is irrelevant.
* COCO category ambiguity counts every annotation, including crowd annotations,
  while a selectable target must itself be noncrowd and have total count one.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from build_gcq_profile_data import (
    CATEGORY_SOURCE,
    ProfileDataError,
    REC_SOURCES,
    canonical_json_bytes,
    canonical_sha256,
    normalize_category_candidate,
    normalize_rec_candidate,
    sha256_bytes,
)


SCHEMA_VERSION = 1
EXPRESSION_NAMESPACE = "gcq-profile-expression-v1"
REF_DATASETS = {
    "refcoco": "jxu124/refcoco",
    "refcocoplus": "jxu124/refcocoplus",
    "refcocog": "jxu124/refcocog",
}
NONTRAIN_SPLITS = {
    "refcoco": ("validation", "test", "testB"),
    "refcocoplus": ("validation", "test", "testB"),
    "refcocog": ("validation", "test"),
}
OUTPUT_NAMES = {
    "rec": "gcq_profile_rec_candidates.json",
    "category": "gcq_profile_category_candidates.json",
    "exclusions": "gcq_profile_excluded_train2014_image_ids.json",
    "metadata": "gcq_profile_candidate_sources.meta.json",
}
COCO_INSTANCES_MEMBER = "annotations/instances_train2014.json"


class ExtractionError(ProfileDataError):
    """Raised when a raw source violates the extraction protocol."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ExtractionError(f"{label} must be an integer")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExtractionError(f"{label} must be a non-empty string")
    return value.strip()


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtractionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ExtractionError(f"{label} must be finite")
    return result


def _parse_object(value: Any, label: str, *, required: bool = True) -> dict[str, Any] | None:
    if value is None and not required:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ExtractionError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ExtractionError(f"{label} must be a JSON object")
    return value


def _box(values: Any, label: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ExtractionError(f"{label} must contain four values")
    return [_require_number(value, f"{label}[{index}]") for index, value in enumerate(values)]


def xyxy_to_xywh(values: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = values
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ExtractionError(f"invalid XYXY box: {list(values)!r}")
    return [x1, y1, x2 - x1, y2 - y1]


def xywh_to_xyxy(values: Sequence[float]) -> list[float]:
    x, y, width, height = values
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ExtractionError(f"invalid XYWH box: {list(values)!r}")
    return [x, y, x + width, y + height]


def _boxes_close(first: Sequence[float], second: Sequence[float]) -> bool:
    return all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-4)
        for a, b in zip(first, second)
    )


def expression_selection_key(
    seed: int,
    source: str,
    ref_id: int | str,
    sentence_id: int | str,
    text: str,
) -> tuple[str, str, str]:
    payload = (
        f"{EXPRESSION_NAMESPACE}\0{seed}\0{source}\0{ref_id}\0"
        f"{sentence_id}\0{text}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), str(sentence_id), text


def choose_expression(
    sentences: Any, *, seed: int, source: str, ref_id: int | str
) -> tuple[str, int | str]:
    if not isinstance(sentences, (list, tuple)) or not sentences:
        raise ExtractionError(f"{source}/{ref_id} has no sentences")
    options = []
    for index, sentence in enumerate(sentences):
        if isinstance(sentence, dict):
            text = sentence.get("sent")
            sentence_id = sentence.get("sent_id")
        else:
            text = sentence
            sentence_id = None
        text = _require_string(text, f"{source}/{ref_id} sentence {index}")
        if sentence_id is None:
            sentence_id = f"text:{text}"
        if isinstance(sentence_id, bool) or not isinstance(sentence_id, (int, str)):
            raise ExtractionError(
                f"{source}/{ref_id} sentence {index} has invalid sent_id"
            )
        options.append((text, sentence_id))
    text, sentence_id = min(
        options,
        key=lambda item: expression_selection_key(
            seed, source, ref_id, item[1], item[0]
        ),
    )
    return text, sentence_id


def _ref_image_geometry(row: Mapping[str, Any], label: str) -> tuple[int, int, int, str]:
    image_id = _require_int(row.get("image_id"), f"{label}.image_id")
    info = _parse_object(row.get("raw_image_info"), f"{label}.raw_image_info")
    assert info is not None
    info_image_id = _require_int(info.get("id"), f"{label}.raw_image_info.id")
    if info_image_id != image_id:
        raise ExtractionError(f"{label} row/image-info image IDs disagree")
    width = _require_int(info.get("width"), f"{label}.width")
    height = _require_int(info.get("height"), f"{label}.height")
    if width <= 0 or height <= 0:
        raise ExtractionError(f"{label} has non-positive image dimensions")
    file_name = _require_string(info.get("file_name"), f"{label}.file_name")
    expected = f"COCO_train2014_{image_id:012d}.jpg"
    if file_name != expected:
        raise ExtractionError(f"{label} expected file_name {expected!r}, got {file_name!r}")
    return image_id, width, height, file_name


def _ref_bbox_xywh(row: Mapping[str, Any], label: str) -> tuple[list[float], int | str | None]:
    source_xyxy = _box(row.get("bbox"), f"{label}.bbox_xyxy")
    converted = xyxy_to_xywh(source_xyxy)
    raw_annotation = _parse_object(
        row.get("raw_anns"), f"{label}.raw_anns", required=False
    )
    annotation_id = row.get("ann_id")
    if raw_annotation is None:
        return converted, annotation_id
    raw_xywh = _box(raw_annotation.get("bbox"), f"{label}.raw_anns.bbox_xywh")
    if not _boxes_close(source_xyxy, xywh_to_xyxy(raw_xywh)):
        raise ExtractionError(
            f"{label} top-level XYXY bbox disagrees with raw XYWH annotation"
        )
    if "image_id" in raw_annotation and raw_annotation["image_id"] != row.get("image_id"):
        raise ExtractionError(f"{label} annotation/image IDs disagree")
    raw_annotation_id = raw_annotation.get("id")
    if annotation_id is not None and raw_annotation_id is not None and annotation_id != raw_annotation_id:
        raise ExtractionError(f"{label} top-level/raw annotation IDs disagree")
    return raw_xywh, raw_annotation_id if raw_annotation_id is not None else annotation_id


def extract_ref_candidates(
    rows: Iterable[Mapping[str, Any]], source: str, *, seed: int = 0
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize one jxu124 RefCOCO-family training split."""
    if source not in REC_SOURCES:
        raise ExtractionError(f"unknown RefCOCO source: {source!r}")
    candidates = []
    input_rows = 0
    for row in rows:
        input_rows += 1
        # Hugging Face split objects do not normally repeat the split name in
        # every row.  An explicit conflicting value is still rejected.
        if row.get("split") not in (None, "train"):
            raise ExtractionError(f"{source} extractor received a nontrain row")
        ref_id = row.get("ref_id")
        if isinstance(ref_id, bool) or not isinstance(ref_id, (int, str)):
            raise ExtractionError(f"{source} ref_id must be an integer or string")
        label = f"{source}/ref:{ref_id}"
        image_id, width, height, file_name = _ref_image_geometry(row, label)
        bbox_xywh, annotation_id = _ref_bbox_xywh(row, label)
        expression, sentence_id = choose_expression(
            row.get("sentences"), seed=seed, source=source, ref_id=ref_id
        )
        candidate = {
            "candidate_id": f"rec:{source}:ref:{ref_id}",
            "task": "rec",
            "source": source,
            "split": "train",
            "ref_id": ref_id,
            "annotation_id": annotation_id,
            "sentence_id": sentence_id,
            "image_id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "bbox_xywh": bbox_xywh,
            "relative_area": bbox_xywh[2] * bbox_xywh[3] / (width * height),
            "expression": expression,
        }
        # Use the downstream schema validator as the final extraction contract.
        normalize_rec_candidate(candidate)
        candidates.append(candidate)

    candidates.sort(key=lambda row: row["candidate_id"])
    ids = [row["candidate_id"] for row in candidates]
    if len(set(ids)) != len(ids):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise ExtractionError(f"{source} has duplicate ref candidate IDs: {duplicates[:3]}")
    stats = {
        "dataset": REF_DATASETS[source],
        "input_train_rows": input_rows,
        "candidates": len(candidates),
        "unique_images": len({row["image_id"] for row in candidates}),
        "candidates_canonical_sha256": canonical_sha256(candidates),
    }
    return candidates, stats


def extract_coco_category_candidates(
    instances: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract noncrowd, officially unambiguous COCO category targets."""
    if not isinstance(instances, Mapping):
        raise ExtractionError("COCO instances payload must be an object")
    raw_images = instances.get("images")
    raw_categories = instances.get("categories")
    raw_annotations = instances.get("annotations")
    if not all(isinstance(value, list) for value in (raw_images, raw_categories, raw_annotations)):
        raise ExtractionError("COCO instances payload lacks images/categories/annotations lists")

    images: dict[int, Mapping[str, Any]] = {}
    for row in raw_images:
        image_id = _require_int(row.get("id"), "COCO image id")
        if image_id in images:
            raise ExtractionError(f"duplicate COCO image id {image_id}")
        images[image_id] = row
    categories: dict[int, str] = {}
    for row in raw_categories:
        category_id = _require_int(row.get("id"), "COCO category id")
        if category_id in categories:
            raise ExtractionError(f"duplicate COCO category id {category_id}")
        categories[category_id] = _require_string(
            row.get("name"), f"COCO category {category_id} name"
        )

    counts: Counter[tuple[int, int]] = Counter()
    annotation_ids: set[int] = set()
    normalized_annotations = []
    crowd_annotations = 0
    for row in raw_annotations:
        annotation_id = _require_int(row.get("id"), "COCO annotation id")
        if annotation_id in annotation_ids:
            raise ExtractionError(f"duplicate COCO annotation id {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = _require_int(row.get("image_id"), f"annotation {annotation_id}.image_id")
        category_id = _require_int(
            row.get("category_id"), f"annotation {annotation_id}.category_id"
        )
        if image_id not in images:
            raise ExtractionError(f"annotation {annotation_id} references unknown image")
        if category_id not in categories:
            raise ExtractionError(f"annotation {annotation_id} references unknown category")
        iscrowd = _require_int(row.get("iscrowd", 0), f"annotation {annotation_id}.iscrowd")
        if iscrowd not in (0, 1):
            raise ExtractionError(f"annotation {annotation_id}.iscrowd must be 0 or 1")
        crowd_annotations += iscrowd
        # Deliberately count crowd and noncrowd rows before target filtering.
        counts[(image_id, category_id)] += 1
        normalized_annotations.append((annotation_id, image_id, category_id, iscrowd, row))

    candidates = []
    ambiguous_noncrowd_targets = 0
    for annotation_id, image_id, category_id, iscrowd, row in sorted(
        normalized_annotations, key=lambda item: item[0]
    ):
        if iscrowd:
            continue
        instance_count = counts[(image_id, category_id)]
        if instance_count != 1:
            ambiguous_noncrowd_targets += 1
            continue
        image = images[image_id]
        width = _require_int(image.get("width"), f"image {image_id}.width")
        height = _require_int(image.get("height"), f"image {image_id}.height")
        if width <= 0 or height <= 0:
            raise ExtractionError(f"image {image_id} has non-positive dimensions")
        file_name = _require_string(image.get("file_name"), f"image {image_id}.file_name")
        expected = f"COCO_train2014_{image_id:012d}.jpg"
        if file_name != expected:
            raise ExtractionError(f"image {image_id} expected file_name {expected!r}")
        bbox_xywh = _box(row.get("bbox"), f"annotation {annotation_id}.bbox_xywh")
        category = categories[category_id]
        candidate = {
            "candidate_id": f"category:coco_train2014:ann:{annotation_id}",
            "task": "coco_grounding",
            "source": CATEGORY_SOURCE,
            "split": "train",
            "annotation_id": annotation_id,
            "category_id": category_id,
            "category": category,
            "category_instance_count": instance_count,
            "image_id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "bbox_xywh": bbox_xywh,
            "relative_area": bbox_xywh[2] * bbox_xywh[3] / (width * height),
        }
        normalize_category_candidate(candidate)
        candidates.append(candidate)

    candidates.sort(key=lambda row: row["candidate_id"])
    stats = {
        "input_images": len(images),
        "input_categories": len(categories),
        "input_annotations": len(normalized_annotations),
        "crowd_annotations_counted_for_ambiguity": crowd_annotations,
        "ambiguous_noncrowd_targets_excluded": ambiguous_noncrowd_targets,
        "candidates": len(candidates),
        "unique_images": len({row["image_id"] for row in candidates}),
        "candidates_canonical_sha256": canonical_sha256(candidates),
    }
    return candidates, stats


def _nontrain_image_id(row: Mapping[str, Any], label: str) -> int:
    return _require_int(row.get("image_id"), f"{label}.image_id")


def build_exclusion_union(
    nontrain_rows: Mapping[str, Mapping[str, Iterable[Mapping[str, Any]]]],
    existing_manifest_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Union cross-variant nontrain IDs and train2014 IDs from prior manifests."""
    excluded: set[int] = set()
    split_stats: dict[str, dict[str, Any]] = {}
    for source in REC_SOURCES:
        if source not in nontrain_rows:
            raise ExtractionError(f"missing nontrain rows for {source}")
        source_stats = {}
        for split in NONTRAIN_SPLITS[source]:
            if split not in nontrain_rows[source]:
                raise ExtractionError(f"missing {source}/{split} nontrain rows")
            ids = [
                _nontrain_image_id(row, f"{source}/{split}")
                for row in nontrain_rows[source][split]
            ]
            excluded.update(ids)
            source_stats[split] = {
                "rows": len(ids),
                "unique_images": len(set(ids)),
                "sorted_image_ids_sha256": canonical_sha256(sorted(set(ids))),
            }
        split_stats[source] = source_stats

    manifest_stats = {}
    for name, rows in sorted((existing_manifest_rows or {}).items()):
        ids = []
        other_split_rows = 0
        for index, row in enumerate(rows):
            image_id = _require_int(row.get("image_id"), f"{name}[{index}].image_id")
            file_name = row.get("file_name")
            if file_name is not None:
                expected_train = f"COCO_train2014_{image_id:012d}.jpg"
                expected_val = f"COCO_val2014_{image_id:012d}.jpg"
                if file_name == expected_val:
                    # Image IDs are split-qualified. A val2014 confirmation row
                    # cannot overlap a train2014 profiling candidate.
                    other_split_rows += 1
                    continue
                if file_name != expected_train:
                    raise ExtractionError(
                        f"{name}[{index}] has an unrecognized split-qualified file name"
                    )
            ids.append(image_id)
        excluded.update(ids)
        manifest_stats[name] = {
            "rows": len(ids),
            "unique_images": len(set(ids)),
            "other_split_rows_verified_disjoint": other_split_rows,
            "sorted_image_ids_sha256": canonical_sha256(sorted(set(ids))),
        }

    output = sorted(excluded)
    stats = {
        "cross_variant_nontrain": split_stats,
        "existing_manifests": manifest_stats,
        "union_unique_images": len(output),
        "union_sorted_image_ids_sha256": canonical_sha256(output),
    }
    return output, stats


def extract_profile_candidate_inputs(
    train_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    nontrain_rows: Mapping[str, Mapping[str, Iterable[Mapping[str, Any]]]],
    coco_instances: Mapping[str, Any],
    *,
    existing_manifest_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pure end-to-end extraction used by the CLI and synthetic tests."""
    rec_candidates = []
    ref_stats = {}
    for source in REC_SOURCES:
        if source not in train_rows:
            raise ExtractionError(f"missing training rows for {source}")
        candidates, stats = extract_ref_candidates(train_rows[source], source, seed=seed)
        rec_candidates.extend(candidates)
        ref_stats[source] = stats
    rec_candidates.sort(key=lambda row: row["candidate_id"])
    if len({row["candidate_id"] for row in rec_candidates}) != len(rec_candidates):
        raise ExtractionError("combined REC candidate IDs are not unique")

    category_candidates, category_stats = extract_coco_category_candidates(coco_instances)
    exclusions, exclusion_stats = build_exclusion_union(
        nontrain_rows, existing_manifest_rows
    )
    exclusion_set = set(exclusions)
    outputs = {
        "rec": rec_candidates,
        "category": category_candidates,
        "exclusions": exclusions,
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "expression_selection_namespace": EXPRESSION_NAMESPACE,
        "seed": seed,
        "sources": {
            "refcoco_family": ref_stats,
            "coco_instances_train2014": category_stats,
            "exclusions": exclusion_stats,
        },
        "outputs": {
            "rec": {
                "file_name": OUTPUT_NAMES["rec"],
                "rows": len(rec_candidates),
                "unique_images": len({row["image_id"] for row in rec_candidates}),
                "sha256": canonical_sha256(rec_candidates),
                "rows_on_excluded_images": sum(
                    row["image_id"] in exclusion_set for row in rec_candidates
                ),
            },
            "category": {
                "file_name": OUTPUT_NAMES["category"],
                "rows": len(category_candidates),
                "unique_images": len(
                    {row["image_id"] for row in category_candidates}
                ),
                "sha256": canonical_sha256(category_candidates),
                "rows_on_excluded_images": sum(
                    row["image_id"] in exclusion_set for row in category_candidates
                ),
            },
            "exclusions": {
                "file_name": OUTPUT_NAMES["exclusions"],
                "rows": len(exclusions),
                "sha256": canonical_sha256(exclusions),
            },
        },
    }
    return outputs, metadata


def load_coco_instances(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load official instances JSON directly or from the standard ann2014 ZIP."""
    artifact = {
        "path": str(path),
        "file_sha256": sha256_file(path),
    }
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if COCO_INSTANCES_MEMBER in names:
                member = COCO_INSTANCES_MEMBER
            else:
                matches = [name for name in names if name.endswith("/instances_train2014.json")]
                if len(matches) != 1:
                    raise ExtractionError(
                        f"expected one instances_train2014 member in {path}: {matches}"
                    )
                member = matches[0]
            data = archive.read(member)
        artifact.update({"member": member, "member_sha256": sha256_bytes(data)})
        payload = json.load(io.TextIOWrapper(io.BytesIO(data), encoding="utf-8"))
    else:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ExtractionError("COCO instances source must decode to an object")
    return payload, artifact


def load_existing_manifests(
    paths: Sequence[Path],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows_by_name = {}
    artifacts = {}
    for path in paths:
        name = path.name
        if name in rows_by_name:
            raise ExtractionError(f"duplicate exclusion-manifest basename: {name}")
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ExtractionError(f"manifest must be a JSON list of objects: {path}")
        rows_by_name[name] = rows
        artifacts[name] = {
            "path": str(path),
            "file_sha256": sha256_file(path),
            "rows": len(rows),
        }
    return rows_by_name, artifacts


def parse_ref_revisions(specs: Sequence[str]) -> dict[str, str]:
    revisions = {}
    for spec in specs:
        if "=" not in spec:
            raise ExtractionError(f"--ref-revision must be SOURCE=REVISION, got {spec!r}")
        source, revision = spec.split("=", 1)
        if source not in REC_SOURCES:
            raise ExtractionError(f"unknown --ref-revision source {source!r}")
        if source in revisions:
            raise ExtractionError(f"duplicate --ref-revision for {source}")
        if not revision or revision in {"main", "master"}:
            raise ExtractionError(f"{source} requires an immutable revision")
        revisions[source] = revision
    missing = sorted(set(REC_SOURCES) - set(revisions))
    if missing:
        raise ExtractionError(f"missing --ref-revision for {missing}")
    return revisions


def load_jxu_ref_sources(
    revisions: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Lazy network/cache adapter; never imported or invoked by unit tests."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ExtractionError("the CLI requires the 'datasets' package") from error

    train = {}
    nontrain = {}
    artifacts = {}
    for source in REC_SOURCES:
        revision = revisions[source]
        dataset = load_dataset(REF_DATASETS[source], revision=revision)
        train[source] = dataset["train"]
        nontrain[source] = {
            split: dataset[split] for split in NONTRAIN_SPLITS[source]
        }
        artifacts[source] = {
            "dataset": REF_DATASETS[source],
            "revision": revision,
            "split_fingerprints": {
                split: getattr(dataset[split], "_fingerprint", None)
                for split in ("train", *NONTRAIN_SPLITS[source])
            },
        }
    return train, nontrain, artifacts


def write_extraction_outputs(
    output_dir: Path,
    outputs: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / file_name for key, file_name in OUTPUT_NAMES.items()}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite output(s): " + ", ".join(existing))
    payloads = {key: canonical_json_bytes(outputs[key]) for key in ("rec", "category", "exclusions")}
    payloads["metadata"] = canonical_json_bytes(metadata)
    for key in ("rec", "category", "exclusions", "metadata"):
        with paths[key].open("xb") as handle:
            handle.write(payloads[key])
            handle.flush()
            os.fsync(handle.fileno())
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-instances", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="repeat for old probe, recovery train/dev, and confirmation manifests",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ref-revision",
        action="append",
        required=True,
        metavar="SOURCE=REVISION",
        help="repeat once per RefCOCO-family repository with its immutable revision",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, nontrain, ref_artifacts = load_jxu_ref_sources(
        parse_ref_revisions(args.ref_revision)
    )
    instances, coco_artifact = load_coco_instances(args.coco_instances)
    manifest_rows, manifest_artifacts = load_existing_manifests(
        args.exclude_manifest
    )
    outputs, metadata = extract_profile_candidate_inputs(
        train,
        nontrain,
        instances,
        existing_manifest_rows=manifest_rows,
        seed=args.seed,
    )
    metadata["source_artifacts"] = {
        "refcoco_family": ref_artifacts,
        "coco_instances": coco_artifact,
        "existing_manifests": manifest_artifacts,
    }
    paths = write_extraction_outputs(args.out_dir, outputs, metadata)
    print(
        json.dumps(
            {
                "paths": {key: str(value) for key, value in paths.items()},
                "output_sha256": {
                    key: metadata["outputs"][key]["sha256"]
                    for key in ("rec", "category", "exclusions")
                },
                "metadata_sha256": canonical_sha256(metadata),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

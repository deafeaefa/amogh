#!/usr/bin/env python3
"""Build deterministic, training-only data for upgraded GCQ profiling.

This module is intentionally independent of Hugging Face datasets.  The CLI
consumes two *normalized* JSON lists produced by an upstream extraction step and
performs only validation, exclusion, stratification, and deterministic
selection.  It writes two image-disjoint 512-row manifests: one for the cheap
coordinate-KL proxy and one for decoded paired-GIoU reranking.

Example::

    python build_gcq_profile_data.py \
      --rec-candidates ref_train_candidates.json \
      --category-candidates coco_train_category_candidates.json \
      --exclude-image-ids forbidden_eval_images.json \
      --exclude-image-ids prior_development_images.json \
      --out-dir "$GCQ_DATA/subsets"

Both candidate files are JSON lists.  Every row has this common schema::

    {
      "candidate_id": "stable-source-specific-id",
      "task": "rec" | "coco_grounding",
      "source": "refcoco" | "refcocoplus" | "refcocog" | "coco_detection",
      "split": "train",
      "image_id": 123,
      "file_name": "COCO_train2014_000000000123.jpg",
      "width": 640,
      "height": 480,
      "bbox_xywh": [x, y, width, height]
    }

REC rows additionally require ``expression`` and ``ref_id``.  Category rows
require ``category``, integer ``category_id``, ``annotation_id``, and
``category_instance_count == 1``.  The latter count must include crowd
annotations; it certifies that the category prompt is unambiguous in the
official annotations.  Optional ``relative_area`` and ``absolute_size`` fields
are checked against the box.  Quartiles, prompts, normalized answers, and UIDs
are derived here.  Exclusion files are either JSON lists of integer image IDs or
objects containing an ``image_ids`` list.

Selection is invariant to candidate input order: all choices are ranked by a
namespaced SHA-256 key over the seed, role, constraint, and ``candidate_id``.
Outputs use canonical JSON and are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
SELECTION_NAMESPACE = "gcq-profile-data-v1"
ROLES = ("proxy", "decode")
TASKS = ("rec", "coco_grounding")
QUARTILES = (1, 2, 3, 4)
REC_SOURCES = ("refcoco", "refcocoplus", "refcocog")
CATEGORY_SOURCE = "coco_detection"
ABSOLUTE_SIZES = ("small", "medium", "large")
ROWS_PER_MANIFEST = 512
ROWS_PER_CELL = 64
DEFAULT_CATEGORY_CAP = 8
DEFAULT_MIN_CATEGORY_SIZE_COUNT = 8
OUTPUT_NAMES = {
    "proxy": "gcq_profile_proxy_train_512.json",
    "decode": "gcq_profile_decode_train_512.json",
    "metadata": "gcq_profile_train.meta.json",
}


class ProfileDataError(ValueError):
    """Raised when normalized inputs or selected manifests violate the protocol."""


class SelectionError(ProfileDataError):
    """Raised when the eligible pool cannot satisfy a frozen selection constraint."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes with a trailing newline."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _require_int(value: Any, label: str) -> int:
    if type(value) is not int:  # bool is deliberately rejected
        raise ProfileDataError(f"{label} must be an integer")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileDataError(f"{label} must be a non-empty string")
    return value.strip()


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileDataError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProfileDataError(f"{label} must be finite")
    return result


def normalize_box_1000(
    bbox_xywh: Sequence[float], width: int, height: int
) -> list[int]:
    x, y, box_width, box_height = bbox_xywh
    xyxy = (x, y, x + box_width, y + box_height)
    dimensions = (width, height, width, height)
    output = [
        max(0, min(1000, round(1000 * value / dimension)))
        for value, dimension in zip(xyxy, dimensions)
    ]
    if not (output[0] < output[2] and output[1] < output[3]):
        raise ProfileDataError(
            f"box collapses after 0..1000 normalization: {bbox_xywh!r}"
        )
    return output


def absolute_size(box_area: float) -> str:
    if box_area < 32**2:
        return "small"
    if box_area < 96**2:
        return "medium"
    return "large"


def _normalize_common(row: Mapping[str, Any], expected_task: str) -> dict[str, Any]:
    candidate_id = _require_nonempty_string(row.get("candidate_id"), "candidate_id")
    task = _require_nonempty_string(row.get("task"), f"{candidate_id}.task")
    if task != expected_task:
        raise ProfileDataError(
            f"{candidate_id}.task is {task!r}, expected {expected_task!r}"
        )
    split = _require_nonempty_string(row.get("split"), f"{candidate_id}.split")
    if split != "train":
        raise ProfileDataError(f"{candidate_id} is not training-only: split={split!r}")
    source = _require_nonempty_string(row.get("source"), f"{candidate_id}.source")
    image_id = _require_int(row.get("image_id"), f"{candidate_id}.image_id")
    if image_id < 0:
        raise ProfileDataError(f"{candidate_id}.image_id must be nonnegative")
    width = _require_int(row.get("width"), f"{candidate_id}.width")
    height = _require_int(row.get("height"), f"{candidate_id}.height")
    if width <= 0 or height <= 0:
        raise ProfileDataError(f"{candidate_id} has non-positive image dimensions")
    file_name = _require_nonempty_string(
        row.get("file_name"), f"{candidate_id}.file_name"
    )
    expected_file_name = f"COCO_train2014_{image_id:012d}.jpg"
    if file_name != expected_file_name:
        raise ProfileDataError(
            f"{candidate_id}.file_name must be {expected_file_name!r}"
        )

    raw_box = row.get("bbox_xywh")
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        raise ProfileDataError(f"{candidate_id}.bbox_xywh must have four values")
    box = [
        _require_number(value, f"{candidate_id}.bbox_xywh[{index}]")
        for index, value in enumerate(raw_box)
    ]
    x, y, box_width, box_height = box
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        raise ProfileDataError(f"{candidate_id} has invalid bbox geometry: {box!r}")
    tolerance = 1e-4
    if x + box_width > width + tolerance or y + box_height > height + tolerance:
        raise ProfileDataError(f"{candidate_id} bbox exceeds image bounds: {box!r}")
    relative = box_width * box_height / (width * height)
    if not (0 < relative <= 1 + tolerance):
        raise ProfileDataError(f"{candidate_id} has invalid relative area {relative}")
    if "relative_area" in row and not math.isclose(
        _require_number(row["relative_area"], f"{candidate_id}.relative_area"),
        relative,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ProfileDataError(f"{candidate_id}.relative_area disagrees with bbox")
    size = absolute_size(box_width * box_height)
    if "absolute_size" in row and row["absolute_size"] != size:
        raise ProfileDataError(f"{candidate_id}.absolute_size disagrees with bbox")

    return {
        "candidate_id": candidate_id,
        "task": task,
        "source": source,
        "split": "train",
        "image_id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "bbox_xywh": box,
        "relative_area": relative,
        "absolute_size": size,
        "bbox_1000": normalize_box_1000(box, width, height),
    }


def normalize_rec_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    result = _normalize_common(row, "rec")
    candidate_id = result["candidate_id"]
    if result["source"] not in REC_SOURCES:
        raise ProfileDataError(
            f"{candidate_id}.source must be one of {REC_SOURCES!r}"
        )
    expression = _require_nonempty_string(
        row.get("expression"), f"{candidate_id}.expression"
    )
    ref_id = row.get("ref_id")
    if isinstance(ref_id, bool) or not isinstance(ref_id, (int, str)):
        raise ProfileDataError(f"{candidate_id}.ref_id must be an integer or string")
    result.update(
        {
            "dataset": result["source"],
            "ref_id": ref_id,
            "expression": expression,
        }
    )
    return result


def normalize_category_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    result = _normalize_common(row, "coco_grounding")
    candidate_id = result["candidate_id"]
    if result["source"] != CATEGORY_SOURCE:
        raise ProfileDataError(
            f"{candidate_id}.source must be {CATEGORY_SOURCE!r}"
        )
    category = _require_nonempty_string(
        row.get("category"), f"{candidate_id}.category"
    )
    category_id = _require_int(
        row.get("category_id"), f"{candidate_id}.category_id"
    )
    annotation_id = row.get("annotation_id")
    if isinstance(annotation_id, bool) or not isinstance(annotation_id, (int, str)):
        raise ProfileDataError(
            f"{candidate_id}.annotation_id must be an integer or string"
        )
    instance_count = _require_int(
        row.get("category_instance_count"),
        f"{candidate_id}.category_instance_count",
    )
    if instance_count != 1:
        raise ProfileDataError(
            f"{candidate_id} is an ambiguous category prompt: "
            f"category_instance_count={instance_count}"
        )
    result.update(
        {
            "dataset": "coco_train2014",
            "annotation_id": annotation_id,
            "category_id": category_id,
            "category": category,
            "category_instance_count": 1,
            "expression": category,
        }
    )
    return result


def derive_quartile_bounds(relative_areas: Iterable[float]) -> tuple[float, float, float]:
    values = sorted(float(value) for value in relative_areas)
    if len(values) < 4:
        raise ProfileDataError("at least four eligible boxes are required for quartiles")
    # Nearest-rank boundaries place exactly divisible, distinct strata on the
    # lower stratum's final observation rather than the next stratum's first.
    bounds = tuple(values[math.ceil(len(values) * q) - 1] for q in (0.25, 0.5, 0.75))
    if not (bounds[0] < bounds[1] < bounds[2]):
        raise ProfileDataError(
            "derived relative-area quartile bounds are not strictly increasing; "
            "provide a more diverse normalized candidate pool"
        )
    return bounds


def area_quartile(value: float, bounds: Sequence[float]) -> int:
    if len(bounds) != 3 or not (bounds[0] < bounds[1] < bounds[2]):
        raise ProfileDataError("quartile bounds must contain three increasing values")
    if value <= bounds[0]:
        return 1
    if value <= bounds[1]:
        return 2
    if value <= bounds[2]:
        return 3
    return 4


def selection_key(
    seed: int, role: str, constraint: str, candidate_id: str
) -> tuple[str, str]:
    payload = (
        f"{SELECTION_NAMESPACE}\0{seed}\0{role}\0{constraint}\0{candidate_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), candidate_id


def _ranked(
    candidates: Iterable[dict[str, Any]], seed: int, role: str, constraint: str
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: selection_key(seed, role, constraint, row["candidate_id"]),
    )


def _select_rec(
    candidates: Sequence[dict[str, Any]],
    *,
    role: str,
    role_index: int,
    seed: int,
    used_images: set[int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for quartile in QUARTILES:
        extra_source = REC_SOURCES[(quartile - 1 + role_index) % len(REC_SOURCES)]
        for source in REC_SOURCES:
            target = 22 if source == extra_source else 21
            pool = (
                row
                for row in candidates
                if row["area_quartile"] == quartile and row["source"] == source
            )
            ranked = _ranked(
                pool, seed, role, f"rec:q{quartile}:source:{source}"
            )
            chosen = []
            for row in ranked:
                if row["image_id"] in used_images:
                    continue
                chosen.append(row)
                used_images.add(row["image_id"])
                if len(chosen) == target:
                    break
            if len(chosen) != target:
                raise SelectionError(
                    f"{role} REC q{quartile}/{source}: selected "
                    f"{len(chosen)}/{target} unique eligible images"
                )
            selected.extend(chosen)
    return selected


def _select_category(
    candidates: Sequence[dict[str, Any]],
    *,
    role: str,
    seed: int,
    used_images: set[int],
    category_cap: int,
    min_absolute_size_count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    category_counts: Counter[int] = Counter()
    quartile_counts: Counter[int] = Counter()

    def admissible(row: dict[str, Any]) -> bool:
        return (
            row["candidate_id"] not in selected_ids
            and row["image_id"] not in used_images
            and category_counts[row["category_id"]] < category_cap
            and quartile_counts[row["area_quartile"]] < ROWS_PER_CELL
        )

    def accept(row: dict[str, Any]) -> None:
        selected.append(row)
        selected_ids.add(row["candidate_id"])
        used_images.add(row["image_id"])
        category_counts[row["category_id"]] += 1
        quartile_counts[row["area_quartile"]] += 1

    # Reserve the predeclared minimum representation for all genuine COCO
    # absolute sizes before filling relative-area cells.
    for size in ABSOLUTE_SIZES:
        pool = _ranked(
            (row for row in candidates if row["absolute_size"] == size),
            seed,
            role,
            f"category:absolute-size:{size}",
        )
        before = len(selected)
        for row in pool:
            if admissible(row):
                accept(row)
                if len(selected) - before == min_absolute_size_count:
                    break
        if len(selected) - before != min_absolute_size_count:
            raise SelectionError(
                f"{role} category {size}: selected "
                f"{len(selected) - before}/{min_absolute_size_count} required rows"
            )

    for quartile in QUARTILES:
        pool = _ranked(
            (row for row in candidates if row["area_quartile"] == quartile),
            seed,
            role,
            f"category:q{quartile}:fill",
        )
        for row in pool:
            if admissible(row):
                accept(row)
                if quartile_counts[quartile] == ROWS_PER_CELL:
                    break
        if quartile_counts[quartile] != ROWS_PER_CELL:
            raise SelectionError(
                f"{role} category q{quartile}: selected "
                f"{quartile_counts[quartile]}/{ROWS_PER_CELL}; category cap, "
                "image disjointness, or source diversity exhausted the pool"
            )
    return selected


def _materialize_manifest(
    selected: Sequence[dict[str, Any]], role: str
) -> list[dict[str, Any]]:
    task_order = {task: index for index, task in enumerate(TASKS)}
    ordered = sorted(
        selected,
        key=lambda row: (
            task_order[row["task"]],
            row["area_quartile"],
            row["source"],
            row["candidate_id"],
        ),
    )
    manifest = []
    for index, row in enumerate(ordered):
        output = {key: value for key, value in row.items() if key != "bbox_1000"}
        output["uid"] = f"gcq_profile_{role}_train_512:{index:05d}"
        output["prompt"] = (
            f"Locate the {row['expression']}, output its bbox_2d in JSON."
        )
        output["answer"] = json.dumps(
            {"bbox_2d": row["bbox_1000"]}, separators=(",", ": ")
        )
        manifest.append(output)
    return manifest


def _manifest_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells = Counter((row["task"], row["area_quartile"]) for row in rows)
    category_counts = Counter(
        row["category_id"] for row in rows if row["task"] == "coco_grounding"
    )
    return {
        "rows": len(rows),
        "unique_images": len({row["image_id"] for row in rows}),
        "sha256": canonical_sha256(rows),
        "task_counts": dict(sorted(Counter(row["task"] for row in rows).items())),
        "task_quartile_counts": {
            f"{task}:q{quartile}": cells[(task, quartile)]
            for task in TASKS
            for quartile in QUARTILES
        },
        "rec_source_counts": dict(
            sorted(
                Counter(
                    row["source"] for row in rows if row["task"] == "rec"
                ).items()
            )
        ),
        "category_absolute_size_counts": dict(
            sorted(
                Counter(
                    row["absolute_size"]
                    for row in rows
                    if row["task"] == "coco_grounding"
                ).items()
            )
        ),
        "category_unique_ids": len(category_counts),
        "category_max_count": max(category_counts.values(), default=0),
    }


def validate_profile_manifests(
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    quartile_bounds: Sequence[float],
    excluded_image_ids: Iterable[int],
    category_cap: int,
    min_absolute_size_count: int,
) -> dict[str, dict[str, Any]]:
    excluded = set(excluded_image_ids)
    all_images: dict[str, set[int]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        if role not in manifests:
            raise ProfileDataError(f"missing {role!r} manifest")
        rows = list(manifests[role])
        if len(rows) != ROWS_PER_MANIFEST:
            raise ProfileDataError(
                f"{role} must contain {ROWS_PER_MANIFEST} rows, found {len(rows)}"
            )
        uids = [row.get("uid") for row in rows]
        if len(set(uids)) != len(uids):
            raise ProfileDataError(f"{role} contains duplicate UIDs")
        images = {row["image_id"] for row in rows}
        if len(images) != len(rows):
            raise ProfileDataError(f"{role} must contain one row per image")
        overlap = images & excluded
        if overlap:
            raise ProfileDataError(
                f"{role} contains {len(overlap)} excluded image IDs"
            )
        all_images[role] = images

        cells = Counter((row["task"], row["area_quartile"]) for row in rows)
        expected_cells = Counter(
            {(task, quartile): ROWS_PER_CELL for task in TASKS for quartile in QUARTILES}
        )
        if cells != expected_cells:
            raise ProfileDataError(f"{role} task/quartile composition is invalid: {cells}")
        for row in rows:
            if row.get("split") != "train":
                raise ProfileDataError(f"{role}/{row.get('uid')} is not training-only")
            expected_quartile = area_quartile(
                float(row["relative_area"]), quartile_bounds
            )
            if row["area_quartile"] != expected_quartile:
                raise ProfileDataError(
                    f"{role}/{row.get('uid')} has incorrect area quartile"
                )

        rec_sources = Counter(
            row["source"] for row in rows if row["task"] == "rec"
        )
        if set(rec_sources) != set(REC_SOURCES) or sorted(rec_sources.values()) != [
            85,
            85,
            86,
        ]:
            raise ProfileDataError(f"{role} REC source balance is invalid: {rec_sources}")

        categories = Counter(
            row["category_id"] for row in rows if row["task"] == "coco_grounding"
        )
        if max(categories.values(), default=0) > category_cap:
            raise ProfileDataError(f"{role} exceeds category cap {category_cap}")
        sizes = Counter(
            row["absolute_size"]
            for row in rows
            if row["task"] == "coco_grounding"
        )
        for size in ABSOLUTE_SIZES:
            if sizes[size] < min_absolute_size_count:
                raise ProfileDataError(
                    f"{role} has only {sizes[size]} {size} category rows; "
                    f"minimum is {min_absolute_size_count}"
                )
        summaries[role] = _manifest_summary(rows)

    cross_overlap = all_images["proxy"] & all_images["decode"]
    if cross_overlap:
        raise ProfileDataError(
            f"proxy/decode manifests overlap on {len(cross_overlap)} images"
        )
    return summaries


def build_profile_manifests(
    rec_candidates: Sequence[Mapping[str, Any]],
    category_candidates: Sequence[Mapping[str, Any]],
    *,
    excluded_image_ids: Iterable[int] = (),
    seed: int = 0,
    category_cap: int = DEFAULT_CATEGORY_CAP,
    min_absolute_size_count: int = DEFAULT_MIN_CATEGORY_SIZE_COUNT,
    quartile_bounds: Sequence[float] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Normalize, select, validate, and describe the two profiling manifests."""
    if category_cap <= 0:
        raise ProfileDataError("category_cap must be positive")
    if min_absolute_size_count <= 0:
        raise ProfileDataError("min_absolute_size_count must be positive")
    if 3 * min_absolute_size_count > 4 * ROWS_PER_CELL:
        raise ProfileDataError("absolute-size minimum exceeds category manifest size")
    excluded = {_require_int(value, "excluded image ID") for value in excluded_image_ids}

    normalized_rec = [normalize_rec_candidate(row) for row in rec_candidates]
    normalized_category = [
        normalize_category_candidate(row) for row in category_candidates
    ]
    all_normalized = normalized_rec + normalized_category
    candidate_ids = [row["candidate_id"] for row in all_normalized]
    if len(set(candidate_ids)) != len(candidate_ids):
        duplicates = [
            key for key, count in Counter(candidate_ids).items() if count > 1
        ]
        raise ProfileDataError(f"duplicate candidate_id values: {duplicates[:3]}")
    normalized_candidates_hash = canonical_sha256(
        sorted(all_normalized, key=lambda row: row["candidate_id"])
    )

    eligible_rec = [row for row in normalized_rec if row["image_id"] not in excluded]
    eligible_category = [
        row for row in normalized_category if row["image_id"] not in excluded
    ]
    eligible = eligible_rec + eligible_category
    if quartile_bounds is None:
        bounds = derive_quartile_bounds(row["relative_area"] for row in eligible)
    else:
        bounds = tuple(float(value) for value in quartile_bounds)
        # Validate before assigning rows.
        area_quartile(float(eligible[0]["relative_area"]), bounds) if eligible else area_quartile(0.5, bounds)

    for row in eligible:
        row["area_quartile"] = area_quartile(row["relative_area"], bounds)

    used_images: set[int] = set()
    manifests: dict[str, list[dict[str, Any]]] = {}
    for role_index, role in enumerate(ROLES):
        selected = _select_rec(
            eligible_rec,
            role=role,
            role_index=role_index,
            seed=seed,
            used_images=used_images,
        )
        selected.extend(
            _select_category(
                eligible_category,
                role=role,
                seed=seed,
                used_images=used_images,
                category_cap=category_cap,
                min_absolute_size_count=min_absolute_size_count,
            )
        )
        manifests[role] = _materialize_manifest(selected, role)

    summaries = validate_profile_manifests(
        manifests,
        quartile_bounds=bounds,
        excluded_image_ids=excluded,
        category_cap=category_cap,
        min_absolute_size_count=min_absolute_size_count,
    )
    canonical_eligible = sorted(eligible, key=lambda row: row["candidate_id"])
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "selection_namespace": SELECTION_NAMESPACE,
        "seed": seed,
        "selection": {
            "roles": list(ROLES),
            "rows_per_manifest": ROWS_PER_MANIFEST,
            "rows_per_task_quartile_cell": ROWS_PER_CELL,
            "candidate_order_invariant": True,
            "image_disjoint_within_and_between_roles": True,
            "rec_sources": list(REC_SOURCES),
            "category_cap_per_manifest": category_cap,
            "minimum_category_rows_per_absolute_size": min_absolute_size_count,
            "quartile_bounds": list(bounds),
            "quartile_source": (
                "derived from eligible normalized training candidates"
                if quartile_bounds is None
                else "caller-supplied training-only bounds"
            ),
        },
        "inputs": {
            "rec_candidates": len(normalized_rec),
            "category_candidates": len(normalized_category),
            "eligible_rec_candidates": len(eligible_rec),
            "eligible_category_candidates": len(eligible_category),
            "normalized_candidates_canonical_sha256": normalized_candidates_hash,
            "eligible_candidates_canonical_sha256": canonical_sha256(
                canonical_eligible
            ),
        },
        "exclusions": {
            "image_ids": len(excluded),
            "sorted_image_ids_sha256": canonical_sha256(sorted(excluded)),
            "selected_overlap": 0,
        },
        "outputs": {
            role: {"file_name": OUTPUT_NAMES[role], **summaries[role]}
            for role in ROLES
        },
        "cross_manifest_image_overlap": 0,
    }
    return manifests, metadata


def _load_json_list(path: Path, label: str) -> list[Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ProfileDataError(f"{label} must be a JSON list: {path}")
    return value


def load_excluded_image_ids(paths: Sequence[Path]) -> set[int]:
    result: set[int] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            value = value.get("image_ids")
        if not isinstance(value, list):
            raise ProfileDataError(
                f"exclusion file must be a list or contain image_ids: {path}"
            )
        result.update(_require_int(item, f"{path} image ID") for item in value)
    return result


def write_profile_outputs(
    output_dir: Path,
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        role: output_dir / OUTPUT_NAMES[role] for role in ROLES
    }
    paths["metadata"] = output_dir / OUTPUT_NAMES["metadata"]
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite output(s): " + ", ".join(existing))
    payloads = {
        role: canonical_json_bytes(manifests[role]) for role in ROLES
    }
    payloads["metadata"] = canonical_json_bytes(metadata)
    for key in (*ROLES, "metadata"):
        with paths[key].open("xb") as handle:
            handle.write(payloads[key])
            handle.flush()
            os.fsync(handle.fileno())
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rec-candidates", type=Path, required=True)
    parser.add_argument("--category-candidates", type=Path, required=True)
    parser.add_argument(
        "--exclude-image-ids",
        type=Path,
        action="append",
        default=[],
        help="repeatable JSON list, or object containing image_ids",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--category-cap", type=int, default=DEFAULT_CATEGORY_CAP)
    parser.add_argument(
        "--min-category-size-count",
        type=int,
        default=DEFAULT_MIN_CATEGORY_SIZE_COUNT,
    )
    parser.add_argument(
        "--quartile-bounds",
        type=float,
        nargs=3,
        default=None,
        metavar=("Q1_MAX", "Q2_MAX", "Q3_MAX"),
        help="optional precomputed training-only relative-area bounds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rec = _load_json_list(args.rec_candidates, "REC candidates")
    category = _load_json_list(args.category_candidates, "category candidates")
    excluded = load_excluded_image_ids(args.exclude_image_ids)
    manifests, metadata = build_profile_manifests(
        rec,
        category,
        excluded_image_ids=excluded,
        seed=args.seed,
        category_cap=args.category_cap,
        min_absolute_size_count=args.min_category_size_count,
        quartile_bounds=args.quartile_bounds,
    )
    paths = write_profile_outputs(args.out_dir, manifests, metadata)
    print(
        json.dumps(
            {
                "paths": {key: str(value) for key, value in paths.items()},
                "manifest_sha256": {
                    role: metadata["outputs"][role]["sha256"] for role in ROLES
                },
                "metadata_sha256": canonical_sha256(metadata),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

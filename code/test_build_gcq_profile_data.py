import copy
from collections import Counter
import json
import random

import pytest

from build_gcq_profile_data import (
    ABSOLUTE_SIZES,
    QUARTILES,
    REC_SOURCES,
    ROLES,
    ROWS_PER_CELL,
    ROWS_PER_MANIFEST,
    ProfileDataError,
    SelectionError,
    build_profile_manifests,
    canonical_json_bytes,
    canonical_sha256,
    validate_profile_manifests,
    write_profile_outputs,
)


RELATIVE_AREAS = {1: 0.005, 2: 0.02, 3: 0.08, 4: 0.20}


def _common(candidate_id, task, source, image_id, relative_area):
    width = height = 400
    # A full-height box makes bbox area exactly relative_area * image area.
    bbox = [0.0, 0.0, relative_area * width, float(height)]
    return {
        "candidate_id": candidate_id,
        "task": task,
        "source": source,
        "split": "train",
        "image_id": image_id,
        "file_name": f"COCO_train2014_{image_id:012d}.jpg",
        "width": width,
        "height": height,
        "bbox_xywh": bbox,
        "relative_area": relative_area,
    }


def candidate_pools():
    rec = []
    category = []
    image_id = 1
    for quartile, relative_area in RELATIVE_AREAS.items():
        # Fifty rows per source and quartile is enough for the two manifests'
        # combined 42/43-row source requirements after exclusions.
        for source in REC_SOURCES:
            for index in range(50):
                row = _common(
                    f"rec:{source}:q{quartile}:{index:03d}",
                    "rec",
                    source,
                    image_id,
                    relative_area,
                )
                row.update(
                    {
                        "ref_id": f"{source}:{quartile}:{index}",
                        "expression": f"{source} object {quartile} {index}",
                    }
                )
                rec.append(row)
                image_id += 1

        # 64 categories x three candidates per quartile provides diversity
        # under the per-manifest cap while keeping every image globally unique.
        for category_id in range(1, 65):
            for repetition in range(3):
                row = _common(
                    f"category:q{quartile}:{category_id:02d}:{repetition}",
                    "coco_grounding",
                    "coco_detection",
                    image_id,
                    relative_area,
                )
                row.update(
                    {
                        "annotation_id": f"ann:{quartile}:{category_id}:{repetition}",
                        "category_id": category_id,
                        "category": f"category {category_id}",
                        "category_instance_count": 1,
                    }
                )
                category.append(row)
                image_id += 1
    return rec, category


def balanced_build():
    rec, category = candidate_pools()
    # Exclude one REC image at every scale; the derived population quartiles
    # remain exactly balanced, and selection must never recover these IDs.
    excluded = {
        next(
            row["image_id"]
            for row in rec
            if row["candidate_id"].startswith(f"rec:refcoco:q{quartile}:")
        )
        for quartile in QUARTILES
    }
    manifests, metadata = build_profile_manifests(
        rec,
        category,
        excluded_image_ids=excluded,
        seed=17,
        category_cap=8,
        min_absolute_size_count=8,
    )
    return rec, category, excluded, manifests, metadata


def test_exact_balance_disjointness_exclusions_caps_scales_and_hashes():
    _, _, excluded, manifests, metadata = balanced_build()
    all_images = {}
    for role in ROLES:
        rows = manifests[role]
        assert len(rows) == ROWS_PER_MANIFEST
        assert len({row["uid"] for row in rows}) == ROWS_PER_MANIFEST
        all_images[role] = {row["image_id"] for row in rows}
        assert len(all_images[role]) == ROWS_PER_MANIFEST
        assert all_images[role].isdisjoint(excluded)

        cells = Counter((row["task"], row["area_quartile"]) for row in rows)
        assert set(cells.values()) == {ROWS_PER_CELL}
        assert len(cells) == 8

        rec_sources = Counter(
            row["source"] for row in rows if row["task"] == "rec"
        )
        assert set(rec_sources) == set(REC_SOURCES)
        assert sorted(rec_sources.values()) == [85, 85, 86]

        category_counts = Counter(
            row["category_id"]
            for row in rows
            if row["task"] == "coco_grounding"
        )
        assert max(category_counts.values()) <= 8
        size_counts = Counter(
            row["absolute_size"]
            for row in rows
            if row["task"] == "coco_grounding"
        )
        assert set(size_counts) == set(ABSOLUTE_SIZES)
        assert all(size_counts[size] >= 8 for size in ABSOLUTE_SIZES)
        assert all(row["split"] == "train" for row in rows)
        assert all(
            json.loads(row["answer"])["bbox_2d"]
            for row in rows
        )

        assert metadata["outputs"][role]["sha256"] == canonical_sha256(rows)
        assert metadata["outputs"][role]["unique_images"] == ROWS_PER_MANIFEST

    assert all_images["proxy"].isdisjoint(all_images["decode"])
    assert metadata["cross_manifest_image_overlap"] == 0
    assert metadata["exclusions"]["image_ids"] == len(excluded)
    assert metadata["exclusions"]["sorted_image_ids_sha256"] == canonical_sha256(
        sorted(excluded)
    )

    summaries = validate_profile_manifests(
        manifests,
        quartile_bounds=metadata["selection"]["quartile_bounds"],
        excluded_image_ids=excluded,
        category_cap=8,
        min_absolute_size_count=8,
    )
    assert summaries["proxy"]["sha256"] == metadata["outputs"]["proxy"]["sha256"]


def test_selection_and_metadata_are_invariant_to_candidate_input_order():
    rec, category = candidate_pools()
    excluded = {
        next(
            row["image_id"]
            for row in rec
            if row["candidate_id"].startswith(f"rec:refcoco:q{quartile}:")
        )
        for quartile in QUARTILES
    }

    first_manifests, first_metadata = build_profile_manifests(
        rec,
        category,
        excluded_image_ids=excluded,
        seed=9,
        category_cap=8,
        min_absolute_size_count=8,
    )
    shuffled_rec = copy.deepcopy(rec)
    shuffled_category = copy.deepcopy(category)
    random.Random(1).shuffle(shuffled_rec)
    random.Random(2).shuffle(shuffled_category)
    second_manifests, second_metadata = build_profile_manifests(
        shuffled_rec,
        shuffled_category,
        excluded_image_ids=reversed(sorted(excluded)),
        seed=9,
        category_cap=8,
        min_absolute_size_count=8,
    )

    assert second_manifests == first_manifests
    assert second_metadata == first_metadata
    assert canonical_json_bytes(second_metadata) == canonical_json_bytes(first_metadata)


def test_infeasible_category_cap_fails_closed():
    rec, category = candidate_pools()
    for row in category:
        row["category_id"] = 1
        row["category"] = "only category"
    with pytest.raises(SelectionError, match="category"):
        build_profile_manifests(
            rec,
            category,
            seed=0,
            category_cap=8,
            min_absolute_size_count=8,
        )


def test_ambiguous_category_candidate_is_rejected_before_selection():
    rec, category = candidate_pools()
    category[0]["category_instance_count"] = 2
    with pytest.raises(ProfileDataError, match="ambiguous category prompt"):
        build_profile_manifests(rec, category)


def test_validator_rejects_cross_manifest_overlap_and_new_exclusion():
    _, _, excluded, manifests, metadata = balanced_build()
    corrupted = copy.deepcopy(manifests)
    corrupted["decode"][0]["image_id"] = corrupted["proxy"][0]["image_id"]
    with pytest.raises(ProfileDataError, match="proxy/decode manifests overlap"):
        validate_profile_manifests(
            corrupted,
            quartile_bounds=metadata["selection"]["quartile_bounds"],
            excluded_image_ids=excluded,
            category_cap=8,
            min_absolute_size_count=8,
        )

    newly_excluded = excluded | {manifests["proxy"][0]["image_id"]}
    with pytest.raises(ProfileDataError, match="excluded image IDs"):
        validate_profile_manifests(
            manifests,
            quartile_bounds=metadata["selection"]["quartile_bounds"],
            excluded_image_ids=newly_excluded,
            category_cap=8,
            min_absolute_size_count=8,
        )


def test_canonical_outputs_match_declared_hashes_and_are_write_once(tmp_path):
    _, _, _, manifests, metadata = balanced_build()
    paths = write_profile_outputs(tmp_path, manifests, metadata)
    for role in ROLES:
        assert paths[role].read_bytes() == canonical_json_bytes(manifests[role])
        assert canonical_sha256(json.loads(paths[role].read_text())) == metadata[
            "outputs"
        ][role]["sha256"]
    assert paths["metadata"].read_bytes() == canonical_json_bytes(metadata)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_profile_outputs(tmp_path, manifests, metadata)

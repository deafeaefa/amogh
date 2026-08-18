import copy
import json

import pytest

from build_gcq_profile_data import canonical_json_bytes, canonical_sha256
from extract_gcq_profile_candidates import (
    NONTRAIN_SPLITS,
    REC_SOURCES,
    ExtractionError,
    build_exclusion_union,
    extract_coco_category_candidates,
    extract_profile_candidate_inputs,
    extract_ref_candidates,
    parse_ref_revisions,
    write_extraction_outputs,
)


def ref_row(source, image_id, ref_id, *, split="train"):
    x, y, width, height = 10.0, 20.0, 30.0, 40.0
    annotation_id = 100_000 + ref_id
    return {
        "split": split,
        "ref_id": ref_id,
        "ann_id": annotation_id,
        "image_id": image_id,
        "bbox": [x, y, x + width, y + height],  # jxu top-level XYXY
        "raw_anns": json.dumps(
            {
                "id": annotation_id,
                "image_id": image_id,
                "bbox": [x, y, width, height],  # COCO raw XYWH
            }
        ),
        "raw_image_info": {
            "id": image_id,
            "width": 100,
            "height": 120,
            "file_name": f"COCO_train2014_{image_id:012d}.jpg",
        },
        "sentences": [
            {"sent_id": ref_id * 10 + 1, "sent": f"{source} left object"},
            {"sent_id": ref_id * 10 + 2, "sent": f"{source} blue object"},
            {"sent_id": ref_id * 10 + 3, "sent": f"{source} rear object"},
        ],
    }


def coco_instances():
    images = [
        {
            "id": image_id,
            "width": 200,
            "height": 100,
            "file_name": f"COCO_train2014_{image_id:012d}.jpg",
        }
        for image_id in (501, 502, 503)
    ]
    return {
        "images": images,
        "categories": [{"id": 7, "name": "cat"}],
        "annotations": [
            # Image 501 is ambiguous only because the crowd annotation counts.
            {"id": 1, "image_id": 501, "category_id": 7, "iscrowd": 0,
             "bbox": [5, 5, 20, 20]},
            {"id": 2, "image_id": 501, "category_id": 7, "iscrowd": 1,
             "bbox": [0, 0, 50, 50]},
            # Image 502 is the sole selectable category target.
            {"id": 3, "image_id": 502, "category_id": 7, "iscrowd": 0,
             "bbox": [10, 10, 30, 25]},
            # Crowd-only targets are counted but never selected.
            {"id": 4, "image_id": 503, "category_id": 7, "iscrowd": 1,
             "bbox": [1, 1, 10, 10]},
        ],
    }


def nontrain_rows():
    output = {}
    next_id = 700
    for source in REC_SOURCES:
        output[source] = {}
        for split in NONTRAIN_SPLITS[source]:
            output[source][split] = [
                {"image_id": next_id},
                {"image_id": next_id + 1},
                {"image_id": next_id + 1},  # duplicate row must collapse
            ]
            next_id += 2
    return output


def train_rows():
    return {
        source: [
            ref_row(source, 100 + source_index * 10, 1),
            ref_row(source, 101 + source_index * 10, 2),
        ]
        for source_index, source in enumerate(REC_SOURCES)
    }


def prior_manifests():
    return {
        "dprobe.json": [
            {
                "image_id": 900,
                "file_name": "COCO_train2014_000000000900.jpg",
            },
            {
                "image_id": 900,
                "file_name": "COCO_train2014_000000000900.jpg",
            },
        ],
        "confirmation.json": [
            {
                "image_id": 901,
                "file_name": "COCO_train2014_000000000901.jpg",
            }
        ],
    }


def test_ref_xyxy_is_converted_and_verified_against_raw_xywh():
    row = ref_row("refcoco", 123, 9)
    candidates, stats = extract_ref_candidates([row], "refcoco", seed=4)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["candidate_id"] == "rec:refcoco:ref:9"
    assert candidate["bbox_xywh"] == [10.0, 20.0, 30.0, 40.0]
    assert candidate["annotation_id"] == 100_009
    assert candidate["expression"] in {sentence["sent"] for sentence in row["sentences"]}
    assert stats["candidates_canonical_sha256"] == canonical_sha256(candidates)

    mismatched = copy.deepcopy(row)
    mismatched["bbox"][2] += 1
    with pytest.raises(ExtractionError, match="XYXY bbox disagrees"):
        extract_ref_candidates([mismatched], "refcoco", seed=4)

    no_split_column = copy.deepcopy(row)
    no_split_column.pop("split")
    assert extract_ref_candidates([no_split_column], "refcoco", seed=4)[0] == candidates


def test_expression_and_candidate_order_are_hash_deterministic():
    rows = [ref_row("refcoco", 123, 9), ref_row("refcoco", 124, 10)]
    first, first_stats = extract_ref_candidates(rows, "refcoco", seed=11)
    reordered = copy.deepcopy(list(reversed(rows)))
    for row in reordered:
        row["sentences"].reverse()
    second, second_stats = extract_ref_candidates(reordered, "refcoco", seed=11)
    assert second == first
    assert second_stats == first_stats


def test_crowd_annotations_participate_in_category_ambiguity_count():
    candidates, stats = extract_coco_category_candidates(coco_instances())
    assert [row["annotation_id"] for row in candidates] == [3]
    assert candidates[0]["category_instance_count"] == 1
    assert candidates[0]["candidate_id"] == "category:coco_train2014:ann:3"
    assert stats["crowd_annotations_counted_for_ambiguity"] == 2
    assert stats["ambiguous_noncrowd_targets_excluded"] == 1
    assert stats["candidates_canonical_sha256"] == canonical_sha256(candidates)


def test_exclusion_union_covers_all_cross_variant_splits_and_manifests():
    sources = nontrain_rows()
    manifests = prior_manifests()
    exclusions, stats = build_exclusion_union(sources, manifests)
    expected = {
        row["image_id"]
        for source in sources.values()
        for rows in source.values()
        for row in rows
    } | {900, 901}
    assert exclusions == sorted(expected)
    assert stats["union_unique_images"] == len(expected)
    assert stats["union_sorted_image_ids_sha256"] == canonical_sha256(exclusions)
    assert set(stats["cross_variant_nontrain"]) == set(REC_SOURCES)
    assert set(stats["existing_manifests"]) == set(manifests)

    with_val = copy.deepcopy(manifests)
    with_val["val_confirmation.json"] = [{
        "image_id": 42,
        "file_name": "COCO_val2014_000000000042.jpg",
    }]
    exclusions_with_val, stats_with_val = build_exclusion_union(sources, with_val)
    assert 42 not in exclusions_with_val
    assert stats_with_val["existing_manifests"]["val_confirmation.json"][
        "other_split_rows_verified_disjoint"
    ] == 1


def test_end_to_end_extraction_is_order_invariant_and_source_hashed():
    train = train_rows()
    nontrain = nontrain_rows()
    instances = coco_instances()
    manifests = prior_manifests()
    first_outputs, first_metadata = extract_profile_candidate_inputs(
        train,
        nontrain,
        instances,
        existing_manifest_rows=manifests,
        seed=23,
    )

    reordered_train = copy.deepcopy(train)
    for rows in reordered_train.values():
        rows.reverse()
        for row in rows:
            row["sentences"].reverse()
    reordered_nontrain = copy.deepcopy(nontrain)
    for source in reordered_nontrain.values():
        for rows in source.values():
            rows.reverse()
    reordered_instances = copy.deepcopy(instances)
    for key in ("images", "categories", "annotations"):
        reordered_instances[key].reverse()
    reordered_manifests = {
        key: list(reversed(value))
        for key, value in reversed(list(manifests.items()))
    }
    second_outputs, second_metadata = extract_profile_candidate_inputs(
        reordered_train,
        reordered_nontrain,
        reordered_instances,
        existing_manifest_rows=reordered_manifests,
        seed=23,
    )

    assert second_outputs == first_outputs
    assert second_metadata == first_metadata
    for key in ("rec", "category", "exclusions"):
        assert first_metadata["outputs"][key]["sha256"] == canonical_sha256(
            first_outputs[key]
        )
    for source in REC_SOURCES:
        source_rows = [
            row for row in first_outputs["rec"] if row["source"] == source
        ]
        assert first_metadata["sources"]["refcoco_family"][source][
            "candidates_canonical_sha256"
        ] == canonical_sha256(source_rows)


def test_canonical_outputs_are_write_once_and_match_hashes(tmp_path):
    outputs, metadata = extract_profile_candidate_inputs(
        train_rows(),
        nontrain_rows(),
        coco_instances(),
        existing_manifest_rows=prior_manifests(),
        seed=5,
    )
    paths = write_extraction_outputs(tmp_path, outputs, metadata)
    for key in ("rec", "category", "exclusions"):
        assert paths[key].read_bytes() == canonical_json_bytes(outputs[key])
        assert canonical_sha256(json.loads(paths[key].read_text())) == metadata[
            "outputs"
        ][key]["sha256"]
    assert paths["metadata"].read_bytes() == canonical_json_bytes(metadata)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_extraction_outputs(tmp_path, outputs, metadata)


def test_ref_revisions_are_per_repository_and_immutable():
    revisions = parse_ref_revisions([
        "refcoco=aaa", "refcocoplus=bbb", "refcocog=ccc"
    ])
    assert revisions == {
        "refcoco": "aaa", "refcocoplus": "bbb", "refcocog": "ccc"
    }
    with pytest.raises(ExtractionError, match="immutable"):
        parse_ref_revisions([
            "refcoco=main", "refcocoplus=bbb", "refcocog=ccc"
        ])
    with pytest.raises(ExtractionError, match="missing"):
        parse_ref_revisions(["refcoco=aaa"])

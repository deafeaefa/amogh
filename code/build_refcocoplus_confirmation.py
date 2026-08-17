#!/usr/bin/env python3
"""Freeze full RefCOCO+ testA/testB manifests for untouched confirmation.

The outputs are created once, before recovery-checkpoint selection, and are
image-disjoint from every recovery training/development row. Existing outputs
are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from datasets import load_dataset


SOURCE_ARROW_SHA256 = {
    "test": "cca834e4b0c880c4ceac39a5bf48e017ee7eefeb5ce50d869fa5deccd6ff6d75",
    "testB": "00861b22908f8f0e897784548b0b510344ea351fab65734ef641afb14524fff4",
}
EXPECTED = {
    "test": {"refs": 1_975, "expressions": 5_726, "images": 750, "label": "testA"},
    "testB": {"refs": 1_798, "expressions": 4_889, "images": 750, "label": "testB"},
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError(f"expected a JSON list: {path}")
    return rows


def source_arrow(cache: Path, split: str) -> Path:
    matches = sorted(
        cache.glob(
            "datasets/jxu124___refcocoplus/default/0.0.0/*/"
            f"refcocoplus-{split}.arrow"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one cached RefCOCO+ {split} Arrow file: {matches}")
    path = matches[0]
    actual = sha256_file(path)
    if actual != SOURCE_ARROW_SHA256[split]:
        raise RuntimeError(
            f"RefCOCO+ {split} source hash changed: {actual} != {SOURCE_ARROW_SHA256[split]}"
        )
    return path


def parse_json_field(value: object, name: str) -> dict:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"{name} must decode to an object")
    return value


def manifest_rows(split_rows: object, split: str) -> list[dict]:
    expected = EXPECTED[split]
    if len(split_rows) != expected["refs"]:
        raise RuntimeError(
            f"RefCOCO+ {split} ref count changed: {len(split_rows)} != {expected['refs']}"
        )
    rows: list[dict] = []
    for reference in split_rows:
        info = parse_json_field(reference["raw_image_info"], "raw_image_info")
        annotation = parse_json_field(reference["raw_anns"], "raw_anns")
        image_id = int(reference["image_id"])
        width = int(info["width"])
        height = int(info["height"])
        bbox = [float(value) for value in annotation["bbox"]]
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"invalid RefCOCO+ geometry for ref {reference['ref_id']}")
        for sentence in reference["sentences"]:
            expression = sentence["sent"] if isinstance(sentence, dict) else sentence
            if not isinstance(expression, str) or not expression.strip():
                raise ValueError(f"empty expression for ref {reference['ref_id']}")
            index = len(rows)
            rows.append(
                {
                    "uid": f"refcocoplus_{expected['label']}:{index:05d}",
                    "dataset": "refcocoplus",
                    "source": "refcocoplus",
                    "task": "rec",
                    "split": expected["label"],
                    "ref_id": int(reference["ref_id"]),
                    "image_id": image_id,
                    "file_name": f"COCO_train2014_{image_id:012d}.jpg",
                    "expression": expression,
                    "bbox_xywh": bbox,
                    "width": width,
                    "height": height,
                    "relative_area": bbox[2] * bbox[3] / (width * height),
                }
            )
    if len(rows) != expected["expressions"]:
        raise RuntimeError(
            f"RefCOCO+ {split} expression count changed: "
            f"{len(rows)} != {expected['expressions']}"
        )
    if len({row["uid"] for row in rows}) != len(rows):
        raise RuntimeError(f"duplicate RefCOCO+ {split} UIDs")
    if len({row["image_id"] for row in rows}) != expected["images"]:
        raise RuntimeError(f"RefCOCO+ {split} image count changed")
    return rows


def write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    data_root = Path(os.environ["GCQ_DATA"])
    cache_root = Path(os.environ["HF_HOME"])
    subset_root = data_root / "subsets"
    paths = {
        "test": subset_root / "refcocoplus_testA_confirm_full.json",
        "testB": subset_root / "refcocoplus_testB_confirm_full.json",
    }
    meta_path = subset_root / "refcocoplus_confirmation.meta.json"
    for path in (*paths.values(), meta_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen confirmation artifact: {path}")

    source_paths = {split: source_arrow(cache_root, split) for split in EXPECTED}
    dataset = load_dataset("jxu124/refcocoplus")
    manifests = {split: manifest_rows(dataset[split], split) for split in EXPECTED}

    forbidden_paths = (
        subset_root / "recovery_train_vqa_replay_12k.json",
        subset_root / "recovery_dev_1k.json",
        subset_root / "dprobe_refcoco_train_512.json",
    )
    forbidden_images = {
        int(row["image_id"])
        for path in forbidden_paths
        for row in load_rows(path)
    }
    confirmation_images = {
        int(row["image_id"])
        for rows in manifests.values()
        for row in rows
    }
    overlap = confirmation_images & forbidden_images
    if overlap:
        raise RuntimeError(f"confirmation images overlap recovery/profiling data: {len(overlap)}")

    manifest_bytes = {split: canonical_json(rows) for split, rows in manifests.items()}
    metadata = {
        "schema_version": 1,
        "role": "untouched grounding confirmation frozen before recovery checkpoint selection",
        "dataset": "jxu124/refcocoplus",
        "splits": {
            EXPECTED[split]["label"]: {
                "manifest": paths[split].name,
                "manifest_sha256": sha256_bytes(manifest_bytes[split]),
                "source_arrow": str(source_paths[split]),
                "source_arrow_sha256": SOURCE_ARROW_SHA256[split],
                "references": EXPECTED[split]["refs"],
                "expressions": len(manifests[split]),
                "images": len({row["image_id"] for row in manifests[split]}),
                "ordered_uid_sha256": sha256_bytes(
                    "".join(f"{row['uid']}\n" for row in manifests[split]).encode("utf-8")
                ),
            }
            for split in EXPECTED
        },
        "unique_confirmation_images": len(confirmation_images),
        "forbidden_image_overlap": len(overlap),
        "forbidden_manifests": {
            path.name: sha256_file(path) for path in forbidden_paths
        },
    }
    metadata_bytes = canonical_json(metadata)
    for split, path in paths.items():
        write_exclusive(path, manifest_bytes[split])
    write_exclusive(meta_path, metadata_bytes)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"metadata_sha256={sha256_bytes(metadata_bytes)}")


if __name__ == "__main__":
    main()

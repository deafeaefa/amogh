"""Build leak-free data manifests for GCQ recovery training.

The default pilot contains exactly 8,000 training examples:
  * 4,500 referring-expression examples (1,500 per RefCOCO variant),
  * 1,500 size-balanced COCO category-grounding examples, and
  * 2,000 COCO caption-replay examples.

All images come from COCO train2014.  We exclude every image used by a
RefCOCO/+/g validation or test split and every image in the allocation probe.
A separate 1,000-example recovery-development set is reserved by image before
the training examples are sampled.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


VARIANTS = ("refcoco", "refcocoplus", "refcocog")
NONTRAIN_SPLITS = {
    "refcoco": ("validation", "test", "testB"),
    "refcocoplus": ("validation", "test", "testB"),
    "refcocog": ("validation", "test"),
}
COCO_SIZE_COUNTS = {"small": 500, "medium": 500, "large": 500}


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_json_field(value):
    return json.loads(value) if isinstance(value, str) else value


def xyxy_to_xywh(box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = (float(x) for x in box)
    return [x1, y1, x2 - x1, y2 - y1]


def normalize_box_1000(box_xywh: Iterable[float], width: int, height: int) -> list[int]:
    x, y, w, h = (float(v) for v in box_xywh)
    xyxy = (x, y, x + w, y + h)
    dims = (width, height, width, height)
    out = [max(0, min(1000, round(1000 * v / d))) for v, d in zip(xyxy, dims)]
    if not (out[0] < out[2] and out[1] < out[3]):
        raise ValueError(f"box collapsed after normalization: {box_xywh} in {width}x{height} -> {out}")
    return out


def bbox_answer(box_xywh: Iterable[float], width: int, height: int) -> str:
    return json.dumps({"bbox_2d": normalize_box_1000(box_xywh, width, height)}, separators=(",", ": "))


def area_quartile_bounds(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 4:
        raise ValueError("need at least four values for quartiles")
    ordered = sorted(values)
    return tuple(ordered[round((len(ordered) - 1) * q)] for q in (0.25, 0.50, 0.75))


def quartile(value: float, bounds: tuple[float, float, float]) -> int:
    if value <= bounds[0]:
        return 0
    if value <= bounds[1]:
        return 1
    if value <= bounds[2]:
        return 2
    return 3


def coco_size(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def choose_unique_images(candidates: list[dict], count: int, used_images: set[int], rng: random.Random) -> list[dict]:
    candidates = candidates[:]
    rng.shuffle(candidates)
    selected = []
    for item in candidates:
        if item["image_id"] in used_images:
            continue
        used_images.add(item["image_id"])
        selected.append(item)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"only found {len(selected)}/{count} candidates with unique images")


def ref_candidates(ds, source: str, excluded: set[int], seed: int) -> tuple[dict[int, list[dict]], tuple[float, float, float]]:
    rng = random.Random(seed)
    raw = []
    for row in ds["train"]:
        image_id = int(row["image_id"])
        if image_id in excluded:
            continue
        info = parse_json_field(row["raw_image_info"])
        box_xywh = xyxy_to_xywh(row["bbox"])
        rel_area = box_xywh[2] * box_xywh[3] / (int(info["width"]) * int(info["height"]))
        expressions = [s["sent"] if isinstance(s, dict) else str(s) for s in row["sentences"]]
        expression = expressions[rng.randrange(len(expressions))].strip()
        raw.append({
            "source": source,
            "task": "rec",
            "image_id": image_id,
            "file_name": f"COCO_train2014_{image_id:012d}.jpg",
            "width": int(info["width"]),
            "height": int(info["height"]),
            "expression": expression,
            "bbox_xywh": box_xywh,
            "relative_area": rel_area,
            "ref_id": int(row["ref_id"]),
            "prompt": f"Locate the {expression}, output its bbox_2d in JSON.",
            "answer": bbox_answer(box_xywh, int(info["width"]), int(info["height"])),
        })
    bounds = area_quartile_bounds([x["relative_area"] for x in raw])
    bins = {q: [] for q in range(4)}
    for item in raw:
        q = quartile(item["relative_area"], bounds)
        item["area_quartile"] = q + 1
        bins[q].append(item)
    return bins, bounds


def load_coco_annotations(annotation_zip: Path):
    with zipfile.ZipFile(annotation_zip) as zf:
        with zf.open("annotations/instances_train2014.json") as f:
            instances = json.load(io.TextIOWrapper(f))
        with zf.open("annotations/captions_train2014.json") as f:
            captions = json.load(io.TextIOWrapper(f))
    return instances, captions


def coco_candidates(instances: dict, excluded: set[int]) -> dict[str, list[dict]]:
    images = {int(x["id"]): x for x in instances["images"]}
    categories = {int(x["id"]): x["name"] for x in instances["categories"]}
    counts = Counter(
        (int(a["image_id"]), int(a["category_id"]))
        for a in instances["annotations"]
        if not a.get("iscrowd", 0)
    )
    bins = {k: [] for k in COCO_SIZE_COUNTS}
    for ann in instances["annotations"]:
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        if image_id in excluded or ann.get("iscrowd", 0) or counts[(image_id, category_id)] != 1:
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        if w <= 0 or h <= 0:
            continue
        info = images[image_id]
        label = categories[category_id]
        size = coco_size(float(ann.get("area", w * h)))
        bins[size].append({
            "source": "coco_detection",
            "task": "coco_grounding",
            "image_id": image_id,
            "file_name": info["file_name"],
            "width": int(info["width"]),
            "height": int(info["height"]),
            "expression": label,
            "category_id": category_id,
            "category": label,
            "size_bucket": size,
            "bbox_xywh": [x, y, w, h],
            "relative_area": w * h / (int(info["width"]) * int(info["height"])),
            "prompt": f"Locate the {label}, output its bbox_2d in JSON.",
            "answer": bbox_answer([x, y, w, h], int(info["width"]), int(info["height"])),
        })
    return bins


def cap_categories(candidates: list[dict], per_category: int, rng: random.Random) -> list[dict]:
    by_category = defaultdict(list)
    for item in candidates:
        by_category[item["category_id"]].append(item)
    out = []
    for category_id in sorted(by_category):
        group = by_category[category_id]
        rng.shuffle(group)
        out.extend(group[:per_category])
    return out


def assign_uids(records: list[dict], prefix: str) -> None:
    for i, record in enumerate(records):
        record["uid"] = f"{prefix}:{i:05d}"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("GCQ_DATA", "data"), "subsets"))
    ap.add_argument("--annotation-zip", default=os.path.join(os.environ.get("GCQ_DATA", "data"), "coco_ann", "ann2014.zip"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    datasets = {name: load_dataset(f"jxu124/{name}") for name in VARIANTS}
    forbidden = set()
    for name, ds in datasets.items():
        for split in NONTRAIN_SPLITS[name]:
            forbidden.update(int(x) for x in ds[split]["image_id"])

    probe_path = Path(args.out_dir) / "dprobe_refcoco_train_512.json"
    with open(probe_path) as f:
        probe_images = {int(x["image_id"]) for x in json.load(f)}
    excluded = forbidden | probe_images

    ref_bins = {}
    quartile_metadata = {}
    for i, name in enumerate(VARIANTS):
        ref_bins[name], quartile_metadata[name] = ref_candidates(datasets[name], name, excluded, args.seed + i)

    instances, captions = load_coco_annotations(Path(args.annotation_zip))
    detection_bins = coco_candidates(instances, excluded)

    used_images: set[int] = set()
    dev = []
    # Recovery-dev: 750 referring expressions (roughly balanced quartiles) and
    # 250 COCO boxes (roughly balanced absolute sizes), all image-disjoint.
    for source in VARIANTS:
        for q, count in enumerate((63, 63, 62, 62)):
            dev.extend(choose_unique_images(ref_bins[source][q], count, used_images, rng))
    for size, count in zip(("small", "medium", "large"), (84, 83, 83)):
        pool = cap_categories(detection_bins[size], per_category=10, rng=rng)
        dev.extend(choose_unique_images(pool, count, used_images, rng))
    assert len(dev) == 1000

    grounding = []
    for source in VARIANTS:
        for q in range(4):
            grounding.extend(choose_unique_images(ref_bins[source][q], 375, used_images, rng))
    for size, count in COCO_SIZE_COUNTS.items():
        pool = cap_categories(detection_bins[size], per_category=10, rng=rng)
        grounding.extend(choose_unique_images(pool, count, used_images, rng))
    assert len(grounding) == 6000

    captions_by_image = defaultdict(list)
    for ann in captions["annotations"]:
        captions_by_image[int(ann["image_id"])].append(str(ann["caption"]).strip())
    caption_replay = []
    by_source = defaultdict(list)
    for item in grounding:
        by_source[item["source"]].append(item)
    for source in (*VARIANTS, "coco_detection"):
        pool = [x for x in by_source[source] if captions_by_image[x["image_id"]]]
        rng.shuffle(pool)
        if len(pool) < 500:
            raise RuntimeError(f"not enough captioned images for {source}: {len(pool)}")
        for base in pool[:500]:
            available = captions_by_image[base["image_id"]]
            answer = available[rng.randrange(len(available))]
            caption_replay.append({
                "source": "coco_captions",
                "replay_group": source,
                "task": "caption",
                "image_id": base["image_id"],
                "file_name": base["file_name"],
                "width": base["width"],
                "height": base["height"],
                "prompt": "Describe this image briefly.",
                "answer": answer,
            })
    assert len(caption_replay) == 2000

    train = grounding + caption_replay
    rng.shuffle(train)
    rng.shuffle(dev)
    assign_uids(train, "recovery_train")
    assign_uids(dev, "recovery_dev")

    # Hard data-hygiene and composition assertions.
    assert len(train) == 8000 and len({x["uid"] for x in train}) == 8000
    assert not ({x["image_id"] for x in grounding} & excluded)
    assert len({x["image_id"] for x in grounding}) == 6000
    assert not ({x["image_id"] for x in grounding} & {x["image_id"] for x in dev})
    assert {x["image_id"] for x in caption_replay} <= {x["image_id"] for x in grounding}
    assert Counter(x["source"] for x in grounding) == Counter({
        "refcoco": 1500, "refcocoplus": 1500, "refcocog": 1500, "coco_detection": 1500
    })
    assert Counter(x.get("size_bucket") for x in grounding if x["source"] == "coco_detection") == Counter(COCO_SIZE_COUNTS)
    for source in VARIANTS:
        assert Counter(x["area_quartile"] for x in grounding if x["source"] == source) == Counter({1: 375, 2: 375, 3: 375, 4: 375})

    out_dir = Path(args.out_dir)
    train_path = out_dir / "recovery_train_8k.json"
    dev_path = out_dir / "recovery_dev_1k.json"
    write_json(train_path, train)
    write_json(dev_path, dev)
    metadata = {
        "schema_version": 1,
        "seed": args.seed,
        "train_examples": len(train),
        "train_grounding": len(grounding),
        "train_caption_replay": len(caption_replay),
        "train_unique_grounding_images": len({x["image_id"] for x in grounding}),
        "dev_examples": len(dev),
        "forbidden_eval_images": len(forbidden),
        "excluded_probe_images": len(probe_images),
        "quartile_bounds": {k: list(v) for k, v in quartile_metadata.items()},
        "train_sha256": sha256_file(train_path),
        "dev_sha256": sha256_file(dev_path),
    }
    write_json(out_dir / "recovery_data_manifest.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

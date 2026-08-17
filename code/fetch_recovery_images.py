"""Download and validate the COCO train2014 images selected for recovery."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


# SCC's TLS proxy presents a certificate that does not match the COCO hostname;
# the public S3 endpoint also serves the immutable files over HTTP.
COCO_BASE = "http://images.cocodataset.org/train2014/"


def valid_image(path: Path, width: int, height: int) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.size == (width, height)
    except (OSError, ValueError):
        return False


def fetch_one(item: dict, image_dir: Path, retries: int) -> tuple[str, str]:
    name = item["file_name"]
    target = image_dir / name
    if target.exists() and valid_image(target, int(item["width"]), int(item["height"])):
        return name, "cached"
    url = COCO_BASE + name
    last_error = None
    for _ in range(retries):
        temp_name = None
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=image_dir, prefix=name + ".", suffix=".tmp", delete=False) as f:
                    temp_name = f.name
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            temp_path = Path(temp_name)
            if not valid_image(temp_path, int(item["width"]), int(item["height"])):
                raise ValueError(f"downloaded image has invalid dimensions: {name}")
            os.replace(temp_path, target)
            return name, "downloaded"
        except Exception as exc:  # network errors vary by Python/platform
            last_error = exc
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)
    return name, f"ERROR: {last_error}"


def main() -> None:
    ap = argparse.ArgumentParser()
    default_subsets = os.path.join(os.environ.get("GCQ_DATA", "data"), "subsets")
    ap.add_argument("--manifests", nargs="+", default=[
        os.path.join(default_subsets, "recovery_train_8k.json"),
        os.path.join(default_subsets, "recovery_dev_1k.json"),
    ])
    ap.add_argument("--image-dir", default=os.path.join(os.environ.get("GCQ_DATA", "data"), "images", "train2014"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    selected = {}
    for manifest in args.manifests:
        with open(manifest) as f:
            for item in json.load(f):
                prior = selected.setdefault(item["image_id"], item)
                if (prior["file_name"], prior["width"], prior["height"]) != (
                    item["file_name"], item["width"], item["height"]
                ):
                    raise ValueError(f"conflicting metadata for image {item['image_id']}")

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    counts = {"cached": 0, "downloaded": 0, "error": 0}
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, item, image_dir, args.retries) for item in selected.values()]
        for done, future in enumerate(as_completed(futures), 1):
            name, status = future.result()
            if status.startswith("ERROR"):
                counts["error"] += 1
                errors.append((name, status))
            else:
                counts[status] += 1
            if done % 100 == 0 or done == len(futures):
                print(f"{done}/{len(futures)} cached={counts['cached']} downloaded={counts['downloaded']} errors={counts['error']}", flush=True)

    if errors:
        for name, error in errors[:20]:
            print(f"{name}: {error}")
        raise SystemExit(f"failed to fetch {len(errors)} images")
    print(f"validated {len(selected)} unique recovery images")


if __name__ == "__main__":
    main()

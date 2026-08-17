#!/usr/bin/env python3
"""Safely fetch only missing COCO val2014 images for fresh VQA confirmation.

Existing destination files are never overwritten.  Every existing and newly
downloaded file is decoded with Pillow; downloads additionally honor the HTTP
Content-Length when supplied and are published atomically only after validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image


EXPECTED_MANIFEST_COUNT = 5_000
EXPECTED_IMAGE_COUNT = 4_571
EXPECTED_MANIFEST_SHA256 = (
    "416aea5f9dd6f4a4cfa7061c3b5ca0af88647e49d5f17ed00a8d83885ea21038"
)
COCO_VAL_RE = re.compile(r"^COCO_val2014_(\d{12})\.jpg$")
# The COCO image host currently presents a certificate whose hostname does not
# cover images.cocodataset.org. The project's existing verified fetch path uses
# the official HTTP endpoint, so use the same endpoint here.
BASE_URL = "http://images.cocodataset.org/val2014"
MAX_IMAGE_BYTES = 64 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen fresh-VQA manifest: {path}")
    digest = sha256_file(path)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"manifest hash mismatch: expected {EXPECTED_MANIFEST_SHA256}, found {digest}"
        )
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or len(rows) != EXPECTED_MANIFEST_COUNT:
        raise ValueError(f"manifest must contain exactly {EXPECTED_MANIFEST_COUNT} rows")

    by_image: dict[int, str] = {}
    question_ids: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"manifest[{index}] must be an object")
        qid = row.get("question_id")
        image_id = row.get("image_id")
        filename = row.get("file_name")
        if type(qid) is not int or type(image_id) is not int:
            raise TypeError(f"manifest[{index}] IDs must be integers")
        if qid in question_ids:
            raise ValueError(f"duplicate question_id {qid} in manifest")
        question_ids.add(qid)
        if not isinstance(filename, str):
            raise TypeError(f"manifest[{index}].file_name must be a string")
        match = COCO_VAL_RE.fullmatch(filename)
        if match is None or int(match.group(1)) != image_id:
            raise ValueError(f"manifest[{index}] has a noncanonical or mismatched filename")
        previous = by_image.setdefault(image_id, filename)
        if previous != filename:
            raise ValueError(f"image_id {image_id} maps to multiple filenames")
    if len(by_image) != EXPECTED_IMAGE_COUNT:
        raise ValueError(f"expected {EXPECTED_IMAGE_COUNT} unique images, found {len(by_image)}")
    return [by_image[image_id] for image_id in sorted(by_image)]


def validate_image(path: Path) -> tuple[int, int, int]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_IMAGE_BYTES:
        raise ValueError(f"invalid image byte size {size}: {path}")
    with Image.open(path) as image:
        image_format = image.format
        width, height = image.size
        image.verify()
    if image_format != "JPEG" or width <= 0 or height <= 0:
        raise ValueError(
            f"expected a positive-size JPEG, found format={image_format!r} "
            f"size={width}x{height}: {path}"
        )
    # verify() checks structure without decoding pixels.  Reopen and load to
    # catch truncated/corrupt streams that only fail during full decoding.
    with Image.open(path) as image:
        image.load()
        if image.size != (width, height):
            raise ValueError(f"image dimensions changed across validation: {path}")
    return size, width, height


def download_once(filename: str, destination: Path, timeout: float) -> tuple[int, int, int]:
    url = f"{BASE_URL}/{quote(filename, safe='')}"
    request = Request(url, headers={"User-Agent": "GCQ-fresh-VQA-fetcher/1.0"})
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".part", dir=destination
    )
    temporary_path = Path(temporary_name)
    try:
        received = 0
        with os.fdopen(descriptor, "wb") as output, urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status != 200:
                raise ValueError(f"HTTP status {status} for {url}")
            content_type = response.headers.get_content_type()
            if content_type not in {"image/jpeg", "image/jpg", "application/octet-stream"}:
                raise ValueError(f"unexpected Content-Type {content_type!r} for {url}")
            length_header = response.headers.get("Content-Length")
            expected_length = int(length_header) if length_header is not None else None
            if expected_length is not None and not 0 < expected_length <= MAX_IMAGE_BYTES:
                raise ValueError(f"invalid Content-Length {expected_length} for {url}")
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                received += len(chunk)
                if received > MAX_IMAGE_BYTES:
                    raise ValueError(f"download exceeds {MAX_IMAGE_BYTES} bytes: {url}")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if expected_length is not None and received != expected_length:
            raise ValueError(
                f"Content-Length mismatch for {url}: expected {expected_length}, got {received}"
            )
        details = validate_image(temporary_path)
        final_path = destination / filename
        try:
            # Linking is an atomic, no-replace publication on this filesystem.
            # It gives stronger no-overwrite behavior than os.rename/os.replace.
            os.link(temporary_path, final_path)
        except FileExistsError:
            # Another process won the race.  Keep its file, never replace it,
            # but require it to pass the same validation.
            return validate_image(final_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return details
    except Exception:
        try:
            os.close(descriptor)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise
        temporary_path.unlink(missing_ok=True)
        raise


def fetch_with_retries(
    filename: str, destination: Path, timeout: float, retries: int
) -> tuple[str, tuple[int, int, int]]:
    final_path = destination / filename
    if final_path.exists():
        return filename, validate_image(final_path)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return filename, download_once(filename, destination, timeout)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if final_path.exists():
                return filename, validate_image(final_path)
            if attempt < retries:
                time.sleep(min(8.0, 0.5 * (2**attempt)))
    assert last_error is not None
    raise RuntimeError(f"failed to download {filename} after {retries + 1} attempts") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = os.environ.get("GCQ_DATA")
    default_manifest = (
        Path(default_data) / "subsets" / "vqa_fresh_confirm_5k.json"
        if default_data
        else None
    )
    default_destination = Path(default_data) / "images" / "val2014" if default_data else None
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--destination", type=Path, default=default_destination)
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads (1-32)")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout seconds")
    parser.add_argument("--retries", type=int, default=3, help="retries after the first attempt (0-8)")
    args = parser.parse_args()
    if args.manifest is None or args.destination is None:
        parser.error("--manifest and --destination are required when GCQ_DATA is unset")
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    if not 1.0 <= args.timeout <= 300.0:
        parser.error("--timeout must be between 1 and 300 seconds")
    if not 0 <= args.retries <= 8:
        parser.error("--retries must be between 0 and 8")
    return args


def main() -> None:
    args = parse_args()
    filenames = load_manifest(args.manifest.resolve())
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    existing = 0
    for filename in filenames:
        path = destination / filename
        if path.exists():
            validate_image(path)
            existing += 1
        else:
            missing.append(filename)
    print(
        f"validated {existing} existing images; "
        f"{len(missing)} of {len(filenames)} images need download"
    )
    if not missing:
        return

    failures: list[tuple[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_with_retries,
                filename,
                destination,
                args.timeout,
                args.retries,
            ): filename
            for filename in missing
        }
        for future in as_completed(futures):
            filename = futures[future]
            try:
                future.result()
                completed += 1
                if completed % 100 == 0 or completed == len(missing):
                    print(f"downloaded and validated {completed}/{len(missing)}")
            except Exception as exc:  # Finish independent downloads, then fail loudly.
                failures.append((filename, str(exc)))

    if failures:
        details = "\n".join(f"  {name}: {error}" for name, error in sorted(failures))
        raise RuntimeError(f"{len(failures)} image downloads failed:\n{details}")

    for filename in filenames:
        validate_image(destination / filename)
    print(f"all {len(filenames)} frozen fresh-VQA images are present and valid")


if __name__ == "__main__":
    main()

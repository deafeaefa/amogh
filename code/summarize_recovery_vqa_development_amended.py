"""Run the frozen development summarizer with one scoped correctness repair.

The launch-frozen summarizer accidentally computed its paired parse-failure
delta over all 1,000 recovery-development rows, then compared it with the
predeclared *primary REC* aggregate over 750 rows.  Every other grounding
gate and paired contrast already uses the primary REC subgroup.  This wrapper
leaves the frozen summarizer and every selection rule untouched, but filters
that one paired parse-failure calculation to ``task == "rec"``.

The frozen module still validates its own launch-time SHA-256, all inputs,
runtime provenance, outputs, metrics, gates, bootstraps, and tie-breaks.  The
companion ``development_summarizer_amendment.json`` records this repair and
the wrapper hash before the amended summarizer is run.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path

import summarize_recovery_vqa_development as frozen
from recovery_utils import sha256_file


_frozen_paired_image_delta = frozen.paired_image_delta


def paired_image_delta_primary_rec(
    reference: list[dict],
    candidate: list[dict],
    image_by_uid: Mapping[str, str],
    field: str = "score",
    resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Use the primary REC subgroup for the grounding parse-failure contrast."""
    if field != "parse_fail":
        return _frozen_paired_image_delta(
            reference,
            candidate,
            dict(image_by_uid),
            field=field,
            resamples=resamples,
            seed=seed,
        )

    primary_reference = [row for row in reference if row.get("task") == "rec"]
    primary_candidate = [row for row in candidate if row.get("task") == "rec"]
    frozen.require(
        len(primary_reference) == frozen.PRIMARY_REC_EXAMPLES,
        "amended parse-failure reference does not contain exactly 750 primary REC rows",
    )
    frozen.require(
        len(primary_candidate) == frozen.PRIMARY_REC_EXAMPLES,
        "amended parse-failure candidate does not contain exactly 750 primary REC rows",
    )
    primary_images = {
        row["uid"]: str(image_by_uid[row["uid"]]) for row in primary_reference
    }
    return _frozen_paired_image_delta(
        primary_reference,
        primary_candidate,
        primary_images,
        field=field,
        resamples=resamples,
        seed=seed,
    )


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    amendment_path = code_dir / "development_summarizer_amendment.json"
    with open(amendment_path) as stream:
        amendment = json.load(stream)
    frozen.require(amendment.get("schema_version") == 1, "invalid amendment schema")
    frozen.require(
        amendment.get("amendment_id")
        == "development-primary-parse-pairing-20260815",
        "unexpected development amendment identifier",
    )
    launch_path = (
        Path(os.environ["GCQ_RUNS"])
        / "recovery_vqa_replay"
        / "development_launch_manifest.json"
    )
    frozen_path = code_dir / "summarize_recovery_vqa_development.py"
    for path_key, hash_key, path in (
        (
            "original_development_launch_manifest",
            "original_development_launch_manifest_sha256",
            launch_path,
        ),
        (
            "original_frozen_summarizer",
            "original_frozen_summarizer_sha256",
            frozen_path,
        ),
        ("amended_wrapper", "amended_wrapper_sha256", Path(__file__).resolve()),
    ):
        frozen.require(
            amendment.get(path_key) == str(path),
            f"development amendment {path_key} mismatch",
        )
        frozen.require(
            amendment.get(hash_key) == sha256_file(path),
            f"development amendment {hash_key} mismatch",
        )
    frozen.paired_image_delta = paired_image_delta_primary_rec
    frozen.main()


if __name__ == "__main__":
    main()

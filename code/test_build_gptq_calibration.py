import copy
import random

import pytest

from build_gptq_calibration import (
    build_calibration_manifest,
    canonical_sha256,
    validate_calibration_manifest,
)


def _build(rows):
    return build_calibration_manifest(
        rows,
        lambda text: [ord(character) for character in text],
        eos_token_id=0,
        samples=3,
        sequence_length=4,
        seed=7,
        dataset_revision="a" * 40,
        tokenizer_class="ToyTokenizer",
    )


def test_calibration_is_equal_length_padding_free_hashed_and_order_invariant():
    rows = [
        {"row_id": "one", "text": "abcdef"},
        {"row_id": "two", "text": "ghijkl"},
        {"row_id": "three", "text": "mnopqr"},
    ]
    first = _build(rows)
    random.Random(9).shuffle(rows)
    second = _build(rows)
    assert first == second
    assert len(first["input_ids"]) == 3
    assert {len(row) for row in first["input_ids"]} == {4}
    assert first["padding"] is False
    assert first["input_ids_sha256"] == canonical_sha256(first["input_ids"])
    validate_calibration_manifest(first)


def test_calibration_rejects_moving_revision_short_stream_and_tampering():
    rows = [{"row_id": "one", "text": "abc"}]
    with pytest.raises(ValueError, match="immutable"):
        build_calibration_manifest(
            rows, lambda text: [1, 2, 3], eos_token_id=0,
            samples=1, sequence_length=2, dataset_revision="main",
        )
    with pytest.raises(ValueError, match="only encoded"):
        build_calibration_manifest(
            rows, lambda text: [1], eos_token_id=0,
            samples=2, sequence_length=2, dataset_revision="a" * 40,
        )
    manifest = _build([
        {"row_id": "one", "text": "abcdef"},
        {"row_id": "two", "text": "ghijkl"},
    ])
    tampered = copy.deepcopy(manifest)
    tampered["input_ids"][0][0] += 1
    with pytest.raises(ValueError, match="input-ID hash"):
        validate_calibration_manifest(tampered)

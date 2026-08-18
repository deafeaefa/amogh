import hashlib

from allocate_gcq_beam import _sha256_json as allocator_sha256
from eval_gcq_plan import artifact_canonical_sha256, canonical_sha256 as state_sha256
from freeze_gcq_upgrade import canonical_sha256 as protocol_sha256
from gcq_profile_metrics import canonical_json_bytes, canonical_sha256 as profile_sha256
from gptq_candidates import canonical_sha256 as candidate_sha256


def test_artifact_and_allocator_hash_domains_are_explicitly_consistent():
    value = {"z": [2, 1], "a": "value"}
    artifact = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    assert artifact == protocol_sha256(value)
    assert artifact == profile_sha256(value)
    assert artifact == candidate_sha256(value)
    assert artifact == artifact_canonical_sha256(value)

    # Allocator state IDs and fingerprints predate the newline-terminated
    # artifact convention.  Their separate no-newline domain is intentional.
    assert allocator_sha256(value) == state_sha256(value)
    assert allocator_sha256(value) != artifact

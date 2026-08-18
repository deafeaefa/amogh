import copy
import json
from pathlib import Path

import pytest
import torch

from gcq_profile_metrics import canonical_sha256
from profile_gcq_projections import (
    BASE_MODEL,
    BASE_REVISION,
    PROFILE_IMPLEMENTATION_FILES,
    RAW_ARTIFACT_KIND,
    PROFILE_SCHEMA_VERSION,
    ProjectionProfileError,
    build_projection_summary,
    coordinate_kl_from_logits,
    exact_projection_swap,
    full_vocabulary_kl,
    locate_coordinate_layout,
    prediction_position_groups,
    sha256_bytes,
    sha256_file,
    validate_candidate_cache_contract,
    validate_bound_protocol,
    validate_profile_manifest,
    validate_raw_profile_record,
    write_profile_outputs,
)


class PieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, token_ids, skip_special_tokens=False):
        assert len(token_ids) == 1
        return self.pieces[token_ids[0]]


def coordinate_layout():
    pieces = [
        "<s>",
        "assistant\n",
        '{"bbox_',
        '2d": [',
        "10",
        ",",
        "2",
        "0",
        ",",
        "300",
        ",",
        "400",
        "]}",
    ]
    answer = '{"bbox_2d": [10,20,300,400]}'
    return locate_coordinate_layout(PieceTokenizer(pieces), list(range(len(pieces))), answer)


def valid_cache_manifest():
    manifest = {
        "schema_version": 1,
        "artifact_kind": "gcq_packed_gptq_projection_bank",
        "recipe": {
            "base_model": BASE_MODEL,
            "revision": BASE_REVISION,
            "bits": [4, 8],
        },
        "recipe_sha256": "recipe-hash",
        "candidates": [
            {
                "module_name": "model.language_model.layers.0.self_attn.q_proj",
                "delta_bytes": 16,
                "w4": {"qdq_sha256": "4" * 64},
                "w8": {"qdq_sha256": "8" * 64},
            }
        ],
    }
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    return manifest


def raw_record(task="rec", quartile=1, uid="u0"):
    layout = coordinate_layout()
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "artifact_kind": RAW_ARTIFACT_KIND,
        "module_name": "model.language_model.layers.0.self_attn.q_proj",
        "uid": uid,
        "image_id": int(uid.removeprefix("u") or 0),
        "task": task,
        "area_quartile": quartile,
        "manifest_sha256": "m" * 64,
        "cache_manifest_sha256": "c" * 64,
        "context_sha256": "x" * 64,
        **layout,
        "w4_coordinate_token_kl": [[1.0], [2.0, 4.0], [3.0], [4.0]],
        "w8_coordinate_token_kl": [[0.5], [1.0, 2.0], [1.5], [2.0]],
    }


def test_numeric_token_groups_exclude_punctuation_and_use_index_minus_one_logits():
    layout = coordinate_layout()
    assert layout["coordinate_token_indices"] == [[4], [6, 7], [9], [11]]
    assert layout["coordinate_token_ids"] == [[4], [6, 7], [9], [11]]
    assert layout["coordinate_prediction_positions"] == [[3], [5, 6], [8], [10]]
    selected = {
        index for group in layout["coordinate_token_indices"] for index in group
    }
    # bbox_2d's 2 and every standalone comma/bracket token are absent.
    assert selected.isdisjoint({3, 5, 8, 10, 12})

    with pytest.raises(ProjectionProfileError, match="index 0"):
        prediction_position_groups([[0], [2], [3], [4]])


def test_coordinate_kl_reads_prediction_position_not_coordinate_input_position():
    layout = coordinate_layout()
    teacher = torch.zeros(13, 5)
    student = teacher.clone()
    # Input token 4 is a coordinate, but its predicting logit is position 3.
    # A perturbation at 4 must therefore be invisible to the coordinate score.
    student[4, 0] = 10.0
    invisible = coordinate_kl_from_logits(
        teacher, student, layout["coordinate_prediction_positions"]
    )
    assert all(value == pytest.approx(0.0) for group in invisible for value in group)

    student[3, 0] = 10.0
    visible = coordinate_kl_from_logits(
        teacher, student, layout["coordinate_prediction_positions"]
    )
    assert visible[0][0] > 0
    assert all(value == pytest.approx(0.0) for group in visible[1:] for value in group)


def test_full_vocabulary_kl_is_fp32_and_uses_non_target_vocabulary_logits():
    teacher = torch.tensor([[2.0, 1.0, -1.0, 0.5]], dtype=torch.bfloat16)
    student = torch.tensor([[2.0, -3.0, 4.0, 0.5]], dtype=torch.bfloat16)
    result = full_vocabulary_kl(teacher, student)
    teacher_fp32 = teacher.float()
    student_fp32 = student.float()
    log_p = torch.log_softmax(teacher_fp32, dim=-1)
    log_q = torch.log_softmax(student_fp32, dim=-1)
    expected = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected)
    # The first/nominal target-class logit is identical, but changes elsewhere
    # still produce positive KL: this is full-vocabulary, not target-token CE.
    assert result.item() > 0


def test_exact_projection_swap_changes_only_target_and_restores_even_on_error():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.target = torch.nn.Linear(2, 2, bias=False)
            self.other = torch.nn.Linear(2, 2, bias=False)

    model = Toy()
    w4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    w8 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    with torch.no_grad():
        model.target.weight.copy_(w4)
    other_before = model.other.weight.detach().clone()

    with exact_projection_swap(model.target, w4=w4, w8=w8):
        torch.testing.assert_close(model.target.weight, w8)
        torch.testing.assert_close(model.other.weight, other_before)
    assert torch.equal(model.target.weight, w4)
    assert torch.equal(model.other.weight, other_before)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with exact_projection_swap(model.target, w4=w4, w8=w8):
            raise RuntimeError("synthetic failure")
    assert torch.equal(model.target.weight, w4)
    assert torch.equal(model.other.weight, other_before)

    with torch.no_grad():
        model.target.weight.add_(1)
    with pytest.raises(ProjectionProfileError, match="not at its persisted W4"):
        with exact_projection_swap(model.target, w4=w4, w8=w8):
            pass


def test_cache_schema_and_pinned_provenance_fail_closed():
    manifest = valid_cache_manifest()
    contract = validate_candidate_cache_contract(manifest, verified_payloads=True)
    assert contract["manifest_sha256"] == manifest["manifest_content_sha256"]
    assert contract["modules"] == 1

    with pytest.raises(ProjectionProfileError, match="payload hashes"):
        validate_candidate_cache_contract(manifest, verified_payloads=False)

    bad_revision = copy.deepcopy(manifest)
    bad_revision["recipe"]["revision"] = "moving-main"
    unhashed = dict(bad_revision)
    unhashed.pop("manifest_content_sha256")
    bad_revision["manifest_content_sha256"] = canonical_sha256(unhashed)
    with pytest.raises(ProjectionProfileError, match="revision differs"):
        validate_candidate_cache_contract(bad_revision, verified_payloads=True)

    corrupt = copy.deepcopy(manifest)
    corrupt["candidates"][0]["delta_bytes"] = 17
    with pytest.raises(ProjectionProfileError, match="content hash mismatch"):
        validate_candidate_cache_contract(corrupt, verified_payloads=True)


def test_launch_protocol_binds_proxy_and_candidate_bank_files():
    protocol = {
        "status": "launch_frozen",
        "model": {"id": BASE_MODEL, "revision": BASE_REVISION},
        "bound_hashes": {
            "proxy_manifest_sha256": "p" * 64,
            "candidate_bank_manifest_sha256": "b" * 64,
        },
        "implementation_files": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in PROFILE_IMPLEMENTATION_FILES
        },
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    validate_bound_protocol(
        protocol,
        proxy_manifest_file_sha256="p" * 64,
        candidate_bank_file_sha256="b" * 64,
    )
    wrong = copy.deepcopy(protocol)
    wrong["bound_hashes"]["proxy_manifest_sha256"] = "x" * 64
    unhashed = dict(wrong)
    unhashed.pop("protocol_sha256")
    wrong["protocol_sha256"] = canonical_sha256(unhashed)
    with pytest.raises(ProjectionProfileError, match="proxy_manifest_sha256 mismatch"):
        validate_bound_protocol(
            wrong,
            proxy_manifest_file_sha256="p" * 64,
            candidate_bank_file_sha256="b" * 64,
        )


def test_manifest_raw_record_provenance_and_eight_cell_summary():
    manifest = []
    records = []
    index = 0
    for task in ("rec", "coco_grounding"):
        for quartile in range(1, 5):
            manifest.append(
                {
                    "uid": f"u{index}",
                    "image_id": index,
                    "task": task,
                    "area_quartile": quartile,
                    "split": "train",
                    "prompt": "Locate the object, output its bbox_2d in JSON.",
                    "answer": '{"bbox_2d": [10,20,300,400]}',
                }
            )
            records.append(raw_record(task, quartile, f"u{index}"))
            index += 1
    contract = validate_profile_manifest(manifest, expected_rows_per_cell=1)
    assert contract["rows"] == 8
    assert len(contract["cells"]) == 8

    for record in records:
        validate_raw_profile_record(
            record,
            manifest_sha256="m" * 64,
            cache_manifest_sha256="c" * 64,
            context_sha256="x" * 64,
        )
    summary = build_projection_summary(
        records[0]["module_name"],
        records,
        candidate_metadata=valid_cache_manifest()["candidates"][0],
        manifest_sha256="m" * 64,
        cache_manifest_sha256="c" * 64,
        context_sha256="x" * 64,
    )
    assert summary["n"] == 8
    assert summary["n_cells"] == 8
    assert set(summary["cells"]) == {
        f"{task}:q{quartile}"
        for task in ("rec", "coco_grounding")
        for quartile in range(1, 5)
    }
    assert summary["repair_macro"] > 0
    unhashed = dict(summary)
    summary_hash = unhashed.pop("summary_sha256")
    assert summary_hash == canonical_sha256(unhashed)

    wrong_context = copy.deepcopy(records[0])
    wrong_context["context_sha256"] = "wrong"
    with pytest.raises(ProjectionProfileError, match="context_sha256 mismatch"):
        validate_raw_profile_record(
            wrong_context,
            manifest_sha256="m" * 64,
            cache_manifest_sha256="c" * 64,
            context_sha256="x" * 64,
        )
    bad_causal = copy.deepcopy(records[0])
    bad_causal["coordinate_prediction_positions"][0][0] += 1
    with pytest.raises(ProjectionProfileError, match="not causal index-1"):
        validate_raw_profile_record(
            bad_causal,
            manifest_sha256="m" * 64,
            cache_manifest_sha256="c" * 64,
            context_sha256="x" * 64,
        )


def test_profile_outputs_are_canonical_hashed_and_write_once(tmp_path):
    record = raw_record()
    summary = {
        "schema_version": 1,
        "artifact_kind": "gcq_projection_coordinate_kl_summary",
        "module_name": record["module_name"],
    }
    paths, metadata = write_profile_outputs(
        tmp_path,
        tag="toy",
        raw_records=[record],
        summaries=[summary],
        metadata={"schema_version": 1, "artifact_kind": "toy-run"},
    )
    raw_bytes = paths["raw"].read_bytes()
    summaries_bytes = paths["summaries"].read_bytes()
    assert metadata["outputs"]["raw"]["sha256"] == sha256_bytes(raw_bytes)
    assert metadata["outputs"]["summaries"]["sha256"] == sha256_bytes(
        summaries_bytes
    )
    assert json.loads(paths["metadata"].read_text()) == metadata
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_profile_outputs(
            tmp_path,
            tag="toy",
            raw_records=[record],
            summaries=[summary],
            metadata={"schema_version": 1},
        )

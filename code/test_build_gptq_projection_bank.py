import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from build_gptq_calibration import canonical_sha256 as calibration_sha256
from gptq_candidates import GPTQCandidateCache, GPTQRecipe, PROJECTION_ROLES, quantize_candidate_pair
from recovery_utils import BASE_MODEL, BASE_REVISION


MODULE_PATH = Path(__file__).with_name("build_gptq_projection_bank.py")


def test_module_import_does_not_require_transformers(monkeypatch):
    imported = []
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            imported.append(name)
            raise AssertionError("Transformers was imported by the CPU helper module")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("_cpu_only_bank_builder_import", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    assert imported == []


from build_gptq_projection_bank import (  # noqa: E402
    CALIBRATION_SAMPLES,
    EXPECTED_CAPTURE_TOKENS,
    EXPECTED_PROJECTIONS,
    ProjectionBankBuildError,
    SEQUENCE_LENGTH,
    build_calibration_batches,
    capture_layer_hessians,
    install_layer_w4,
    move_candidate_pair_to_cpu,
    persist_candidate_cache,
    validate_and_group_projections,
    validate_calibration_provenance,
    validate_hessian_capture,
    validate_protocol,
)


def valid_protocol():
    return {
        "schema_version": 1,
        "protocol_id": "gcq-projection-gptq-beam-v2",
        "status": "design_frozen_inputs_unbound",
        "bound_hashes": None,
        "model": {
            "id": BASE_MODEL,
            "revision": BASE_REVISION,
            "decoder_layers": 28,
            "projection_suffixes": list(PROJECTION_ROLES),
            "expected_projection_count": 196,
            "expected_decoder_weight_count": 1_409_286_144,
        },
        "quantization": {
            "algorithm": "GPTQ second-order error compensation",
            "candidate_bits": [4, 8],
            "group_size": 128,
            "block_size": 128,
            "percdamp": 0.01,
            "scheme": "symmetric signed absmax, qmax=2^(bits-1)-1, no zero point",
            "scale_dtype": "float16",
            "prefix_policy": "all earlier decoder layers installed from their cached W4 candidates",
            "candidate_pair_rule": (
                "W4 and W8 for one projection use identical pristine BF16 weight "
                "and independent clones of one immutable Hessian"
            ),
            "calibration": {
                "source": "Salesforce/wikitext wikitext-2-raw-v1 train",
                "role": "standard text-only quantizer calibration",
                "examples": 128,
                "sequence_length": 512,
                "selection_seed": 20260817,
                "padding_allowed": False,
                "revision_and_token_ids_must_be_hashed": True,
            },
        },
    }


def _refresh_calibration_hashes(value):
    value["input_ids_sha256"] = calibration_sha256(value["input_ids"])
    value.pop("manifest_content_sha256", None)
    value["manifest_content_sha256"] = calibration_sha256(value)
    return value


def valid_calibration():
    input_ids = [list(range(SEQUENCE_LENGTH)) for _ in range(CALIBRATION_SAMPLES)]
    value = {
        "schema_version": 1,
        "role": "standard_text_gptq_calibration",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "dataset": {
            "id": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "train",
            "revision": "1" * 40,
            "fingerprint": "fixture-fingerprint",
        },
        "selection": {
            "namespace": "gcq-gptq-calibration-v1",
            "seed": 20260817,
            "rule": "fixture",
            "source_rows": [{
                "row_id": "immutable-row-0",
                "text_sha256": "2" * 64,
                "encoded_tokens_including_separator": EXPECTED_CAPTURE_TOKENS,
                "tokens_used_before_cutoff": EXPECTED_CAPTURE_TOKENS,
            }],
        },
        "tokenizer": {
            "model": BASE_MODEL,
            "revision": BASE_REVISION,
            "class": "Qwen2TokenizerFast",
            "add_special_tokens": False,
            "eos_token_id": 151645,
        },
        "samples": CALIBRATION_SAMPLES,
        "sequence_length": SEQUENCE_LENGTH,
        "padding": False,
        "attention_mask": "all ones",
        "input_ids": input_ids,
    }
    return _refresh_calibration_hashes(value)


def test_checked_in_protocol_and_exact_calibration_are_accepted():
    checked_in = __import__("json").loads(Path(__file__).with_name("gcq_upgrade_protocol.json").read_text())
    checked_contract = validate_protocol(checked_in)
    fixture_contract = validate_protocol(valid_protocol())
    assert checked_contract.recipe == fixture_contract.recipe
    summary = validate_calibration_provenance(valid_calibration(), checked_contract)
    assert summary["samples"] == 128
    assert summary["sequence_length"] == 512
    assert summary["padding"] is False
    assert summary["dataset"]["revision"] == "1" * 40


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value["model"].__setitem__("revision", "main"), "40-character"),
        (lambda value: value["model"].__setitem__("id", "Qwen/not-the-model"), "differs"),
        (lambda value: value["quantization"].__setitem__("candidate_bits", [8, 4]), "candidate_bits"),
        (lambda value: value["quantization"]["calibration"].__setitem__("examples", 127), "128 x 512"),
    ],
)
def test_protocol_rejects_unpinned_or_changed_build_contract(mutate, match):
    value = valid_protocol()
    mutate(value)
    with pytest.raises(ProjectionBankBuildError, match=match):
        validate_protocol(value)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda value: (
                value.__setitem__("samples", 127),
                value.__setitem__("input_ids", value["input_ids"][:127]),
            ),
            "exactly 128",
        ),
        (
            lambda value: (
                value.__setitem__("sequence_length", 511),
                value.__setitem__("input_ids", [row[:511] for row in value["input_ids"]]),
            ),
            "exactly 512",
        ),
        (lambda value: value["input_ids"][0].__setitem__(0, True), "invalid token ID"),
        (lambda value: value.__setitem__("padding", True), "padding-free"),
        (lambda value: value["dataset"].__setitem__("revision", "main"), "40-character"),
        (lambda value: value["tokenizer"].__setitem__("revision", "3" * 40), "differs"),
        (lambda value: value["selection"].__setitem__("seed", 7), "differs"),
    ],
)
def test_calibration_rejects_wrong_provenance_or_non_exact_shape(mutate, match):
    value = valid_calibration()
    mutate(value)
    _refresh_calibration_hashes(value)
    with pytest.raises(ProjectionBankBuildError, match=match):
        validate_calibration_provenance(value, validate_protocol(valid_protocol()))


def _tiny_projection_layout():
    projections = {}
    for layer in range(28):
        for role in PROJECTION_ROLES:
            name = f"language_model.layers.{layer}.{role}"
            projections[name] = torch.nn.Linear(1, 1, bias=False)
    return projections


def test_projection_layout_requires_exact_28_by_7_and_role_shapes():
    shapes = {role: (1, 1) for role in PROJECTION_ROLES}
    projections = _tiny_projection_layout()
    grouped = validate_and_group_projections(
        projections,
        expected_weight_count=EXPECTED_PROJECTIONS,
        expected_shapes=shapes,
        expected_dtype=None,
    )
    assert list(grouped) == list(range(28))
    assert all(len(modules) == 7 for modules in grouped.values())
    with pytest.raises(ProjectionBankBuildError, match="expected pristine torch.bfloat16"):
        validate_and_group_projections(
            projections,
            expected_weight_count=EXPECTED_PROJECTIONS,
            expected_shapes=shapes,
        )

    missing = dict(projections)
    missing.pop("language_model.layers.27.mlp.down_proj")
    with pytest.raises(ProjectionBankBuildError, match="expected 196"):
        validate_and_group_projections(
            missing,
            expected_weight_count=None,
            expected_shapes=shapes,
            expected_dtype=None,
        )

    wrong_shape = dict(projections)
    wrong_shape["language_model.layers.0.self_attn.q_proj"] = torch.nn.Linear(2, 1, bias=False)
    with pytest.raises(ProjectionBankBuildError, match=r"shape \(1, 2\); expected \(1, 1\)"):
        validate_and_group_projections(
            wrong_shape,
            expected_weight_count=None,
            expected_shapes=shapes,
            expected_dtype=None,
        )


class TinyCaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 3)
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(3, 2, bias=False) for _ in PROJECTION_ROLES
        ])

    def forward(self, input_ids, attention_mask, use_cache):
        assert torch.all(attention_mask == 1)
        assert use_cache is False
        hidden = self.embedding(input_ids)
        for projection in self.projections:
            projection(hidden)
        return hidden


def test_layer_hessian_capture_counts_every_padding_free_token_and_unhooks():
    torch.manual_seed(10)
    model = TinyCaptureModel()
    modules = [(f"projection-{index}", module) for index, module in enumerate(model.projections)]
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    batches = [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}]
    hessians, counts = capture_layer_hessians(model, modules, batches, expected_tokens=8)
    assert counts == {name: 8 for name, _ in modules}
    expected_x = model.embedding(ids).reshape(-1, 3).detach()
    expected_hessian = expected_x.T @ expected_x
    for hessian in hessians.values():
        torch.testing.assert_close(hessian, expected_hessian)
    assert all(not module._forward_hooks for _, module in modules)

    wrong_counts = dict(counts)
    wrong_counts[modules[0][0]] = 7
    with pytest.raises(ProjectionBankBuildError, match="captured 7"):
        validate_hessian_capture(modules, hessians, wrong_counts, expected_tokens=8)
    wrong_hessians = dict(hessians)
    wrong_hessians[modules[0][0]] = torch.eye(2)
    with pytest.raises(ProjectionBankBuildError, match="shape/dtype"):
        validate_hessian_capture(modules, wrong_hessians, counts, expected_tokens=8)


def test_runtime_batches_are_exact_and_padding_free():
    batches = build_calibration_batches(valid_calibration()["input_ids"], batch_size=17, device="cpu")
    assert sum(batch["input_ids"].shape[0] for batch in batches) == 128
    assert all(batch["input_ids"].shape[1] == 512 for batch in batches)
    assert sum(batch["input_ids"].numel() for batch in batches) == 128 * 512
    assert all(torch.all(batch["attention_mask"] == 1) for batch in batches)
    assert all(batch["input_ids"].dtype == torch.long for batch in batches)


def _positive_hessian(width):
    activation = torch.randn(width * 4, width)
    return activation.T @ activation + torch.eye(width)


def _small_pair(module):
    recipe = GPTQRecipe(group_size=2, block_size=2)
    return move_candidate_pair_to_cpu(
        quantize_candidate_pair(module.weight.detach(), _positive_hessian(3), recipe=recipe)
    ), recipe


def test_matched_pair_moves_to_cpu_and_only_w4_is_installed():
    torch.manual_seed(11)
    module = torch.nn.Linear(3, 2, bias=False)
    name = "language_model.layers.0.self_attn.q_proj"
    pair, _ = _small_pair(module)
    assert pair[4].source_weight_sha256 == pair[8].source_weight_sha256
    assert pair[4].hessian_sha256 == pair[8].hessian_sha256
    assert all(
        tensor.device.type == "cpu"
        for candidate in pair.values()
        for tensor in (candidate.codes, candidate.scales, candidate.qdq)
    )
    trace = install_layer_w4([(name, module)], {name: pair})
    assert torch.equal(module.weight, pair[4].qdq)
    assert trace[0]["module_name"] == name
    assert trace[0]["installed_w4_sha256"] == pair[4].qdq_sha256


def test_packed_cache_is_verified_and_write_once(tmp_path):
    torch.manual_seed(12)
    module = torch.nn.Linear(3, 2, bias=False)
    name = "language_model.layers.0.self_attn.q_proj"
    pair, recipe = _small_pair(module)
    cache = GPTQCandidateCache.build(
        {name: pair},
        recipe=recipe,
        strict_qwen=False,
        provenance={"fixture": True},
    )
    destination = tmp_path / "packed-bank"
    loaded = persist_candidate_cache(cache, destination, expected_candidates=1)
    assert loaded.verified_payloads is True
    assert loaded.names == (name,)
    assert (destination / "manifest.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        persist_candidate_cache(cache, destination, expected_candidates=1)

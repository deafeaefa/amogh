import copy
import json

import pytest
import torch

from gptq_candidates import (
    EXPECTED_PROJECTIONS,
    GPTQCandidateCache,
    GPTQCandidateError,
    GPTQRecipe,
    PROJECTION_ROLES,
    enumerate_qwen_projections,
    gptq_quantize_linear,
    pack_signed_codes,
    quantize_candidate_pair,
    tensor_sha256,
    unpack_signed_codes,
    validate_qwen_projections,
    canonical_sha256,
)


def positive_hessian(width):
    torch.manual_seed(3)
    activations = torch.randn(width * 3, width)
    return activations.T @ activations + torch.eye(width) * 0.1


def small_recipe():
    return GPTQRecipe(group_size=3, block_size=4, percdamp=0.01)


class ToyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 2, bias=False)
        self.k_proj = torch.nn.Linear(2, 2, bias=False)
        self.v_proj = torch.nn.Linear(2, 2, bias=False)
        self.o_proj = torch.nn.Linear(2, 2, bias=False)


class ToyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(2, 2, bias=False)
        self.up_proj = torch.nn.Linear(2, 2, bias=False)
        self.down_proj = torch.nn.Linear(2, 2, bias=False)


class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()


class ToyLanguageModel(torch.nn.Module):
    def __init__(self, layers=28):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyLayer() for _ in range(layers)])


class ToyQwen(torch.nn.Module):
    def __init__(self, layers=28):
        super().__init__()
        self.language_model = ToyLanguageModel(layers)


def test_qwen_projection_enumeration_is_exact_28_by_7():
    model = ToyQwen()
    projections = enumerate_qwen_projections(model, expected_weight_count=None)
    assert len(projections) == EXPECTED_PROJECTIONS
    summary = validate_qwen_projections(projections, expected_weight_count=None)
    assert summary["role_counts"] == {role: 28 for role in PROJECTION_ROLES}
    broken = dict(projections)
    broken.pop(next(iter(broken)))
    with pytest.raises(GPTQCandidateError, match="expected 196"):
        validate_qwen_projections(broken, expected_weight_count=None)


def test_gptq_is_nonmutating_and_pair_order_independent():
    torch.manual_seed(4)
    weight = torch.randn(5, 7)
    hessian = positive_hessian(7)
    weight_before = weight.clone()
    hessian_before = hessian.clone()
    w4_first = gptq_quantize_linear(weight, hessian, bits=4, recipe=small_recipe())
    w8_second = gptq_quantize_linear(weight, hessian, bits=8, recipe=small_recipe())
    w8_first = gptq_quantize_linear(weight, hessian, bits=8, recipe=small_recipe())
    w4_second = gptq_quantize_linear(weight, hessian, bits=4, recipe=small_recipe())
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)
    torch.testing.assert_close(hessian, hessian_before, rtol=0, atol=0)
    assert torch.equal(w4_first.codes, w4_second.codes)
    assert torch.equal(w4_first.scales, w4_second.scales)
    assert torch.equal(w8_first.codes, w8_second.codes)
    assert torch.equal(w8_first.scales, w8_second.scales)
    pair = quantize_candidate_pair(weight, hessian, recipe=small_recipe())
    assert pair[4].source_weight_sha256 == pair[8].source_weight_sha256
    assert pair[4].hessian_sha256 == pair[8].hessian_sha256


@pytest.mark.parametrize("bits", [4, 8])
def test_signed_pack_roundtrip_tail_groups_and_exact_bytes(bits):
    torch.manual_seed(5)
    weight = torch.randn(3, 5)
    candidate = gptq_quantize_linear(
        weight, positive_hessian(5), bits=bits, recipe=small_recipe()
    )
    payload = pack_signed_codes(candidate.codes, bits)
    expected = (candidate.numel * bits + 7) // 8
    assert len(payload) == expected == candidate.logical_code_bytes
    assert torch.equal(
        unpack_signed_codes(payload, bits, candidate.shape), candidate.codes.cpu()
    )
    assert candidate.scales.shape == (3, 2)  # ceil(5 / group_size=3)
    assert candidate.logical_scale_bytes == 12
    torch.testing.assert_close(
        candidate.dequantize(dtype=weight.dtype, device=weight.device),
        candidate.qdq,
        rtol=0,
        atol=0,
    )


def test_dead_columns_and_invalid_hessian_fail_or_quantize_cleanly():
    weight = torch.randn(2, 3)
    hessian = torch.eye(3)
    hessian[1, 1] = 0
    candidate = gptq_quantize_linear(weight, hessian, bits=4, recipe=small_recipe())
    assert torch.count_nonzero(candidate.codes[:, 1]) == 0
    with pytest.raises(GPTQCandidateError, match="shape"):
        gptq_quantize_linear(weight, torch.eye(2), bits=4, recipe=small_recipe())
    invalid = torch.eye(3)
    invalid[0, 0] = -1
    with pytest.raises(GPTQCandidateError, match="nonnegative"):
        gptq_quantize_linear(weight, invalid, bits=4, recipe=small_recipe())


def _two_projection_cache(model):
    recipe = small_recipe()
    names = [
        "language_model.layers.0.self_attn.q_proj",
        "language_model.layers.0.self_attn.k_proj",
    ]
    pairs = {}
    for name in names:
        module = dict(model.named_modules())[name]
        pairs[name] = quantize_candidate_pair(
            module.weight.detach(), positive_hessian(2), recipe=recipe
        )
    return GPTQCandidateCache.build(pairs, recipe=recipe, strict_qwen=False), pairs


def test_packed_cache_composition_restore_order_and_byte_audit(tmp_path):
    torch.manual_seed(8)
    model = ToyQwen(layers=1)
    cache, pairs = _two_projection_cache(model)
    bank_dir = cache.save(tmp_path / "bank")
    loaded = GPTQCandidateCache.load(bank_dir, verify_hashes=True)
    q_name, k_name = loaded.names

    loaded.compose(model, [q_name])
    assert tensor_sha256(dict(model.named_modules())[q_name].weight) == pairs[q_name][8].qdq_sha256
    assert tensor_sha256(dict(model.named_modules())[k_name].weight) == pairs[k_name][4].qdq_sha256
    loaded.compose(model, [k_name], previous_promotions=[q_name])
    assert tensor_sha256(dict(model.named_modules())[q_name].weight) == pairs[q_name][4].qdq_sha256
    assert tensor_sha256(dict(model.named_modules())[k_name].weight) == pairs[k_name][8].qdq_sha256
    loaded.compose(model, [], previous_promotions=[k_name])
    all_w4_hashes = {
        name: tensor_sha256(dict(model.named_modules())[name].weight)
        for name in loaded.names
    }
    loaded.compose(model, [k_name, q_name], previous_promotions=[])
    loaded.compose(model, [], previous_promotions=[q_name, k_name])
    assert all_w4_hashes == {
        name: tensor_sha256(dict(model.named_modules())[name].weight)
        for name in loaded.names
    }

    composition = loaded.composition_manifest([q_name])
    rows = loaded.manifest["candidates"]
    expected_payload = sum(
        row["w8" if row["module_name"] == q_name else "w4"]["logical_payload_bytes"]
        for row in rows
    )
    assert composition["logical_packed_decoder_payload_bytes"] == expected_payload
    assert composition["added_code_bytes_over_uniform_w4"] == rows[0]["delta_bytes"]
    assert all(
        (bank_dir / row[f"w{bits}"][kind]).stat().st_size
        == row[f"w{bits}"]["code_bytes" if kind == "codes_file" else "scale_bytes"]
        for row in rows
        for bits in (4, 8)
        for kind in ("codes_file", "scales_file")
    )
    out = loaded.write_composition_manifest(tmp_path / "composition.json", [q_name])
    assert json.loads(out.read_text())["promotions"] == [q_name]
    packed = loaded.export_packed_composition(tmp_path / "packed-composition", [q_name])
    packed_manifest = json.loads((packed / "manifest.json").read_text())
    assert packed_manifest["payload_file_bytes"] == expected_payload
    assert packed_manifest["composition"]["average_decoder_payload_bits"] > packed_manifest["composition"]["average_decoder_code_bits"]
    assert {row["bits"] for row in packed_manifest["candidates"]} == {4, 8}
    assert sum(
        path.stat().st_size for path in (packed / "tensors").iterdir()
    ) == expected_payload
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        loaded.export_packed_composition(packed, [q_name])


def test_cache_detects_manifest_and_payload_corruption(tmp_path):
    model = ToyQwen(layers=1)
    cache, _ = _two_projection_cache(model)
    bank = cache.save(tmp_path / "bank")
    manifest = json.loads((bank / "manifest.json").read_text())
    code_path = bank / manifest["candidates"][0]["w4"]["codes_file"]
    payload = bytearray(code_path.read_bytes())
    payload[0] ^= 1
    code_path.write_bytes(payload)
    with pytest.raises(GPTQCandidateError, match="corrupt W4 code"):
        GPTQCandidateCache.load(bank, verify_hashes=True)

    second = cache.save(tmp_path / "bank2")
    tampered_manifest = json.loads((second / "manifest.json").read_text())
    tampered_manifest["recipe"]["group_size"] += 1
    (second / "manifest.json").write_text(json.dumps(tampered_manifest))
    with pytest.raises(GPTQCandidateError, match="manifest content hash"):
        GPTQCandidateCache.load(second)


def test_persisted_candidates_require_payload_verification(tmp_path):
    model = ToyQwen(layers=1)
    cache, _ = _two_projection_cache(model)
    bank = cache.save(tmp_path / "bank")
    unverified = GPTQCandidateCache.load(bank, verify_hashes=False)
    with pytest.raises(GPTQCandidateError, match="without hash verification"):
        unverified.candidate(unverified.names[0], 4)


def test_cache_rejects_self_consistent_noncanonical_metadata(tmp_path):
    model = ToyQwen(layers=1)
    cache, _ = _two_projection_cache(model)
    bank = cache.save(tmp_path / "bank")
    manifest_path = bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["candidates"][0]["w4"]["code_bytes"] += 1
    manifest.pop("manifest_content_sha256")
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(GPTQCandidateError, match="code-byte mismatch"):
        GPTQCandidateCache.load(bank, verify_hashes=False)


def test_recipe_rejects_noncanonical_variants():
    with pytest.raises(GPTQCandidateError, match="bits"):
        GPTQRecipe(bits=(3, 4, 8))
    with pytest.raises(GPTQCandidateError, match="float16"):
        GPTQRecipe(scale_dtype="float32")

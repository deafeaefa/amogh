"""Simulated (quant-dequant) round-to-nearest weight quantization for GCQ.

Applies groupwise symmetric RTN in place to the LLM transformer blocks ONLY
(name filter: 'language_model.layers'), leaving the vision tower, embeddings,
and lm_head at BF16 — per the paper's vision-tower/head policy.

Simulated quantization reproduces the numerics deployment kernels compute
(dequantize-then-matmul) and needs no extra toolchain; used for the
measurement grid and profiling. Real GPTQ checkpoints are produced separately.
"""
import torch

LLM_FILTER = "language_model.layers"

def rtn_quant_tensor_(w: torch.Tensor, bits: int, group: int):
    """In-place groupwise symmetric RTN quant-dequant of a 2D weight [out, in]."""
    out_f, in_f = w.shape
    g = group if (group > 0 and in_f % group == 0) else in_f
    qmax = 2 ** (bits - 1) - 1
    w32 = w.detach().to(torch.float32).view(out_f, in_f // g, g)
    scale = w32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(w32 / scale).clamp_(-qmax, qmax)
    w.copy_((q * scale).view(out_f, in_f).to(w.dtype))

@torch.no_grad()
def apply_rtn(model, bits: int, group: int = 128, only_module: str = None, skip_module: str = None,
              promote: dict = None):
    """Quantize all LLM-block Linear weights to `bits`.

    only_module: if set, quantize ONLY modules whose name contains this substring.
    skip_module: if set, skip modules whose name contains this substring.
    promote: optional {name_substring: bits} overrides (e.g. 8-bit promotions).
    Returns list of (name, bits) applied.
    """
    applied = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear): continue
        if LLM_FILTER not in name: continue
        if only_module and only_module not in name: continue
        if skip_module and skip_module in name: continue
        b = bits
        if promote:
            for sub, pb in promote.items():
                if sub in name: b = pb; break
        rtn_quant_tensor_(mod.weight, b, group)
        applied.append((name, b))
    return applied

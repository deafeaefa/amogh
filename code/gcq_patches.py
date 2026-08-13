"""Performance patch for Qwen3-VL: replace the vision patch-embed Conv3d with its
exact linear-algebra equivalent.

The patch conv has kernel_size == stride == (t, p, p), and pixel_values arrive
already unfolded as [num_patches, C*t*p*p] in (C, t, p, p) order — the same
order as Conv3d weight [E, C, t, p, p] flattened. So
    Conv3d(x.view(-1, C, t, p, p))  ==  F.linear(x, W.view(E, -1), b)
element-for-element (up to GEMM float reassociation). cuDNN handles thousands of
kernel-sized micro-convs catastrophically (~4.5 s/image); the GEMM is ~ms.

Verified at install time by compare_outputs() (max |Δ| on real inputs).
"""
import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as _m

_orig_forward = _m.Qwen3VLVisionPatchEmbed.forward

def _fast_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    w = self.proj.weight
    return F.linear(hidden_states.to(w.dtype), w.view(w.shape[0], -1), self.proj.bias)

def apply_fast_patch_embed():
    _m.Qwen3VLVisionPatchEmbed.forward = _fast_forward

def revert():
    _m.Qwen3VLVisionPatchEmbed.forward = _orig_forward

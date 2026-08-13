"""Minimal faithful GPTQ (Frantar et al.) for Qwen3-VL-2B LLM blocks.

Standard algorithm: per-layer Hessian H = sum x x^T from calibration activations,
sequential layer propagation (later layers see earlier layers already quantized),
column-blocked quantization with Cholesky-based error compensation, groupwise
symmetric scales (group=128 input channels). Produces a quant-dequant BF16
checkpoint saved via save_pretrained -> loads in the existing harness unchanged.

Supports per-group bit overrides (promote dict) so GCQ-on-GPTQ reuses promote files.

Usage:
  python gptq_own.py --bits 4 --out DIR [--nsamples 128] [--promote-file F] [--device cuda:0]
"""
import os, json, argparse, time, re
import torch
import gcq_patches; gcq_patches.apply_fast_patch_embed()
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset
from quant_utils import LLM_FILTER

@torch.no_grad()
def gptq_quantize_linear(W, H, bits, group=128, blocksize=128, percdamp=0.01):
    """Quantize weight W [out, in] given Hessian H [in, in]; returns quant-dequant W."""
    W = W.clone().float()
    out_f, in_f = W.shape
    qmax = 2 ** (bits - 1) - 1
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    H += torch.eye(in_f, device=H.device) * damp
    # inverse via Cholesky (upper), as in reference implementation
    Hc = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hc)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    scales = torch.zeros(out_f, max(in_f // group, 1), device=W.device)
    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        Wb = W[:, i1:i2].clone()
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j
            g = col // group
            if col % group == 0:  # new group: compute scale from remaining (unquantized) block
                gend = min(col + group, in_f)
                scales[:, g] = W[:, col:gend].abs().amax(dim=1).clamp_min(1e-8) / qmax
            s = scales[:, g]
            w = Wb[:, j]
            q = torch.clamp(torch.round(w / s), -qmax, qmax) * s
            err = (w - q) / Hb[j, j]
            Wb[:, j] = q
            if j + 1 < i2 - i1:
                Wb[:, j+1:] -= err.unsqueeze(1) * Hb[j, j+1:].unsqueeze(0)
            Eb[:, j] = err
        W[:, i1:i2] = Wb
        if i2 < in_f:
            W[:, i2:] -= Eb @ Hinv[i1:i2, i2:]
    return W

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--promote-file", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    promote = {}
    if args.promote_file:
        pj = json.load(open(args.promote_file))
        promote = {s: pj["bits"] for s in pj["substrings"]}

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",
                                                        dtype=torch.bfloat16, device_map=args.device).eval()

    # calibration: wikitext chat-templated text (standard text calibration)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    chunks, buf = [], ""
    for row in ds:
        t = row["text"].strip()
        if not t: continue
        buf += " " + t
        if len(buf) > 1500:
            chunks.append(buf.strip()); buf = ""
        if len(chunks) >= args.nsamples: break
    batches = []
    for i in range(0, len(chunks), 8):
        msgs = [[{"role": "user", "content": [{"type": "text", "text": c}]}] for c in chunks[i:i+8]]
        enc = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                            return_dict=True, return_tensors="pt", padding=True)
        batches.append({k: v.to(args.device) for k, v in enc.items()})
    print(f"calibration: {len(chunks)} chunks in {len(batches)} batches")

    # group linears by decoder layer (sequential propagation)
    layer_linears = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and LLM_FILTER in n:
            l = int(re.search(r"layers\.(\d+)\.", n).group(1))
            layer_linears.setdefault(l, []).append((n, m))

    t0 = time.time()
    for l in sorted(layer_linears):
        mods = layer_linears[l]
        H = {n: torch.zeros(m.weight.shape[1], m.weight.shape[1], device=args.device)
             for n, m in mods}
        counts = {n: 0 for n, _ in mods}
        hooks = []
        def mk(n):
            def hook(module, inp, out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                H[n] += x.T @ x
                counts[n] += x.shape[0]
            return hook
        for n, m in mods: hooks.append(m.register_forward_hook(mk(n)))
        with torch.no_grad():
            for b in batches: model(**b)
        for h in hooks: h.remove()
        for n, m in mods:
            bits = args.bits
            for sub, pb in promote.items():
                if sub in n: bits = pb; break
            Wq = gptq_quantize_linear(m.weight.data, H[n], bits, args.group)
            m.weight.data.copy_(Wq.to(m.weight.dtype))
        del H
        torch.cuda.empty_cache()
        print(f"layer {l:2d} done ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"SAVED {args.out}")

if __name__ == "__main__":
    main()

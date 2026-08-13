"""Localization-sensitivity profiling for GCQ (Week 3, Intervention step 1).

Direction matches the allocator: start from the FULLY W4-RTN-quantized model,
promote ONE module at a time to 8-bit, and measure the reduction in
coordinate-token KL vs the BF16 teacher on D_probe:

    s_m = KL_base - KL_(+m)     (bigger = promoting m helps grounding more)

Teacher-forced on ground-truth answers rendered in the model's native output
format; C = token positions of the digits/commas/brackets inside the bbox_2d
array. Teacher log-probs are computed once and kept on-GPU; each config is one
forward pass over D_probe. Writes rows incrementally to
$GCQ_RUNS/sensitivity.csv.

Usage: profile_sensitivity.py --device cuda:0 --modules 0:28 [--limit N] [--batch 16]
"""
import os, re, json, csv, time, argparse
import torch
import gcq_patches; gcq_patches.apply_fast_patch_embed()
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
from quant_utils import LLM_FILTER, rtn_quant_tensor_

def gt_answer_string(r):
    x, y, w, h = r["bbox_xywh"]
    W, H = r["width"], r["height"]
    b = [round(x/W*1000), round(y/H*1000), round((x+w)/W*1000), round((y+h)/H*1000)]
    return '```json\n[\n\t{"bbox_2d": [%d, %d, %d, %d], "label": "%s"}\n]\n```' % (b[0], b[1], b[2], b[3], r["expression"])

def coord_char_span(ans):
    m = re.search(r'"bbox_2d"\s*:\s*(\[[^\]]*\])', ans)
    return m.span(1)  # the [x1, y1, x2, y2] array incl. brackets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--probe", choices=["rec", "vqa"], default="rec",
                    help="rec: coordinate-token KL on D_probe; vqa: answer-token KL on the disjoint VQA probe")
    ap.add_argument("--modules", default="0:56", help="slice a:b of the sorted LLM-linear list")
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="sensitivity.csv")
    ap.add_argument("--base-ckpt", default="", help="profile from this quant-dequant checkpoint as the base (e.g. GPTQ-W4) instead of in-memory RTN")
    args = ap.parse_args()

    data_dir = os.environ["GCQ_DATA"]; runs_dir = os.environ["GCQ_RUNS"]
    probe_file = "dprobe_refcoco_train_512.json" if args.probe == "rec" else "vqa_probe_512.json"
    recs = json.load(open(os.path.join(data_dir, "subsets", probe_file)))[:args.limit]

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    processor.tokenizer.padding_side = "right"
    tok = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",
                                                        dtype=torch.bfloat16, device_map=args.device).eval()

    # ---- collect target linear modules (block modules only), stable order ----
    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and LLM_FILTER in n]
    # group attn/mlp per layer: profile at (layer, kind) granularity
    groups = {}
    for n, m in linears:
        layer = int(re.search(r"layers\.(\d+)\.", n).group(1))
        kind = "attn" if "self_attn" in n else "mlp"
        groups.setdefault((layer, kind), []).append((n, m))
    keys = sorted(groups)
    a, b = args.modules.split(":"); sel = keys[int(a):int(b)]
    print(f"{len(keys)} module-groups total; profiling {len(sel)}: {sel[0]}..{sel[-1]}")

    cpu_copy = {n: m.weight.detach().cpu().clone() for k in keys for n, m in groups[k]}
    base_copy = None
    if args.base_ckpt:
        import glob
        from safetensors.torch import load_file
        sd = {}
        for f in glob.glob(os.path.join(args.base_ckpt, "*.safetensors")): sd.update(load_file(f))
        base_copy = {n: sd[n + ".weight"] for k in keys for n, m in groups[k]}
        print(f"base = checkpoint {args.base_ckpt} ({len(base_copy)} linears)")

    # ---- build teacher-forced batches and coordinate-token indices ----
    batches = []
    for i in range(0, len(recs), args.batch):
        chunk = recs[i:i+args.batch]
        msgs, answers = [], []
        for r in chunk:
            if args.probe == "rec":
                imdir, ans = "train2014", gt_answer_string(r)
                prompt = f"Locate the {r['expression']}, output its bbox_2d in JSON."
            else:
                imdir, ans = "val2014", r["answer"]
                prompt = r["question"] + " Answer with a single word or phrase."
            img = Image.open(os.path.join(data_dir, "images", imdir, r["file_name"])).convert("RGB")
            msgs.append([{"role":"user","content":[{"type":"image","image":img},
                        {"type":"text","text":prompt}]},
                        {"role":"assistant","content":[{"type":"text","text":ans}]}])
            answers.append(ans)
        inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                               return_dict=True, return_tensors="pt", padding=True)
        # locate coordinate tokens per row: decode the actual tail tokens (context-correct
        # vs re-tokenizing the substring, which BPE renders differently) and match char offsets
        pos = []  # list of (row, tokpos) — position of coordinate TOKEN t (KL uses logits at t-1)
        for row, ans in enumerate(answers):
            s, e = coord_char_span(ans) if args.probe == "rec" else (0, len(ans))
            coord_str = ans[s:e]
            n_real = int(inputs["attention_mask"][row].sum().item())
            ids = inputs["input_ids"][row][:n_real].tolist()  # right padding -> real prefix
            K = min(96, len(ids))
            pieces = [tok.decode([tid]) for tid in ids[-K:]]
            text = "".join(pieces)
            j = text.rfind(coord_str)
            assert j >= 0, f"coord substring not found in decoded tail, row {row}"
            off = 0; sel_toks = []
            for i, p in enumerate(pieces):
                if off < j + len(coord_str) and off + len(p) > j:
                    sel_toks.append(len(ids) - K + i)
                off += len(p)
            min_toks = 4 if args.probe == "rec" else 1
            assert len(sel_toks) >= min_toks, f"too few target tokens row {row}: {sel_toks}"
            pos.extend((row, t) for t in sel_toks)
        batches.append((inputs, pos))
    ncoord = sum(len(p) for _, p in batches)
    print(f"{len(batches)} batches, {ncoord} coordinate-token positions")

    dev = args.device
    @torch.no_grad()
    def coord_logprobs(store_probs=False):
        """Forward all batches; return list of [n_pos, V] log-probs (fp16 on GPU) at coord positions."""
        out = []
        for inputs, pos in batches:
            inp = {k: v.to(dev) for k, v in inputs.items()}
            logits = model(**inp).logits
            rows = torch.tensor([r for r, _ in pos], device=dev)
            cols = torch.tensor([t - 1 for _, t in pos], device=dev)  # logits at t-1 predict token t
            sel_logits = logits[rows, cols].float()
            out.append(torch.log_softmax(sel_logits, dim=-1).half())
            del logits
        return out

    t0 = time.time()
    teacher = coord_logprobs()
    print(f"teacher pass {time.time()-t0:.0f}s")

    @torch.no_grad()
    def kl_vs_teacher():
        tot = 0.0
        for t_lp, s_lp in zip(teacher, coord_logprobs()):
            p = t_lp.float().exp()
            tot += (p * (t_lp.float() - s_lp.float())).sum().item()
        return tot / ncoord

    def set_bits(key, bits):
        for n, m in groups[key]:
            with torch.no_grad():
                if base_copy is not None and bits == 4:
                    m.weight.copy_(base_copy[n].to(m.weight.device))  # checkpoint base (e.g. GPTQ-4)
                else:
                    m.weight.copy_(cpu_copy[n].to(m.weight.device))
                    if bits: rtn_quant_tensor_(m.weight, bits, 128)

    # ---- base: everything W4 ----
    for k in keys: set_bits(k, 4)
    kl_base = kl_vs_teacher()
    print(f"KL_base (all W4) = {kl_base:.5f}")

    out_path = os.path.join(runs_dir, args.out)
    new = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["layer", "kind", "kl_promoted", "kl_base", "s_m", "n_pos", "n_probe"])
        for k in sel:
            t0 = time.time()
            set_bits(k, 8)
            kl_m = kl_vs_teacher()
            set_bits(k, 4)
            s_m = kl_base - kl_m
            w.writerow([k[0], k[1], f"{kl_m:.6f}", f"{kl_base:.6f}", f"{s_m:.6f}", ncoord, len(recs)])
            f.flush()
            print(f"layer {k[0]:2d} {k[1]:4s}: KL={kl_m:.5f} s_m={s_m:+.5f} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()

"""ODinW-13 held-out out-of-domain grounding eval (touched once, Week 4).

Frozen slice: 500 images sampled seed-0, stratified by dataset. For each GT-present
category: prompt "Locate every {c}...", parse ALL emitted bbox_2d, greedy-IoU-match
to that category's GT boxes. Reports micro precision/recall/F1@0.5 overall and
per-dataset. GT bboxes are xyxy pixels; predictions are [0,1000]-normalized.

Usage: eval_odinw.py --tag bf16_odinw [--rtn-bits 4] [--promote-file F] [--device cuda:0]
"""
import os, re, json, csv, time, argparse, random
from collections import defaultdict
import torch
import gcq_patches; gcq_patches.apply_fast_patch_embed()
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset

BOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]')

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rtn-bits", type=int, default=0)
    ap.add_argument("--promote-file", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--images", type=int, default=500)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    runs = os.environ["GCQ_RUNS"]

    ds = load_dataset("kcz358/ODinW-13", split="test")
    by_set = defaultdict(list)
    for i, name in enumerate(ds["dataset_name"]): by_set[name].append(i)
    rng = random.Random(0)
    per = max(1, args.images // len(by_set))
    idx = []
    for name in sorted(by_set):
        pool = by_set[name][:]; rng.shuffle(pool); idx.extend(pool[:per])
    idx = idx[:args.images]

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", max_pixels=1003520)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",
                                                        dtype=torch.bfloat16, device_map=args.device).eval()
    if args.rtn_bits:
        from quant_utils import apply_rtn
        promote = None
        if args.promote_file:
            pj = json.load(open(args.promote_file))
            promote = {s: pj["bits"] for s in pj["substrings"]}
        applied = apply_rtn(model, args.rtn_bits, 128, promote=promote)
        print(f"RTN W{args.rtn_bits} applied ({sum(1 for _,b in applied if b==8)} groups at 8-bit)")

    # build (image_idx, category) queries for GT-present categories
    queries = []
    for i in idx:
        labs = set(ds[i]["labels"])
        for c in sorted(labs): queries.append((i, c))
    print(f"{len(idx)} images -> {len(queries)} (image, category) queries")

    stats = defaultdict(lambda: [0, 0, 0])  # dataset -> [tp, fp, fn]
    flog = open(os.path.join(runs, f"{args.tag}.odinw.jsonl"), "w")
    t0 = time.time()
    for b in range(0, len(queries), args.batch):
        chunk = queries[b:b+args.batch]
        msgs = []
        for i, c in chunk:
            img = ds[i]["image"].convert("RGB")
            msgs.append([{"role": "user", "content": [{"type": "image", "image": img},
                        {"type": "text", "text": f"Locate every {c}, output all bbox_2d in JSON."}]}])
        inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for (i, c), text in zip(chunk, texts):
            ex = ds[i]; W, H = ex["image"].size
            gts = [bb for bb, lb in zip(ex["bboxes"], ex["labels"]) if lb == c]
            preds = [[int(x1)*W/1000, int(y1)*H/1000, int(x2)*W/1000, int(y2)*H/1000]
                     for x1, y1, x2, y2 in BOX_RE.findall(text)][:20]
            used = [False]*len(gts); tp = 0
            for p in preds:
                best, bj = 0.0, -1
                for j, g in enumerate(gts):
                    if used[j]: continue
                    v = iou(p, g)
                    if v > best: best, bj = v, j
                if best >= 0.5: used[bj] = True; tp += 1
            st = stats[ex["dataset_name"]]
            st[0] += tp; st[1] += len(preds)-tp; st[2] += len(gts)-tp
            for j, g in enumerate(gts):
                flog.write(json.dumps({"ds": ex["dataset_name"], "cat": c,
                                       "area": (g[2]-g[0])*(g[3]-g[1]), "matched": used[j]}) + "\n")
            flog.flush()
        if (b//args.batch) % 10 == 0:
            print(f"  {b+len(chunk)}/{len(queries)} ({time.time()-t0:.0f}s)", flush=True)

    TP = sum(s[0] for s in stats.values()); FP = sum(s[1] for s in stats.values()); FN = sum(s[2] for s in stats.values())
    P = TP/max(TP+FP, 1); R = TP/max(TP+FN, 1); F1 = 2*P*R/max(P+R, 1e-9)
    print(f"\nRESULT tag={args.tag} ODinW-13 micro: P={P:.4f} R={R:.4f} F1={F1:.4f} (TP={TP} FP={FP} FN={FN})")
    for name in sorted(stats):
        tp, fp, fn = stats[name]
        p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
        print(f"  {name:24s} P={p:.3f} R={r:.3f} n_gt={tp+fn}")
    with open(os.path.join(runs, "results.csv"), "a", newline="") as f:
        csv.writer(f).writerow([args.tag, "Qwen/Qwen3-VL-2B-Instruct", "odinw", f"P={P:.4f} R={R:.4f}", TP+FN,
                                f"{F1:.4f}", "", "", "", "", "", False, int(time.time()-t0)])

if __name__ == "__main__":
    main()

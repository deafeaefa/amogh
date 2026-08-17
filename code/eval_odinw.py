"""ODinW-13 out-of-domain grounding evaluation.

For each GT-present category: prompt "Locate every {c}...", parse all emitted
bbox_2d values, and perform maximum-cardinality, maximum-IoU bipartite matching.
Reports micro precision/recall/F1@0.5 overall and per dataset.  ``--images N``
uses an exact-size, seed-0 balanced sample with exhausted-dataset redistribution;
``--images 0`` evaluates the complete test split.

Usage: eval_odinw.py --tag bf16_odinw [--rtn-bits 4] [--promote-file F] [--device cuda:0]
"""
import os, re, json, csv, time, argparse, random
from collections import defaultdict
import torch
import numpy as np
import gcq_patches; gcq_patches.apply_fast_patch_embed()
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment

BOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]')

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def stratified_indices(by_set, requested, seed=0):
    """Return exactly ``requested`` balanced indices, redistributing shortages."""
    total = sum(len(v) for v in by_set.values())
    if requested <= 0 or requested >= total:
        return list(range(total))
    rng = random.Random(seed)
    pools = {}
    for name in sorted(by_set):
        pools[name] = list(by_set[name])
        rng.shuffle(pools[name])
    cursors = {name: 0 for name in pools}
    selected = []
    while len(selected) < requested:
        progressed = False
        for name in sorted(pools):
            cursor = cursors[name]
            if cursor < len(pools[name]):
                selected.append(pools[name][cursor])
                cursors[name] += 1
                progressed = True
                if len(selected) == requested:
                    break
        if not progressed:
            raise RuntimeError(f"could only select {len(selected)}/{requested} ODinW images")
    return selected

def match_boxes(preds, gts, threshold=0.5):
    """Maximum-cardinality matching, using total IoU to break equal-cardinality ties."""
    if not preds or not gts:
        return []
    overlaps = np.asarray([[iou(p, g) for g in gts] for p in preds], dtype=np.float64)
    # A valid edge is worth more than the maximum possible sum of all IoUs, so
    # the assignment first maximizes the number of thresholded matches.
    score = overlaps + (overlaps >= threshold) * (min(len(preds), len(gts)) + 1.0)
    pred_rows, gt_cols = linear_sum_assignment(-score)
    return [(int(p), int(g), float(overlaps[p, g]))
            for p, g in zip(pred_rows, gt_cols) if overlaps[p, g] >= threshold]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--rtn-bits", type=int, default=0)
    ap.add_argument("--rtn-group", type=int, default=128)
    ap.add_argument("--a8", action="store_true", help="simulated 8-bit dynamic activation quant on LLM linears")
    ap.add_argument("--promote-file", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--images", type=int, default=500)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=1003520)
    ap.add_argument("--adapter-dir", default="", help="GCQ recovery adapter; attached after base quantization")
    args = ap.parse_args()
    if "gptq" in args.tag.lower() and args.model.startswith("Qwen/"):
        raise SystemExit("refusing: tag says gptq but --model is the base HF id; pass the checkpoint path")
    runs = os.environ["GCQ_RUNS"]

    ds = load_dataset("kcz358/ODinW-13", split="test")
    by_set = defaultdict(list)
    for i, name in enumerate(ds["dataset_name"]): by_set[name].append(i)
    idx = stratified_indices(by_set, args.images, seed=0)

    revision = None
    if args.adapter_dir:
        from recovery_utils import read_adapter_manifest, validate_adapter_quantization
        manifest = read_adapter_manifest(args.adapter_dir)
        validate_adapter_quantization(manifest, args.model, args.rtn_bits, args.rtn_group,
                                      args.promote_file, args.max_pixels)
        revision = manifest["base_revision"]
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", revision=revision,
                                             max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.model, revision=revision,
                                                        dtype=torch.bfloat16, device_map=args.device)
    if args.rtn_bits:
        from quant_utils import apply_rtn
        promote = None
        if args.promote_file:
            pj = json.load(open(args.promote_file))
            promote = {s: pj["bits"] for s in pj["substrings"]}
        applied = apply_rtn(model, args.rtn_bits, args.rtn_group, promote=promote)
        print(f"RTN W{args.rtn_bits} applied ({sum(1 for _,b in applied if b==8)} groups at 8-bit)")
    if args.a8:
        from quant_utils import apply_a8
        print(f"A8 hooks on {apply_a8(model)} linears")
    if args.adapter_dir:
        from recovery_utils import attach_adapter
        model = attach_adapter(model, args.adapter_dir)
        print(f"attached recovery adapter {args.adapter_dir}")
    model.eval()

    # build (image_idx, category) queries for GT-present categories
    queries = []
    for i in idx:
        labs = set(ds[i]["labels"])
        for c in sorted(labs): queries.append((i, c))
    n_gt_total = sum(sum(1 for label in ds[i]["labels"] if label == category) for i, category in queries)
    print(f"{len(idx)} images -> {len(queries)} (image, category) queries -> {n_gt_total} GT boxes")

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
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated = out[:, inputs["input_ids"].shape[1]:]
        texts = processor.batch_decode(generated, skip_special_tokens=True)
        eos_id = processor.tokenizer.eos_token_id
        for row, ((i, c), text) in enumerate(zip(chunk, texts)):
            ex = ds[i]; W, H = ex["image"].size
            gts = [bb for bb, lb in zip(ex["bboxes"], ex["labels"]) if lb == c]
            preds = []
            for values in BOX_RE.findall(text):
                x1, y1, x2, y2 = (int(v) for v in values)
                if x1 < x2 and y1 < y2:
                    preds.append([x1*W/1000, y1*H/1000, x2*W/1000, y2*H/1000])
            matches = match_boxes(preds, gts, threshold=0.5)
            matched_gt = {g for _, g, _ in matches}
            tp = len(matches)
            token_ids = generated[row].tolist()
            truncated = len(token_ids) >= args.max_new_tokens and (eos_id is None or eos_id not in token_ids)
            st = stats[ex["dataset_name"]]
            st[0] += tp; st[1] += len(preds)-tp; st[2] += len(gts)-tp
            image_id = ex.get("image_id", i)
            flog.write(json.dumps({"record_type": "query", "dataset_index": i, "image_id": image_id,
                                   "ds": ex["dataset_name"], "cat": c, "n_gt": len(gts),
                                   "n_pred": len(preds), "tp": tp, "fp": len(preds)-tp,
                                   "fn": len(gts)-tp, "truncated": truncated,
                                   "pred_raw": text.strip()[:1000]}) + "\n")
            for j, g in enumerate(gts):
                flog.write(json.dumps({"record_type": "gt", "dataset_index": i, "image_id": image_id,
                                       "ds": ex["dataset_name"], "cat": c,
                                       "area": (g[2]-g[0])*(g[3]-g[1]),
                                       "relative_area": (g[2]-g[0])*(g[3]-g[1])/(W*H),
                                       "matched": j in matched_gt}) + "\n")
            flog.flush()
        if (b//args.batch) % 10 == 0:
            print(f"  {b+len(chunk)}/{len(queries)} ({time.time()-t0:.0f}s)", flush=True)

    TP = sum(s[0] for s in stats.values()); FP = sum(s[1] for s in stats.values()); FN = sum(s[2] for s in stats.values())
    P = TP/max(TP+FP, 1); R = TP/max(TP+FN, 1); F1 = 2*P*R/max(P+R, 1e-9)
    print(f"\nRESULT tag={args.tag} ODinW-13 micro: P={P:.4f} R={R:.4f} F1={F1:.4f} "
          f"(images={len(idx)} queries={len(queries)} GT={n_gt_total} TP={TP} FP={FP} FN={FN})")
    for name in sorted(stats):
        tp, fp, fn = stats[name]
        p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
        print(f"  {name:24s} P={p:.3f} R={r:.3f} n_gt={tp+fn}")
    with open(os.path.join(runs, "results.csv"), "a", newline="") as f:
        csv.writer(f).writerow([args.tag, "Qwen/Qwen3-VL-2B-Instruct", "odinw",
                                f"images={len(idx)} queries={len(queries)} GT={n_gt_total} P={P:.4f} R={R:.4f}", TP+FN,
                                f"{F1:.4f}", "", "", "", "", "", False, int(time.time()-t0)])

if __name__ == "__main__":
    main()

"""REC (referring-expression comprehension) evaluation harness for GCQ.

Scores a Qwen3-VL model (BF16 or quantized) on a frozen subset:
  acc@IoU0.5, mean GIoU, parse-failure rate, size-stratified acc (COCO areas).
Writes one CSV row to $GCQ_RUNS/results.csv and per-sample records to
$GCQ_RUNS/<tag>.rec.jsonl. Greedy decoding; deterministic.

Usage:
  python eval_rec.py --model Qwen/Qwen3-VL-2B-Instruct --subset rec_eval_refcoco_val_1k \
      --tag bf16_receval --batch 16 [--limit N] [--blank-image] [--load-4bit-dir DIR]
"""
import os, re, io, json, csv, time, argparse
import torch
import gcq_patches; gcq_patches.apply_fast_patch_embed()
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

BOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]')

def parse_box(text):
    m = BOX_RE.search(text)
    if not m: return None
    b = [int(g) for g in m.groups()]
    return b if b[0] < b[2] and b[1] < b[3] else None

def to_pixels(box1000, w, h):
    return [box1000[0]*w/1000.0, box1000[1]*h/1000.0, box1000[2]*w/1000.0, box1000[3]*h/1000.0]

def iou_giou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1,bx1), max(ay1,by1), min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    aa = (ax2-ax1)*(ay2-ay1); ab = (bx2-bx1)*(by2-by1)
    union = aa + ab - inter
    iou = inter/union if union > 0 else 0.0
    cx1, cy1, cx2, cy2 = min(ax1,bx1), min(ay1,by1), max(ax2,bx2), max(ay2,by2)
    hull = (cx2-cx1)*(cy2-cy1)
    giou = iou - (hull-union)/hull if hull > 0 else iou
    return iou, giou

def size_bucket(area):
    if area < 32**2: return "small"
    if area < 96**2: return "medium"
    return "large"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blank-image", action="store_true", help="image-blind floor run")
    ap.add_argument("--rtn-bits", type=int, default=0, help="apply simulated RTN at this width to LLM blocks")
    ap.add_argument("--rtn-group", type=int, default=128)
    ap.add_argument("--max-pixels", type=int, default=1003520, help="1280*28*28, Qwen-standard eval cap; identical for ALL configs")
    args = ap.parse_args()

    data_dir = os.environ["GCQ_DATA"]; runs_dir = os.environ["GCQ_RUNS"]
    os.makedirs(runs_dir, exist_ok=True)
    with open(os.path.join(data_dir, "subsets", args.subset + ".json")) as f:
        recs = json.load(f)
    if args.limit: recs = recs[:args.limit]

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "left"
    print("image processor max_pixels:", getattr(processor.image_processor, "max_pixels", "N/A"))
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    if args.rtn_bits:
        from quant_utils import apply_rtn
        applied = apply_rtn(model, args.rtn_bits, args.rtn_group)
        print(f"RTN W{args.rtn_bits} g{args.rtn_group} applied to {len(applied)} LLM linears")
    print(f"model {args.model} loaded {time.time()-t0:.0f}s")

    out_path = os.path.join(runs_dir, args.tag + ".rec.jsonl")
    n = correct = parsefail = 0
    giou_sum = 0.0
    by_size = {"small": [0,0], "medium": [0,0], "large": [0,0]}
    t0 = time.time()
    with open(out_path, "w") as fout:
        for i in range(0, len(recs), args.batch):
            chunk = recs[i:i+args.batch]
            msgs, imgs = [], []
            for r in chunk:
                sub = "train2014" if "train2014" in r["file_name"] else "val2014"
                if args.blank_image:
                    img = Image.new("RGB", (r["width"], r["height"]), (127,127,127))
                else:
                    img = Image.open(os.path.join(data_dir, "images", sub, r["file_name"])).convert("RGB")
                imgs.append(img)
                msgs.append([{"role":"user","content":[{"type":"image","image":img},
                    {"type":"text","text":f"Locate the {r['expression']}, output its bbox_2d in JSON."}]}])
            inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, text in zip(chunk, texts):
                gt = r["bbox_xywh"]; gt_xyxy = [gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]]
                box = parse_box(text)
                n += 1
                bucket = size_bucket(gt[2]*gt[3]); by_size[bucket][1] += 1
                if box is None:
                    parsefail += 1; iou = 0.0; giou = -1.0
                else:
                    iou, giou = iou_giou(to_pixels(box, r["width"], r["height"]), gt_xyxy)
                hit = iou >= 0.5
                if hit: correct += 1; by_size[bucket][0] += 1
                giou_sum += giou
                fout.write(json.dumps({"uid": r["uid"], "pred_raw": text.strip()[:200],
                                       "box1000": box, "iou": round(iou,4), "giou": round(giou,4), "hit": hit}) + "\n")
            fout.flush()
            if (i//args.batch) % 10 == 0:
                el = time.time()-t0
                print(f"  {n}/{len(recs)} acc={correct/max(n,1):.3f} ({el:.0f}s, {n/max(el,1):.1f}/s)", flush=True)

    acc = correct/n; mgiou = giou_sum/n; pf = parsefail/n
    sizes = {k: (v[0]/v[1] if v[1] else None) for k, v in by_size.items()}
    print(f"\nRESULT tag={args.tag} model={args.model} subset={args.subset} n={n}")
    print(f"  acc@0.5={acc:.4f} meanGIoU={mgiou:.4f} parsefail={pf:.4f} by_size={sizes}")
    csv_path = os.path.join(runs_dir, "results.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        wcsv = csv.writer(f)
        if new: wcsv.writerow(["tag","model","task","subset","n","acc","mean_giou","parse_fail",
                              "acc_small","acc_medium","acc_large","blank_image","seconds"])
        wcsv.writerow([args.tag, args.model, "rec", args.subset, n, f"{acc:.4f}", f"{mgiou:.4f}", f"{pf:.4f}",
                       sizes["small"], sizes["medium"], sizes["large"], args.blank_image, int(time.time()-t0)])

if __name__ == "__main__":
    main()

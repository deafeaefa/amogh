"""VQAv2 + POPE evaluation harness for GCQ (general-capability constraint metrics).

VQAv2: standard soft accuracy min(#humans_matching/3, 1) on the frozen 5k subset.
POPE: yes/no accuracy + F1 on the three official COCO variants.
Writes rows to $GCQ_RUNS/results.csv and per-sample records to $GCQ_RUNS/<tag>.<task>.jsonl.

Usage:
  python eval_vqa.py --model Qwen/Qwen3-VL-2B-Instruct --tag bf16 --task vqa [--limit N] [--blank-image]
  python eval_vqa.py --model ... --tag bf16 --task pope
"""
import os, re, json, csv, time, argparse
import torch
import gcq_patches; gcq_patches.apply_fast_patch_embed()
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

def norm_ans(s):
    s = s.lower().strip().rstrip(".")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task", choices=["vqa", "pope"], required=True)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blank-image", action="store_true")
    ap.add_argument("--rtn-bits", type=int, default=0)
    ap.add_argument("--rtn-group", type=int, default=128)
    ap.add_argument("--a8", action="store_true", help="simulated 8-bit dynamic activation quant on LLM linears")
    ap.add_argument("--max-pixels", type=int, default=1003520)
    ap.add_argument("--promote-file", default="")
    args = ap.parse_args()

    data_dir = os.environ["GCQ_DATA"]; runs_dir = os.environ["GCQ_RUNS"]
    os.makedirs(runs_dir, exist_ok=True)

    if args.task == "vqa":
        with open(os.path.join(data_dir, "subsets", "vqa_val_5k.json")) as f:
            items = json.load(f)
        for it in items: it["prompt"] = it["question"] + " Answer with a single word or phrase."
    else:
        items = []
        for v in ["random", "popular", "adversarial"]:
            with open(os.path.join(data_dir, "pope", f"coco_pope_{v}.json")) as f:
                for line in f:
                    d = json.loads(line)
                    items.append(dict(uid=f"pope_{v}:{d['question_id']}", variant=v, file_name=d["image"],
                                      prompt=d["text"] + " Answer yes or no.", label=d["label"]))
    if args.limit: items = items[:args.limit]

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    if args.rtn_bits:
        from quant_utils import apply_rtn
        promote = None
        if args.promote_file:
            with open(args.promote_file) as pf:
                pj = json.load(pf)
            promote = {s: pj["bits"] for s in pj["substrings"]}
        applied = apply_rtn(model, args.rtn_bits, args.rtn_group, promote=promote)
        n8 = sum(1 for _, b in applied if b == 8)
        print(f"RTN W{args.rtn_bits} g{args.rtn_group} applied to {len(applied)} LLM linears ({n8} at 8-bit)")
    if args.a8:
        from quant_utils import apply_a8
        print(f"A8 hooks on {apply_a8(model)} linears")

    out_path = os.path.join(runs_dir, f"{args.tag}.{args.task}.jsonl")
    n = 0; score_sum = 0.0
    pope_stats = {}  # variant -> [tp, fp, tn, fn]
    t0 = time.time()
    with open(out_path, "w") as fout:
        for i in range(0, len(items), args.batch):
            chunk = items[i:i+args.batch]
            msgs = []
            for it in chunk:
                if args.blank_image:
                    img = Image.new("RGB", (448, 448), (127,127,127))
                else:
                    img = Image.open(os.path.join(data_dir, "images", "val2014", it["file_name"])).convert("RGB")
                msgs.append([{"role":"user","content":[{"type":"image","image":img},
                             {"type":"text","text":it["prompt"]}]}])
            inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for it, text in zip(chunk, texts):
                pred = norm_ans(text)
                n += 1
                if args.task == "vqa":
                    matches = sum(1 for a in it["answers"] if norm_ans(a) == pred)
                    s = min(matches/3.0, 1.0)
                    score_sum += s
                    fout.write(json.dumps({"uid": f"vqa:{it['question_id']}", "pred": pred, "score": round(s,3)}) + "\n")
                else:
                    yes = pred.startswith("yes")
                    gt_yes = it["label"] == "yes"
                    st = pope_stats.setdefault(it["variant"], [0,0,0,0])
                    if yes and gt_yes: st[0] += 1
                    elif yes and not gt_yes: st[1] += 1
                    elif not yes and not gt_yes: st[2] += 1
                    else: st[3] += 1
                    score_sum += 1.0 if yes == gt_yes else 0.0
                    fout.write(json.dumps({"uid": it["uid"], "pred": pred, "correct": yes == gt_yes}) + "\n")
            fout.flush()
            if (i//args.batch) % 20 == 0:
                el = time.time()-t0
                print(f"  {n}/{len(items)} running={score_sum/max(n,1):.3f} ({el:.0f}s, {n/max(el,1):.1f}/s)", flush=True)

    acc = score_sum/n
    print(f"\nRESULT tag={args.tag} task={args.task} model={args.model} n={n} acc={acc:.4f}")
    extra = ""
    if args.task == "pope":
        for v, (tp,fp,tn,fn) in sorted(pope_stats.items()):
            p = tp/max(tp+fp,1); r = tp/max(tp+fn,1); f1 = 2*p*r/max(p+r,1e-9)
            a = (tp+tn)/max(tp+fp+tn+fn,1)
            extra += f"{v}:acc={a:.3f},f1={f1:.3f} "
            print(f"  {v}: acc={a:.4f} f1={f1:.4f}")
    csv_path = os.path.join(runs_dir, "results.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["tag","model","task","subset","n","acc","mean_giou","parse_fail",
                           "acc_small","acc_medium","acc_large","blank_image","seconds"])
        w.writerow([args.tag, args.model, args.task, extra.strip() or args.task, n, f"{acc:.4f}", "", "",
                    "", "", "", args.blank_image, int(time.time()-t0)])

if __name__ == "__main__":
    main()

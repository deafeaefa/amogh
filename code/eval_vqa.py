"""VQAv2 + POPE evaluation harness for GCQ (general-capability constraints).

VQAv2 uses official answer normalization and leave-one-annotator-out soft
accuracy on the frozen 5k subset. POPE requires an explicit leading yes/no;
malformed generations are counted as incorrect and reported separately.
Writes rows to $GCQ_RUNS/results.csv and per-sample records to $GCQ_RUNS/<tag>.<task>.jsonl.

Usage:
  python eval_vqa.py --model Qwen/Qwen3-VL-2B-Instruct --tag bf16 --task vqa [--start N] [--limit N] [--blank-image]
  python eval_vqa.py --model ... --tag bf16 --task pope
"""
import os, re, json, csv, time, argparse
import torch
import gcq_patches; gcq_patches.apply_fast_patch_embed()
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
from recovery_utils import BASE_REVISION, sha256_file

VQA_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "id've": "i'd've", "i'dve": "i'd've", "im": "i'm", "ive": "i've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "shes": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've",
}
VQA_NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}
VQA_ARTICLES = {"a", "an", "the"}
VQA_PERIOD_RE = re.compile(r"(?!<=\d)(\.)(?!\d)")
VQA_COMMA_RE = re.compile(r"(\d)(,)(\d)")
VQA_PUNCTUATION = (';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
                   '_', '-', '>', '<', '@', '`', ',', '?', '!')


def vqa_normalize(text):
    """Normalize an answer according to the official VQAv2 evaluator."""
    text = str(text).replace("\n", " ").replace("\t", " ").strip()
    output = text
    for punctuation in VQA_PUNCTUATION:
        if (punctuation + " " in text or " " + punctuation in text
                or VQA_COMMA_RE.search(text)):
            output = output.replace(punctuation, "")
        else:
            output = output.replace(punctuation, " ")
    output = VQA_PERIOD_RE.sub("", output)
    words = []
    for word in output.lower().split():
        word = VQA_NUMBER_MAP.get(word, word)
        if word not in VQA_ARTICLES:
            words.append(VQA_CONTRACTIONS.get(word, word))
    return " ".join(words)


def vqa_soft_score(prediction, answers):
    """Average agreement over the ten leave-one-annotator-out VQA subsets."""
    if len(answers) != 10:
        raise ValueError(f"VQAv2 requires exactly 10 human answers, found {len(answers)}")
    prediction = vqa_normalize(prediction)
    normalized_answers = [vqa_normalize(answer) for answer in answers]
    accuracies = []
    for held_out in range(len(normalized_answers)):
        matches = sum(
            answer == prediction
            for index, answer in enumerate(normalized_answers)
            if index != held_out
        )
        accuracies.append(min(1.0, matches / 3.0))
    return sum(accuracies) / len(accuracies), prediction


def parse_yes_no(text):
    match = re.match(r"^\s*(yes|no)\b", str(text), flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def slice_eval_items(items, start=0, limit=0):
    """Take a deterministic contiguous slice and reject empty/invalid requests."""
    if start < 0:
        raise ValueError("--start must be nonnegative")
    if limit < 0:
        raise ValueError("--limit must be nonnegative")
    if start >= len(items):
        raise ValueError(f"--start {start} is outside a dataset of {len(items)} rows")
    if limit and start + limit > len(items):
        raise ValueError(
            f"requested slice [{start}:{start + limit}] exceeds a dataset of {len(items)} rows"
        )
    selected = items[start:] if limit == 0 else items[start:start + limit]
    if not selected:
        raise ValueError("evaluation slice is empty")
    return selected

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=BASE_REVISION,
                    help="immutable base-model revision used by every compared arm")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task", choices=["vqa", "pope"], required=True)
    ap.add_argument("--vqa-file", default="",
                    help="optional frozen VQA JSON manifest; defaults to subsets/vqa_val_5k.json")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--start", type=int, default=0,
                    help="zero-based offset into the frozen evaluation order")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blank-image", action="store_true")
    ap.add_argument("--rtn-bits", type=int, default=0)
    ap.add_argument("--rtn-group", type=int, default=128)
    ap.add_argument("--a8", action="store_true", help="simulated 8-bit dynamic activation quant on LLM linears")
    ap.add_argument("--max-pixels", type=int, default=1003520)
    ap.add_argument("--promote-file", default="")
    ap.add_argument("--adapter-dir", default="", help="GCQ recovery adapter; attached after base quantization")
    args = ap.parse_args()

    data_dir = os.environ["GCQ_DATA"]; runs_dir = os.environ["GCQ_RUNS"]
    os.makedirs(runs_dir, exist_ok=True)

    input_paths = []
    if args.task == "vqa":
        vqa_path = os.path.abspath(
            args.vqa_file or os.path.join(data_dir, "subsets", "vqa_val_5k.json")
        )
        input_paths.append(vqa_path)
        with open(vqa_path) as f:
            items = json.load(f)
        for it in items: it["prompt"] = it["question"] + " Answer with a single word or phrase."
    else:
        if args.vqa_file:
            ap.error("--vqa-file is only valid with --task vqa")
        items = []
        for v in ["random", "popular", "adversarial"]:
            pope_path = os.path.abspath(os.path.join(data_dir, "pope", f"coco_pope_{v}.json"))
            input_paths.append(pope_path)
            with open(pope_path) as f:
                for line in f:
                    d = json.loads(line)
                    items.append(dict(uid=f"pope_{v}:{d['question_id']}", variant=v, file_name=d["image"],
                                      prompt=d["text"] + " Answer yes or no.", label=d["label"]))
    try:
        items = slice_eval_items(items, start=args.start, limit=args.limit)
    except ValueError as exc:
        ap.error(str(exc))

    revision = args.revision
    if args.adapter_dir:
        from recovery_utils import read_adapter_manifest, validate_adapter_quantization
        manifest = read_adapter_manifest(args.adapter_dir)
        validate_adapter_quantization(manifest, args.model, args.rtn_bits, args.rtn_group,
                                      args.promote_file, args.max_pixels, base_revision=revision)
    processor = AutoProcessor.from_pretrained(args.model, revision=revision,
                                             max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.model, revision=revision,
                                                        dtype=torch.bfloat16, device_map=args.device)
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
    if args.adapter_dir:
        from recovery_utils import attach_adapter
        model = attach_adapter(model, args.adapter_dir)
        print(f"attached recovery adapter {args.adapter_dir}")
    model.eval()

    out_path = os.path.join(runs_dir, f"{args.tag}.{args.task}.jsonl")
    n = 0; score_sum = 0.0
    pope_stats = {}  # variant -> tp/fp/tn/fn/malformed/total
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
                n += 1
                if args.task == "vqa":
                    s, pred = vqa_soft_score(text, it["answers"])
                    score_sum += s
                    fout.write(json.dumps({"uid": f"vqa:{it['question_id']}", "pred": pred, "score": round(s,3)}) + "\n")
                else:
                    pred = str(text).strip().lower()
                    decision = parse_yes_no(text)
                    gt_yes = it["label"] == "yes"
                    st = pope_stats.setdefault(it["variant"], {
                        "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                        "malformed": 0, "total": 0,
                    })
                    st["total"] += 1
                    correct = False
                    if decision is None:
                        st["malformed"] += 1
                        if gt_yes:
                            st["fn"] += 1
                    elif decision == "yes" and gt_yes:
                        st["tp"] += 1; correct = True
                    elif decision == "yes" and not gt_yes:
                        st["fp"] += 1
                    elif decision == "no" and not gt_yes:
                        st["tn"] += 1; correct = True
                    else:
                        st["fn"] += 1
                    score_sum += float(correct)
                    fout.write(json.dumps({"uid": it["uid"], "pred": pred,
                                           "parsed_answer": decision,
                                           "parse_fail": decision is None,
                                           "correct": correct}) + "\n")
            fout.flush()
            if (i//args.batch) % 20 == 0:
                el = time.time()-t0
                print(f"  {n}/{len(items)} running={score_sum/max(n,1):.3f} ({el:.0f}s, {n/max(el,1):.1f}/s)", flush=True)

    acc = score_sum/n
    print(f"\nRESULT tag={args.tag} task={args.task} model={args.model} n={n} acc={acc:.4f}")
    extra = ""
    malformed_total = 0
    variant_metrics = {}
    if args.task == "pope":
        for v, st in sorted(pope_stats.items()):
            tp, fp, tn, fn = st["tp"], st["fp"], st["tn"], st["fn"]
            p = tp/max(tp+fp,1); r = tp/max(tp+fn,1); f1 = 2*p*r/max(p+r,1e-9)
            a = (tp+tn)/max(st["total"],1)
            malformed_total += st["malformed"]
            variant_metrics[v] = {**st, "accuracy": a, "precision": p, "recall": r, "f1": f1}
            extra += f"{v}:acc={a:.3f},f1={f1:.3f},malformed={st['malformed']} "
            print(f"  {v}: acc={a:.4f} f1={f1:.4f} malformed={st['malformed']}")
    metrics = {
        "tag": args.tag, "model": args.model, "base_revision": revision,
        "task": args.task, "n": n, "accuracy": acc,
        "start": args.start, "requested_limit": args.limit,
        "input_files": [
            {"path": path, "sha256": sha256_file(path)} for path in input_paths
        ],
        "vqa_evaluator": "official_normalization_leave_one_annotator_out" if args.task == "vqa" else None,
        "pope_variants": variant_metrics if args.task == "pope" else None,
        "parse_fail": malformed_total / n if args.task == "pope" else None,
    }
    with open(os.path.join(runs_dir, f"{args.tag}.{args.task}.metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    csv_path = os.path.join(runs_dir, "results.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["tag","model","task","subset","n","acc","mean_giou","parse_fail",
                           "acc_small","acc_medium","acc_large","blank_image","seconds"])
        w.writerow([args.tag, args.model, args.task, extra.strip() or args.task, n, f"{acc:.4f}", "",
                    f"{malformed_total/n:.4f}" if args.task == "pope" else "",
                    "", "", "", args.blank_image, int(time.time()-t0)])

if __name__ == "__main__":
    main()

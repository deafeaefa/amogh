"""Build the frozen, committed data subsets for GCQ (seed 0, deterministic).

Outputs (in $GCQ_DATA/subsets/):
  rec_eval_refcoco_val_1k.json   - 1,000 expressions from RefCOCO val (reporting subset)
  ddev_refcoco_val_1k.json       - 1,000 expressions, disjoint refs from the above (selection ONLY)
  dprobe_refcoco_train_512.json  - 512 expressions from RefCOCO train, images excluded from
                                   ALL variants' val/test images (cross-variant hygiene)
  vqa_val_5k.json                - 5,000 VQAv2 val questions (paired constraint set)
  image_manifest.txt             - unique COCO images needed by all subsets

Hygiene enforced here:
  * D_dev and eval subset share no ref_id (and no image overlap between eval and D_dev refs
    is required by the protocol, but refs are image-disjoint sampled to be safe).
  * D_probe excludes any image_id present in val/testA/testB/test of ANY RefCOCO variant.
  * RefCOCOg is the umd split (verified: val 2,573 / test 5,023 refs).
Each expression record: {uid, dataset, split, ref_id, image_id, file_name, expression, bbox_xywh, width, height}
"""
import os, json, random, zipfile, io
from datasets import load_dataset

SEED = 0
OUT = os.path.join(os.environ["GCQ_DATA"], "subsets")
os.makedirs(OUT, exist_ok=True)
rng = random.Random(SEED)

def records(ds, dataset_name, split_name):
    recs = []
    for row in ds:
        info = json.loads(row["raw_image_info"]) if isinstance(row["raw_image_info"], str) else row["raw_image_info"]
        ann = json.loads(row["raw_anns"]) if isinstance(row["raw_anns"], str) else row["raw_anns"]
        for s in row["sentences"]:
            sent = s["sent"] if isinstance(s, dict) else s
            recs.append(dict(
                dataset=dataset_name, split=split_name, ref_id=row["ref_id"],
                image_id=row["image_id"],
                file_name=f"COCO_train2014_{row['image_id']:012d}.jpg",
                expression=sent, bbox_xywh=ann["bbox"],
                width=info["width"], height=info["height"]))
    return recs

print("loading refcoco variants...")
rc  = load_dataset("jxu124/refcoco")
rcp = load_dataset("jxu124/refcocoplus")
rcg = load_dataset("jxu124/refcocog")

# --- forbidden images: any image in any variant's val/test splits ---
forbidden = set()
for ds, splits in [(rc, ["validation", "test", "testB"]),
                   (rcp, ["validation", "test", "testB"]),
                   (rcg, ["validation", "test"])]:
    for sp in splits:
        forbidden.update(ds[sp]["image_id"])
print(f"forbidden (val/test) images across variants: {len(forbidden)}")

# --- RefCOCO val: sample refs image-disjointly into eval(1k expr) and ddev(1k expr) ---
val_recs = records(rc["validation"], "refcoco", "val")
by_image = {}
for r in val_recs:
    by_image.setdefault(r["image_id"], []).append(r)
images = sorted(by_image)
rng.shuffle(images)

def take(images_iter, n_expr):
    out, used = [], []
    for img in images_iter:
        if len(out) >= n_expr: break
        out.extend(by_image[img]); used.append(img)
    return out[:n_expr], used

half = len(images) // 2
eval_recs, eval_imgs = take(images[:half], 1000)
ddev_recs, ddev_imgs = take(images[half:], 1000)
assert not (set(eval_imgs) & set(ddev_imgs)), "eval/ddev image overlap!"

# --- D_probe: RefCOCO train, excluding forbidden images ---
train_recs = [r for r in records(rc["train"], "refcoco", "train") if r["image_id"] not in forbidden]
rng.shuffle(train_recs)
probe_recs = train_recs[:512]

# --- VQAv2 val 5k ---
vqa_zip = os.path.join(os.environ["GCQ_DATA"], "vqa", "v2_Questions_Val_mscoco.zip")
ann_zip = os.path.join(os.environ["GCQ_DATA"], "vqa", "v2_Annotations_Val_mscoco.zip")
with zipfile.ZipFile(vqa_zip) as z:
    qs = json.load(io.TextIOWrapper(z.open(z.namelist()[0])))["questions"]
with zipfile.ZipFile(ann_zip) as z:
    anns = json.load(io.TextIOWrapper(z.open(z.namelist()[0])))["annotations"]
ans_by_qid = {a["question_id"]: a for a in anns}
rng.shuffle(qs)
vqa = [dict(question_id=q["question_id"], image_id=q["image_id"],
            file_name=f"COCO_val2014_{q['image_id']:012d}.jpg",
            question=q["question"],
            answers=[a["answer"] for a in ans_by_qid[q["question_id"]]["answers"]],
            multiple_choice_answer=ans_by_qid[q["question_id"]]["multiple_choice_answer"])
       for q in qs[:5000]]

# --- POPE images (from the three json files) ---
pope_files = set()
for v in ["random", "popular", "adversarial"]:
    with open(os.path.join(os.environ["GCQ_DATA"], "pope", f"coco_pope_{v}.json")) as f:
        for line in f:
            pope_files.add(json.loads(line)["image"])

# --- write everything, with uids ---
def dump(name, recs):
    for i, r in enumerate(recs): r["uid"] = f"{name}:{i}"
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(recs, f)
    print(f"{name}: {len(recs)} records")

dump("rec_eval_refcoco_val_1k", eval_recs)
dump("ddev_refcoco_val_1k", ddev_recs)
dump("dprobe_refcoco_train_512", probe_recs)
dump("vqa_val_5k", vqa)

train_imgs = {r["file_name"] for r in eval_recs} | {r["file_name"] for r in ddev_recs} | {r["file_name"] for r in probe_recs}
val_imgs = {v["file_name"] for v in vqa} | pope_files
with open(os.path.join(OUT, "image_manifest.txt"), "w") as f:
    for fn in sorted(train_imgs): f.write(f"train2014/{fn}\n")
    for fn in sorted(val_imgs): f.write(f"val2014/{fn}\n")
print(f"image manifest: {len(train_imgs)} train2014 + {len(val_imgs)} val2014 images")

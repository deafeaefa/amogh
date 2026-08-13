"""Smoke test: load Qwen3-VL-2B BF16, run grounding prompts on 3 COCO images,
parse the emitted bbox_2d JSON. Pass = parseable boxes on all prompts."""
import os, re, json, sys, time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
IMDIR = os.path.join(os.environ["GCQ_DATA"], "images/val2014")

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
print(f"loaded in {time.time()-t0:.0f}s | mem {torch.cuda.memory_allocated()/1e9:.1f} GB")

CASES = [
    ("COCO_val2014_000000000042.jpg", "the dog"),
    ("COCO_val2014_000000000073.jpg", "the motorcycle"),
    ("COCO_val2014_000000000133.jpg", "the vase of flowers"),
]

BOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')

ok = 0
for fname, expr in CASES:
    img = Image.open(os.path.join(IMDIR, fname)).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": f"Locate the {expr}, output its bbox_2d in JSON."}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = BOX_RE.search(text)
    box = [int(g) for g in m.groups()] if m else None
    W, H = img.size
    px = [box[0]*W//1000, box[1]*H//1000, box[2]*W//1000, box[3]*H//1000] if box else None
    print(f"{fname} | '{expr}' -> raw: {text.strip()[:110]}")
    print(f"   parsed [0,1000]: {box} -> pixels ({W}x{H}): {px}")
    if box and box[0] < box[2] and box[1] < box[3]:
        ok += 1

print(f"\nSMOKE TEST: {ok}/{len(CASES)} boxes parsed and well-formed -> {'PASS' if ok == len(CASES) else 'FAIL'}")
sys.exit(0 if ok == len(CASES) else 1)

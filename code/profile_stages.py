"""Profile where the 5s/sample goes: image load vs processor vs generate vs decode."""
import os, json, time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

data_dir = os.environ["GCQ_DATA"]
with open(os.path.join(data_dir, "subsets", "rec_eval_refcoco_val_1k.json")) as f:
    recs = json.load(f)[:16]

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
processor.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

t = time.time(); imgs = []
for r in recs:
    imgs.append(Image.open(os.path.join(data_dir, "images", "train2014", r["file_name"])).convert("RGB"))
t_img = time.time() - t

msgs = [[{"role":"user","content":[{"type":"image","image":im},
        {"type":"text","text":f"Locate the {r['expression']}, output its bbox_2d in JSON."}]}]
        for im, r in zip(imgs, recs)]

t = time.time()
inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt", padding=True)
t_proc = time.time() - t
inputs = inputs.to(model.device)
print("input_ids shape:", inputs["input_ids"].shape, "| pixel_values:", inputs["pixel_values"].shape if "pixel_values" in inputs else "?")

torch.cuda.synchronize(); t = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize(); t_gen = time.time() - t

t = time.time()
texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
t_dec = time.time() - t

gen_tokens = out.shape[1] - inputs["input_ids"].shape[1]
print(f"batch=16 | image_load {t_img:.1f}s | processor {t_proc:.1f}s | generate {t_gen:.1f}s ({gen_tokens} steps, {t_gen/max(gen_tokens,1)*1000:.0f} ms/step) | decode {t_dec:.1f}s")
print(f"TOTAL {(t_img+t_proc+t_gen+t_dec):.1f}s for 16 samples")

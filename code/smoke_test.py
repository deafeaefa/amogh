"""Load Qwen3-VL-2B BF16 and require parseable boxes on three COCO images."""
from __future__ import annotations

import os
import re
import sys
import time


MODEL = "Qwen/Qwen3-VL-2B-Instruct"
CASES = [
    ("COCO_val2014_000000000042.jpg", "the dog"),
    ("COCO_val2014_000000000073.jpg", "the motorcycle"),
    ("COCO_val2014_000000000133.jpg", "the vase of flowers"),
]
BOX_RE = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
)


def main() -> int:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    image_dir = os.path.join(os.environ["GCQ_DATA"], "images/val2014")
    started = time.time()
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    print(
        f"loaded in {time.time() - started:.0f}s | "
        f"mem {torch.cuda.memory_allocated() / 1e9:.1f} GB"
    )

    passed = 0
    for file_name, expression in CASES:
        with Image.open(os.path.join(image_dir, file_name)) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Locate the {expression}, output its bbox_2d in JSON."
            )},
        ]}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        text = processor.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        match = BOX_RE.search(text)
        box = [int(group) for group in match.groups()] if match else None
        width, height = image.size
        pixels = (
            [
                box[0] * width // 1000,
                box[1] * height // 1000,
                box[2] * width // 1000,
                box[3] * height // 1000,
            ]
            if box
            else None
        )
        print(f"{file_name} | {expression!r} -> raw: {text.strip()[:110]}")
        print(f"   parsed [0,1000]: {box} -> pixels ({width}x{height}): {pixels}")
        if box and box[0] < box[2] and box[1] < box[3]:
            passed += 1

    status = "PASS" if passed == len(CASES) else "FAIL"
    print(f"\nSMOKE TEST: {passed}/{len(CASES)} boxes parsed and well-formed -> {status}")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())

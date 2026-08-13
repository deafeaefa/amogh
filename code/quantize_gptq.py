"""Real GPTQ quantization of Qwen3-VL-2B via gptqmodel (text calibration = field status quo).

Usage: venv/bin/python quantize_gptq.py --bits 4 [--group 128] [--out DIR]
Saves a loadable GPTQ checkpoint; vision tower / embeddings / lm_head remain unquantized
(gptqmodel's qwen3_vl support targets the LLM decoder blocks, matching our RTN policy).
"""
import os, argparse, time
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = args.out or os.path.join(os.environ.get("GCQ_CKPTS", "/tmp"), f"qwen3vl2b-gptq-w{args.bits}g{args.group}")
    os.makedirs(out, exist_ok=True)

    # standard text calibration: wikitext-2 chunks (the field's default practice)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts, buf = [], ""
    for row in ds:
        t = row["text"].strip()
        if not t: continue
        buf += " " + t
        if len(buf) > 2000:
            texts.append(buf.strip()); buf = ""
        if len(texts) >= args.nsamples: break
    print(f"calibration: {len(texts)} text chunks")

    qc = QuantizeConfig(bits=args.bits, group_size=args.group)
    t0 = time.time()
    model = GPTQModel.load("Qwen/Qwen3-VL-2B-Instruct", qc)
    model.quantize(texts, batch_size=4)
    print(f"quantized in {time.time()-t0:.0f}s")
    model.save(out)
    print(f"SAVED {out}")

if __name__ == "__main__":
    main()

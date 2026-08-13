"""Greedy localization-sensitivity-guided bit allocation (GCQ core).

Reads $GCQ_RUNS/sensitivity.csv, ranks module-groups by s_m per extra byte,
promotes greedily to 8-bit under an average-bits budget over LLM linears,
and writes promotion configs:
  $GCQ_RUNS/promote_gcq_b{B}.json          - grounding-driven (ours)
  $GCQ_RUNS/promote_random{S}_b{B}.json    - random controls at matched budget (seeds)
Prints the allocation table for the paper.

Usage: allocate.py --budget 4.25 [--seeds 1 2 3]
"""
import os, json, csv, argparse, random, re
import torch
from transformers import AutoModelForImageTextToText
from quant_utils import LLM_FILTER

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=4.25)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    args = ap.parse_args()
    runs = os.environ["GCQ_RUNS"]

    # ---- module sizes from the model (CPU, once) ----
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16, device_map="cpu")
    sizes = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and LLM_FILTER in n:
            layer = int(re.search(r"layers\.(\d+)\.", n).group(1))
            kind = "attn" if "self_attn" in n else "mlp"
            sizes[(layer, kind)] = sizes.get((layer, kind), 0) + m.weight.numel()
    del model
    total = sum(sizes.values())

    # ---- sensitivity ----
    s = {}
    with open(os.path.join(runs, "sensitivity.csv")) as f:
        for row in csv.DictReader(f):
            if row["layer"] == "layer": continue  # duplicated header from parallel writers
            s[(int(row["layer"]), row["kind"])] = float(row["s_m"])
    missing = set(sizes) - set(s)
    assert not missing, f"sensitivity missing for {sorted(missing)[:5]}..."

    budget_params = (args.budget - 4.0) / 4.0 * total
    print(f"total quantizable params {total/1e6:.1f}M | promotion budget {budget_params/1e6:.1f}M ({args.budget} avg bits)")

    ranked = sorted(sizes, key=lambda k: s[k] / sizes[k], reverse=True)  # s_m per extra byte
    chosen, used = [], 0
    for k in ranked:
        if used + sizes[k] <= budget_params:
            chosen.append(k); used += sizes[k]
    print(f"GCQ allocation ({len(chosen)} groups, {used/1e6:.1f}M params, avg bits {4 + 4*used/total:.3f}):")
    for k in sorted(chosen): print(f"  layer {k[0]:2d} {k[1]:4s} s_m={s[k]:+.5f} params={sizes[k]/1e6:.1f}M")

    def substrings(keys):
        return [f"layers.{l}.self_attn" if kind == "attn" else f"layers.{l}.mlp" for l, kind in keys]

    with open(os.path.join(runs, f"promote_gcq_b{args.budget}.json"), "w") as f:
        json.dump({"bits": 8, "substrings": substrings(chosen), "avg_bits": 4 + 4*used/total,
                   "params_promoted": used, "groups": [list(k) for k in sorted(chosen)]}, f, indent=1)

    # ---- random controls at matched (<=) budget ----
    for seed in args.seeds:
        rng = random.Random(seed)
        pool = list(sizes); rng.shuffle(pool)
        rchosen, rused = [], 0
        for k in pool:
            if rused + sizes[k] <= used:  # match the GCQ byte budget
                rchosen.append(k); rused += sizes[k]
        with open(os.path.join(runs, f"promote_random{seed}_b{args.budget}.json"), "w") as f:
            json.dump({"bits": 8, "substrings": substrings(rchosen), "avg_bits": 4 + 4*rused/total,
                       "params_promoted": rused, "groups": [list(k) for k in sorted(rchosen)]}, f, indent=1)
        print(f"random seed {seed}: {len(rchosen)} groups, {rused/1e6:.1f}M params")

if __name__ == "__main__":
    main()

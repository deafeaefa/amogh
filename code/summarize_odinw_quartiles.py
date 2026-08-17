"""Size-stratified held-out detection recall with image-clustered intervals.

Quartile edges are computed once from the pooled ground-truth relative-area
distribution (identical across models: same images, same queries), so the bins
are outcome-independent.  Reports per-quartile recall for BF16 / uniform W4 /
GCQ plus paired image-clustered bootstrap CIs for W4-BF16 and GCQ-W4.

Usage:
  summarize_odinw_quartiles.py --bf16 R/bf16_odinwFULL.odinw.jsonl \
      --w4 R/w4rtn_odinwFULL.odinw.jsonl --gcq R/gcq_b425_odinwFULL.odinw.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


def load_gt(path):
    """Return list of (key, image_id, relative_area, matched) for GT records."""
    rows = []
    per_key = defaultdict(int)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("record_type") != "gt":
                continue
            # a (query, gt-index) key: dataset_index + category + running index
            k = (r["dataset_index"], r["cat"], per_key[(r["dataset_index"], r["cat"])])
            per_key[(r["dataset_index"], r["cat"])] += 1
            rows.append((k, r["image_id"], r["relative_area"], bool(r["matched"])))
    return rows


def recall(matched):
    return 100.0 * float(np.mean(matched)) if len(matched) else float("nan")


def clustered_ci(pairs, image_ids, resamples, rng):
    """pairs: array of per-box differences; cluster resample by image."""
    clusters = defaultdict(list)
    for d, img in zip(pairs, image_ids):
        clusters[img].append(d)
    vals = [np.asarray(v, dtype=float) for v in clusters.values()]
    n = len(vals)
    boot = np.empty(resamples)
    for b in range(resamples):
        idx = rng.integers(0, n, n)
        chunk = np.concatenate([vals[i] for i in idx])
        boot[b] = chunk.mean() * 100.0
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--w4", required=True)
    ap.add_argument("--gcq", required=True)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sets = {name: load_gt(p) for name, p in
            (("bf16", args.bf16), ("w4", args.w4), ("gcq", args.gcq))}
    keys = [set(k for k, _, _, _ in rows) for rows in sets.values()]
    common = set.intersection(*keys)
    print(f"GT boxes per model: {[len(v) for v in sets.values()]}; aligned: {len(common)}")

    aligned = {}
    for name, rows in sets.items():
        aligned[name] = {k: (img, area, m) for k, img, area, m in rows if k in common}

    order = sorted(common)
    areas = np.array([aligned["bf16"][k][1] for k in order])
    imgs = np.array([aligned["bf16"][k][0] for k in order])
    edges = np.percentile(areas, [25, 50, 75])
    qidx = np.digitize(areas, edges)  # 0..3

    rng = np.random.default_rng(args.seed)
    m = {name: np.array([aligned[name][k][2] for k in order], dtype=float)
         for name in ("bf16", "w4", "gcq")}

    print(f"\nrelative-area quartile edges: {edges[0]:.5f} {edges[1]:.5f} {edges[2]:.5f}")
    print(f"{'quartile':10s} {'n':>6s} {'BF16':>7s} {'W4':>7s} {'GCQ':>7s}"
          f" {'W4-BF16':>9s} {'95% CI':>18s} {'GCQ-W4':>8s} {'95% CI':>18s}")
    for q in range(4):
        sel = qidx == q
        n = int(sel.sum())
        d1 = m["w4"][sel] - m["bf16"][sel]
        d2 = m["gcq"][sel] - m["w4"][sel]
        lo1, hi1 = clustered_ci(d1, imgs[sel], args.resamples, rng)
        lo2, hi2 = clustered_ci(d2, imgs[sel], args.resamples, rng)
        rel = 100.0 * d1.mean() / (m["bf16"][sel].mean() or float("nan"))
        print(f"Q{q+1:<9d} {n:6d} {recall(m['bf16'][sel]):7.1f} {recall(m['w4'][sel]):7.1f}"
              f" {recall(m['gcq'][sel]):7.1f} {d1.mean()*100:+9.2f} [{lo1:+6.2f},{hi1:+6.2f}]"
              f" {d2.mean()*100:+8.2f} [{lo2:+6.2f},{hi2:+6.2f}]   rel {rel:+.0f}%")

    print("\noverall:")
    d1 = m["w4"] - m["bf16"]; d2 = m["gcq"] - m["w4"]
    lo1, hi1 = clustered_ci(d1, imgs, args.resamples, rng)
    lo2, hi2 = clustered_ci(d2, imgs, args.resamples, rng)
    print(f"  recall BF16={recall(m['bf16']):.1f} W4={recall(m['w4']):.1f} GCQ={recall(m['gcq']):.1f}")
    print(f"  W4-BF16 {d1.mean()*100:+.2f} [{lo1:+.2f},{hi1:+.2f}]  "
          f"GCQ-W4 {d2.mean()*100:+.2f} [{lo2:+.2f},{hi2:+.2f}]")


if __name__ == "__main__":
    main()

"""Render the single two-panel figure for the AAAI/AIR-FM 4-page paper.

(a) How much W4 damage each metric reveals (the 'benchmarks hide it' evidence).
(b) Per-layer attention sensitivity for the grounding vs VQA probes (the band).
Okabe-Ito colors, grayscale-safe hatching, sized 7.0in x 2.15in for figure*.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.environ["GCQ_RUNS"]
OUT = "/usr4/spclpgm/eric1/GCQ/paper-aaai/fig_aaai.pdf"
C_G, C_V, C_GRAY = "#0072B2", "#D55E00", "#9AA0A6"

def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.15), constrained_layout=True)

# ---- Panel A: damage revealed per metric (W4 - BF16, absolute points) ----
labels = ["VQAv2", "POPE", "REC\n@0.5", "mean\nGIoU", "IoU\n.50:.95"]
vals   = [-3.14, -0.62, -3.78, -5.58, -6.64]
err_lo = [0, 0, 3.78-2.95, 5.58-4.83, 6.64-5.82]
err_hi = [0, 0, 4.62-3.78, 6.34-5.58, 7.45-6.64]
colors = [C_GRAY, C_GRAY, C_G, C_G, C_G]
bars = axA.bar(labels, vals, yerr=[err_lo, err_hi], width=0.62, color=colors,
               capsize=2.5, error_kw=dict(lw=0.9, ecolor="#333"))
for b in bars[:2]:
    b.set_hatch("//"); b.set_edgecolor("#555")
axA.axhline(0, color="#333", lw=0.8)
for b, v, e in zip(bars, vals, err_lo):
    axA.annotate(f"{v:.1f}", (b.get_x()+b.get_width()/2, v-e), ha="center",
                 va="top", fontsize=7.5, xytext=(0, -2), textcoords="offset points")
axA.set_ylim(-8.4, 1.2)
axA.set_ylabel("W4 $-$ BF16 (points)", fontsize=8.5)
axA.tick_params(labelsize=8)
axA.set_title("(a) Damage revealed depends on the metric", fontsize=9)
axA.annotate("usually reported", (0.5, 0.55), fontsize=7, color="#555", ha="center")
style(axA)

# ---- Panel B: per-layer attention sensitivity, grounding vs VQA ----
def load(fname):
    d = {}
    for r in csv.DictReader(open(os.path.join(RUNS, fname))):
        if r["layer"] == "layer":
            continue
        d[(int(r["layer"]), r["kind"])] = 1000 * float(r["s_m"])
    return d

rec_s, vqa_s = load("sensitivity.csv"), load("sensitivity_vqa.csv")
layers = sorted({l for l, k in rec_s if k == "attn"})
w = 0.42
rec_v = [rec_s[(l, "attn")] for l in layers]
vqa_v = [vqa_s[(l, "attn")] for l in layers]
rec_n = [v / max(abs(x) for x in rec_v) for v in rec_v]
vqa_n = [v / max(abs(x) for x in vqa_v) for v in vqa_v]
axB.bar([l - w/2 for l in layers], rec_n, width=w, color=C_G, label="grounding probe")
axB.bar([l + w/2 for l in layers], vqa_n, width=w, color=C_V, label="VQA probe",
        hatch="///", edgecolor="#7a3300", lw=0.25)
axB.axvspan(9.5, 17.5, color=C_G, alpha=0.10, lw=0)
axB.axhline(0, color="#888", lw=0.6)
axB.annotate("protected band", (13.5, 1.02), ha="center", fontsize=7, color=C_G)
axB.set_ylim(-1.15, 1.45)
axB.set_xlabel("Decoder layer (attention modules)", fontsize=8.5)
axB.set_ylabel("sensitivity\n(per-probe max = 1)", fontsize=8.5)
axB.set_xticks(range(0, 28, 4))
axB.tick_params(labelsize=8)
axB.legend(frameon=False, fontsize=7, loc="lower left", ncol=2,
           handlelength=1.2, columnspacing=1.0)
axB.set_title(r"(b) Band-localized, unrelated to VQA ($\rho\!=\!0.001$)", fontsize=9)
style(axB)

fig.savefig(OUT)
print("wrote", OUT)

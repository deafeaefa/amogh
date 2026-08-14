"""Render Figure 1 (retention + allocation comparison) and Figure 2 (sensitivity maps)
from results.csv / sensitivity*.csv into paper/ as PDF.

Colors: Okabe-Ito (CVD-safe): blue #0072B2 grounding/attn, vermillion #D55E00 VQA,
green #009E73 POPE, orange #E69F00 mlp. Floors: REC 3.0, VQA 36.84, POPE 50.0.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.environ["GCQ_RUNS"]
PAPER = "/usr4/spclpgm/eric1/GCQ/paper"
C_G, C_V, C_P, C_M = "#0072B2", "#D55E00", "#009E73", "#E69F00"
GRAY = "#9AA0A6"

def style(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)

def rprime(s, bf, floor): return 100 * (s - floor) / (bf - floor)

# ---------------- Figure 1 ----------------
# Panel A: floor-corrected retention vs avg bits (uniform RTN points)
bits =      [8,     4,     3]
rec_r  = [rprime(a, 88.90, 3.00) for a in (88.70, 84.50, 0.00)]
vqa_r  = [None] + [rprime(a, 81.14, 36.84) for a in (77.95, 1.34)]
pope_r = [None] + [rprime(a, 89.39, 50.00) for a in (88.77, 50.42)]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 1.95), constrained_layout=True)
xs = range(len(bits))
axA.plot(xs, rec_r, "-o", color=C_G, lw=2, ms=5, label="Grounding (REC)")
axA.plot(xs[1:], vqa_r[1:], "-s", color=C_V, lw=2, ms=5, label="VQAv2")
axA.plot(xs[1:], pope_r[1:], "-^", color=C_P, lw=2, ms=5, label="POPE")
axA.scatter([1], [rprime(86.60, 88.90, 3.00)], marker="D", s=42, color=C_G, zorder=5)
axA.scatter([1], [rprime(87.30, 88.90, 3.00)], marker="D", s=42, facecolor="white", edgecolor=C_G, zorder=5)
axA.annotate("GCQ 4.25", (1, rprime(86.60, 88.90, 3.00)), textcoords="offset points",
             xytext=(10, -9), fontsize=7.5, color=C_G)
axA.annotate("GCQ 4.5", (1, rprime(87.30, 88.90, 3.00)), textcoords="offset points",
             xytext=(10, 4), fontsize=7.5, color=C_G)
axA.annotate("global\ncollapse", (2, 8), textcoords="offset points", xytext=(-4, 12),
             fontsize=7.5, color="#555", ha="right")
axA.set_xticks(list(xs)); axA.set_xticklabels(["W8", "W4", "W3"])
axA.set_ylabel("Floor-corrected retention (%)")
axA.set_xlabel("Uniform RTN weight width")
axA.set_ylim(-4, 104)
axA.legend(frameon=False, fontsize=7.5, loc="lower left")
axA.set_title("(a) Capability retention under RTN", fontsize=9)
style(axA)

# Panel B: allocation policies at matched memory (avg 4.25 bits)
labels = ["Uniform\nW4", "Random\n(3 seeds)", "VQA-\ndriven", "GCQ\n(ours)"]
vals   = [84.50, 84.87, 85.10, 86.60]
errs   = [0, 0.31, 0, 0]
colors = [GRAY, GRAY, C_V, C_G]
bars = axB.bar(labels, vals, yerr=errs, width=0.62, color=colors, capsize=3,
               error_kw=dict(lw=1, ecolor="#555"))
axB.axhline(88.90, color="#333", lw=1, ls=(0, (4, 3)))
axB.annotate("BF16 ceiling 88.9", (3.45, 88.9), fontsize=7.5, color="#333",
             ha="right", va="bottom")
for b, v in zip(bars, vals):
    axB.annotate(f"{v:.1f}", (b.get_x() + b.get_width()/2, v), ha="center",
                 va="bottom", fontsize=8)
axB.set_ylim(82, 90.4)
axB.set_ylabel("REC acc@0.5 (%)")
axB.set_title("(b) Allocation policy at matched memory (+88M params)", fontsize=9)
style(axB)
fig.savefig(os.path.join(PAPER, "fig1_retention.pdf"))
print("fig1 written")

# ---------------- Figure 2 ----------------
def load_sens(fname):
    d = {}
    for r in csv.DictReader(open(os.path.join(RUNS, fname))):
        if r["layer"] == "layer": continue
        d[(int(r["layer"]), r["kind"])] = 1000 * float(r["s_m"])  # milli-nats
    return d

rec_s, vqa_s = load_sens("sensitivity.csv"), load_sens("sensitivity_vqa.csv")
layers = sorted({l for l, _ in rec_s})
fig2, axes = plt.subplots(2, 1, figsize=(8.6, 2.15), sharex=True, constrained_layout=True)
for ax, (title, d) in zip(axes, [("Grounding probe (coordinate-token KL reduction)", rec_s),
                                  ("VQA probe (answer-token KL reduction)", vqa_s)]):
    w = 0.4
    ax.bar([l - w/2 for l in layers], [d[(l, "attn")] for l in layers], width=w,
           color=C_G, label="attention")
    ax.bar([l + w/2 for l in layers], [d[(l, "mlp")] for l in layers], width=w,
           color=C_M, label="MLP")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_ylabel("$s_m$ (m·nats)")
    ax.set_title(title, fontsize=9)
    style(ax)
axes[0].axvspan(9.5, 17.5, color=C_G, alpha=0.08, lw=0)
top = max(rec_s.values())
axes[0].set_ylim(None, top * 1.28)
axes[0].annotate("layers 10–17 attention band", (13.5, top * 1.14),
                 ha="center", fontsize=7.5, color=C_G)
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
axes[1].set_xlabel("Transformer layer")
axes[1].set_xticks(range(0, 28, 2))
fig2.savefig(os.path.join(PAPER, "fig2_sensitivity.pdf"))
print("fig2 written")

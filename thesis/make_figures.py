# -*- coding: utf-8 -*-
"""Generate thesis figures (vector PDF) — publication-quality styling.

Palette: Paul Tol "bright" (colorblind-safe, standard in the literature).
Typography: Times New Roman serif to match the document body.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT)

# ---------------------------------------------------------------- style
C = {
    "blue":   "#4477AA",
    "red":    "#CC3311",
    "green":  "#009988",
    "purple": "#AA4499",
    "grey":   "#888888",
    "orange": "#EE7733",
}

# custom sequential map: white -> Tol blue -> dark navy (heatmaps)
from matplotlib.colors import LinearSegmentedColormap
SEQ = LinearSegmentedColormap.from_list(
    "tolseq", ["#f7f7f7", "#c6dbef", "#6baed6", "#2171b5", "#084594"])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

def style_ax(ax):
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)

# ---------------------------------------------------------------- Fig 1: scaling curve
d = json.load(open(os.path.join(ROOT, "out", "ens_lite_3class_v3.json"),
                   encoding="utf-8"))

pts = [
    ("1K",    952,    0.7883, 0.7237),
    ("5K",    4751,   0.8033, 0.7535),
    ("19K",   19001,  0.8433, 0.8016),
    ("48K",   47501,  0.8550, 0.8146),
    ("160K",  160000, 0.8750, 0.8463),
]
names = [p[0] for p in pts]
ns = np.array([p[1] for p in pts], dtype=float)
acc = np.array([p[2] for p in pts])
f1 = np.array([p[3] for p in pts])

fig, ax = plt.subplots(figsize=(3.35, 2.35))
style_ax(ax)
ax.plot(ns, acc, "o-", color=C["blue"], ms=4.5, lw=1.4,
        label="Accuracy", zorder=3)
ax.plot(ns, f1, "s-", color=C["red"], ms=4, lw=1.4,
        label="Macro-F1", zorder=3)
for x, a in zip(ns, acc):
    ax.annotate(f"{a:.3f}", (x, a), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=6.5, color=C["blue"])
# deployed-ensemble reference lines: accuracy (solid dots) and macro-F1
ax.axhline(0.8950, color=C["grey"], ls=":", lw=0.9)
ax.axhline(0.8594, color=C["grey"], ls=(0, (3, 2, 1, 2)), lw=0.9)
ax.text(900, 0.902, "deployed acc = 0.895",
        fontsize=6.5, color=C["grey"], ha="right")
ax.text(900, 0.852, "deployed F1 = 0.859",
        fontsize=6.5, color=C["grey"], ha="right")
ax.set_xscale("log")
ax.set_xlabel("Pseudo-labeled training rows (log scale)")
ax.set_ylabel("Gold-600 score")
ax.set_xticks(ns)
ax.set_xticklabels(names)
ax.set_ylim(0.70, 0.92)
ax.legend(loc="lower right")
ax.set_title("Scaling with pseudo-labeled data (no gold in training)")
fig.savefig(f"{OUT}/fig_scaling.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: confusion matrix
cm = np.array(d["oof"]["confusion"])
labels = ["Positive", "Neutral", "Negative"]

fig, ax = plt.subplots(figsize=(2.9, 2.6))
im = ax.imshow(cm, cmap=SEQ, aspect="equal")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Deployed v3 confusion matrix ($n=600$)")
thr = cm.max() * 0.55
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thr else "black",
                fontsize=9, fontweight="bold")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.5)
fig.savefig(f"{OUT}/fig_confusion.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 3: calibration
y = np.array(d["y"]); probs = np.array(d["probs"])

def ece_and_bins(probs_in, T=1.0, n_bins=10):
    # match web/app.py: logits = log(p)/T, then softmax
    logits = np.log(np.clip(probs_in, 1e-9, 1.0)) / T
    z = np.exp(logits - logits.max(1, keepdims=True))
    z = z / z.sum(1, keepdims=True)
    p = z.max(1); c = (z.argmax(1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, accs, sizes = [], [], []
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p <= hi if hi == 1 else p < hi)
        if m.sum() == 0:
            continue
        xs.append(p[m].mean()); accs.append(c[m].mean()); sizes.append(m.sum())
        ece += m.sum() / len(p) * abs(c[m].mean() - p[m].mean())
    return np.array(xs), np.array(accs), np.array(sizes), ece

x0, a0, s0, e0 = ece_and_bins(probs, 1.0)
x1, a1, s1, e1 = ece_and_bins(probs, 0.37)

# confidence histogram (sample counts per equal-width bin)
def conf_hist(probs_in, T=1.0, n_bins=10):
    logits = np.log(np.clip(probs_in, 1e-9, 1.0)) / T
    z = np.exp(logits - logits.max(1, keepdims=True))
    z = z / z.sum(1, keepdims=True)
    p = z.max(1)
    bins = np.linspace(0, 1, n_bins + 1)
    counts = [int(((p >= lo) & (p <= hi if hi == 1 else p < hi)).sum())
              for lo, hi in zip(bins[:-1], bins[1:])]
    return bins, counts

hb, hc = conf_hist(probs, 1.0)

fig, (ax, axh) = plt.subplots(
    2, 1, figsize=(4.4, 3.1), sharex=True,
    gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08})
style_ax(ax)
ax.plot([0, 1], [0, 1], color=C["grey"], ls="--", lw=0.9, zorder=1)
ax.plot(x0, a0, "o-", color=C["red"], ms=4, lw=1.3,
        label=f"Uncalibrated (ECE={e0:.3f})", zorder=3)
ax.plot(x1, a1, "s-", color=C["green"], ms=4, lw=1.3,
        label=f"$T$=0.37 (ECE={e1:.3f})", zorder=3)
ax.set_ylabel("Accuracy")
ax.set_title("Calibration via temperature scaling")
ax.legend(loc="lower left")
ax.set_xlim(0.3, 1.02); ax.set_ylim(0.45, 1.02)
axh.bar(hb[:-1], hc, width=0.1, color=C["grey"], alpha=0.55,
        align="edge", zorder=3)
axh.set_xlabel("Confidence")
axh.set_ylabel("Count", fontsize=7.5)
axh.tick_params(labelsize=7)
axh.spines.top.set_visible(False)
axh.spines.right.set_visible(False)
axh.grid(False)
fig.savefig(f"{OUT}/fig_calibration.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 4: rating crosstab
ct = json.load(open(os.path.join(ROOT, "out", "corpus_rating_crosstab.json"),
                    encoding="utf-8"))
rc = ct["rating_pred_row_normalized"]
M = np.array([[rc[str(r)][c] for c in ("pos", "neu", "neg")]
              for r in range(1, 6)])

fig, ax = plt.subplots(figsize=(3.1, 2.5))
im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(3)); ax.set_yticks(range(5))
ax.set_xticklabels(["Positive", "Neutral", "Negative"])
ax.set_yticklabels([f"{r}-star" for r in range(1, 6)])
ax.set_xlabel("Predicted sentiment"); ax.set_ylabel("Star rating")
ax.set_title("Rating vs. predicted sentiment (272K pool)")
for i in range(5):
    for j in range(3):
        ax.text(j, i, f"{M[i,j]*100:.1f}", ha="center", va="center",
                fontsize=7,
                color="white" if M[i, j] > 0.52 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Row share", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.5)
fig.savefig(f"{OUT}/fig_rating_crosstab.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 5: lab500 comparison
fig, ax = plt.subplots(figsize=(4.4, 2.4))
style_ax(ax)
cats = ["v3 vs human", "Qwen3-4B vs human",
        "v3 vs consensus", "v3 binary vs human"]
ns_ = ["$n$=500", "$n$=500", "$n$=331", "$n$=255"]
vals = [0.604, 0.662, 0.740, 0.918]
cols = [C["blue"], C["purple"], C["orange"], C["green"]]
bars = ax.bar(range(4), vals, color=cols, width=0.6, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
            ha="center", fontsize=7.5)
ax.axhline(0.5, color=C["grey"], ls=":", lw=0.8)
ax.text(3.42, 0.512, "chance", fontsize=6.5, color=C["grey"], ha="right")
ax.set_xticks(range(4))
ax.set_xticklabels([f"{c}\n{n}" for c, n in zip(cats, ns_)], fontsize=6.8)
ax.tick_params(axis="x", pad=2)
ax.set_ylabel("Accuracy")
ax.set_ylim(0.4, 1.0)
ax.set_title("Out-of-distribution check: fresh corpus sample")
fig.savefig(f"{OUT}/fig_lab500.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 6: lab2000 OOD + IAA
# Numbers: mix rows are 5-fold cross-fit on ood1000 (mix_crossfit.py);
# SFT/LLM rows from sft_results + ds_*_labels.jsonl.
fig, (axa, axb) = plt.subplots(
    1, 2, figsize=(6.6, 2.55),
    gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.16})
style_ax(axa); style_ax(axb)

systems = ["v3\ngold-600", "SFT\ngold+al1k", "mix\ngold+al17k",
           "DS-V4.1-\nFlash", "DS-V4-\nPro"]
acc3 = [0.650, 0.714, 0.766, 0.820, 0.844]
bin_ = [0.850, 0.880, 0.918, 0.976, np.nan]
cols_sys = [C["blue"], C["blue"], C["green"], C["purple"], C["purple"]]
x = np.arange(len(systems))
w_ = 0.36
b1 = axa.bar(x - w_ / 2, acc3, w_, color=cols_sys, alpha=0.92, zorder=3,
             label="3-class")
b2 = axa.bar(x + w_ / 2, [v if not np.isnan(v) else 0 for v in bin_],
             w_, color=cols_sys, alpha=0.45, hatch="///", zorder=3,
             label="binary (pos/neg)")
for i, v in enumerate(acc3):
    axa.text(x[i] - w_ / 2, v + 0.012, f"{v:.3f}", ha="center",
             fontsize=6, color=C["blue"])
for i, v in enumerate(bin_):
    if not np.isnan(v):
        axa.text(x[i] + w_ / 2, v + 0.012, f"{v:.3f}", ha="center",
                 fontsize=6, color=C["blue"])
axa.axhline(0.715, color=C["red"], ls="--", lw=0.9, zorder=2)
axa.text(4.45, 0.722, "human IAA 71.5%", fontsize=6.2, color=C["red"],
         ha="right")
axa.set_xticks(x)
axa.set_xticklabels(systems, fontsize=6.4)
axa.set_ylabel("Accuracy on ood1000")
axa.set_ylim(0.4, 1.02)
axa.legend(loc="upper left", fontsize=6.3, handlelength=1.2,
           handletextpad=0.4, borderpad=0.3, labelspacing=0.3)
axa.set_title("(a) OOD accuracy by system", fontsize=8.5)

# panel b: raw pairwise agreement on the same 200 rows
axb.bar([0, 1], [0.835, 0.715], color=[C["purple"], C["orange"]],
        width=0.55, zorder=3)
axb.text(0, 0.845, "83.5%", ha="center", fontsize=7.5, fontweight="bold")
axb.text(1, 0.725, "71.5%", ha="center", fontsize=7.5, fontweight="bold")
axb.set_xticks([0, 1])
axb.set_xticklabels(["two LLM\nannotators\n(diff.\nfamilies)",
                     "two\nhumans"], fontsize=6.4)
axb.set_ylabel("Raw agreement rate")
axb.set_ylim(0.5, 1.0)
axb.set_title("(b) Inter-annotator agreement (n=200)", fontsize=8.5)
fig.savefig(f"{OUT}/fig_lab2000.pdf")
plt.close(fig)

print("figures regenerated")
for f in sorted(os.listdir(OUT)):
    if f.startswith("fig_"):
        print(f, os.path.getsize(os.path.join(OUT, f)))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the NAACL SRW paper (paper/).

fig1_pipeline.pdf  — model + data flow (audit-aware)
fig2_provenance.pdf — annotation-source provenance of eval sets
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "paper"
plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

# ---------------- Figure 1: pipeline ----------------
fig, ax = plt.subplots(figsize=(3.4, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis("off")

def box(x, y, w, h, text, fc="#dbeafe", ec="#1d4ed8", fs=7.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                 fc=fc, ec=ec, lw=0.9))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

def arr(x1, y1, x2, y2, style="-|>", lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=9, lw=lw, linestyle=ls,
                 color="#374151", shrinkA=1, shrinkB=1))

# data sources (left column)
box(0.15, 11.6, 2.5, 1.15, "Gold-600\n600 human labels", "#dcfce7", "#16a34a")
box(0.15, 9.9, 2.5, 1.15, "911K corpus\nstar-rule + stu3c pseudo", "#fef9c3", "#a16207")
box(0.15, 8.2, 2.5, 1.15, "AL pools\n~17K LLM labels", "#fef9c3", "#a16207")

# channels (middle)
box(3.3, 11.6, 3.0, 1.15, "MacBERT student\n(per-fold distill, w=1)", "#dbeafe")
box(3.3, 9.9, 3.0, 1.15, "MegatronBERT-1.3B\nfrozen MLP judge (w=2)", "#dbeafe")
box(3.3, 8.2, 3.0, 1.15, "Corpus neutral\ndetector (Leak A!)", "#fee2e2", "#dc2626", fs=6.8)
box(3.3, 6.4, 3.0, 1.15, "Neutral head\n(frozen feats, w=0.6 in pn)", "#dbeafe", fs=6.8)

# blend (right)
box(7.1, 9.4, 2.75, 1.7, "dep = (s + 2j)/3\npn = .6·head + .4·corpus\nspec → final blend", "#ede9fe", "#6d28d9", fs=6.8)
box(7.1, 7.2, 2.75, 1.15, "Lexicon guard\n(ultra-short slang)", "#f1f5f9", "#64748b", fs=6.8)

arr(2.65, 12.15, 3.3, 12.15); arr(2.65, 10.45, 3.3, 10.45)
arr(2.65, 8.75, 3.3, 8.75, ls="--"); arr(2.65, 8.75, 3.3, 6.95, ls="--")
arr(6.3, 12.15, 7.1, 10.6); arr(6.3, 10.45, 7.1, 10.6)
arr(6.3, 8.75, 7.1, 10.0); arr(6.3, 6.95, 7.1, 7.8)
arr(7.1, 9.4, 7.1, 8.35)

ax.text(5.0, 13.55, "Deployed 3-class pipeline (weights frozen)", ha="center",
        fontsize=8, fontweight="bold")
ax.text(5.0, 5.35, "Leak B: soft targets trained on val-fold info (α=1.0 removes)\n"
        "Leak A: corpus detector labels ← stu3c ← all-600 gold",
        ha="center", fontsize=6.3, color="#dc2626")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_pipeline.pdf")
plt.close(fig)

# ---------------- Figure 2: annotation provenance ----------------
fig, ax = plt.subplots(figsize=(3.3, 2.5))
sets = ["gold-600", "ood1000", "fresh500", "lab500", "AL train\n(~17K)"]
human = [100, 20, 0, 0, 1.6]          # % human-adjudicated (479/2000→ood20%, al279/17k≈1.6%)
llm   = [0, 80, 100, 100, 98.4]
# stacked bars: bottom=human, top=llm
x = np.arange(len(sets))
b1 = ax.bar(x, human, 0.62, color="#16a34a", label="human-adjudicated")
b2 = ax.bar(x, llm, 0.62, bottom=human, color="#f59e0b", label="LLM consensus/teacher")
for xi, (h, l) in enumerate(zip(human, llm)):
    if h > 8: ax.text(xi, h/2, f"{h:.0f}%", ha="center", va="center", fontsize=6.5, color="white")
    if l > 15: ax.text(xi, h + l/2, f"{l:.0f}%", ha="center", va="center", fontsize=6.5, color="white")
ax.set_xticks(x); ax.set_xticklabels(sets, fontsize=6.8)
ax.set_ylabel("label provenance (%)", fontsize=7.5)
ax.set_ylim(0, 112)
ax.legend(fontsize=6.5, loc="upper right", frameon=False)
ax.set_title("Annotation-source composition of each split", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_provenance.pdf")
plt.close(fig)
print("figures written")

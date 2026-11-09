#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the NAACL SRW paper (paper/).

fig1_pipeline.pdf  — horizontal model+data flow (audit-aware)
fig2_provenance.pdf — annotation-source provenance of eval sets
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "paper"
plt.rcParams.update({
    "font.size": 7.5, "font.family": "serif",
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

# ---------------- Figure 1: pipeline (horizontal) ----------------
fig, ax = plt.subplots(figsize=(6.9, 2.9))
ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")

def box(x, y, w, h, text, fc="#dbeafe", ec="#1d4ed8", fs=6.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                 fc=fc, ec=ec, lw=0.9))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs)

def arr(x1, y1, x2, y2, ls="-", lw=0.9, color="#374151"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=8, lw=lw, linestyle=ls, color=color,
                 shrinkA=1, shrinkB=1))

# column headers
ax.text(1.35, 5.55, "Training signals", ha="center", fontsize=7.5, fontweight="bold", color="#374151")
ax.text(4.8, 5.55, "Model channels", ha="center", fontsize=7.5, fontweight="bold", color="#374151")
ax.text(9.3, 5.55, "Neutral expert", ha="center", fontsize=7.5, fontweight="bold", color="#374151")
ax.text(12.6, 5.55, "Output", ha="center", fontsize=7.5, fontweight="bold", color="#374151")

# training signals (left)
box(0.1, 3.9, 2.5, 1.15, "Gold-600\n600 human labels", "#dcfce7", "#16a34a")
box(0.1, 2.35, 2.5, 1.15, "911K corpus\nstar-rule + stu3c pseudo", "#fef9c3", "#a16207")
box(0.1, 0.8, 2.5, 1.15, "AL pools\n~17K LLM labels", "#fef9c3", "#a16207")

# channels (middle)
box(3.6, 3.9, 2.4, 1.15, "MacBERT student\n(w=1)", "#dbeafe")
box(3.6, 2.35, 2.4, 1.15, "MegatronBERT-1.3B\nfrozen MLP judge (w=2)", "#dbeafe", fs=6.4)

# neutral expert (right-middle)
box(6.9, 3.15, 2.4, 1.7, "pn = 0.6·head\\\n+ 0.4·corpus\\\n(w=0.5 blend)", "#ede9fe", "#6d28d9", fs=6.4)
box(6.9, 1.1, 2.4, 1.5, "dep = (s+2j)/3\nspec → final", "#f1f5f9", "#64748b", fs=6.6)

# output
box(10.2, 1.9, 2.2, 1.5, "3-class\n(+ binary head,\nlexicon guard)", "#dcfce7", "#16a34a", fs=6.6)

# arrows
arr(2.6, 4.45, 3.6, 4.45); arr(2.6, 2.9, 3.6, 2.9)
arr(2.6, 1.35, 3.6, 4.45, ls="--"); arr(2.6, 1.35, 6.9, 3.6, ls="--")
arr(6.0, 4.45, 6.9, 3.85); arr(6.0, 2.9, 6.9, 3.6, ls=":")
arr(9.3, 3.6, 10.2, 2.7); arr(9.3, 1.85, 10.2, 2.4, ls=":")

# leak annotations
ax.text(5.0, 0.15, "Leak B: soft targets (α=0.5) carry val-fold info → α=1.0 removes",
        ha="center", fontsize=6.3, color="#dc2626")
ax.text(8.1, 0.15, "Leak A: corpus detector labels ← stu3c ← all 600 gold",
        ha="center", fontsize=6.3, color="#dc2626")
ax.plot([4.8,4.8],[0.35,0.5],color="#dc2626",lw=0.5)
ax.plot([8.1,8.1],[0.35,0.5],color="#dc2626",lw=0.5)

fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/fig1_pipeline.pdf")
plt.close(fig)

# ---------------- Figure 2: annotation provenance ----------------
fig, ax = plt.subplots(figsize=(3.3, 2.4))
sets = ["gold-600", "ood1000", "fresh500", "lab500", "AL train\n(~17K)"]
human = [100, 20, 0, 0, 1.6]
llm   = [0, 80, 100, 100, 98.4]
x = np.arange(len(sets))
ax.bar(x, human, 0.62, color="#16a34a", label="human-adjudicated")
ax.bar(x, llm, 0.62, bottom=human, color="#f59e0b", label="LLM consensus/teacher")
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

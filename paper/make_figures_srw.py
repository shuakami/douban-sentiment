#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the NAACL SRW paper (paper/).

fig3_al_scaling.pdf — Dev-1000 accuracy vs LLM-labelled pool size
(fig_star_length.pdf is made by make_fig_star_length.py;
 fig_h400_curve.pdf by ../scripts/make_fig_h400_curve.py after spec S3)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import os
OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.size": 7.5, "font.family": "serif",
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

# ---------------- Dev-1000 scaling curve (verified values, 01-VERIFIED-STATE.md) ----
fig, ax = plt.subplots(figsize=(3.0, 1.85))
k = [1, 6, 11, 14, 17]; acc = [72.0, 74.2, 74.9, 75.5, 77.1]
ax.plot(k, acc, marker="o", ms=3.5, lw=1.2, color="#1d4ed8")
for x, y in zip(k, acc):
    ax.text(x, y + 0.55, f"{y:.1f}", ha="center", fontsize=6.5)
ax.axhline(65.0, color="#b91c1c", lw=0.9, ls="--")
ax.text(17.2, 65.5, "GoldEns (no LLM labels): 65.0", ha="right", fontsize=6.3, color="#b91c1c")
ax.set_xticks(k); ax.set_ylim(63, 79.5)
ax.set_xlabel("LLM-labelled training reviews (thousands)")
ax.set_ylabel("Dev-1000 accuracy (%)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/fig3_al_scaling.pdf")
plt.close(fig)
print("figures written")

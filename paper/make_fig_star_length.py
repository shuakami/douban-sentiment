#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_star_length.pdf — star-rule agreement vs review length, by label source.

Reads analysis/a1_star_agreement_triangulation.json (produced by
analysis/a1_star_agreement_triangulation.py; CPU-only, de-identified files).
Usage: python make_fig_star_length.py <a1_json> <out_pdf>
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "../analysis/a1_star_agreement_triangulation.json"
dst = sys.argv[2] if len(sys.argv) > 2 else "fig_star_length.pdf"
d = {p["name"]: p for p in json.load(open(src, encoding="utf-8"))["populations"]}
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})
bins = ["<=20", "21-50", "51-100", ">100"]
series = [
    ("human400 (corpus, length-stratified, blind human)", "Human-400 (blind human)", "#b91c1c", "o", "-"),
    ("ood1000 all (uniform corpus; final labels)", "Dev-1000 (released labels)", "#1d4ed8", "s", "-"),
    ("ood1000 DeepSeek-Flash zero-shot labels", "Dev-1000 (DeepSeek-Flash)", "#15803d", "^", "--"),
    ("ood1000 LLM pass A only", "Dev-1000 (LLM pass A)", "#7c3aed", "v", ":"),
]
fig, ax = plt.subplots(figsize=(3.05, 2.05))
for key, lab, c, m, ls in series:
    y = [d[key]["by_len"][b]["agree_pct"] for b in bins]
    ax.plot(range(4), y, marker=m, ms=3.5, lw=1.1, color=c, ls=ls, label=lab)
g = d["gold600 (single film, human)"]["agree_pct"]
ax.axhline(g, color="#374151", lw=0.8, ls=(0, (4, 2)))
ax.text(-0.12, g + 0.8, f"Gold-600 (single film): {g:.1f}%", fontsize=6.3, color="#374151")
ax.set_xticks(range(4)); ax.set_xticklabels(["$\\leq$20", "21\u201350", "51\u2013100", "$>$100"])
ax.set_xlabel("Review length (characters)"); ax.set_ylabel("Star-rule agreement (%)")
ax.set_ylim(34, 80); ax.grid(axis="y", lw=0.3, alpha=0.5)
ax.legend(fontsize=5.6, frameon=False, loc="lower right", handlelength=2.2, labelspacing=0.25)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(dst)
print("wrote", dst)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate neutral-specialist combinations for 3-class sentiment.

The specialist is a binary neutral-vs-rest model (OOF probs (n,2)).
Its pos/neg split is borrowed from the deployed 3-class ensemble so the
constructed 3-class channel is:
    P3_spec = [(1-pn)*r_pos, pn, (1-pn)*r_neg]
with r = deployed pos/neg ratio renormalized.

Combination formulas tested (all on OOF, 600 rows):
  A) ensemble mean:   dep + w * spec          (w swept)
  B) neutral-column blend: dep pos/neg kept, neutral col = a*dep + (1-a)*pn
  C) logit-average of dep and spec channels

Usage: python scripts/eval_neu_decomp.py out/distill_macbert_neu_*.json [...]
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from ensemble_eval import load_result, score  # noqa: E402


def load_spec(path):
    r = json.load(open(path, encoding="utf-8"))
    p = np.array(r["probs"], dtype=np.float64)  # (n,2) [non-neu, neu]
    assert p.shape[1] == 2 and len(p) == 600, f"{path}: bad shape {p.shape}"
    return os.path.basename(path), p[:, 1]      # p_neu


def main():
    files = sys.argv[1:] or sorted(
        glob.glob(os.path.join(ROOT, "out", "distill_*_neu_*.json")) +
        glob.glob(os.path.join(ROOT, "out", "headmlp_neu_*.json")))
    if not files:
        print("no specialist result files found")
        return

    _, dep, y = load_result("ens_lite_3class.json")   # (n,3) [pos,neu,neg]
    dep = np.asarray(dep, dtype=np.float64)
    dep = dep / dep.sum(1, keepdims=True)
    acc0, f10 = score("3class", y, dep)[:2]
    print(f"deployed:            acc={acc0:.4f} f1={f10:.4f}")

    # pos/neg ratio from deployed (renormalized over non-neutral mass)
    posneg = dep[:, [0, 2]]
    posneg = posneg / posneg.sum(1, keepdims=True)

    for path in files:
        name, pn = load_spec(path)
        pn = np.clip(pn, 1e-6, 1 - 1e-6)

        # specialist-only 3-class channel
        spec = np.stack([(1 - pn) * posneg[:, 0], pn, (1 - pn) * posneg[:, 1]], axis=1)
        acc, f1 = score("3class", y, spec)[:2]
        print(f"\n== {name}  (specialist alone) acc={acc:.4f} f1={f1:.4f}")

        # A) mean ensemble sweep
        best = None
        for w in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            pr = (dep + w * spec) / (1 + w)
            acc, f1 = score("3class", y, pr)[:2]
            tag = f"A mean w={w}"
            print(f"  {tag:16s} acc={acc:.4f} f1={f1:.4f}")
            if best is None or f1 > best[1]:
                best = (tag, f1, acc)

        # B) neutral-column blend sweep
        for a in [0.3, 0.5, 0.7]:
            newneu = a * dep[:, 1] + (1 - a) * pn
            scale = (1 - newneu) / np.clip(1 - dep[:, 1], 1e-6, None)
            pr = np.stack([dep[:, 0] * scale, newneu, dep[:, 2] * scale], axis=1)
            pr = pr / pr.sum(1, keepdims=True)
            acc, f1 = score("3class", y, pr)[:2]
            print(f"  B neu-blend a={a}  acc={acc:.4f} f1={f1:.4f}")

        # C) logit average
        eps = 1e-6
        l_dep = np.log(np.clip(dep, eps, 1))
        l_sp = np.log(np.clip(spec, eps, 1))
        for w in [0.5, 1.0, 2.0]:
            lg = l_dep + w * l_sp
            lg -= lg.max(1, keepdims=True)
            pr = np.exp(lg)
            pr /= pr.sum(1, keepdims=True)
            acc, f1 = score("3class", y, pr)[:2]
            print(f"  C logit-avg w={w} acc={acc:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate al17kds mixes on human400 vs HUMAN labels.

Same frozen-weight protocol as the fresh500 mix eval: mix weights were
selected on ood1000 via 5-fold cross-fit (3class modal w_sft=0.6; binary
fold-median w_sft=0.5) — NOT re-tuned on human400. The small sweep is
diagnostic only.

Inputs (under <DATA_ROOT>/human400/):
  human400_v3_probs.npz            {probs3 (400,3), probsb (400,2)}
  human400_sft_probs_al17k/*.npy   per-(task,seed) SFT probs
Labels: repo data/human400_labels.jsonl (`label` field)

Usage: python scripts/eval_mix_human400.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "human400"
PDIR = OUT / "human400_sft_probs_al17k"
IN = ROOT / "data" / "human400_labels.jsonl"
N3 = ["正面", "中性", "负面"]


def mcnemar(y, p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    a = int(((p1 == y) & (p2 != y)).sum())
    b = int(((p1 != y) & (p2 == y)).sum())
    p = binomtest(min(a, b), a + b, 0.5).pvalue if a + b else 1.0
    return a, b, float(p)


def met(p, y):
    yp = p.argmax(1)
    return (round(float(accuracy_score(y, yp)), 4),
            round(float(f1_score(y, yp, average="macro")), 4))


def perclass3(p, y):
    yp = p.argmax(1)
    out = {}
    for i, l in enumerate(N3):
        out[l] = {
            "p": round(float(precision_score(y, yp, labels=[i],
                                             average=None,
                                             zero_division=0)[0]), 3),
            "r": round(float(recall_score(y, yp, labels=[i],
                                          average=None,
                                          zero_division=0)[0]), 3),
        }
    out["confusion"] = confusion_matrix(y, yp, labels=[0, 1, 2]).tolist()
    return out


def main():
    rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
    rows = [r for r in rows if r.get("label")]
    y3 = np.array([N3.index(r["label"]) for r in rows])
    isneu = y3 == 1
    bidx = np.where(~isneu)[0]
    yb = (y3 == 0).astype(int)  # 1=正面 0=负面 (col1=pos convention)
    print(f"human400: n={len(rows)} pos={int((y3==0).sum())} "
          f"neu={int((y3==1).sum())} neg={int((y3==2).sum())} "
          f"binary_n={len(bidx)}", flush=True)

    z = np.load(OUT / "human400_v3_probs.npz")
    pv3 = z["probs3"]
    pvb = z["probsb"][bidx]
    ybb = yb[bidx]

    seeds3 = [s for s in (42, 123, 456, 789, 999)
              if (PDIR / f"3class_e4_s{s}.npy").exists()]
    p3 = [np.load(PDIR / f"3class_e4_s{s}.npy") for s in seeds3]
    pm3 = np.mean(p3, axis=0) if p3 else None
    seedsb = [s for s in (42, 123, 789, 999)
              if (PDIR / f"binary_e8_s{s}.npy").exists()]
    pb = [np.load(PDIR / f"binary_e8_s{s}.npy")[bidx] for s in seedsb]
    pmb = np.mean(pb, axis=0) if pb else None
    print(f"loaded 3class seeds={seeds3} binary seeds={seedsb}", flush=True)

    print("\n=== 3class ===")
    print("label dist:", {l: int((y3 == i).sum()) for i, l in
                           enumerate(N3)})
    print("v3 alone      :", met(pv3, y3), perclass3(pv3, y3))
    if pm3 is not None:
        mix = 0.6 * pm3 + 0.4 * pv3
        print("al17k mix w.6 :", met(mix, y3), perclass3(mix, y3))
        print("al17k SFT aln :", met(pm3, y3))
        print("McNemar mix vs v3 :", mcnemar(y3, mix.argmax(1),
                                          pv3.argmax(1)))
        print("McNemar SFT vs v3 :", mcnemar(y3, pm3.argmax(1),
                                          pv3.argmax(1)))
        for w in (0.4, 0.5, 0.6, 0.7, 0.8):
            print(f"  w_sft={w}:", met(w * pm3 + (1 - w) * pv3, y3))
        for s, pp in zip(seeds3, p3):
            print(f"  s{s} mix w.6:", met(0.6 * pp + 0.4 * pv3, y3))

    print("\n=== binary (non-neutral only) ===")
    print("v3 alone      :", met(pvb, ybb))
    qwen_path = OUT / "qwen3_4b_human400_probs.npy"
    if qwen_path.exists():
        pq = np.load(qwen_path)
        print("Qwen3-4B 3c   :", met(pq, y3), perclass3(pq, y3))
        print("Qwen vs mix   :", mcnemar(y3, pq.argmax(1),
                                     (0.6 * pm3 + 0.4 * pv3).argmax(1))
              if pm3 is not None else "n/a")
    if pmb is not None:
        for w in (0.3, 0.4, 0.5, 0.6, 0.7):
            print(f"  w_sft={w}:", met(w * pmb + (1 - w) * pvb, ybb))
        mixb = 0.5 * pmb + 0.5 * pvb
        print("McNemar w=0.5 mix vs v3:",
              mcnemar(ybb, mixb.argmax(1), pvb.argmax(1)))
        print("al17k SFT alon:", met(pmb, ybb))
        for s, pp in zip(seedsb, pb):
            print(f"  s{s} mix w.5:", met(0.5 * pp + 0.5 * pvb, ybb))


if __name__ == "__main__":
    main()

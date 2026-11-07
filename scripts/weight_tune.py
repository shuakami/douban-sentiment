#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coordinate-ascent weight tuning on the deployed ensemble's channels."""
import os, json, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_eval import load_result, score, OUT
from sklearn.metrics import accuracy_score, f1_score

task = sys.argv[1]
ens_file = sys.argv[2]

r = json.load(open(os.path.join(OUT, ens_file), encoding="utf-8"))
y = np.array(r["y"], dtype=np.int64)
inputs = r["inputs"]
counts = np.array(r["best"]["counts"], dtype=int)
P = []
names = []
for i, c in enumerate(counts):
    if c > 0:
        _, p, _ = load_result(inputs[i])
        for _ in range(c):
            P.append(p); names.append(os.path.basename(inputs[i]))
P = np.stack(P)  # (k, n, cls)
k = P.shape[0]
w = np.ones(k)

def ev(w):
    avg = np.tensordot(w, P, axes=(0, 0)) / w.sum()
    return score(task, y, avg)[:2]

best = ev(w)
print(f"start equal: acc={best[0]:.4f} f1={best[1]:.4f}")
grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
for rd in range(4):
    improved = False
    for j in range(k):
        for g in grid:
            w2 = w.copy(); w2[j] = g
            a, f = ev(w2)
            if f > best[1] + 1e-12 or (abs(f - best[1]) < 1e-12 and a > best[0]):
                best = (a, f); w = w2; improved = True
    print(f"round {rd}: acc={best[0]:.4f} f1={best[1]:.4f} w={np.round(w,2).tolist()}")
    if not improved:
        break

avg = np.tensordot(w, P, axes=(0, 0)) / w.sum()
if task == "binary":
    bt, bf, ba = 0.5, -1, 0
    for t in np.arange(0.35, 0.66, 0.005):
        pred = (avg[:, -1] >= t).astype(int)
        f = f1_score(y, pred, average="macro"); a = accuracy_score(y, pred)
        if f > bf:
            bf, bt, ba = f, t, a
    print(f"thresh: t={bt:.3f} acc={ba:.4f} f1={bf:.4f}")
print("weights:", {names[i]: round(float(w[i]), 3) for i in range(k)})

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid-search the binary decision threshold on ensemble OOF probabilities."""
import json
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = sys.argv[1] if len(sys.argv) > 1 else "out/ens_binary_all.json"
r = json.load(open(ROOT, encoding="utf-8"))
probs = np.array(r["probs"], dtype=np.float32)
y = np.array(r["y"], dtype=np.int64)
print(f"file={ROOT} n={len(y)} method={r.get('best_method')}")
best = None
for t in np.arange(0.30, 0.71, 0.01):
    pred = (probs[:, -1] >= t).astype(int)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    if best is None or f1 > best[1]:
        best = (t, f1, acc)
    if abs(t - 0.5) < 0.005 or t in (0.4, 0.45, 0.55, 0.6):
        print(f"t={t:.2f} acc={acc:.4f} macro_f1={f1:.4f}")
print(f"BEST t={best[0]:.2f} acc={best[2]:.4f} macro_f1={best[1]:.4f}")

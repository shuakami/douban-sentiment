#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Warm-start forward selection: begin from the deployed ensemble's channel
multiset, greedily try adding any channel from the full pool."""
import os, json, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_eval import load_result, score, OUT
from run_ens_all import collect

task = sys.argv[1] if len(sys.argv) > 1 else "3class"
ens_file = sys.argv[2] if len(sys.argv) > 2 else "ens_3class_allch.json"
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 5

r = json.load(open(os.path.join(OUT, ens_file), encoding="utf-8"))
y = np.array(r["y"], dtype=np.int64)
inputs = r["inputs"]
counts = np.array(r["best"]["counts"], dtype=int)
sel_names = set()
base = []
for i, c in enumerate(counts):
    if c > 0:
        sel_names.add(os.path.basename(inputs[i]))
        _, p, _ = load_result(inputs[i])
        for _ in range(c):
            base.append(p)

pool = []
for path in collect(task):
    name = os.path.basename(path)
    if name in sel_names:
        continue
    _, p, yy = load_result(path)
    if np.array_equal(yy, y):
        pool.append((name, p))
print(f"pool={len(pool)} base_channels={len(base)}")
P = np.stack(base).mean(axis=0)
acc, f1, _ = score(task, y, P)
print(f"start: acc={acc:.4f} macro_f1={f1:.4f}")

cur = list(base)
cur_names = set(sel_names)
for rd in range(rounds):
    best = None
    for name, p in pool:
        if name in cur_names:
            continue
        avg = (np.stack(cur).sum(axis=0) + p) / (len(cur) + 1)
        a, f, _ = score(task, y, avg)
        if best is None or f > best[0]:
            best = (f, a, name, p)
    if best is None or best[0] <= f1 + 1e-12:
        print(f"round {rd}: no improvement (best f1={best[0]:.4f})")
        break
    f, a, name, p = best
    cur.append(p); cur_names.add(name)
    avg = np.stack(cur).mean(axis=0)
    acc, f1, _ = score(task, y, avg)
    print(f"round {rd}: +{name} -> acc={acc:.4f} macro_f1={f1:.4f} (nch={len(cur)})")

print("final channels:")
for n in sorted(cur_names):
    print("  ", n)

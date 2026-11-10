#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate ctrl500 random-vs-AL SFT probs on human400.

Reads E:/GB77_data/human400/human400_sft_probs_ctrl500_{random,al}/
{task}_e{ep}_s{seed}.npy, averages probs over seeds, argmax vs the blind
human labels, reports 3-class acc/macro-F1 and binary acc/macro-F1
(non-neutral rows only, matching the paper's binary protocol).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

DATA = Path(r"E:/GB77_data/human400")
N3 = ["正面", "中性", "负面"]

rows = [json.loads(l) for l in open(DATA / "human400_labels.jsonl",
                                    encoding="utf-8")]
rows = [r for r in rows if r.get("label")]
y3 = np.array([N3.index(r["label"]) for r in rows])
mask_bin = y3 != 1  # non-neutral

CONFIGS = {"3class": (4, [42, 123, 456, 789, 999]),
           "binary": (8, [42, 123, 789, 999])}


def load_probs(arm, task):
    ep, seeds = CONFIGS[task]
    d = DATA / f"human400_sft_probs_ctrl500_{arm}"
    ps = []
    for s in seeds:
        f = d / f"{task}_e{ep}_s{s}.npy"
        if f.exists():
            ps.append(np.load(f))
    if not ps:
        return None
    return np.mean(ps, axis=0)


for arm in ["random", "al"]:
    for task in ["3class", "binary"]:
        p = load_probs(arm, task)
        if p is None:
            print(f"{arm}/{task}: no probs yet")
            continue
        pred = p.argmax(1)
        if task == "3class":
            acc = accuracy_score(y3, pred)
            f1 = f1_score(y3, pred, average="macro")
            print(f"{arm}/{task}: n_seeds_files acc={acc:.4f} macroF1={f1:.4f}")
        else:
            # binary: col1=pos convention (1=正面 0=负面), non-neutral rows
            yb = ((y3 == 0).astype(int))[mask_bin]
            pb = pred[mask_bin]
            acc = accuracy_score(yb, pb)
            f1 = f1_score(yb, pb, average="macro")
            print(f"{arm}/{task}: n={mask_bin.sum()} acc={acc:.4f} macroF1={f1:.4f}")
print("DONE")

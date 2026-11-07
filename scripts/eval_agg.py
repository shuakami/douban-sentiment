#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare aggregation rules (mean/median/trimmed/geometric) on the deployed
ensemble channel selections, using stored OOF probs."""
import os, json, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_eval import load_result, score, OUT


def agg(task, y, P, kind):
    # P: (n_ch, n, n_cls) probs of selected channels (counts>0)
    if kind == "mean":
        return P.mean(axis=0)
    if kind == "median":
        return np.median(P, axis=0)
    if kind == "trim":  # per-sample per-class trimmed mean, drop min&max
        s = np.sort(P, axis=0)
        if P.shape[0] >= 4:
            return s[1:-1].mean(axis=0)
        return s.mean(axis=0)
    if kind == "gmean":
        return np.exp(np.log(np.clip(P, 1e-6, 1)).mean(axis=0))
    if kind == "mean_logit":  # average logits then softmax-ish (binary: logit avg)
        eps = 1e-6
        if P.shape[2] == 2:
            lo = np.log(np.clip(P[..., -1], eps, 1 - eps) / np.clip(1 - P[..., -1], eps, 1))
            m = lo.mean(axis=0)
            p1 = 1 / (1 + np.exp(-m))
            return np.stack([1 - p1, p1], axis=-1)
        L = np.log(np.clip(P, eps, 1))
        L = L - L.max(axis=-1, keepdims=True)
        e = np.exp(L).mean(axis=0)
        return e / e.sum(axis=-1, keepdims=True)
    raise ValueError(kind)


def run(ens_file):
    path = os.path.join(OUT, ens_file)
    r = json.load(open(path, encoding="utf-8"))
    task = r["task"]
    y = np.array(r["y"], dtype=np.int64)
    inputs = r["inputs"]
    counts = np.array(r["best"]["counts"], dtype=int)
    sel = [i for i, c in enumerate(counts) if c > 0]
    probs = []
    for i in sel:
        _, p, _ = load_result(inputs[i])
        for _ in range(counts[i]):
            probs.append(p)
    P = np.stack(probs)  # (n_ch, n, cls)
    print(f"== {ens_file} task={task} channels={P.shape[0]}")
    print("   sel:", [os.path.basename(inputs[i]) for i in sel])
    for kind in ["mean", "median", "trim", "gmean", "mean_logit"]:
        pr = agg(task, y, P, kind)
        acc, f1, _ = score(task, y, pr)
        print(f"   {kind:10s} acc={acc:.4f} macro_f1={f1:.4f}")
    # binary threshold sweep on mean
    if task == "binary":
        pr = agg(task, y, P, "mean")
        best = (0, 0.5)
        for t in np.arange(0.30, 0.71, 0.01):
            pred = (pr[:, -1] >= t).astype(int)
            from sklearn.metrics import accuracy_score, f1_score
            a = accuracy_score(y, pred); f = f1_score(y, pred, average="macro")
            if f > best[1]:
                best = (t, f, a)
        print(f"   thresh sweep best t={best[0]:.2f} acc={best[2]:.4f} f1={best[1]:.4f}")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        run(f)

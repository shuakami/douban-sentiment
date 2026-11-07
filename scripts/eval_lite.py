#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the deployed lite ensemble from OOF probs and write ens_lite_*.json.

Deployed config (student : mb13b-feature MLP head = 1:2 for both tasks):
  binary: distill_macbert_bin + headmlp_bin_mb13b  -> out/ens_lite_binary.json
  3class: distill_macbert_3c  + headmlp_3c_mb13b   -> out/ens_lite_3class.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from ensemble_eval import load_result, score  # noqa: E402

COMBOS = {
    "ens_lite_binary": {
        "channels": [("macbert_distill_bin", "distill_macbert_bin_a50_t2_e8_m256", 1.0),
                     ("mb13b_mlp_bin", "headmlp_bin_mb13b", 2.0)],
        "task": "binary",
    },
    "ens_lite_3class": {
        "channels": [("macbert_distill_3c", "distill_macbert_3c_a50_t2_e8_m256", 1.0),
                     ("mb13b_mlp_3c", "headmlp_3c_mb13b", 2.0)],
        "task": "3class",
    },
}

for tag, cfg in COMBOS.items():
    task = cfg["task"]
    items = []
    y = None
    for name, fname, w in cfg["channels"]:
        _, p, yy = load_result(fname + ".json")
        if y is None:
            y = yy
        else:
            assert np.array_equal(y, yy), f"label mismatch {fname}"
        items.append((name, p, w))
    probs = sum(w * p for _, p, w in items) / sum(w for _, _, w in items)
    acc, f1 = score(task, y, probs)[:2]
    pred = probs.argmax(1) if task == "3class" else (probs[:, 1] >= 0.5).astype(int)
    res = {
        "model": "lite:" + "+".join(n for n, _, _ in items),
        "task": task,
        "channels": [{"name": n, "file": f, "weight": w}
                     for n, f, w in cfg["channels"]],
        "best": {"method": "weighted_mean", "acc": float(acc), "macro_f1": float(f1),
                 "pred": pred.tolist()},
        "y": y.tolist(),
        "probs": probs.tolist(),
    }
    path = os.path.join(ROOT, "out", tag + ".json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{tag}: acc={acc:.4f} macro_f1={f1:.4f} -> {path}")

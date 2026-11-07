#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subset-aware OOF eval for augmented runs.

Reports acc / macro-F1 on: all rows, base (non-aug) rows, aug rows.
Usage:
  python scripts/eval_aug_combo.py --task binary \
      --model distill_macbert_bin_a50_t2_e8_m256_aug.json:1 \
      --model headmlp_bin_mb13b_aug.json:2
"""
import os, json, argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data", "reviews_600.jsonl")


def load_result(tag):
    path = tag if os.path.isabs(tag) else os.path.join(OUT, os.path.basename(tag))
    r = json.load(open(path, encoding="utf-8"))
    return np.array(r["probs"], dtype=np.float64), np.array(r["y"], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["3class", "binary"], required=True)
    ap.add_argument("--model", action="append", required=True, help="tag[:weight]")
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    rows = [r for r in rows if r.get("label_semantic")]
    if args.task == "binary":
        sel = [i for i, r in enumerate(rows) if r["label_semantic"] != "中性"]
    else:
        sel = list(range(len(rows)))
    y = np.array([{"正面": 0, "中性": 1, "负面": 2}[rows[i]["label_semantic"]] if args.task == "3class"
                  else (0 if rows[i]["label_semantic"] == "负面" else 1) for i in sel])
    aug_mask = np.array([rows[i].get("src") == "aug" for i in sel])
    print(f"# task={args.task} n={len(y)} aug={aug_mask.sum()} base={(~aug_mask).sum()}")

    probs_list, weights, names = [], [], []
    for spec in args.model:
        tag, _, w = spec.partition(":")
        p, yy = load_result(tag.strip())
        assert len(p) == len(y), f"{tag}: n mismatch {len(p)} vs {len(y)}"
        assert np.array_equal(yy, y), f"{tag}: label mismatch"
        probs_list.append(p)
        weights.append(float(w) if w else 1.0)
        names.append(tag.strip())
    w = np.array(weights)
    probs = sum(wi * p for wi, p in zip(w, probs_list)) / w.sum()

    def pred_of(pp):
        return pp.argmax(1) if args.task == "3class" else (pp[:, -1] >= 0.5).astype(int)

    for label, m in [("all ", np.ones(len(y), bool)), ("base", ~aug_mask), ("aug ", aug_mask)]:
        pred = pred_of(probs[m])
        acc = accuracy_score(y[m], pred)
        f1 = f1_score(y[m], pred, average="macro")
        print(f"{label}: acc={acc:.4f} macro_f1={f1:.4f}  (n={m.sum()})")

    # per-model breakdown on subsets
    print("\n# per-channel:")
    for nm, p in zip(names, probs_list):
        for label, m in [("all", np.ones(len(y), bool)), ("base", ~aug_mask), ("aug", aug_mask)]:
            pred = pred_of(p[m])
            print(f"  {nm:50s} {label}: acc={accuracy_score(y[m],pred):.4f} f1={f1_score(y[m],pred,average='macro'):.4f}")


if __name__ == "__main__":
    main()

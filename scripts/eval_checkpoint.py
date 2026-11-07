#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate <data_root>/models/<tag>_goldprobs.npy on the gold-600 set.

Usage: python scripts/eval_checkpoint.py --tag macbert_3c_pseudo_p55k_e2 --task 3class
Prints accuracy / macro-F1 / per-class report. For --task neu, probs are
2-class [非中性, 中性] and labels are 中性 vs rest.
"""
import argparse, json, os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from _paths import DATA_ROOT

GOLD = "data/reviews_600_base.jsonl"
OUTDIR = str(DATA_ROOT / "models")
N3 = ["正面", "中性", "负面"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task", choices=["binary", "3class", "neu"],
                    default="3class")
    ap.add_argument("--probs", default="", help="override probs path")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    if args.task == "binary":
        keep = [i for i, r in enumerate(rows)
                if r["label_semantic"] != "中性"]
        y = np.array([0 if rows[i]["label_semantic"] == "负面" else 1
                      for i in keep])
        names = ["负面", "正面"]
    elif args.task == "neu":
        keep = list(range(len(rows)))
        y = np.array([1 if r["label_semantic"] == "中性" else 0
                      for r in rows])
        names = ["非中性", "中性"]
    else:
        keep = list(range(len(rows)))
        y = np.array([N3.index(r["label_semantic"]) for r in rows])
        names = N3

    p = args.probs or os.path.join(OUTDIR, args.tag + "_goldprobs.npy")
    probs = np.load(p)
    if len(probs) != len(y):
        if len(probs) == 648 and len(y) == 600:
            probs = probs[:600]  # goldprobs ran on reviews_600.jsonl; base rows first
        elif args.task == "binary" and len(probs) == 600:
            probs = probs[keep]
    pred = probs.argmax(1)

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    print(f"{args.tag} {args.task}: acc={acc:.4f} macro_f1={f1:.4f}")
    print(classification_report(y, pred, target_names=names, digits=4))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep LR head variants on cached frozen-backbone features.

For each cached feature npy: pooling slice (full/cls/mean), C grid,
class_weight in {balanced, None}; 5-fold OOF (seed 42, same as the
frozen_* runs) and writes ensemble_eval-compatible json for promising ones.
"""
import os, sys, json, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, HERE)
import gpu_finetune as gf  # noqa: E402

N3 = gf.N3


def oof_lr(feats, y, c, cw, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n_classes = int(y.max() + 1)
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    for tr, va in skf.split(feats, y):
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=c, max_iter=3000, class_weight=cw)),
        ])
        pipe.fit(feats[tr], y[tr])
        oof[va] = pipe.predict_proba(feats[va])
    if n_classes == 2:
        pred = (oof[:, 1] >= 0.5).astype(int)
    else:
        pred = oof.argmax(1)
    return oof, float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True, help="npy under out/ or absolute")
    ap.add_argument("--model_name", required=True, help="label written into json")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--half", type=int, default=0, help="0=full,1=cls half,2=mean half")
    ap.add_argument("--cs", default="0.003,0.01,0.03,0.1,0.3,1.0,3.0")
    ap.add_argument("--cws", default="balanced,None")
    ap.add_argument("--save_all", action="store_true")
    args = ap.parse_args()

    path = args.feat if os.path.isabs(args.feat) else os.path.join(OUT, args.feat)
    feats_all = np.load(path)
    rows = gf.load_rows("label_semantic")

    if args.half == 1:
        feats_all = feats_all[:, : feats_all.shape[1] // 2]
    elif args.half == 2:
        feats_all = feats_all[:, feats_all.shape[1] // 2 :]
    slice_name = {0: "full", 1: "cls", 2: "mean"}[args.half]

    results = []
    for task in ["binary", "3class"]:
        texts, y = gf.make_labels(rows, task)
        idx = [i for i, r in enumerate(rows)
               if not (task == "binary" and r["label_semantic"] == "中性")]
        feats = feats_all[idx]
        for cw_s in args.cws.split(","):
            cw = None if cw_s == "None" else "balanced"
            for c in [float(x) for x in args.cs.split(",")]:
                oof, acc, f1 = oof_lr(feats, y, c, cw)
                results.append({"task": task, "C": c, "cw": cw_s, "acc": acc, "f1": f1})
                print(f"{task} {slice_name} C={c} cw={cw_s}: acc={acc:.4f} f1={f1:.4f}", flush=True)
                # save every variant that beats the known baselines enough to matter
                base = {"binary": 0.9331, "3class": 0.8233}[task]
                if args.save_all or acc >= base - 0.01:
                    tag = f"{args.tag}_{task}_{slice_name}_C{c}_cw{cw_s.replace('balanced','bal').replace('None','none')}"
                    res = {
                        "model": args.model_name + f" [frozen {slice_name} -> LR C={c} cw={cw_s}]",
                        "task": task, "label_key": "label_semantic",
                        "n": len(y), "n_classes": int(y.max() + 1),
                        "config": {"feat": os.path.basename(path), "slice": slice_name,
                                   "C": c, "class_weight": cw_s},
                        "oof": {"acc": acc, "macro_f1": f1},
                        "y": y.tolist(), "probs": oof.tolist(),
                    }
                    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w", encoding="utf-8"),
                              ensure_ascii=False)

    results.sort(key=lambda r: -r["f1"])
    print("\ntop5:")
    for r in results[:5]:
        print(f"  {r['task']} {slice_name} C={r['C']} cw={r['cw']}: acc={r['acc']:.4f} f1={r['f1']:.4f}")


if __name__ == "__main__":
    main()

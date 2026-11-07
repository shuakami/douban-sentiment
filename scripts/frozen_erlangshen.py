#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen-feature LR on arbitrary HF backbones (Erlangshen 110M/330M etc.).

Same recipe as frozen_megabert.py: CLS+mean features -> StandardScaler+LR,
5-fold OOF probs written in ensemble_eval-compatible json.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, HERE)
import gpu_finetune as gf  # noqa: E402
import frozen_megabert as fm  # noqa: E402

N3 = gf.N3


def run_task(task, feats, y, c):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(y), int(y.max() + 1)), dtype=np.float32)
    for tr, va in skf.split(feats, y):
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=c, max_iter=3000, class_weight="balanced")),
        ])
        pipe.fit(feats[tr], y[tr])
        oof[va] = pipe.predict_proba(feats[va])
    if task == "3class":
        pred = oof.argmax(1)
        names = N3
    else:
        pred = (oof[:, 1] >= 0.5).astype(int)
        names = ["负面", "正面"]
    return oof, {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion": confusion_matrix(y, pred).tolist(),
        "report": classification_report(y, pred, target_names=names, digits=4, output_dict=True),
        "pred": pred.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cs", default="0.01,0.1,1.0")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# frozen {args.model} device={device} max_len={args.max_len}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)

    rows = gf.load_rows("label_semantic")
    all_texts = [r["text"] for r in rows]
    t0 = time.time()
    feats_all = fm.extract_features(model, tok, all_texts, args.max_len, device, args.bs)
    print(f"# features {feats_all.shape} in {time.time()-t0:.1f}s", flush=True)

    os.makedirs(OUT, exist_ok=True)
    for task in ["binary", "3class"]:
        texts, y = gf.make_labels(rows, task)
        idx = [i for i, r in enumerate(rows)
               if not (task == "binary" and r["label_semantic"] == "中性")]
        feats = feats_all[idx]
        for c in [float(x) for x in args.cs.split(",")]:
            oof, m = run_task(task, feats, y, c)
            tag = args.tag or f"frozen_{os.path.basename(args.model)}_{task}"
            res = {
                "model": args.model + " [frozen CLS+mean -> LR]",
                "task": task,
                "label_key": "label_semantic",
                "n": len(y),
                "n_classes": int(y.max() + 1),
                "config": {"max_len": args.max_len, "C": c, "bs": args.bs},
                "oof": m,
                "y": y.tolist(),
                "probs": oof.tolist(),
            }
            path = os.path.join(OUT, f"{tag}_{task}_C{c}.json")
            json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"{task} C={c}: acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train production LR heads on frozen backbone features (any HF model).

Generalizes train_mb13b_heads.py: extracts CLS+mean features once per
(model, max_len), caches them under out/, then fits LogisticRegression on
ALL labeled rows for each requested task/C and joblib-dumps pipelines.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, HERE)
import gpu_finetune as gf  # noqa: E402
import frozen_megabert as fm  # noqa: E402


def get_features(model_id, max_len, bs, device, cache_path):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    rows = gf.load_rows("label_semantic")
    texts = [r["text"] for r in rows]
    t0 = time.time()
    feats = fm.extract_features(model, tok, texts, max_len, device, bs)
    print(f"# features {model_id} {feats.shape} in {time.time()-t0:.1f}s", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    os.makedirs(OUT, exist_ok=True)
    np.save(cache_path, feats)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out_dir", required=True, help="dir under models/ for joblibs")
    ap.add_argument("--tag", required=True, help="feature cache tag")
    ap.add_argument("--heads", required=True,
                    help="comma list of task:C e.g. binary:0.1,3class:0.01,3class:1.0")
    ap.add_argument("--cw", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--slice", choices=["full", "cls", "mean"], default="full",
                    help="feature slice: cls/mean take one half of cls+mean concat")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe = args.tag.replace("/", "_")
    cache = os.path.join(OUT, f"feats_{safe}_m{args.max_len}.npy")
    feats_all = get_features(args.model, args.max_len, args.bs, device, cache)
    if args.slice != "full":
        h = feats_all.shape[1] // 2
        feats_all = feats_all[:, :h] if args.slice == "cls" else feats_all[:, h:]
    rows = gf.load_rows("label_semantic")

    out_dir = os.path.join(ROOT, "models", args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for spec in args.heads.split(","):
        task, c = spec.split(":")
        c = float(c)
        texts, y = gf.make_labels(rows, task)
        idx = [i for i, r in enumerate(rows)
               if not (task == "binary" and r["label_semantic"] == "中性")]
        feats = feats_all[idx]
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=c, max_iter=3000,
                                      class_weight=None if args.cw == "none" else "balanced")),
        ])
        pipe.fit(feats, y)
        name = f"{task}_C{c}_m{args.max_len}"
        path = os.path.join(out_dir, name + ".joblib")
        joblib.dump({"pipeline": pipe, "task": task, "C": c,
                     "max_len": args.max_len, "n_features": int(feats.shape[1]),
                     "slice": args.slice, "class_weight": args.cw,
                     "classes": pipe.classes_.tolist()}, path)
        pred = pipe.predict(feats)
        print(f"{name}: train acc={accuracy_score(y,pred):.4f} "
              f"f1={f1_score(y,pred,average='macro'):.4f} -> {path}", flush=True)

    print("done ->", out_dir)


if __name__ == "__main__":
    main()

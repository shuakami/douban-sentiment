#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train production LR heads on MegatronBert-1.3B frozen features and save them.

Extracts CLS+mean features once (same recipe as frozen_megabert.py OOF runs),
then fits LogisticRegression on ALL labeled rows for each task/C used by the
chosen ensembles, and joblib-dumps the sklearn pipelines under models/megabert13b/.
"""
import os, sys, json, time
import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gpu_finetune as gf  # noqa: E402
import frozen_megabert as fm  # noqa: E402

MODEL = fm.MODEL
OUT_MODELS = os.path.join(ROOT, "models", "megabert13b")
FEAT_CACHE = os.path.join(ROOT, "out", "mb13b_feats.npy")
MAX_LEN = 256
BS = 16

# (task, C) heads needed by the deployed ensembles
HEADS = [("binary", 1.0), ("3class", 0.01), ("3class", 0.1), ("3class", 1.0)]


def get_features(device):
    if os.path.exists(FEAT_CACHE):
        return np.load(FEAT_CACHE)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    rows = gf.load_rows("label_semantic")
    texts = [r["text"] for r in rows]
    t0 = time.time()
    feats = fm.extract_features(model, tok, texts, MAX_LEN, device, BS)
    print(f"# features {feats.shape} in {time.time()-t0:.1f}s", flush=True)
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    np.save(FEAT_CACHE, feats)
    return feats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats_all = get_features(device)
    rows = gf.load_rows("label_semantic")
    os.makedirs(OUT_MODELS, exist_ok=True)

    for task, c in HEADS:
        texts, y = gf.make_labels(rows, task)
        idx = [i for i, r in enumerate(rows)
               if not (task == "binary" and r["label_semantic"] == "中性")]
        feats = feats_all[idx]
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=c, max_iter=3000, class_weight="balanced")),
        ])
        pipe.fit(feats, y)
        name = f"{task}_C{c}"
        path = os.path.join(OUT_MODELS, name + ".joblib")
        joblib.dump({"pipeline": pipe, "task": task, "C": c,
                     "max_len": MAX_LEN, "n_features": int(feats.shape[1]),
                     "classes": pipe.classes_.tolist()}, path)
        # sanity: training-set score (optimistic, just a smoke check)
        pred = pipe.predict(feats)
        print(f"{name}: train acc={accuracy_score(y,pred):.4f} f1={f1_score(y,pred,average='macro'):.4f} -> {path}", flush=True)

    meta = {"model": MODEL, "max_len": MAX_LEN, "feature": "cls+mean_pool",
            "n_features": int(feats_all.shape[1]), "heads": [f"{t}_C{c}" for t, c in HEADS]}
    with open(os.path.join(OUT_MODELS, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("saved meta ->", os.path.join(OUT_MODELS, "meta.json"))


if __name__ == "__main__":
    main()

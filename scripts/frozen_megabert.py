#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen-feature LR on Erlangshen-MegatronBert-1.3B-Sentiment (GPU feature extraction).

Mirrors the handoff's best CPU recipe (frozen backbone + linear head) but with the
1.3B sentiment backbone on the RX 9070 GRE. Writes OOF probability jsons compatible
with ensemble_eval.py.
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

MODEL = "IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment"
N3 = gf.N3


@torch.no_grad()
def extract_features(model, tok, texts, max_len, device, bs=16):
    feats = []
    model.eval()
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=device.type == "cuda"):
            out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mean = (out.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
        cls = out[:, 0].float()
        feats.append(torch.cat([cls, mean], dim=1).cpu())
    return torch.cat(feats).numpy()


def run_task(task, feats, y, texts_len, c, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
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
    res = {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion": confusion_matrix(y, pred).tolist(),
        "report": classification_report(y, pred, target_names=names, digits=4, output_dict=True),
        "pred": pred.tolist(),
    }
    return oof, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    print(f"# megabert frozen features device={device} max_len={args.max_len}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)

    rows = gf.load_rows("label_semantic")
    all_texts = [r["text"] for r in rows]
    t0 = time.time()
    feats_all = extract_features(model, tok, all_texts, args.max_len, device, args.bs)
    print(f"# features {feats_all.shape} in {time.time()-t0:.1f}s", flush=True)

    os.makedirs(OUT, exist_ok=True)
    for task in ["binary", "3class"]:
        texts, y = gf.make_labels(rows, task)
        idx = [i for i, r in enumerate(rows)
               if not (task == "binary" and r["label_semantic"] == "中性")]
        feats = feats_all[idx]
        for c in [0.01, 0.1, 1.0]:
            oof, m = run_task(task, feats, y, len(texts), c)
            tag = args.tag or f"frozen_mb13b_{task}_C{c}"
            fname = f"{tag}_{task}_C{c}.json" if args.tag else f"{tag}_C{c}.json"
            res = {
                "model": MODEL + " [frozen CLS+mean -> LR]",
                "task": task,
                "label_key": "label_semantic",
                "n": len(y),
                "n_classes": int(y.max() + 1),
                "config": {"max_len": args.max_len, "C": c, "bs": args.bs},
                "oof": m,
                "y": y.tolist(),
                "probs": oof.tolist(),
            }
            path = os.path.join(OUT, fname)
            json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"{task} C={c}: acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()

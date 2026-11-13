#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict human400 probs with a train_final_distill checkpoint and score.

Usage:
  python scripts/predict_human400.py --model models/eexp_base --arm base

Saves E:/GB77_data/human400/human400_sft_probs_eexp/{arm}.npy and prints
3-class acc/macro-F1 (+per-class) and binary acc/macro-F1 on non-neutral
rows, matching the paper's human400 protocol.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _paths import DATA_ROOT  # noqa: E402
import gpu_finetune as gf  # noqa: E402

IN = DATA_ROOT / "human400" / "human400_labels.jsonl"
OUTD = DATA_ROOT / "human400" / "human400_sft_probs_eexp"
N3 = ["正面", "中性", "负面"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir (under models/ or absolute)")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    mdir = Path(args.model)
    if not mdir.is_absolute() and not str(mdir).startswith("models"):
        mdir = HERE.parent / "models" / mdir
    elif not mdir.is_absolute():
        mdir = HERE.parent / mdir
    tok = AutoTokenizer.from_pretrained(str(mdir))
    model = AutoModelForSequenceClassification.from_pretrained(str(mdir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
    rows = [r for r in rows if r.get("label")]
    texts = [(r["text"] or "").strip()[:256] for r in rows]
    y3 = np.array([N3.index(r["label"]) for r in rows])

    probs = []
    with torch.inference_mode():
        for i in range(0, len(texts), args.bs):
            enc = tok(texts[i:i + args.bs], truncation=True,
                      max_length=args.max_len, padding=True,
                      return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(**enc).logits.float()
            probs.append(F.softmax(logits, dim=-1).cpu())
    p = torch.cat(probs).numpy().astype(np.float32)

    OUTD.mkdir(parents=True, exist_ok=True)
    np.save(OUTD / f"{args.arm}.npy", p)

    from sklearn.metrics import accuracy_score, f1_score
    pred = p.argmax(1)
    acc = accuracy_score(y3, pred)
    f1 = f1_score(y3, pred, average="macro")
    per = f1_score(y3, pred, average=None, labels=[0, 1, 2])
    rec = [float((pred[y3 == c] == c).mean()) for c in range(3)]
    mask = y3 != 1
    yb = (y3[mask] == 0).astype(int)  # 1=正面
    pb = pred[mask]
    bacc = accuracy_score(yb, pb)
    bf1 = f1_score(yb, pb, average="macro")
    print(f"{args.arm}: 3class n={len(y3)} acc={acc:.4f} macroF1={f1:.4f} "
          f"per-class F1(pos/neu/neg)={per[0]:.3f}/{per[1]:.3f}/{per[2]:.3f} "
          f"recall={rec[0]:.3f}/{rec[1]:.3f}/{rec[2]:.3f}")
    print(f"{args.arm}: binary n={mask.sum()} acc={bacc:.4f} macroF1={bf1:.4f}")
    print(f"saved {OUTD / (args.arm + '.npy')}", flush=True)


if __name__ == "__main__":
    main()

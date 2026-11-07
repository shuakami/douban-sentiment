#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Silver-standard cross-domain evaluation on fresh corpus rows.

Samples comments.pkl stratified by star rating (rows NOT necessarily in the
pseudo pool — true out-of-sample), scores them with a checkpoint, and
reports agreement with the rating-implied label. Ratings are noisy, so this
is a lower-bound sanity check for cross-movie generalization, not gold.

Usage: python scripts/silver_eval.py --model <ckpt> --tag stu3c --n-per-rating 800
"""
import os, sys, json, argparse, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bigdata import predict_probs  # noqa
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from _paths import DATA_ROOT

PKL = str(DATA_ROOT / "clean" / "comments.pkl")
OUT = Path(__file__).resolve().parents[1] / "out"
N3 = ["pos", "neu", "neg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-per-rating", type=int, default=800)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    df = pd.read_pickle(PKL)
    df = df[df["len"] >= 4]
    parts = []
    for r in [1, 2, 3, 4, 5]:
        g = df[df["rating"] == r]
        parts.append(g.sample(min(len(g), args.n_per_rating),
                              random_state=args.seed))
    s = pd.concat(parts).sample(frac=1.0, random_state=args.seed)
    texts = s["content"].tolist()
    rating = s["rating"].astype(int).to_numpy()
    print(f"silver eval: {len(texts)} rows", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    t0 = time.time()
    probs = predict_probs(model, tok, texts, args.max_len,
                          device, bs=args.batch, amp="bf16")
    print(f"infer {time.time()-t0:.0f}s", flush=True)

    pred3 = probs.argmax(1)
    # rating-implied 3-class: 1-2->neg, 3->neu, 4-5->pos
    y3 = np.select([rating <= 2, rating == 3, rating >= 4], [2, 1, 0])
    # binary view: drop rating==3
    m = rating != 3
    yb = (rating[m] >= 4).astype(int)          # 1=pos 0=neg
    pb = np.where(pred3[m] == 0, 1,
                  np.where(pred3[m] == 2, 0, -1))  # -1 = predicted neu
    valid = pb >= 0

    res = {
        "model": args.model, "n": len(texts),
        "3class_vs_rating": {
            "acc": float(accuracy_score(y3, pred3)),
            "macro_f1": float(f1_score(y3, pred3, average="macro")),
        },
        "binary_vs_rating": {
            "acc": float(accuracy_score(yb[valid], pb[valid])),
            "macro_f1": float(f1_score(yb[valid], pb[valid],
                                       average="macro")),
            "coverage": float(valid.mean()),
        },
        "per_rating_pred_dist": {
            str(r): np.bincount(pred3[rating == r], minlength=3).tolist()
            for r in [1, 2, 3, 4, 5]},
    }
    json.dump(res, open(os.path.join(OUT, f"silver_{args.tag}.json"), "w"),
              indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cheap baseline: TF-IDF (char n-grams) + logistic regression on the same
training data as the LLM-17K student (Gold-600 + extra_al17k_SWAP_ds),
evaluated on Human-400. CPU-only; artifact for the paper's baseline table.

Writes E:/GB77_data/human400/tfidf_human400_preds.json  {acc, macro_f1, ...}
and per-row predictions for McNemar reuse.
"""
import json, sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_ROOT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAB = DATA_ROOT / "lab2000"
H4 = DATA_ROOT / "human400" / "human400_labels.jsonl"
LAB3 = {"正面": 0, "中性": 1, "负面": 2}


def main():
    gold = [json.loads(l) for l in open(ROOT / "data/reviews_600_base.jsonl", encoding="utf-8")]
    extra = [json.loads(l) for l in open(LAB / "extra_al17k_SWAP_ds.jsonl", encoding="utf-8")]
    extra = [r for r in extra if r.get("final") in LAB3]
    texts = [r["text"] for r in gold] + [r["text"] for r in extra]
    y = np.array([LAB3[r["label_semantic"]] for r in gold] +
                 [LAB3[r["final"]] for r in extra])

    h4 = [json.loads(l) for l in open(ROOT / "data/human400_labels.jsonl", encoding="utf-8")]
    h4 = [r for r in h4 if r.get("label")]
    xt = [r["text"] for r in h4]
    yt = np.array([LAB3[r["label"]] for r in h4])

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                          max_features=300000, sublinear_tf=True)
    X = vec.fit_transform(texts)
    Xt = vec.transform(xt)
    print(f"train={X.shape} vocab={len(vec.vocabulary_)}")

    clf = LogisticRegression(max_iter=1000, C=4.0, class_weight=None)
    clf.fit(X, y)
    p = clf.predict(Xt)
    acc = accuracy_score(yt, p)
    f1 = f1_score(yt, p, average="macro")
    neu_r = float((p[yt == 1] == 1).mean())
    rec = [float((p[yt == k] == k).mean()) for k in range(3)]
    print(f"TFIDF+LR Human-400: acc={100*acc:.2f} macroF1={100*f1:.2f} "
          f"neuR={100*neu_r:.1f} recalls={[round(100*r,1) for r in rec]}")

    # binary view: non-neutral gold rows, pred mapped pos/neg (neutral=error)
    nn = yt != 1
    yb = (yt[nn] == 0).astype(int)
    pb = np.where(p[nn] == 0, 1, np.where(p[nn] == 2, 0, -1))
    bin_acc = float((pb == yb).mean())
    print(f"binary_acc={100*bin_acc:.2f} (n={nn.sum()})")

    json.dump({"acc": round(100 * acc, 2), "macro_f1": round(100 * f1, 2),
               "neutral_recall": round(100 * neu_r, 1),
               "recall_pos_neu_neg": [round(100 * r, 1) for r in rec],
               "binary_acc": round(100 * bin_acc, 2),
               "pred": p.tolist()},
              open(DATA_ROOT / "human400" / "tfidf_human400_preds.json", "w"))
    print("wrote tfidf_human400_preds.json")


if __name__ == "__main__":
    main()

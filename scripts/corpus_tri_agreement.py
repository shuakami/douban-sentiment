#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corpus-wide 3-class agreement between stu3c pseudo-labels and star-rating
tri-labels (1-2*->neg, 3*->neu, 4-5*->pos), plus the same on the confident
(max-prob >= 0.7) pool.

Paper claim (PAPER.md / thesis): pseudo-labels and ratings agree at only
47.8% corpus-wide (54.0% on the confident pool), vs 71.2% on gold-600 —
the corpus-scale quantification of the rating-sentiment gap.

Writes out/corpus_tri_agreement.json.
"""
import os, sys, json
import numpy as np
import pandas as pd
from _paths import DATA_ROOT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSEUDO = str(DATA_ROOT / "pseudo")


def tri_labels(rating):
    # train_bigdata.py tri map: 0=pos(4-5) 1=neu(3) 2=neg(1-2)
    return np.where(rating <= 2, 2, np.where(rating == 3, 1, 0))


def main():
    meta = pd.read_pickle(os.path.join(PSEUDO, "stu3c_meta.pkl"))
    probs = np.load(os.path.join(PSEUDO, "stu3c_probs.npy"), mmap_mode="r")
    n = len(meta)
    pred = np.empty(n, dtype=np.int8)
    conf = np.empty(n, dtype=np.float32)
    for i in range(0, n, 100000):
        p = np.asarray(probs[i:i + 100000])
        pred[i:i + 100000] = p.argmax(1)
        conf[i:i + 100000] = p.max(1)

    tri = tri_labels(meta["rating"].values.astype(int))
    out = {
        "n_corpus": int(n),
        "agreement_3class_vs_tri_rating": float((pred == tri).mean()),
    }
    keep = conf >= 0.7
    out["confident_pool_n"] = int(keep.sum())
    out["agreement_confident_pool"] = float((pred[keep] == tri[keep]).mean())

    dst = os.path.join(ROOT, "out", "corpus_tri_agreement.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False))
    print("wrote", dst)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()

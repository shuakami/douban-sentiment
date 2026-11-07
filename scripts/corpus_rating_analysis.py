#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Star-rating vs model-prediction crosstab on the confident pseudo pool.

Paper motivation evidence: quantifies at scale (271,905 reviews) how poorly
Douban star ratings proxy semantic sentiment — e.g. only ~59% of 5-star
reviews are semantically positive, and 3-star reviews skew negative.
Writes out/corpus_rating_crosstab.json.
"""
import os, sys, json
import numpy as np
import pandas as pd
from _paths import DATA_ROOT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSEUDO = str(DATA_ROOT / "pseudo")
N3 = ["pos", "neu", "neg"]


def main():
    meta = pd.read_pickle(os.path.join(PSEUDO, "stu3c_meta.pkl"))
    probs = np.load(os.path.join(PSEUDO, "stu3c_probs.npy"))
    keep = probs.max(1) >= 0.7
    m = meta[keep].reset_index(drop=True)
    p = probs[keep]
    m["pred"] = p.argmax(1)
    m["rating"] = m["rating"].astype(int)

    ct = pd.crosstab(m["rating"], m["pred"], normalize="index").round(4)
    counts = m["rating"].value_counts().sort_index()

    # agreement between rating-implied binary label and model pos/neg
    bin_y = np.select([m["rating"] <= 2, m["rating"] >= 4], [0, 1], default=-1)
    mask = bin_y >= 0
    bin_pred = np.where(m["pred"].to_numpy()[mask] == 2, 0,
                        np.where(m["pred"].to_numpy()[mask] == 0, 1, -1))
    valid = bin_pred >= 0
    agree = float((bin_y[mask][valid] == bin_pred[valid]).mean())

    out = {
        "pool_n": int(len(m)),
        "rating_pred_row_normalized": {
            str(r): {N3[c]: float(ct.loc[r, c]) for c in ct.columns}
            for r in ct.index},
        "rating_counts": {str(k): int(v) for k, v in counts.items()},
        "rating_vs_model_binary_agreement": agree,
    }
    dst = os.path.join(ROOT, "out", "corpus_rating_crosstab.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

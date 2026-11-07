#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build neu_v2 pseudo-label pool: confident neutrals + hard/easy non-neutrals.

Strata (from stu3c 3-class probs, conf>=0.7 pool of 271,905):
  neu      : argmax==1                       (68,568 rows, P(neu)>=0.7)
  hard_neg : argmax!=1 and P(neu)>=0.15       (~49.5K boundary rows)
  easy_neg : argmax!=1 and P(neu)<0.15        (random sample to reach ~2:1
             non-neu:neu ratio matching corpus prior)

Writes <data_root>/pseudo/neu_v2_meta.pkl + neu_v2_probs.npy in the same
format train_bigdata.py --pseudo expects (argmax encodes the binary label:
1=中性 for neu rows, 0=非中性 for non-neu rows).
"""
import os
import numpy as np
import pandas as pd
from _paths import DATA_ROOT

PSEUDO = str(DATA_ROOT / "pseudo")
HARD_T = 0.15
EASY_PER_HARD = 1.8  # easy_neg ~= 1.8 * hard_neg -> non-neu ~2:1 vs neu


def main():
    meta = pd.read_pickle(os.path.join(PSEUDO, "stu3c_meta.pkl"))
    probs = np.load(os.path.join(PSEUDO, "stu3c_probs.npy"))
    conf = probs.max(1)
    keep = conf >= 0.7
    m = meta[keep].reset_index(drop=True)
    p = probs[keep]
    am = p.argmax(1)
    pn = p[:, 1]

    neu = am == 1
    hard = (~neu) & (pn >= HARD_T)
    easy = (~neu) & (pn < HARD_T)
    n_easy = min(int(hard.sum() * EASY_PER_HARD), int(easy.sum()))
    easy_idx = np.flatnonzero(easy)
    rng = np.random.RandomState(42)
    easy_pick = rng.choice(easy_idx, size=n_easy, replace=False)

    sel = np.concatenate([
        np.flatnonzero(neu),
        np.flatnonzero(hard),
        easy_pick,
    ])
    m = m.iloc[sel].reset_index(drop=True)

    # synthetic 3-col probs: argmax==1 -> 中性, else 非中性
    y = np.zeros(len(m), dtype=int)
    y[: int(neu.sum())] = 1
    out_probs = np.zeros((len(m), 3), dtype=np.float32)
    out_probs[y == 1] = [0.0, 1.0, 0.0]
    out_probs[y == 0] = [0.5, 0.0, 0.5]

    m.to_pickle(os.path.join(PSEUDO, "neu_v2_meta.pkl"))
    np.save(os.path.join(PSEUDO, "neu_v2_probs.npy"), out_probs)
    print(f"neu_v2: total={len(m)} neu={int(y.sum())} non-neu={int((y==0).sum())}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append gold-600 rows (semantic labels, one-hot probs) to a pseudo pool,
with optional oversampling, for gold-mixed corpus training.

Usage: python scripts/mix_gold_into_pool.py --pseudo stu3c_v3 --repeat 50 --tag stu3c_v3g50
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from _paths import DATA_ROOT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSEUDO = str(DATA_ROOT / "pseudo")
GOLD = os.path.join(ROOT, "data", "reviews_600_base.jsonl")
N3 = ["正面", "中性", "负面"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo", required=True)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    meta = pd.read_pickle(os.path.join(PSEUDO, args.pseudo + "_meta.pkl"))
    probs = np.load(os.path.join(PSEUDO, args.pseudo + "_probs.npy"))
    assert len(meta) == len(probs)

    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    g = pd.DataFrame({
        "cid": [-1 - i for i in range(len(rows))],
        "uid": "gold", "mid": -1, "rating": 0.0, "votes": 0,
        "len": [len(r["text"]) for r in rows],
        "content": [r["text"] for r in rows],
    })
    y = np.array([N3.index(r["label_semantic"]) for r in rows])
    gp = np.zeros((len(rows), probs.shape[1]), dtype=np.float32)
    gp[np.arange(len(rows)), y] = 1.0

    meta_out = pd.concat([meta] + [g] * args.repeat,
                         ignore_index=True)
    probs_out = np.concatenate([probs] + [gp] * args.repeat)
    meta_out.to_pickle(os.path.join(PSEUDO, args.tag + "_meta.pkl"))
    np.save(os.path.join(PSEUDO, args.tag + "_probs.npy"), probs_out)
    print(f"{args.tag}: pool={len(meta)} + gold {len(rows)}x{args.repeat}"
          f" -> {len(meta_out)}", flush=True)


if __name__ == "__main__":
    main()

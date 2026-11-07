#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""List high-confidence errors from an ensemble/result JSON for manual review."""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "reviews_600_base.jsonl")
OUT = os.path.join(ROOT, "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--task", choices=["binary", "3class"], default="binary")
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()
    path = args.result if os.path.exists(args.result) else os.path.join(OUT, args.result if args.result.endswith(".json") else args.result + ".json")
    res = json.load(open(path, encoding="utf-8"))
    probs = np.array(res["probs"], dtype=np.float32)
    y = np.array(res["y"], dtype=np.int64)
    pred = probs.argmax(axis=1) if args.task == "3class" else (probs[:, -1] >= 0.5).astype(int)
    conf = probs.max(axis=1)
    err = np.where(pred != y)[0]
    order = err[np.argsort(-conf[err])]
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    if args.task == "binary":
        rows = [r for r in rows if r.get("label_semantic") and r["label_semantic"] != "中性"]
    print(f"errors={len(err)} / {len(y)}")
    for rank, i in enumerate(order[: args.top], 1):
        r = rows[i]
        lab = r["label_semantic"]
        yid = {"正面": 0 if args.task == "3class" else 1, "中性": 1, "负面": 2 if args.task == "3class" else 0}[lab]
        print(f"#{rank} idx={i} id={r['id']} star={r['star']} label={lab} pred_idx={pred[i]} p={probs[i].round(4).tolist()} conf={conf[i]:.4f}")
        print(r["text"][:220].replace("\n", " "))
        print("-" * 80)


if __name__ == "__main__":
    main()

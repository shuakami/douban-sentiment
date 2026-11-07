#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random-subset ensemble search over top single-channel OOF probs.

Complements greedy forward selection: samples random channel subsets,
evaluates equal-weight mean, reports the best. Guards against greedy
path-dependence.
"""
import os, sys, json, glob, argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, HERE)
import ensemble_eval as ee  # noqa: E402


def collect(task):
    pats = {"binary": ["erlangshen*bin*.json", "macbert*bin*.json", "tfidf*bin*.json",
                       "zs*bin*.json", "frozen*bin*.json", "swp*bin*.json", "lora*bin*.json"],
            "3class": ["erlangshen*3class*.json", "macbert*3class*.json",
                       "tfidf*3class*.json", "frozen*3class*.json", "swp*3class*.json",
                       "lora*3class*.json", "lora*3c*.json"]}[task]
    files = {}
    for pat in pats:
        for f in glob.glob(os.path.join(OUT, pat)):
            b = os.path.basename(f)
            if b.startswith("ens_"):
                continue
            try:
                r = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if r.get("task") == task and "probs" in r and "y" in r:
                files[b] = f
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["binary", "3class"])
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_k", type=int, default=8)
    args = ap.parse_args()

    files = collect(args.task)
    names, probs, y = [], [], None
    for b in sorted(files):
        r, p, yy = ee.load_result(files[b])
        acc, f1, _ = ee.score(args.task, yy, p)
        names.append(b)
        probs.append(p.astype(np.float64))
        if y is None:
            y = yy
    P = np.stack(probs)
    singles = np.array([ee.score(args.task, y, p)[1] for p in probs])
    order = np.argsort(-singles)
    top_idx = order[:args.top]
    print(f"# {args.task}: {len(files)} channels, searching top {len(top_idx)}")
    for i in top_idx:
        print(f"  {singles[i]:.4f} {names[i]}")

    rng = np.random.default_rng(args.seed)
    best = (-1, None, None)
    Pt = P[top_idx]
    nt = len(top_idx)
    for it in range(args.iters):
        k = int(rng.integers(2, args.max_k + 1))
        idx = rng.choice(nt, size=k, replace=False)
        avg = Pt[idx].mean(0)
        _, f1, _ = ee.score(args.task, y, avg)
        if f1 > best[0]:
            acc, _, _ = ee.score(args.task, y, avg)
            best = (f1, acc, idx.copy())
            print(f"it{it}: f1={f1:.4f} acc={acc:.4f} k={k} "
                  f"{[names[top_idx[i]] for i in idx]}", flush=True)
    f1, acc, idx = best
    sel = [names[top_idx[i]] for i in idx]
    print(f"\nBEST f1={f1:.4f} acc={acc:.4f} channels={sel}")
    out = {"task": args.task, "channels": sel, "acc": acc, "macro_f1": f1}
    tag = f"randsearch_{args.task}_s{args.seed}"
    json.dump(out, open(os.path.join(OUT, tag + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved", os.path.join(OUT, tag + ".json"))


if __name__ == "__main__":
    main()

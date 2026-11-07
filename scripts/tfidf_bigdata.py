#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TF-IDF baseline on rating-derived weak labels from the big corpus,
evaluated on the gold 600 semantic labels."""
import os, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
from _paths import DATA_ROOT

PKL = str(DATA_ROOT / "clean" / "comments.pkl")
GOLD = Path(__file__).resolve().parents[1] / "data" / "reviews_600_base.jsonl"
LAB3 = {"正面": 0, "中性": 1, "负面": 2}
N3 = ["正面", "中性", "负面"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmap", choices=["bin_strict", "bin_all", "tri"], default="bin_strict")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--per-class", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--ngram", default="1-2")
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pseudo", default="")
    ap.add_argument("--conf", type=float, default=0.0)
    args = ap.parse_args()

    if args.pseudo:
        pdir = str(DATA_ROOT / "pseudo")
        meta = pd.read_pickle(os.path.join(pdir, args.pseudo + "_meta.pkl"))
        probs = np.load(os.path.join(pdir, args.pseudo + "_probs.npy"))
        conf = probs.max(1)
        m = meta[conf >= args.conf].copy()
        m["y"] = probs[conf >= args.conf].argmax(1)
    else:
        df = pd.read_pickle(PKL)
        df = df[df["len"] >= args.min_len]
        r = df["rating"].astype(int)
        if args.rmap == "bin_strict":
            m = df[(r <= 2) | (r == 5)].copy()
            m["y"] = (m["rating"].astype(int) == 5).astype(int)
        elif args.rmap == "bin_all":
            m = df[(r <= 2) | (r >= 4)].copy()
            m["y"] = (m["rating"].astype(int) >= 4).astype(int)
        else:
            m = df.copy()
            m["y"] = np.select([r >= 4, r == 3, r <= 2], [0, 1, 2])
    if args.per_class:
        m = pd.concat([g.sample(min(len(g), args.per_class), random_state=args.seed)
                       for _, g in m.groupby("y")])
    if len(m) > args.n:
        m = m.sample(args.n, random_state=args.seed)
    print(f"pool={len(m)} dist={m['y'].value_counts().sort_index().to_dict()}", flush=True)

    lo, hi = args.ngram.split("-")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(int(lo), int(hi)),
                          min_df=3, max_features=args.max_features, sublinear_tf=True)
    X = vec.fit_transform(m["content"])
    print("X:", X.shape, flush=True)
    clf = LogisticRegression(C=args.C, max_iter=1000, n_jobs=-1)
    clf.fit(X, m["y"])
    # corpus holdout acc (fit monitor only)
    # gold eval
    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    rows = [x for x in rows if x.get("label_semantic")]
    texts = [x["text"] for x in rows]
    is3 = args.rmap == "tri" or (args.pseudo and int(m["y"].max()) > 1)
    if is3:
        y = np.array([LAB3[x["label_semantic"]] for x in rows]); names = N3
    else:
        keep = [i for i, x in enumerate(rows) if x["label_semantic"] != "中性"]
        texts = [texts[i] for i in keep]
        y = np.array([1 if rows[i]["label_semantic"] == "正面" else 0 for i in keep])
        names = ["负面", "正面"]
    Xg = vec.transform(texts)
    pred = clf.predict(Xg)
    probs = clf.predict_proba(Xg)  # columns follow clf.classes_ (always sorted 0..k-1)
    acc = accuracy_score(y, pred); f1 = f1_score(y, pred, average="macro")
    print(f"GOLD600 {args.rmap}: acc={acc:.4f} macro_f1={f1:.4f}", flush=True)
    print(classification_report(y, pred, target_names=names, digits=4), flush=True)
    res = {"tag": args.tag, "args": vars(args), "n": len(m),
           "gold": {"acc": acc, "macro_f1": f1}, "is3": bool(is3)}
    os.makedirs(str(DATA_ROOT / "results"), exist_ok=True)
    json.dump(res, open(str(DATA_ROOT / "results" / f"{args.tag}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    np.save(str(DATA_ROOT / "results" / f"{args.tag}_goldprobs.npy"), probs.astype(np.float32))
    np.save(str(DATA_ROOT / "results" / f"{args.tag}_gdyy.npy"), y.astype(np.int64))


if __name__ == "__main__":
    main()

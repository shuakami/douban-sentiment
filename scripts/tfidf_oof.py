#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TF-IDF + LogisticRegression out-of-fold probabilities for ensembling."""
import os, json, time, argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "reviews_600_base.jsonl")

OUT = os.path.join(ROOT, "out")
LAB3 = {"正面": 0, "中性": 1, "负面": 2}
N3 = ["正面", "中性", "负面"]


def load_rows(label_key="label_semantic", name="reviews_600_base.jsonl"):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", name), encoding="utf-8")]
    return [r for r in rows if r.get(label_key)]


def make_labels(rows, task):
    texts, ys = [], []
    for r in rows:
        lab = r["label_semantic"]
        if task == "3class":
            y = LAB3[lab]
        elif task == "neu":
            y = 1 if lab == "中性" else 0
        elif task == "binary":
            if lab == "中性":
                continue
            y = 1 if lab == "正面" else 0
        else:
            raise ValueError(task)
        texts.append(r["text"])
        ys.append(y)
    return texts, np.array(ys, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["3class", "binary", "neu"], default="3class")
    ap.add_argument("--data", default="reviews_600_base.jsonl")
    ap.add_argument("--label_key", default="label_semantic")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--analyzer", default="char_wb")
    ap.add_argument("--ngram", default="1-3")
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_features", type=int, default=200000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = load_rows(args.label_key, args.data)
    texts, y = make_labels(rows, args.task)
    n_classes = int(y.max() + 1)
    n1, n2 = map(int, args.ngram.split("-"))
    print(f"# tfidf task={args.task} n={len(y)} classes={n_classes} analyzer={args.analyzer} ngram={args.ngram}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    t0 = time.time()
    for k, (tr, va) in enumerate(skf.split(texts, y), 1):
        vec = TfidfVectorizer(analyzer=args.analyzer, ngram_range=(n1, n2), min_df=args.min_df,
                              sublinear_tf=True, max_features=args.max_features)
        clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced", n_jobs=-1)
        m = Pipeline([("v", vec), ("c", clf)])
        m.fit([texts[i] for i in tr], y[tr])
        oof[va] = m.predict_proba([texts[i] for i in va])
        print(f"fold{k} done {time.time()-t0:.1f}s")

    if args.task == "3class":
        pred = oof.argmax(axis=1); names = N3
    elif args.task == "neu":
        pred = (oof[:, 1] >= 0.5).astype(int); names = ["非中性", "中性"]
    else:
        pred = (oof[:, 1] >= 0.5).astype(int); names = ["负面", "正面"]
    res = {
        "model": f"tfidf_{args.analyzer}_{args.ngram}_C{args.C}",
        "task": args.task,
        "label_key": args.label_key,
        "n": len(y),
        "n_classes": n_classes,
        "config": vars(args),
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "report": classification_report(y, pred, target_names=names, digits=4, output_dict=True),
        "confusion": confusion_matrix(y, pred).tolist(),
        "pred": pred.tolist(),
        "y": y.tolist(),
        "probs": oof.tolist(),
        "total_secs": time.time() - t0,
    }
    tag = args.tag or f"tfidf_{args.task}_{args.analyzer}_{args.ngram.replace('-', '_')}_C{args.C}"
    path = os.path.join(OUT, tag + ".json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"acc={res['acc']:.4f} macro_f1={res['macro_f1']:.4f} total={res['total_secs']:.1f}s saved {path}")


if __name__ == "__main__":
    main()

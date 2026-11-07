#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TF-IDF + 线性模型基线：分层 K 折交叉验证，支持两套标签视图与三/二分类。

用途：
  1) 作为 BERT 微调的性能下限参照（若 BERT 打不过 TF-IDF，说明数据或标注有问题）
  2) 快速探查「弱监督标签A」与「语义标签B」哪个更容易学

用法：
  python3.11 baseline_tfidf.py --label rule
  python3.11 baseline_tfidf.py --label semantic --binary drop_neutral
  python3.11 baseline_tfidf.py --label both --folds 5
"""
import json, argparse, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "reviews_600_base.jsonl")

POS, NEU, NEG = 0, 1, 2
NAMES3 = ["正面", "中性", "负面"]


def load(label_key, binary):
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    texts, ys, kept = [], [], []
    for r in rows:
        lab = r[label_key]
        if lab is None:
            continue
        if binary == "drop_neutral":
            if lab == "中性":
                continue
            y = 1 if lab == "正面" else 0
        elif binary == "neg_vs_rest":
            y = 1 if lab == "负面" else 0
        else:
            y = {"正面": POS, "中性": NEU, "负面": NEG}[lab]
        texts.append(r["text"])
        ys.append(y)
        kept.append(r)
    return texts, np.array(ys), kept


def make_model(kind):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2,
                          sublinear_tf=True, max_features=200000)
    if kind == "lr":
        clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced", n_jobs=-1)
    else:
        clf = LinearSVC(C=0.5, class_weight="balanced")
    return Pipeline([("v", vec), ("c", clf)])


def cv_eval(texts, y, kind, folds, seed=42):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1m, f1w = [], [], []
    oof = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(texts, y):
        m = make_model(kind)
        m.fit([texts[i] for i in tr], y[tr])
        pred = m.predict([texts[i] for i in te])
        oof[te] = pred
        accs.append(accuracy_score(y[te], pred))
        f1m.append(f1_score(y[te], pred, average="macro"))
        f1w.append(f1_score(y[te], pred, average="weighted"))
    return (np.mean(accs), np.std(accs), np.mean(f1m), np.std(f1m),
            np.mean(f1w), oof, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["rule", "semantic", "both"], default="both")
    ap.add_argument("--binary", choices=["none", "drop_neutral", "neg_vs_rest"], default="none")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model", choices=["lr", "svm", "all"], default="all")
    args = ap.parse_args()

    keys = ["label_rule", "label_semantic"] if args.label == "both" else [f"label_{args.label}"]
    models = ["lr", "svm"] if args.model == "all" else [args.model]

    print(f"# TF-IDF 基线 | 二分类模式={args.binary} | {args.folds} 折分层CV\n")
    print("| 标签视图 | 模型 | 样本数 | Acc | Macro-F1 | Weighted-F1 |")
    print("|---|---|---|---|---|---|")
    best = None
    for k in keys:
        texts, y, _ = load(k, args.binary)
        for mk in models:
            acc, acc_s, f1m, f1s, f1w, oof, yt = cv_eval(texts, y, mk, args.folds)
            tag = "弱监督A(星级)" if k == "label_rule" else "语义B"
            print(f"| {tag} | {mk} | {len(y)} | {acc:.4f}±{acc_s:.3f} | "
                  f"{f1m:.4f}±{f1s:.3f} | {f1w:.4f} |")
            if best is None or f1m > best[0]:
                best = (f1m, k, mk, oof, yt)
    print("\n## 最优配置的 OOF 详细报告\n")
    f1m, k, mk, oof, yt = best
    names = (NAMES3 if args.binary == "none" else
             ["负面", "正面"] if args.binary == "drop_neutral" else ["非负面", "负面"])
    print(f"标签视图={k} 模型={mk}  Macro-F1={f1m:.4f}\n")
    print(classification_report(yt, oof, target_names=names, digits=4))
    print("混淆矩阵（行=真实，列=预测）:\n")
    print("      " + "".join(f"{n:>8}" for n in names))
    cm = confusion_matrix(yt, oof)
    for i, n in enumerate(names):
        print(f"{n:>6}" + "".join(f"{v:>8}" for v in cm[i]))


if __name__ == "__main__":
    main()

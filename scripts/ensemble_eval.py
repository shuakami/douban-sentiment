#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensemble OOF probabilities produced by gpu_finetune.py / tfidf_oof.py."""
import os, json, argparse, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
N3 = ["正面", "中性", "负面"]


def load_result(path):
    if not os.path.exists(path):
        cand = os.path.join(OUT, path)
        if not cand.endswith(".json"):
            cand += ".json"
        path = cand
    elif not path.endswith(".json") and os.path.isdir(path):
        path = os.path.join(path, "out")
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    return r, np.array(r["probs"], dtype=np.float32), np.array(r["y"], dtype=np.int64)


def pred_from_probs(task, probs):
    if task == "3class":
        return probs.argmax(axis=1)
    return (probs[:, -1] >= 0.5).astype(int)


def score(task, y, probs):
    pred = pred_from_probs(task, probs)
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro")), pred


def report_dict(task, y, pred):
    names = N3 if task == "3class" else ["负面", "正面"]
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "report": classification_report(y, pred, target_names=names, digits=4, output_dict=True),
        "confusion": confusion_matrix(y, pred).tolist(),
    }


def greedy_ensemble(task, y, probs_list, iters=30):
    n = len(y)
    counts = np.zeros(len(probs_list), dtype=int)
    best_probs = None
    best_f1 = -1
    history = []
    for it in range(iters):
        cand_best = None
        for j in range(len(probs_list)):
            c = counts.copy(); c[j] += 1
            avg = sum(counts[i] * probs_list[i] for i in range(len(probs_list)))
            avg = avg + probs_list[j]
            avg = avg / c.sum()
            _, f1, _ = score(task, y, avg)
            if cand_best is None or f1 > cand_best[0]:
                cand_best = (f1, j, avg)
        if cand_best is None:
            break
        f1, j, avg = cand_best
        if f1 <= best_f1 + 1e-12 and it > 0:
            break
        counts[j] += 1
        best_f1 = f1
        best_probs = avg
        history.append({"iter": it + 1, "add": j, "macro_f1": f1, "counts": counts.tolist()})
    return counts, best_probs, best_f1, history


def stack_ensemble(task, y, probs_list, folds=5, seed=42):
    # Leave-one-fold-out logistic stacker on concatenated base probabilities.
    feats = [p.reshape(len(y), -1) for p in probs_list]
    X = np.concatenate(feats, axis=1)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=int)
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(X[tr], y[tr])
        oof_pred[va] = clf.predict(X[va])
    return oof_pred, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="result json paths or out/ tags")
    ap.add_argument("--task", choices=["3class", "binary"], default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--greedy_iters", type=int, default=30)
    args = ap.parse_args()

    results = []
    probs_list = []
    y = None
    for inp in args.inputs:
        r, p, yy = load_result(inp)
        if y is None:
            y = yy
            task = args.task or r.get("task") or ("3class" if p.shape[1] == 3 else "binary")
        else:
            if not np.array_equal(y, yy):
                raise ValueError(f"label mismatch in {inp}")
            task = task or args.task or r.get("task")
        results.append(r)
        probs_list.append(p)
    if args.task:
        task = args.task

    print(f"# ensemble task={task} n={len(y)} models={[r.get('model', r.get('tag')) for r in results]}")
    singles = []
    for r, p in zip(results, probs_list):
        acc, f1, pred = score(task, y, p)
        singles.append({"model": r.get("model", r.get("tag")), "acc": acc, "macro_f1": f1})
        print(f"single {r.get('model', r.get('tag'))}: acc={acc:.4f} macro_f1={f1:.4f}")

    mean_probs = np.mean(probs_list, axis=0)
    acc, f1, pred_mean = score(task, y, mean_probs)
    print(f"mean: acc={acc:.4f} macro_f1={f1:.4f}")

    counts, greedy_probs, greedy_f1, hist = greedy_ensemble(task, y, probs_list, args.greedy_iters)
    gacc, gf1, gpred = score(task, y, greedy_probs)
    print(f"greedy: acc={gacc:.4f} macro_f1={gf1:.4f} counts={counts.tolist()}")

    stack_pred, X = stack_ensemble(task, y, probs_list)
    srep = report_dict(task, y, stack_pred)
    print(f"stack: acc={srep['acc']:.4f} macro_f1={srep['macro_f1']:.4f}")

    methods = {
        "mean": {"probs": mean_probs, "acc": acc, "macro_f1": f1, "pred": pred_mean.tolist()},
        "greedy": {"probs": greedy_probs, "acc": gacc, "macro_f1": gf1, "pred": gpred.tolist(),
                   "counts": counts.tolist(), "history": hist},
        "stack": {"pred": stack_pred.tolist(), "acc": srep["acc"], "macro_f1": srep["macro_f1"],
                  "report": srep["report"], "confusion": srep["confusion"]},
    }
    best_name = max(methods, key=lambda k: methods[k]["macro_f1"])
    best = methods[best_name]
    tag = args.tag or "ensemble_" + "_".join(os.path.basename(inp).replace(".json", "") for inp in args.inputs)[:80]
    res = {
        "task": task,
        "n": len(y),
        "inputs": [os.path.basename(inp) for inp in args.inputs],
        "singles": singles,
        "methods": {k: {kk: vv for kk, vv in v.items() if kk != "probs"} for k, v in methods.items()},
        "best_method": best_name,
        "best": {k: v for k, v in best.items() if k != "probs"},
        "y": y.tolist(),
        "probs": best.get("probs", mean_probs).tolist() if best_name in ("mean", "greedy") else None,
        "total_secs": None,
    }
    path = os.path.join(OUT, tag + ".json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"best={best_name} acc={best['acc']:.4f} macro_f1={best['macro_f1']:.4f} saved {path}")


if __name__ == "__main__":
    main()

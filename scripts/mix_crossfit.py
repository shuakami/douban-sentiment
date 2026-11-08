#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Honest mix-weight evaluation: stratified 5-fold cross-fit on ood1000.

For every v3+SFT mix used in the paper, the fusion weight w is selected
on the training folds only; each row's prediction uses a weight that
never saw that row's label. Reports OOF accuracy / macro-F1 and saves
OOF probability matrices for downstream analysis (McNemar, subsets, ECE).

Class orders (verified in mix_analysis.py):
  3-class probs3 / SFT 3class : [正面, 中性, 负面]
  binary probsb / SFT binary  : [负面, 正面]  (model order)
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest

LAB = Path(os.environ.get("LAB2000_DIR",
                          r"E:\GB77_data\lab2000"))
RES = LAB / "sft_results"
N3 = ["正面", "中性", "负面"]
N3i = {n: i for i, n in enumerate(N3)}
WGRID = np.arange(0.0, 1.01, 0.05)


def acc_f1(yt, yp, ncls=3):
    yt = np.asarray(yt); yp = np.asarray(yp)
    acc = float((yt == yp).mean())
    f1s = []
    for c in range(ncls):
        tp = int(((yt == c) & (yp == c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return acc, float(np.mean(f1s))


def ece(probs, yt, n_bins=10):
    probs = np.asarray(probs); yt = np.asarray(yt)
    conf = probs.max(1); pred = probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf <= hi if hi == 1 else conf < hi)
        if m.sum() == 0:
            continue
        e += m.sum() / len(yt) * abs((pred[m] == yt[m]).mean() - conf[m].mean())
    return float(e)


def mcnemar(y, p1, p2):
    p1 = np.asarray(p1); p2 = np.asarray(p2)
    a = int(((p1 == y) & (p2 != y)).sum())
    b = int(((p1 != y) & (p2 == y)).sum())
    p = binomtest(min(a, b), a + b, 0.5).pvalue if a + b else 1.0
    return a, b, float(p)


def sft(fn):
    d = json.load(open(RES / fn, encoding="utf-8"))
    return np.array(d["ood1000"]["probs"], dtype=np.float64)


def crossfit_mix(pv, ps, yt, n_splits=5, seed=0, task="3c"):
    """Stratified n_splits CV: tune w on train folds, predict held-out.

    Returns OOF prob matrix (n,k), list of chosen w per fold, and metrics.
    For binary, yt in {0,1} and probs are (n,2).
    """
    n = len(yt)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(pv)
    ws = []
    for tr, te in skf.split(np.zeros(n), yt):
        best_w, best_a = 0.5, -1.0
        for w in WGRID:
            pm = (1 - w) * pv[tr] + w * ps[tr]
            yp = pm.argmax(1) if task == "3c" else (pm[:, 1] >= 0.5).astype(int)
            a = float((yp == yt[tr]).mean())
            if a > best_a:
                best_a, best_w = a, w
        ws.append(float(best_w))
        oof[te] = (1 - best_w) * pv[te] + best_w * ps[te]
    return oof, ws


def metrics_from_oof(oof, yt, task="3c"):
    if task == "3c":
        yp = oof.argmax(1)
    else:
        yp = (oof[:, 1] >= 0.5).astype(int)
    a, f = acc_f1(yt, yp, oof.shape[1])
    return a, f, ece(oof, yt), yp


def main():
    z = np.load(LAB / "v3_probs.npz", allow_pickle=False)
    rows = [json.loads(l) for l in open(LAB / "lab2000_final.jsonl",
                                        encoding="utf-8")]
    ood = [i for i, r in enumerate(rows)
           if r["set"] == "ood1000" and r["final"]]
    y3 = np.array([N3i[rows[i]["final"]] for i in ood])
    pv3 = z["probs3"][ood]
    pvb_all = z["probsb"][ood]          # model order [neg,pos], all 1000
    binmask = np.array([rows[i]["final"] != "中性" for i in ood])
    yb = np.array([1 if rows[i]["final"] == "正面" else 0
                   for i in np.array(ood)[binmask]])
    pvb = pvb_all[binmask]              # (673,2)

    # ---- config table: name -> (3c spec, bin spec) -----------------------
    # spec = list of sft_results filenames whose probs are averaged (seeds),
    #        or None to skip a task.
    CFG = {
        "gold_al":      (["gold_al_3class_e8.json"],
                         ["gold_al_binary_e8.json"]),
        "gold_ds":      (["gold_ds_3class_e8.json"],
                         ["gold_ds_binary_e8.json"]),
        "al6kh":        (["gold_extra_al6kh_3class_e4.json"],
                         ["gold_extra_al6kh_binary_e8.json"]),
        "al11kds_s42":  (["gold_extra_al11kds_3class_e4.json"],
                         ["gold_extra_al11kds_binary_e8.json"]),
        "al11kds_s123": (["gold_extra_al11kds_3class_e4_s123.json"],
                         ["gold_extra_al11kds_binary_e4_s123.json"]),
        "al11kds_s456": (["gold_extra_al11kds_3class_e4_s456.json"],
                         ["gold_extra_al11kds_binary_e4_s456.json"]),
        "al11kds_avg":  (["gold_extra_al11kds_3class_e4.json",
                          "gold_extra_al11kds_3class_e4_s123.json",
                          "gold_extra_al11kds_3class_e4_s456.json"],
                         ["gold_extra_al11kds_binary_e8.json"]),
        "al11kds0":     (["gold_extra_al11kds0_3class_e4.json"],
                         ["gold_extra_al11kds0_binary_e4.json"]),
        "al11kq":       (["gold_extra_al11kq_3class_e4.json"],
                         ["gold_extra_al11kq_binary_e8.json"]),
        "al14kds_s42":  (["gold_extra_al14kds_3class_e4.json"],
                         ["gold_extra_al14kds_binary_e8.json"]),
        "al14kds_s123": (["gold_extra_al14kds_3class_e4_s123.json"],
                         ["gold_extra_al14kds_binary_e4_s123.json"]),
        "al14kds_s456": (["gold_extra_al14kds_3class_e4_s456.json"],
                         None),
        "al14kds_avg":  (["gold_extra_al14kds_3class_e4.json",
                          "gold_extra_al14kds_3class_e4_s123.json",
                          "gold_extra_al14kds_3class_e4_s456.json"],
                         ["gold_extra_al14kds_binary_e8.json"]),
        "al17kds_s42":  (["gold_extra_al17kds_3class_e4.json"],
                         ["gold_extra_al17kds_binary_e4.json"]),
        "al17kds_s123": (["gold_extra_al17kds_3class_e4_s123.json"],
                         ["gold_extra_al17kds_binary_e4_s123.json"]),
        "al17kds_s456": (["gold_extra_al17kds_3class_e4_s456.json"],
                         None),
        "al17kds_avg":  (["gold_extra_al17kds_3class_e4.json",
                          "gold_extra_al17kds_3class_e4_s123.json",
                          "gold_extra_al17kds_3class_e4_s456.json"],
                         ["gold_extra_al17kds_binary_e4.json"]),
        "al17kds_s789": (["gold_extra_al17kds_3class_e4_s789.json"],
                         ["gold_extra_al17kds_binary_e4_s789.json"]),
        "al17kds_e8":   (None,
                         ["gold_extra_al17kds_binary_e8.json"]),
        "al17kds_e8avg": (None,
                         ["gold_extra_al17kds_binary_e8.json",
                          "gold_extra_al17kds_binary_e8_s123.json",
                          "gold_extra_al17kds_binary_e8_s789.json"]),
        "al17kds_avg4": (["gold_extra_al17kds_3class_e4.json",
                          "gold_extra_al17kds_3class_e4_s123.json",
                          "gold_extra_al17kds_3class_e4_s456.json",
                          "gold_extra_al17kds_3class_e4_s789.json"],
                         ["gold_extra_al17kds_binary_e4.json",
                          "gold_extra_al17kds_binary_e4_s123.json",
                          "gold_extra_al17kds_binary_e4_s789.json"]),
        "al3kds":       (["gold_extra_al3kds_3class_e4.json"],
                         ["gold_extra_al3kds_binary_e4.json"]),
    }

    # sanity: reproduce oracle numbers first for key configs
    print("=== oracle sanity (full-set tuned w) ===")
    for name in ["gold_al", "al17kds_avg", "al14kds_avg"]:
        f3, fb = CFG[name]
        ps3 = np.mean([sft(f) for f in f3], axis=0)
        best = max((( (1-w)*pv3 + w*ps3 ).argmax(1) == y3).mean()
                   for w in WGRID)
        print(f"{name} 3c oracle-best acc={best:.4f}")

    out = {"configs": {}, "y3": y3.tolist(), "yb": yb.tolist(),
           "binmask": binmask.tolist()}
    np.savez_compressed(LAB / "mix_oof_probs.npz",
                        pv3_oof=pv3, pvb_oof=pvb, y3=y3, yb=yb,
                        binmask=binmask)

    print("\n=== 5-fold cross-fit mixes ===")
    oof_store = {}
    for name, (f3, fb) in CFG.items():
        rec = {}
        if f3:
            ps3 = np.mean([sft(f) for f in f3], axis=0)
            assert ps3.shape == (1000, 3), (name, ps3.shape)
            oof3, ws3 = crossfit_mix(pv3, ps3, y3, task="3c")
            a, f, e, yp = metrics_from_oof(oof3, y3, "3c")
            rec["3c"] = {"acc": round(a, 4), "f1": round(f, 4),
                         "ece": round(e, 4), "w_folds": ws3}
            oof_store[f"{name}__3c"] = oof3
        if fb:
            psb = np.mean([sft(f) for f in fb], axis=0)
            assert psb.shape == (673, 2), (name, psb.shape)
            oofb, wsb = crossfit_mix(pvb, psb, yb, task="bin")
            a, f, e, yp = metrics_from_oof(oofb, yb, "bin")
            rec["bin"] = {"acc": round(a, 4), "f1": round(f, 4),
                          "ece": round(e, 4), "w_folds": wsb}
            oof_store[f"{name}__bin"] = oofb
        out["configs"][name] = rec
        print(name, json.dumps(rec))

    # also save v3-alone and sft-alone references
    a3, f3_, e3, yp3 = metrics_from_oof(pv3, y3, "3c")
    ab, fb_, eb, ypb = metrics_from_oof(pvb, yb, "bin")
    out["v3"] = {"3c": {"acc": round(a3, 4), "f1": round(f3_, 4),
                        "ece": round(e3, 4)},
                 "bin": {"acc": round(ab, 4), "f1": round(fb_, 4),
                         "ece": round(eb, 4)}}
    print("v3", out["v3"])

    np.savez_compressed(LAB / "mix_oof_probs.npz",
                        **{k: v for k, v in oof_store.items()},
                        y3=y3, yb=yb, binmask=binmask,
                        pv3_oof=pv3, pvb_oof=pvb)
    json.dump(out, open(LAB / "mix_crossfit_results.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nsaved mix_crossfit_results.json + mix_oof_probs.npz")


if __name__ == "__main__":
    main()

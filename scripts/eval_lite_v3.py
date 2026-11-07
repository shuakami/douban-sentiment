#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild the deployed 3-class lite ensembles (v2 / v3) from OOF probs.

Both share the same deployed pos/neg channel:
    dep = (distill_macbert_3c + 2 * headmlp_3c_mb13b) / 3   (= ens_lite_3class.json)

Neutral-expert specialist (binary neutral-vs-rest -> 3-class channel):
    posneg = dep[:, [pos, neg]] renormalized
    spec   = [(1-pn)*posneg_pos, pn, (1-pn)*posneg_neg]
    final  = (dep + 0.5 * spec) / 1.5

pn differs per version:
  v2: pn = mean(p_neu(macbert_neu_s123), p_neu(head_neu_h256))
  v3: pn = 0.6 * p_neu(head_neu_h256) + 0.4 * p_neu(neu_corpus_detector)
      where the corpus detector probs on gold-600 live in out/cascade_pneu.npy
      (produced by eval_cascade.py --neu <detector probs>).

Writes out/ens_lite_3class_v2.json and out/ens_lite_3class_v3.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, HERE)
from ensemble_eval import load_result, score  # noqa: E402

DEP_FILE = "ens_lite_3class"          # dep=(stu+2*judge)/3, probs (n,3) [pos,neu,neg]
STUDENT_NEU = "distill_macbert_neu_a50_t2_e8_m256_s123"   # v2 specialist student
HEAD_NEU = "headmlp_neu_mb13b_h256"                     # mb13b-feature neu MLP head
CORPUS_PNEU = os.path.join(OUT, "cascade_pneu.npy")     # neu_pseudo_p60k_e2 on gold


def load_probs(tag):
    _, p, y = load_result(os.path.join(OUT, tag + ".json"))
    return np.asarray(p, dtype=np.float64), y


def p_neu_2col(tag):
    p, _ = load_probs(tag)
    assert p.shape[1] == 2, f"{tag}: expected (n,2) [non-neu, neu], got {p.shape}"
    return np.clip(p[:, 1], 1e-6, 1 - 1e-6)


def build(dep, pn):
    dep = dep / dep.sum(1, keepdims=True)
    posneg = dep[:, [0, 2]]
    posneg = posneg / posneg.sum(1, keepdims=True)
    spec = np.stack([(1 - pn) * posneg[:, 0], pn, (1 - pn) * posneg[:, 1]], axis=1)
    return (dep + 0.5 * spec) / 1.5


def emit(tag, model, recipe, probs, y):
    acc, f1, pred = score("3class", y, probs)
    res = {
        "model": model,
        "task": "3class",
        "n": int(len(y)),
        "oof": {
            "acc": float(acc),
            "macro_f1": float(f1),
            "confusion": confusion_matrix(y, pred).tolist(),
            "pred": pred.tolist(),
        },
        "recipe": recipe,
        "y": y.tolist(),
        "probs": np.asarray(probs, dtype=np.float64).tolist(),
    }
    path = os.path.join(OUT, tag + ".json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{tag}: acc={acc:.4f} macro_f1={f1:.4f} -> {path}")
    return res


def main():
    dep, y = load_probs(DEP_FILE)
    assert len(dep) == 600

    s_pn = p_neu_2col(STUDENT_NEU)
    h_pn = p_neu_2col(HEAD_NEU)
    d_pn = np.clip(np.load(CORPUS_PNEU).astype(np.float64), 1e-6, 1 - 1e-6)
    assert d_pn.shape == (600,)

    base_recipe = ("dep=(stu+2*judge)/3 ; spec=[(1-pn)*r_pos,pn,(1-pn)*r_neg] ; "
                   "final=(dep+0.5*spec)/1.5 ; ")

    emit("ens_lite_3class_v2", "lite_v2_neu_expert",
         base_recipe + "pn=mean(macbert_neu_s123, head_neu_h256)",
         build(dep, (s_pn + h_pn) / 2), y)

    emit("ens_lite_3class_v3", "lite_v3_neu_corpus",
         base_recipe + "pn=0.6*head_neu_h256+0.4*neu_corpus_detector",
         build(dep, 0.6 * h_pn + 0.4 * d_pn), y)


if __name__ == "__main__":
    main()

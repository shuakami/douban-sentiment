#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-stage neutral injection experiment.

Train a neutral-vs-rest detector (LR on frozen features, 5-fold OOF) and
inject its probability into the 3-class ensemble's 中性 column, rescaling
正面/负面 proportionally. All parts are OOF-honest.
"""
import os, sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, HERE)
import gpu_finetune as gf  # noqa: E402

FEATS = {
    "mb256": "feats_mb13b_m256_m256.npy",
    "mb128": "feats_mb13b_m128_m128.npy",
    "erl110": "feats_erl110_m256.npy",
    "erl330": "feats_erl330_m256.npy",
}
LAB3 = gf.LAB3
N3 = gf.N3


def detector_oof(feats, y_bin, c=0.1, cw="balanced", seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_bin), dtype=np.float32)
    for tr, va in skf.split(feats, y_bin):
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=c, max_iter=3000, class_weight=cw)),
        ])
        pipe.fit(feats[tr], y_bin[tr])
        oof[va] = pipe.predict_proba(feats[va])[:, 1]
    return oof


def main():
    rows = gf.load_rows("label_semantic")
    y3 = np.array([LAB3[r["label_semantic"]] for r in rows], dtype=np.int64)
    y_neu = (y3 == 1).astype(np.int64)

    picks7 = ['erlangshen110_3class_e5_b32_m128_eager',
              'erlangshen110_3class_e5_b32_m256_eager',
              'erlangshen330_3class_e5_b32_m256',
              'frozen_erl110_3class_C0.01',
              'frozen_mb13b_m128_3class_C0.01',
              'swp_mb256mean_3class_mean_C0.003_cwbal',
              'swp_mb256mean_3class_mean_C0.01_cwnone']
    base = np.mean([np.array(json.load(open(os.path.join(OUT, f + '.json'),
                                          encoding='utf-8'))['probs'], dtype=np.float64)
                    for f in picks7], axis=0)
    print(f"base ens7+? acc={accuracy_score(y3, base.argmax(1)):.4f} "
          f"f1={f1_score(y3, base.argmax(1), average='macro'):.4f}")

    dets = {}
    for name, fn in FEATS.items():
        feats = np.load(os.path.join(OUT, fn))
        for c in [0.01, 0.1, 1.0]:
            p = detector_oof(feats, y_neu, c=c)
            # standalone neutral-detector quality on 中性 rows
            pred = (p >= 0.5).astype(int)
            f1n = f1_score(y_neu, pred)
            dets[f"{name}_C{c}"] = p
            print(f"det {name} C={c}: neutral-F1={f1n:.4f} mean_p_neu={p[y_neu==1].mean():.3f} "
                  f"mean_p_other={p[y_neu==0].mean():.3f}")
            # inject
            for alpha in [0.5, 0.75, 1.0]:
                inj = base.copy()
                new_neu = alpha * p + (1 - alpha) * inj[:, 1]
                rest = inj[:, 0] + inj[:, 2]
                scale = np.where(rest > 1e-9, (1 - new_neu) / np.maximum(rest, 1e-9), 0.5)
                inj[:, 0] = inj[:, 0] * scale
                inj[:, 2] = inj[:, 2] * scale
                inj[:, 1] = new_neu
                acc = accuracy_score(y3, inj.argmax(1))
                f1 = f1_score(y3, inj.argmax(1), average="macro")
                print(f"  inject a={alpha}: acc={acc:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()

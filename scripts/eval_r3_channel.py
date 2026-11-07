#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick check: does the r3 pseudo student add signal to the deployed v3?"""
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from _paths import DATA_ROOT


def load(f):
    d = json.load(open(f"out/{f}.json"))
    return np.array(d["probs"]), np.array(d["y"])


stu, y = load("distill_macbert_3c_a50_t2_e8_m256")
judge, _ = load("headmlp_3c_mb13b")
nhead, _ = load("headmlp_neu_mb13b_h256")
det = np.load(str(DATA_ROOT / "models" / "neu_pseudo_p60k_e2_goldprobs_600.npy"))[:, 1]
r3 = np.load(str(DATA_ROOT / "models" / "macbert_3c_r3pseudo_e2_goldprobs.npy"))[:600]
old = np.load(str(DATA_ROOT / "models" / "macbert_3c_pseudo_p55k_e2_goldprobs.npy"))[:600]

dep = (stu + 2 * judge) / 3
posneg = dep[:, 0] + dep[:, 2]
rp = dep[:, 0] / posneg
rn = dep[:, 2] / posneg
pn = 0.6 * nhead[:, 1] + 0.4 * det
spec = np.stack([(1 - pn) * rp, pn, (1 - pn) * rn], 1)
v3 = (dep + 0.5 * spec) / 1.5


def ev(P):
    return (round(accuracy_score(y, P.argmax(1)), 4),
            round(f1_score(y, P.argmax(1), average="macro"), 4))


print("dep", ev(dep))
print("v3 ", ev(v3))
for tag, extra in [("r3", r3), ("old", old)]:
    for w in [0.25, 0.5, 1.0]:
        dep2 = (stu + 2 * judge + w * extra) / (3 + w)
        s = dep2[:, 0] + dep2[:, 2]
        rp2 = dep2[:, 0] / s
        rn2 = dep2[:, 2] / s
        spec2 = np.stack([(1 - pn) * rp2, pn, (1 - pn) * rn2], 1)
        print(f"v3+{tag}dep w={w}", ev((dep2 + 0.5 * spec2) / 1.5))
        print(f"v3+{tag}direct w={w}", ev((v3 + w * extra) / (1 + w)))

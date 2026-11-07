#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
picks = ['frozen_erl110_binary_C0.1', 'frozen_erl330_binary_C0.1',
         'frozen_macbert_binary_C0.01', 'frozen_mb13b_binary_C1.0_C1.0']
P = []
y = None
for f in picks:
    rr = json.load(open(os.path.join(OUT, f + '.json'), encoding='utf-8'))
    p = np.array(rr['probs'], dtype=np.float64)
    yy = np.array(rr['y'])
    if y is None:
        y = yy
    P.append(p)
P = np.stack(P)
y = y.astype(int)


def sc(idx):
    pred = P[idx].mean(0).argmax(1)
    return accuracy_score(y, pred), f1_score(y, pred, average='macro')


print('all4      :', sc([0, 1, 2, 3]))
for i, f in enumerate(picks):
    rest = [j for j in range(4) if j != i]
    print(f'w/o {f:34s}:', sc(rest))
# deployed triple for comparison
tri = ['tfidf_binary_charwb_1_3_C4', 'frozen_mb13b_binary_C1.0_C1.0', 'frozen_erl110_binary_C0.1']
Pt = []
for f in tri:
    rr = json.load(open(os.path.join(OUT, f + '.json'), encoding='utf-8'))
    Pt.append(np.array(rr['probs'], dtype=np.float64))
Pt = np.stack(Pt)
pred = Pt.mean(0).argmax(1)
print('deployed3 :', accuracy_score(y, pred), f1_score(y, pred, average='macro'))
# 4ch minus macbert, plus erl330 only variants
for combo, idxs in {
    'mb+erl110+erl330 (no macbert)': [0, 1, 3],
    'mb+erl110+macbert': [0, 2, 3],
    'mb+erl330+macbert': [1, 2, 3],
    'erl110+erl330+macbert': [0, 1, 2],
}.items():
    print(f'{combo:32s}:', sc(idxs))

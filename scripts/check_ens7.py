#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os, sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
picks = ['erlangshen110_3class_e5_b32_m128_eager', 'erlangshen110_3class_e5_b32_m256_eager',
         'erlangshen330_3class_e5_b32_m256', 'frozen_erl110_3class_C0.01',
         'frozen_mb13b_m128_3class_C0.01', 'swp_mb256mean_3class_mean_C0.003_cwbal',
         'swp_mb256mean_3class_mean_C0.01_cwnone']
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


def sc(idx, PP=None):
    PP = P if PP is None else PP
    pred = PP[idx].mean(0).argmax(1)
    return accuracy_score(y, pred), f1_score(y, pred, average='macro')


print('all7      :', sc(list(range(7))))
for i, f in enumerate(picks):
    rest = [j for j in range(7) if j != i]
    print(f'w/o {f[:44]:46s}:', sc(rest))
print('6ch no C0.003bal:', sc([0, 1, 2, 3, 4, 6]))
print('6ch no C0.01none:', sc([0, 1, 2, 3, 4, 5]))
extra = json.load(open(os.path.join(OUT, 'erlangshen110_3class_e5_b32_m256_s123_eager.json'), encoding='utf-8'))
pe = np.array(extra['probs'], dtype=np.float64)
P8 = np.concatenate([P, pe[None]], 0)
pred = P8[list(range(8))].mean(0).argmax(1)
print('7ch + s123  :', accuracy_score(y, pred), f1_score(y, pred, average='macro'))

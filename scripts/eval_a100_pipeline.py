"""Rebuild deployed-pipeline OOF metrics with alpha=1.0 (Leak-B-clean) channels.

Replicates eval_lite.py / eval_lite_v3.py math exactly.
Corpus detector (Leak A) is unchanged in the 'kept' variants.
"""
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

OUT = 'out/'

def load(tag):
    d = json.load(open(f'{OUT}{tag}.json', encoding='utf-8'))
    return np.asarray(d['probs'], dtype=np.float64), np.asarray(d['y'])

def report(name, probs, y, task):
    if task == 'binary':
        pred = (probs[:, 1] >= 0.5).astype(int)
    else:
        pred = probs.argmax(1)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='macro')
    print(f'{name}: acc={acc:.4f} macro_f1={f1:.4f} n={len(y)}')
    return acc, f1

# ---------- 3-class ----------
s50, y3 = load('distill_macbert_3c_a50_t2_e8_m256')
s100, _ = load('distill_macbert_3c_a100_t2_e8_m256')
j50, _ = load('headmlp_3c_mb13b')
j100, _ = load('headmlp_3c_mb13b_a100')
n50, _ = load('headmlp_neu_mb13b_h256')
n100, _ = load('headmlp_neu_mb13b_a100_h256')
dpn = np.clip(np.load(f'{OUT}cascade_pneu.npy').astype(np.float64), 1e-6, 1 - 1e-6)

def build3c(s, j, h_pn, corpus_pn=None, spec_w=0.5):
    dep = (s + 2 * j) / 3
    dep = dep / dep.sum(1, keepdims=True)
    pn = h_pn if corpus_pn is None else 0.6 * h_pn + 0.4 * corpus_pn
    posneg = dep[:, [0, 2]]
    posneg = posneg / posneg.sum(1, keepdims=True)
    spec = np.stack([(1 - pn) * posneg[:, 0], pn, (1 - pn) * posneg[:, 1]], axis=1)
    return (dep + spec_w * spec) / (1 + spec_w)

print('=== 3-class ===')
# reference: should reproduce 0.8950/0.8594
report('REF a0.5 + corpus (deployed v3)', build3c(s50, j50, np.clip(n50[:,1],1e-6,1-1e-6), dpn), y3, '3class')
report('a1.0 student+judge+head, corpus kept', build3c(s100, j100, np.clip(n100[:,1],1e-6,1-1e-6), dpn), y3, '3class')
report('a1.0 all, corpus REMOVED (pn=head only)', build3c(s100, j100, np.clip(n100[:,1],1e-6,1-1e-6), None), y3, '3class')
report('a0.5 all, corpus REMOVED', build3c(s50, j50, np.clip(n50[:,1],1e-6,1-1e-6), None), y3, '3class')

# ---------- binary ----------
sb50, yb = load('distill_macbert_bin_a50_t2_e8_m256')
sb100, _ = load('distill_macbert_bin_a100_t2_e8_m256')
jb50, _ = load('headmlp_bin_mb13b')
jb100, _ = load('headmlp_bin_mb13b_a100')

def build_bin(s, j):
    return (s + 2 * j) / 3

print('=== binary ===')
report('REF a0.5 (deployed)', build_bin(sb50, jb50), yb, 'binary')
report('a1.0 student+head', build_bin(sb100, jb100), yb, 'binary')

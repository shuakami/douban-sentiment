"""Dedup-592 counterpart of eval_a100_pipeline.py.

Same ensemble math, dedup-aligned OOF artifacts from dedup_retrain.sh.
Compares 600-row (leaky-dup) vs 592-row (dedup) estimates side by side.
"""
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

OUT = 'out/'

def load(tag):
    d = json.load(open(f'{OUT}{tag}.json', encoding='utf-8'))
    return np.asarray(d['probs'], dtype=np.float64), np.asarray(d['y'])

def report(name, probs, y, task='3class'):
    if task == 'binary':
        pred = (probs[:, 1] >= 0.5).astype(int)
    else:
        pred = probs.argmax(1)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='macro')
    print(f'{name}: acc={acc:.4f} macro_f1={f1:.4f} n={len(y)}')
    return acc, f1

def build3c(s, j, h_pn, corpus_pn=None, spec_w=0.5):
    dep = (s + 2 * j) / 3
    dep = dep / dep.sum(1, keepdims=True)
    pn = h_pn if corpus_pn is None else 0.6 * h_pn + 0.4 * corpus_pn
    posneg = dep[:, [0, 2]]
    posneg = posneg / posneg.sum(1, keepdims=True)
    spec = np.stack([(1 - pn) * posneg[:, 0], pn, (1 - pn) * posneg[:, 1]], axis=1)
    return (dep + spec_w * spec) / (1 + spec_w)

def build_bin(s, j):
    return (s + 2 * j) / 3

print('=== 3-class DEDUP-592 ===')
s50, y3 = load('dedup_distill_3c_a50')
s100, _ = load('dedup_distill_3c_a100')
j50, _ = load('dedup_head_3c_a50')
j100, _ = load('dedup_head_3c_a100')
n50, _ = load('dedup_head_neu_a50')
n100, _ = load('dedup_head_neu_a100')
dpn = np.clip(np.load(f'{OUT}cascade_pneu_dedup592.npy').astype(np.float64), 1e-6, 1 - 1e-6)

report('a0.5 deployed-equiv + corpus', build3c(s50, j50, np.clip(n50[:, 1], 1e-6, 1 - 1e-6), dpn), y3)
report('a1.0 + corpus kept', build3c(s100, j100, np.clip(n100[:, 1], 1e-6, 1 - 1e-6), dpn), y3)
report('a1.0 corpus REMOVED', build3c(s100, j100, np.clip(n100[:, 1], 1e-6, 1 - 1e-6), None), y3)
report('a0.5 corpus REMOVED', build3c(s50, j50, np.clip(n50[:, 1], 1e-6, 1 - 1e-6), None), y3)

print('=== binary DEDUP (n=503 non-neutral) ===')
sb50, yb = load('dedup_distill_bin_a50')
sb100, _ = load('dedup_distill_bin_a100')
jb50, _ = load('dedup_head_bin_a50')
jb100, _ = load('dedup_head_bin_a100')
report('a0.5 deployed-equiv', build_bin(sb50, jb50), yb, 'binary')
report('a1.0', build_bin(sb100, jb100), yb, 'binary')

print('=== reference: original 600-row (from eval_a100_pipeline) ===')
print('3c a1.0+corpus: 0.8400-ish | a1.0 no-corpus: 0.8333-ish | bin a1.0: 0.929-ish')

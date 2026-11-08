# -*- coding: utf-8 -*-
"""McNemar: DS-Flash vs al17k mix / v3 on 992 joint-parseable ood1000 rows."""
import json
import numpy as np
from math import comb
from pathlib import Path

LAB = Path(r"E:\GB77_data\lab2000")
N3 = ["正面", "中性", "负面"]
N3i = {n: i for i, n in enumerate(N3)}

rows = [json.loads(l) for l in open(LAB / "lab2000_final.jsonl", encoding="utf-8")]
# ood rows in file order -> index i
ood_idx = [r["i"] for r in rows if r["set"] == "ood1000"]
gold = {r["i"]: N3i[r["final"]] for r in rows
        if r["set"] == "ood1000" and r["final"] in N3i}

ds = {}
for l in open(LAB / "ds_deepseek_flash_labels.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r.get("label") in N3i:
        ds[r["i"]] = N3i[r["label"]]

z = np.load(LAB / "mix_oof_probs.npz", allow_pickle=False)
y3 = z["y3"]  # gold labels aligned to ood1000 order?
print("y3 shape", y3.shape, "first", y3[:5])

# map: probs arrays are indexed by position in ood1000 (0..999); i field = row id
# assume probs row j corresponds to ood_idx[j]
pos_of_i = {i: j for j, i in enumerate(ood_idx)}


def binom_p(k, n):
    if n == 0:
        return 1.0
    lo = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2 * lo / (2 ** n))


def eval_sys(name, probs):
    # probs: (1000,3); restrict to rows where DS has label
    idxs = [pos_of_i[i] for i in ds if i in pos_of_i]
    yt = np.array([gold[i] for i in ds if i in pos_of_i])
    yp_mix = probs[idxs].argmax(1)
    yp_ds = np.array([ds[i] for i in ds if i in pos_of_i])
    n = len(idxs)
    acc_mix = (yp_mix == yt).mean()
    acc_ds = (yp_ds == yt).mean()
    # McNemar mix vs ds
    a = int(((yp_mix == yt) & (yp_ds != yt)).sum())  # mix-only correct
    b = int(((yp_mix != yt) & (yp_ds == yt)).sum())  # ds-only correct
    p = binom_p(min(a, b), a + b)
    print(f"{name}: n={n} mix_acc={acc_mix:.4f} ds_acc={acc_ds:.4f} "
          f"discordant mix-only={a} ds-only={b} p={p:.3e}")


for key in ["al17kds_avg__3c", "al14kds_avg__3c", "gold_al__3c"]:
    eval_sys(key, z[key])
# v3 OOF probs?
if "pv3_oof" in z:
    print("pv3_oof", z["pv3_oof"].shape)

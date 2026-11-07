#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build r3 pseudo-labels: deployed v3 ensemble probs on the stu3c pool.

Combines (all aligned to the conf>=0.7 pool of 271,905 rows):
  stu   : stu3c_probs.npy            (MacBERT 3c student, w=1)
  judge : mlp_3c_mb13b_m256.pt on mb13b_pool_c07 feats (w=2)
  neu   : 0.6*mlp_neu_mb13b_m256_h256.pt + 0.4*detector pool probs
Recipe identical to deployed v3 (out/ens_lite_3class_v3.json).

Writes <data_root>/pseudo/stu3c_v3_{meta,probs}.npy for
train_bigdata.py --pseudo stu3c_v3.
"""
import os, sys, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_head as dh  # noqa
from _paths import DATA_ROOT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSEUDO = str(DATA_ROOT / "pseudo")
FEAT = str(DATA_ROOT / "features" / "mb13b_pool_c07_feats.npy")
FEAT_META = str(DATA_ROOT / "features" / "mb13b_pool_c07_meta.pkl")
DET = os.path.join(PSEUDO, "neu_corpus_pool_probs.npy")
JUDGE = os.path.join(ROOT, "models", "mlp_3c_mb13b_m256.pt")
NEUHEAD = os.path.join(ROOT, "models", "mlp_neu_mb13b_m256_h256.pt")
BS = 4096


def head_probs(pt_path, X, device):
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    net = dh.MLP(int(blob["in_dim"]), int(blob["n_classes"]),
                 int(blob["hidden"])).to(device)
    net.load_state_dict(blob["state_dict"])
    net.eval()
    mean = blob["scaler_mean"].astype(np.float32)
    scale = blob["scaler_scale"].astype(np.float32)
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), BS):
            x = torch.tensor((X[i:i + BS] - mean) / scale,
                             dtype=torch.float32, device=device)
            out.append(F.softmax(net(x).float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stu_meta = pd.read_pickle(os.path.join(PSEUDO, "stu3c_meta.pkl"))
    stu_probs = np.load(os.path.join(PSEUDO, "stu3c_probs.npy"))
    conf = stu_probs.max(1)
    keep = np.flatnonzero(conf >= 0.7)
    stu = stu_probs[keep]
    print(f"pool={len(stu)}", flush=True)

    fmeta = pd.read_pickle(FEAT_META)
    assert len(fmeta) == len(stu), "feat meta / pool size mismatch"
    X = np.load(FEAT)
    assert X.shape[0] == len(stu)

    judge = head_probs(JUDGE, X, device)
    print("judge done", judge.shape, flush=True)
    nhead = head_probs(NEUHEAD, X, device)
    print("neu head done", nhead.shape, flush=True)
    det = np.load(DET)
    assert det.shape[0] == len(stu)

    # v3 recipe
    dep = (stu + 2 * judge) / 3
    posneg = dep[:, 0] + dep[:, 2]
    rp = dep[:, 0] / posneg
    rn = dep[:, 2] / posneg
    pn = 0.6 * nhead[:, 1] + 0.4 * det[:, 1]
    spec = np.stack([(1 - pn) * rp, pn, (1 - pn) * rn], 1)
    final = (dep + 0.5 * spec) / 1.5

    meta = stu_meta.iloc[keep].reset_index(drop=True)
    meta.to_pickle(os.path.join(PSEUDO, "stu3c_v3_meta.pkl"))
    np.save(os.path.join(PSEUDO, "stu3c_v3_probs.npy"),
            final.astype(np.float32))
    print("saved stu3c_v3", final.shape,
          "dist", np.bincount(final.argmax(1)).tolist(), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal MegatronBERT-1.3B corpus feature extractor (proven-stable path).

Replicates the inline recipe that works: bf16 weights, inference_mode,
per-chunk .cpu() sync, incremental fp16 checkpoint every 200 chunks.
"""
import os, sys, json, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frozen_megabert import extract_features  # noqa
from _paths import DATA_ROOT

PSEUDO = str(DATA_ROOT / "pseudo")
FEATDIR = str(DATA_ROOT / "features")
MODEL = "IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment"
TAG = "mb13b_stu3c_c07_p25k"
PER_CLASS = 25000
CONF = 0.7
BS = 16
MAX_LEN = 256


def main():
    meta = pd.read_pickle(os.path.join(PSEUDO, "stu3c_meta.pkl"))
    probs = np.load(os.path.join(PSEUDO, "stu3c_probs.npy"))
    conf = probs.max(1)
    m = meta[conf >= CONF].reset_index(drop=True)
    m["y"] = probs[conf >= CONF].argmax(1)
    m = pd.concat([g.sample(min(len(g), PER_CLASS), random_state=42)
                   for _, g in m.groupby("y")]).reset_index(drop=True)
    texts = m["content"].tolist()
    print(f"n={len(texts)}", flush=True)

    os.makedirs(FEATDIR, exist_ok=True)
    m[["cid", "mid", "rating", "votes", "len", "y", "content"]].to_pickle(
        os.path.join(FEATDIR, TAG + "_meta.pkl"))

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
    device = torch.device("cuda")

    ckpt = os.path.join(FEATDIR, TAG + "_ckpt.npy")
    ckpt_json = ckpt + ".json"
    feats, start, t0 = [], 0, time.time()
    if os.path.exists(ckpt) and os.path.exists(ckpt_json):
        try:
            dr = json.load(open(ckpt_json))["done_rows"]
            dr -= dr % BS
            if 0 < dr < len(texts):
                feats.append(np.load(ckpt).astype(np.float32))
                start = dr
                print(f"resume {dr}", flush=True)
        except Exception as e:
            print("bad ckpt, restarting:", e, flush=True)
            feats, start = [], 0

    for i in range(start, len(texts), BS):
        chunk = extract_features(model, tok, texts[i:i + BS], MAX_LEN,
                                 device, bs=BS)
        feats.append(chunk)
        done = i + BS
        if (i // BS) % 20 == 0:
            print(f"{min(done, len(texts))}/{len(texts)} "
                  f"{time.time()-t0:.0f}s", flush=True)
            tmp = ckpt + ".tmp.npy"
            np.save(tmp, np.concatenate(feats).astype(np.float16))
            os.replace(tmp, ckpt)
            json.dump({"done_rows": min(done, len(texts))},
                      open(ckpt_json, "w"))
    X = np.concatenate(feats).astype(np.float32)
    out = os.path.join(FEATDIR, TAG + "_feats.npy")
    tmp = out + ".tmp.npy"
    np.save(tmp, X)
    os.replace(tmp, out)
    json.dump({"tag": TAG, "n": len(m), "feat_dim": X.shape[1],
               "secs": time.time() - t0},
              open(os.path.join(FEATDIR, TAG + ".json"), "w"), indent=1)
    print("saved", X.shape, flush=True)


if __name__ == "__main__":
    main()

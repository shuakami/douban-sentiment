#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract frozen MegatronBERT-1.3B features for pseudo-labeled corpus rows.
Saves <data_root>/features/<tag>_feats.npy (+ _meta.pkl aligned)."""
import os, sys, json, argparse, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo", default="stu3c")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--per-class", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    meta = pd.read_pickle(os.path.join(PSEUDO, args.pseudo + "_meta.pkl"))
    probs = np.load(os.path.join(PSEUDO, args.pseudo + "_probs.npy"))
    conf = probs.max(1)
    m = meta[conf >= args.conf].reset_index(drop=True)
    m["y"] = probs[conf >= args.conf].argmax(1)
    if args.per_class:
        m = pd.concat([g.sample(min(len(g), args.per_class), random_state=42)
                       for _, g in m.groupby("y")])
    m = m.reset_index(drop=True)
    print(f"subset n={len(m)} dist={m['y'].value_counts().sort_index().to_dict()}", flush=True)
    os.makedirs(FEATDIR, exist_ok=True)
    m[["cid", "mid", "rating", "votes", "len", "y", "content"]].to_pickle(
        os.path.join(FEATDIR, args.tag + "_meta.pkl"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model,
                                    torch_dtype=torch.bfloat16).to(device)
    t0 = time.time()
    texts = m["content"].tolist()
    ckpt_path = os.path.join(FEATDIR, args.tag + "_ckpt.npy")
    feats = []
    done_rows = 0
    start_i = 0
    ckpt_json = ckpt_path + ".json"
    if os.path.exists(ckpt_json) and os.path.exists(ckpt_path):
        meta_ck = json.load(open(ckpt_json))
        dr = int(meta_ck.get("done_rows", 0))
        dr -= dr % args.bs
        if 0 < dr < len(texts):
            feats.append(np.load(ckpt_path).astype(np.float32))
            done_rows = dr
            start_i = dr
            print(f"resuming from {dr} rows", flush=True)
    for i in range(start_i, len(texts), args.bs):
        chunk = extract_features(model, tok, texts[i:i + args.bs],
                                 args.max_len, device, bs=args.bs)
        feats.append(chunk)
        done_rows += len(texts[i:i + args.bs])
        if (i // args.bs) % 200 == 0:
            el = time.time() - t0
            print(f"  {done_rows}/{len(texts)} feats elapsed={el:.0f}s", flush=True)
            np.save(ckpt_path, np.concatenate(feats).astype(np.float16))
            json.dump({"done_rows": done_rows, "n_total": len(texts),
                       "secs": el},
                      open(ckpt_path + ".json", "w"))
    X = np.concatenate(feats).astype(np.float32)
    print(f"feats {X.shape} in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(FEATDIR, exist_ok=True)
    np.save(os.path.join(FEATDIR, args.tag + "_feats.npy"), X)
    m[["cid", "mid", "rating", "votes", "len", "y", "content"]].to_pickle(
        os.path.join(FEATDIR, args.tag + "_meta.pkl"))
    json.dump({"tag": args.tag, "n": len(m), "feat_dim": X.shape[1],
               "max_len": args.max_len, "secs": time.time() - t0},
              open(os.path.join(FEATDIR, args.tag + ".json"), "w"), indent=1)
    print("saved", args.tag, flush=True)


if __name__ == "__main__":
    main()

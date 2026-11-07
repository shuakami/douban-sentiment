#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pseudo-label the big corpus with a fine-tuned checkpoint.
Saves per-row probabilities aligned to comments.pkl row order (subset kept)."""
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bigdata import predict_probs, TokenizedDS  # noqa
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from _paths import DATA_ROOT

PKL = str(DATA_ROOT / "clean" / "comments.pkl")
OUT = str(DATA_ROOT / "pseudo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_pickle(PKL)
    df = df[df["len"] >= args.min_len].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"scoring {len(df)} rows with {args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    t0 = time.time()
    probs = predict_probs(model, tok, df["content"].tolist(), args.max_len,
                          device, bs=args.batch, amp="bf16")
    print(f"done in {time.time()-t0:.0f}s shape={probs.shape}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, args.tag + "_probs.npy"), probs.astype(np.float32))
    df[["cid", "uid", "mid", "rating", "votes", "len", "content"]].to_pickle(
        os.path.join(OUT, args.tag + "_meta.pkl"))
    meta = {"model": args.model, "n": len(df), "max_len": args.max_len,
            "secs": time.time() - t0}
    json.dump(meta, open(os.path.join(OUT, args.tag + ".json"), "w"), indent=1)
    print("saved", args.tag, flush=True)


if __name__ == "__main__":
    main()

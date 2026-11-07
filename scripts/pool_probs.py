#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score an arbitrary model on the stu3c pseudo-label pool rows only.

Unlike pseudo_label.py (which scores all of comments.pkl), this reads
<data_root>/pseudo/stu3c_meta.pkl and scores exactly those texts, saving
probs aligned to pool row order. Used for r3 teacher construction.
"""
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
from train_bigdata import predict_probs  # noqa
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from _paths import DATA_ROOT

PSEUDO = str(DATA_ROOT / "pseudo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--conf", type=float, default=0.7,
                    help="keep stu3c pool rows with max prob >= this; "
                         "0 = score all meta rows")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_pickle(os.path.join(PSEUDO, "stu3c_meta.pkl"))
    if args.conf > 0:
        probs0 = np.load(os.path.join(PSEUDO, "stu3c_probs.npy"))
        meta = meta[probs0.max(1) >= args.conf].reset_index(drop=True)
    texts = meta["content"].tolist()
    print(f"scoring {len(texts)} pool rows with {args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    t0 = time.time()
    probs = predict_probs(model, tok, texts, args.max_len,
                          device, bs=args.batch, amp="bf16")
    print(f"done in {time.time()-t0:.0f}s shape={probs.shape}", flush=True)

    np.save(os.path.join(PSEUDO, args.tag + "_probs.npy"),
            probs.astype(np.float32))
    json.dump({"model": args.model, "n": len(texts),
               "max_len": args.max_len, "secs": time.time() - t0},
              open(os.path.join(PSEUDO, args.tag + ".json"), "w"), indent=1)
    print("saved", args.tag, flush=True)


if __name__ == "__main__":
    main()

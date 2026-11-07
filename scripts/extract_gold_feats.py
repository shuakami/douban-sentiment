#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract frozen-backbone features for the gold set -> out/feats_*.npy.

Produces the feature files consumed by distill_head.py / train_final_head.py.
Loads the backbone in bf16 — the same dtype web/app.py uses at serving time —
so reproduced heads see train/serve-consistent features.

Usage: extract_gold_feats.py [--model MODEL] [--max-len 256]
                             [--out out/feats_mb13b_m256.npy]
"""
import os, sys, json, argparse

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frozen_megabert import extract_features  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="out/feats_mb13b_m256.npy")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(
        os.path.join(ROOT, "data", "reviews_600_base.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r.get("label_semantic")]
    texts = [r["text"] for r in rows]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    X = extract_features(model, tok, texts, args.max_len,
                         torch.device("cuda"), bs=args.bs)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, X.astype(np.float32))
    print("saved", X.shape, "->", out)


if __name__ == "__main__":
    main()

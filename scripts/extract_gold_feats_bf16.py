#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract gold-600 MegatronBERT-1.3B features with bf16 weights.

Matches extract_mb13b_corpus.py dtype so corpus-trained heads see
consistent features at train/eval time. Saves out/feats_mb13b_bf16.npy.
"""
import os, sys, json

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
MODEL = "IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment"


def main():
    rows = [json.loads(l) for l in open(
        os.path.join(ROOT, "data", "reviews_600_base.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r.get("label_semantic")]
    texts = [r["text"] for r in rows]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
    X = extract_features(model, tok, texts, 256, torch.device("cuda"), bs=16)
    np.save(os.path.join(ROOT, "out", "feats_mb13b_bf16.npy"), X.astype(np.float32))
    print("saved", X.shape)


if __name__ == "__main__":
    main()

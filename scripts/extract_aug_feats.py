#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract mb13b m256 CLS+mean features for the 48 aug rows and build the
combined 648-row feature file used by distill_head.py.

  out/feats_mb13b_m256_aug.npy  — 48 x 4096 (aug rows only)
  out/feats_mb13b_m256_v2.npy   — 648 x 4096 (600 base + 48 aug, row order)
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gpu_finetune as gf  # noqa: E402
import frozen_megabert as fm  # noqa: E402
from transformers import AutoTokenizer, AutoModel  # noqa: E402

OUT = os.path.join(ROOT, "out")
MODEL = fm.MODEL


def main():
    rows = gf.load_rows("label_semantic")
    aug = [r for r in rows if r.get("src") == "aug"]
    texts = [r["text"] for r in aug]
    print(f"# extracting mb13b m256 features for {len(texts)} aug texts", flush=True)

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    feats_aug = fm.extract_features(model, tok, texts, 256, device, 16)
    del model
    torch.cuda.empty_cache()
    print(f"# aug feats {feats_aug.shape}", flush=True)

    base_path = os.path.join(OUT, "feats_mb13b_m256.npy")
    base = np.load(base_path)
    assert base.shape[0] == 600 and feats_aug.shape[0] == len(aug)
    v2 = np.concatenate([base, feats_aug], axis=0).astype(np.float32)
    assert v2.shape[0] == len(rows) == 648

    np.save(os.path.join(OUT, "feats_mb13b_m256_aug.npy"), feats_aug)
    np.save(os.path.join(OUT, "feats_mb13b_m256_v2.npy"), v2)
    print(f"saved feats_mb13b_m256_aug.npy {feats_aug.shape} and feats_mb13b_m256_v2.npy {v2.shape}")


if __name__ == "__main__":
    main()

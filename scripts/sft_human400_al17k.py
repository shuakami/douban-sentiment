#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retrain al17kds SFT (gold-600 + extra_al17k_ds) and predict human400.

Same recipe/seeds as the fresh500 al17k run (models were not checkpointed,
so they are retrained — identical data, seeds, epochs, max_len=256).
Saves probs npy per (task, seed) to <DATA_ROOT>/human400/
human400_sft_probs_al17k/.

Headline configs (match fresh500_sft_probs_al17k naming):
  3class e4 seeds 42 123 456 789 999
  binary  e8 seeds 42 123 789 999

Resilient mode: run ONE (task, seed) per process invocation so a ROCm
c10 AbortHandler crash only loses that seed; the npy-exists check makes
every invocation resumable:

  python scripts/sft_human400_al17k.py --task 3class --seed 123
  python scripts/sft_human400_al17k.py --task binary  --seed 42
  python scripts/sft_human400_al17k.py --all        # in-process batch

Requires DOUBAN_DATA_ROOT with lab2000/extra_al17k_ds.jsonl and
human400_labels.jsonl input from repo data/.
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=<cache>
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments here —
# it triggers c10_cuda_check_implementation failures on this ROCm build.

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _paths import DATA_ROOT  # noqa: E402
import sft_lab2000 as S  # noqa: E402
import gpu_finetune as gf  # noqa: E402

LAB = DATA_ROOT / "lab2000"
OUT = DATA_ROOT / "human400"
PDIR = OUT / "human400_sft_probs_al17k"
MODEL = S.MODEL
EXTRA = LAB / "extra_al17k_ds.jsonl"
IN = ROOT / "data" / "human400_labels.jsonl"

CONFIGS = {"3class": (4, [42, 123, 456, 789, 999]),
           "binary": (8, [42, 123, 789, 999])}


def run(task, epochs, seeds, items, ftexts, tok):
    tr_items = S.filter_items(items, task)
    texts, y = S.to_xy(tr_items, task)
    n_classes = int(y.max() + 1)
    print(f"[{task}] n_train={len(texts)} n_classes={n_classes} "
          f"epochs={epochs} seeds={seeds}", flush=True)
    for seed in seeds:
        out = PDIR / f"{task}_e{epochs}_s{seed}.npy"
        if out.exists():
            print(f"skip {out.name} (exists)", flush=True)
            continue
        ns = argparse.Namespace(
            epochs=epochs, batch=32, lr=2e-5, weight_decay=0.01,
            warmup_ratio=0.1, clip=1.0, max_len=256, seed=seed,
            class_weight=0, amp="bf16")
        model, device, secs = S.train_all(MODEL, tok, texts, y,
                                          n_classes, ns)
        p = S.predict(ftexts, model, tok, device, 256, 64, "bf16")
        np.save(out, p.astype(np.float32))
        print(f"{task}_e{epochs}_s{seed}: train_secs={secs:.0f} "
              f"saved {p.shape}", flush=True)
        # No gc.collect()/empty_cache() between seeds — allocator re-init
        # on ROCm/Windows is what triggered c10 AbortHandler crashes.
        del model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["3class", "binary"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    PDIR.mkdir(parents=True, exist_ok=True)

    gold = S.load_gold()
    extra = S.load_all(str(EXTRA))
    items = gold + extra
    rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
    rows = [r for r in rows if r.get("label")]
    ftexts = [(r["text"] or "").strip()[:256] for r in rows]
    tok = __import__("transformers").AutoTokenizer.from_pretrained(MODEL)

    if args.all:
        for task, (ep, seeds) in CONFIGS.items():
            run(task, ep, seeds, items, ftexts, tok)
    else:
        ep, seeds = CONFIGS[args.task]
        ep = args.epochs or ep
        seeds = [args.seed] if args.seed is not None else seeds
        run(args.task, ep, seeds, items, ftexts, tok)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

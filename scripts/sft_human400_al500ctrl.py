#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random-vs-AL control at n=500 for human400 evaluation.

Trains the al17k SFT recipe (gold-600 + 500 DeepSeek-labelled rows) twice:
once with 500 uniformly-random corpus rows (fresh500 labels) and once with
the first 500 uncertainty-selected AL rows (extra_al17k_ds head). Same
teacher family (DeepSeek), same budget, same seeds/epochs — isolates the
effect of sampling strategy.

Resilient per-(arm,task,seed) invocation:

  python scripts/sft_human400_al500ctrl.py --arm random --task 3class --seed 42
  python scripts/sft_human400_al500ctrl.py --arm al     --task binary --seed 999
  python scripts/sft_human400_al500ctrl.py --arm all    --task 3class   # all seeds

Requires DOUBAN_DATA_ROOT with lab2000/ctrl500_{random,al}_ds.jsonl and
human400/human400_labels.jsonl (text-bearing copy).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _paths import DATA_ROOT  # noqa: E402
import sft_lab2000 as S  # noqa: E402
import gpu_finetune as gf  # noqa: E402

LAB = DATA_ROOT / "lab2000"
OUT = DATA_ROOT / "human400"
MODEL = S.MODEL
IN = DATA_ROOT / "human400" / "human400_labels.jsonl"

CONFIGS = {"3class": (4, [42, 123, 456, 789, 999]),
           "binary": (8, [42, 123, 789, 999])}


def run(arm, task, epochs, seeds, items, ftexts, tok):
    pdir = OUT / f"human400_sft_probs_ctrl500_{arm}"
    pdir.mkdir(parents=True, exist_ok=True)
    tr_items = S.filter_items(items, task)
    texts, y = S.to_xy(tr_items, task)
    n_classes = int(y.max() + 1)
    print(f"[{arm}/{task}] n_train={len(texts)} n_classes={n_classes} "
          f"epochs={epochs} seeds={seeds}", flush=True)
    for seed in seeds:
        out = pdir / f"{task}_e{epochs}_s{seed}.npy"
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
        print(f"{arm}/{task}_e{epochs}_s{seed}: train_secs={secs:.0f} "
              f"saved {p.shape}", flush=True)
        del model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["random", "al", "all"], required=True)
    ap.add_argument("--task", choices=["3class", "binary"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    arms = ["random", "al"] if args.arm == "all" else [args.arm]
    gold = S.load_gold()
    rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
    rows = [r for r in rows if r.get("label")]
    ftexts = [(r["text"] or "").strip()[:256] for r in rows]
    tok = __import__("transformers").AutoTokenizer.from_pretrained(MODEL)

    tasks = [args.task] if args.task else list(CONFIGS)
    for arm in arms:
        extra = S.load_all(str(LAB / f"ctrl500_{arm}_ds.jsonl"))
        items = gold + extra
        for task in tasks:
            ep, seeds = CONFIGS[task]
            ep = args.epochs or ep
            seeds = [args.seed] if args.seed is not None else seeds
            if args.all and args.seed is None:
                run(arm, task, ep, seeds, items, ftexts, tok)
            else:
                run(arm, task, ep, seeds, items, ftexts, tok)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

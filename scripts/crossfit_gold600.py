#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-fit gold-600 evaluation for the gold+al SFT recipe.

The gold_al/al6k SFT models train on ALL 600 gold rows, so their gold-600
accuracy is resubstitution-inflated. To honestly compare a candidate
v4 (mix of v3 OOF probs + SFT probs) against deployed v3 on gold-600,
we train 5 fold-models: each holds out 1/5 of gold-600 (stratified),
trains on the remaining 480 gold + ALL al1000 (+ optional extra AL rows
with LLM labels), and predicts the held-out fold. The resulting gold-600
probabilities are honest; mixing them with v3's OOF probabilities gives
an honest gold-600 estimate of the deployed mix.

Outputs E:/GB77_data/lab2000/crossfit_gold600.json with per-fold probs.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpu_finetune as gf  # noqa: E402
import sft_lab2000 as sl  # noqa: E402

LAB = Path(r"E:\GB77_data\lab2000\lab2000_final.jsonl")
GOLD = HERE.parent / "data" / "reviews_600_base.jsonl"
MODEL = "hfl/chinese-macbert-base"
OUT = Path(r"E:\GB77_data\lab2000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra_lab", default=None,
                    help="optional JSONL of extra AL rows with `final` labels")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--task", default="3class", choices=["3class", "binary"])
    ap.add_argument("--tag", default="gold600")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")

    gold = sl.load_gold()                       # 600 rows, label_semantic
    al = sl.load_lab("al1000", LAB)             # released-label AL (mixed provenance)
    extra = sl.load_all(args.extra_lab) if args.extra_lab else []
    train_pool = gold + al + extra
    print(f"train_pool={len(train_pool)} (gold={len(gold)} al={len(al)} "
          f"extra={len(extra)})", flush=True)

    # stratified 5-fold over gold only
    rng = np.random.RandomState(args.seed)
    yt_g = np.array([gf.LAB3[g["label"]] for g in gold])
    folds = [[] for _ in range(5)]
    for c in range(3):
        idx = np.nonzero(yt_g == c)[0]
        idx = rng.permutation(idx)
        for j, i in enumerate(idx):
            folds[j % 5].append(int(i))

    tok = AutoTokenizer.from_pretrained(MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gold_probs = np.zeros((len(gold), 3 if args.task == "3class" else 2),
                          dtype=np.float32)
    gold_yt = yt_g if args.task == "3class" else np.array(
        [1 if g["label"] == "正面" else 0 for g in gold])

    t0 = time.time()
    for fold in range(5):
        held = set(folds[fold])
        tr_gold = [g for i, g in enumerate(gold) if i not in held]
        items = sl.filter_items(tr_gold + al + extra, args.task)
        texts = [it["text"] for it in items]
        y = np.array([gf.LAB3[it["label"]] if args.task == "3class"
                      else (1 if it["label"] == "正面" else 0)
                      for it in items], dtype=np.int64)
        n_classes = int(y.max() + 1)

        ns = argparse.Namespace(epochs=args.epochs, batch=32, lr=2e-5,
                                weight_decay=0.01, warmup_ratio=0.1,
                                clip=1.0, max_len=256, seed=args.seed,
                                class_weight=0, amp="bf16")
        model, dev, secs = sl.train_all(MODEL, tok, texts, y, n_classes, ns)

        # predict held-out gold fold
        htexts = [gold[i]["text"] for i in sorted(held)]
        p = sl.predict(htexts, model, tok, dev, 256, 64, "bf16")
        for k, i in enumerate(sorted(held)):
            gold_probs[i] = p[k]
        del model
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        acc = float((gold_probs[sorted(held)].argmax(1) ==
                     gold_yt[sorted(held)]).mean()) if args.task == "3class" \
            else float(((gold_probs[sorted(held)][:, 1] >= 0.5).astype(int)
                        == gold_yt[sorted(held)]).mean())
        print(f"fold {fold}: held={len(held)} train={len(texts)} "
              f"secs={secs:.0f} fold_acc={acc:.4f}", flush=True)

    if args.task == "3class":
        yp = gold_probs.argmax(1)
    else:
        yp = (gold_probs[:, 1] >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, f1_score
    res = {"task": args.task, "epochs": args.epochs,
           "n_extra": len(extra), "seed": args.seed,
           "acc": float(accuracy_score(gold_yt, yp)),
           "macro_f1": float(f1_score(gold_yt, yp, average="macro")),
           "probs": gold_probs.tolist(), "y": gold_yt.tolist(),
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT / f"crossfit_{args.tag}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print("gold600 crossfit:", res["acc"], res["macro_f1"], flush=True)


if __name__ == "__main__":
    main()

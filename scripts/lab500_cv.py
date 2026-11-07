#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5-fold CV fine-tune of MacBERT on the 500 LLM-labelled corpus reviews.

Measures the corpus-domain ceiling: can a model trained on corpus-style
labels (terse, ambiguous reviews) beat the deployed v3 ensemble's 60.4%
agreement with the same labels? Also reports gold-600 transfer per fold.

Usage: python scripts/lab500_cv.py --tag lab500_cv [--epochs 3]
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
LAB3 = {"正面": 0, "中性": 1, "负面": 2}
GOLD = os.path.join(ROOT, "data", "reviews_600_base.jsonl")


class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len, padding=True)
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def collate(batch):
    keys = [k for k in batch[0] if k != "labels"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


@torch.no_grad()
def predict(model, tok, texts, max_len, device, bs=64):
    model.eval()
    ds = DS(texts, [0] * len(texts), tok, max_len)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
    probs = []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**b).logits.float()
        probs.append(F.softmax(logits, dim=-1).cpu())
    return torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-macbert-base")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "lab500_llm.jsonl"))
    ap.add_argument("--label-key", default="label_semantic")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    texts = [r["text"] for r in rows]
    y = np.array([LAB3[r[args.label_key]] for r in rows])
    print(f"n={len(y)} dist={np.bincount(y).tolist()}", flush=True)

    gold_rows = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    gold_texts = [r["text"] for r in gold_rows]
    gold_y = np.array([LAB3[r["label_semantic"]] for r in gold_rows])

    oof = np.zeros((len(y), 3))
    gold_probs = np.zeros((len(gold_y), 3))
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(texts, y)):
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=3, ignore_mismatched_sizes=True).to(device)
        ds = DS([texts[i] for i in tr], y[tr], tok, args.max_len)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed + fold))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = len(dl) * args.epochs
        sched = get_cosine_schedule_with_warmup(opt, int(steps * 0.1), steps)
        model.train()
        for ep in range(args.epochs):
            for b in dl:
                b = {k: v.to(device) for k, v in b.items()}
                labels = b.pop("labels")
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(**b).logits.float(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
        oof[va] = predict(model, tok, [texts[i] for i in va], args.max_len, device)
        gold_probs += predict(model, tok, gold_texts, args.max_len, device) / args.folds
        pred = oof[va].argmax(1)
        print(f"fold{fold} val acc={accuracy_score(y[va],pred):.4f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    pred = oof.argmax(1)
    res = {
        "tag": args.tag, "n": len(y), "folds": args.folds, "epochs": args.epochs,
        "lab500_acc": float(accuracy_score(y, pred)),
        "lab500_f1": float(f1_score(y, pred, labels=[0,1,2], average="macro")),
        "lab500_recall": [float((pred[y==c]==c).mean()) for c in range(3)],
        "lab500_cm": confusion_matrix(y, pred, labels=[0,1,2]).tolist(),
        "gold_acc": float(accuracy_score(gold_y, gold_probs.argmax(1))),
        "gold_f1": float(f1_score(gold_y, gold_probs.argmax(1), labels=[0,1,2], average="macro")),
        "gold_recall": [float((gold_probs.argmax(1)[gold_y==c]==c).mean()) for c in range(3)],
        "secs": time.time() - t0,
    }
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    np.save(os.path.join(OUT, args.tag + "_oof.npy"), oof)
    np.save(os.path.join(OUT, args.tag + "_gold.npy"), gold_probs)
    json.dump(res, open(os.path.join(OUT, args.tag + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

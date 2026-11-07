#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune MacBERT on the large Douban corpus (rating-derived weak labels),
evaluate transfer on the gold 600 semantic labels.

Labels (project convention): 3class -> 0=正面 1=中性 2=负面 ; binary -> 0=负面 1=正面
Rating maps (--rmap):
  bin_strict : 1-2星->neg, 5星->pos          (cleanest, drops 3/4星)
  bin_all    : 1-2星->neg, 4-5星->pos        (drops 3星)
  tri        : 1-2星->neg, 3星->neu, 4-5星->pos
"""
import os, json, time, random, argparse, gc, warnings
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging
from sklearn.metrics import accuracy_score, f1_score, classification_report
from _paths import DATA_ROOT

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = str(DATA_ROOT / "clean" / "comments.pkl")
GOLD = os.path.join(ROOT, "data", "reviews_600_base.jsonl")
OUTDIR = str(DATA_ROOT / "models")
LAB3 = {"正面": 0, "中性": 1, "负面": 2}
N3 = ["正面", "中性", "负面"]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TokenizedDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        enc = tok(list(texts), truncation=True, max_length=max_len, padding=False)
        self.enc = {k: [torch.tensor(x, dtype=torch.long) for x in v] for k, v in enc.items()}
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        out = {k: v[i] for k, v in self.enc.items()}
        out["labels"] = self.labels[i]
        return out


def collate(batch):
    keys = [k for k in batch[0] if k != "labels"]
    out = {k: torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True, padding_value=0) for k in keys}
    out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


def autocast_ctx(device, amp):
    enabled = device.type == "cuda" and amp in ("bf16", "fp16")
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    return torch.autocast(device_type="cuda", dtype=dt, enabled=enabled)


def predict_probs(model, tok, texts, max_len, device, bs=64, amp="bf16"):
    model.eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok(list(texts[i:i + bs]), truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with autocast_ctx(device, amp):
                logits = model(**enc).logits.float()
            probs.append(F.softmax(logits, dim=-1).cpu())
    return torch.cat(probs).numpy()


def gold_eval(model, tok, task, device, max_len, amp):
    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    rows = [r for r in rows if r.get("label_semantic")]
    texts = [r["text"] for r in rows]
    if task == "neu":
        y = np.array([1 if r["label_semantic"] == "中性" else 0 for r in rows])
        names = ["非中性", "中性"]
    elif task == "binary":
        keep = [i for i, r in enumerate(rows) if r["label_semantic"] != "中性"]
        texts = [texts[i] for i in keep]
        y = np.array([1 if rows[i]["label_semantic"] == "正面" else 0 for i in keep])
        names = ["负面", "正面"]
    else:
        y = np.array([LAB3[rows[i]["label_semantic"]] for i in range(len(rows))])
        names = N3
    probs = predict_probs(model, tok, texts, max_len, device, 64, amp)
    if task in ("binary", "neu"):
        pred = (probs[:, 1] >= 0.5).astype(int)
    else:
        pred = probs.argmax(1)
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "report": classification_report(y, pred, target_names=names, digits=4, output_dict=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-macbert-base")
    ap.add_argument("--task", choices=["binary", "3class", "neu"], default="binary")
    ap.add_argument("--rmap", choices=["bin_strict", "bin_all", "tri"], default=None)
    ap.add_argument("--n", type=int, default=200000, help="max train samples (0=all)")
    ap.add_argument("--per-class", type=int, default=0, help="cap per class")
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--pseudo", default="", help="pseudo-label tag under the data root (pseudo/)")
    ap.add_argument("--conf", type=float, default=0.0, help="min max-prob for pseudo labels")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    rmap = args.rmap or ("tri" if args.task == "3class" else "bin_strict")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pseudo:
        pdir = str(DATA_ROOT / "pseudo")
        meta = pd.read_pickle(os.path.join(pdir, args.pseudo + "_meta.pkl"))
        probs = np.load(os.path.join(pdir, args.pseudo + "_probs.npy"))
        conf = probs.max(1)
        m = meta.copy()
        m["y"] = probs.argmax(1)
        if args.task == "neu":
            m["y"] = (m["y"] == 1).astype(int)  # 1=中性
        if args.conf > 0:
            m = m[conf >= args.conf].copy()
        print(f"pseudo={args.pseudo} pool={len(m)} dist={m['y'].value_counts().sort_index().to_dict()}", flush=True)
    else:
        df = pd.read_pickle(PKL)
        df = df[df["len"] >= args.min_len]
        r = df["rating"].astype(int)
        if rmap == "bin_strict":
            m = df[(r <= 2) | (r == 5)].copy()
            m["y"] = (m["rating"].astype(int) == 5).astype(int)  # 0=neg 1=pos
        elif rmap == "bin_all":
            m = df[(r <= 2) | (r >= 4)].copy()
            m["y"] = (m["rating"].astype(int) >= 4).astype(int)
        else:  # tri: 0=pos(4-5) 1=neu(3) 2=neg(1-2)
            m = df.copy()
            m["y"] = np.select([r >= 4, r == 3, r <= 2], [0, 1, 2])
        print(f"rmap={rmap} pool={len(m)} dist={m['y'].value_counts().sort_index().to_dict()}", flush=True)

    if args.per_class:
        parts = [g.sample(min(len(g), args.per_class), random_state=args.seed)
                 for _, g in m.groupby("y")]
        m = pd.concat(parts)
    if args.n and len(m) > args.n:
        m = m.sample(args.n, random_state=args.seed)
    m = m.sample(frac=1.0, random_state=args.seed)
    texts = m["content"].tolist()
    y = m["y"].to_numpy()
    print(f"train n={len(y)} dist={np.bincount(y).tolist()}", flush=True)

    n_val = min(5000, len(y) // 20)
    val_texts, val_y = texts[:n_val], y[:n_val]
    tr_texts, tr_y = texts[n_val:], y[n_val:]

    tok = AutoTokenizer.from_pretrained(args.model)
    n_classes = int(y.max() + 1)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_classes, ignore_mismatched_sizes=True).to(device)
    model.train()

    ds = TokenizedDS(tr_texts, tr_y, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dl) * args.epochs
    warmup = max(1, int(total_steps * 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device, args.amp):
                logits = model(**batch).logits.float()
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 500 == 0:
                print(f"  step {step}/{total_steps} loss={loss.item():.4f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        vp = predict_probs(model, tok, val_texts[:2000], args.max_len, device, 64, args.amp)
        vacc = accuracy_score(val_y[:2000], vp.argmax(1))
        print(f"epoch {ep+1}/{args.epochs} corpus-val acc={vacc:.4f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    gold = gold_eval(model, tok, args.task, device, args.max_len, args.amp)
    print(f"GOLD600 {args.task}: acc={gold['acc']:.4f} macro_f1={gold['macro_f1']:.4f}", flush=True)
    print(json.dumps(gold["report"], ensure_ascii=False, indent=1), flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    ckpt = os.path.join(OUTDIR, args.tag)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    res = {"tag": args.tag, "args": vars(args), "rmap": rmap, "n_train": len(tr_y),
           "train_secs": time.time() - t0, "gold": gold,
           "corpus_label_dist": np.bincount(y).tolist()}
    json.dump(res, open(os.path.join(OUTDIR, args.tag + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved", ckpt, flush=True)


if __name__ == "__main__":
    main()

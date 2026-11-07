#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ONE final distilled student on all labeled rows and save a production
checkpoint under models/. Same recipe as distill.py (hard CE + soft KL from the
deployed ensemble's OOF probs), but a single fit on 100% of the data instead of
5-fold CV — the OOF probs are already honest out-of-fold teacher targets.
"""
import os, sys, json, argparse, time
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging
from sklearn.model_selection import train_test_split

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import distill as ds  # noqa: E402


def train_full(model_name, tok, texts, y, soft, args, n_classes, seed, out_dir):
    ds.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn = None if args.attn_implementation == "none" else args.attn_implementation
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes, ignore_mismatched_sizes=True,
        attn_implementation=attn).to(device)
    model.train()

    idx = list(range(len(texts)))
    alpha_vec = np.full(len(texts), args.alpha, dtype=np.float32)
    data = ds.TokenizedDS(texts, y, soft, alpha_vec, idx, tok, args.max_len)
    dl = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=0,
                    collate_fn=ds.collate, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(dl) * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    cw = ds.class_weights(y, n_classes, device) if args.class_weight else None

    t0 = time.time()
    for ep in range(args.epochs):
        tot = 0.0
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            soft_t = batch.pop("soft")
            batch.pop("alpha", None)
            opt.zero_grad(set_to_none=True)
            with ds.autocast_ctx(device, args.amp):
                logits = model(**batch).logits.float()
            ce = F.cross_entropy(logits, labels, weight=cw)
            logp = F.log_softmax(logits / args.tau, dim=-1)
            target = F.softmax(soft_t / args.tau, dim=-1)
            kl = F.kl_div(logp, target, reduction="batchmean") * args.tau * args.tau
            loss = args.alpha * ce + (1.0 - args.alpha) * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            tot += float(loss)
        print(f"epoch {ep+1}/{args.epochs} loss={tot/len(dl):.4f} elapsed={time.time()-t0:.1f}s", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    meta = {"model": model_name, "task": args.task, "n": len(y), "n_classes": n_classes,
            "config": vars(args), "train_secs": time.time() - t0}
    with open(os.path.join(out_dir, "distill_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"saved {out_dir} in {meta['train_secs']:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["3class", "binary", "neu"], required=True)
    ap.add_argument("--out", required=True, help="output dir under models/")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight", type=int, default=1)
    ap.add_argument("--attn_implementation", choices=["sdpa", "eager", "none"], default="sdpa")
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    ap.add_argument("--data", default="reviews_600_base.jsonl")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")

    rows = ds.load_rows(args.data)
    texts, y = ds.make_labels(rows, args.task)
    n_classes = int(y.max() + 1)
    if args.teacher:
        tr = json.load(open(os.path.join(ds.OUT, args.teacher), encoding="utf-8"))
        teacher_probs = np.array(tr["probs"], dtype=np.float32)
        teacher_tag = args.teacher
    else:
        teacher_probs, teacher_tag = ds.load_teacher(args.task)
    assert len(teacher_probs) == len(y), f"teacher n={len(teacher_probs)} != y n={len(y)}"

    print(f"# final-distill model={args.model} task={args.task} n={len(y)} "
          f"classes={n_classes} teacher={teacher_tag} -> {args.out}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, "models", args.out)
    train_full(args.model, tok, texts, y, teacher_probs, args, n_classes, args.seed, out_dir)


if __name__ == "__main__":
    main()

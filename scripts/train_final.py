#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one final model on all labeled rows and save it under models/."""
import os, sys, json, time, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_finetune as gf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["3class", "binary"], default="3class")
    ap.add_argument("--label_key", default="label_semantic")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight", type=int, default=1)
    ap.add_argument("--attn_implementation", choices=["sdpa", "eager", "none"], default="eager")
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    ap.add_argument("--out", required=True, help="directory under models/")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    gf.set_seed(args.seed)

    rows = gf.load_rows(args.label_key)
    texts, y = gf.make_labels(rows, args.task)
    n_classes = int(y.max() + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    attn = None if args.attn_implementation == "none" else args.attn_implementation
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_classes, ignore_mismatched_sizes=True,
        attn_implementation=attn).to(device)

    indices = np.arange(len(y))
    ds = gf.TokenizedDS(texts, y, indices, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    collate_fn=gf.collate, generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(dl) * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    cw = gf.class_weights(y, n_classes, device) if args.class_weight else None

    model.train()
    t0 = time.time()
    for ep in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            opt.zero_grad(set_to_none=True)
            with gf.autocast_ctx(device, args.amp):
                logits = model(**batch).logits.float()
            loss = F.cross_entropy(logits, labels, weight=cw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
        print(f"epoch {ep+1}/{args.epochs} loss={float(loss):.4f} elapsed={time.time()-t0:.1f}s")

    out_dir = os.path.join(MODELS, args.out)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    meta = {"model": args.model, "task": args.task, "label_key": args.label_key,
            "n": len(y), "n_classes": n_classes, "config": vars(args),
            "id2label": {str(i): (gf.N3[i] if args.task == "3class" else ("负面" if i == 0 else "正面")) for i in range(n_classes)},
            "train_secs": time.time() - t0}
    with open(os.path.join(out_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"saved {out_dir} in {meta['train_secs']:.1f}s")


if __name__ == "__main__":
    main()

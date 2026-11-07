#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoRA fine-tune of Erlangshen-MegatronBert-1.3B for sentiment (5-fold OOF).

The 1.3B backbone does not fit full-parameter AdamW in 12 GB VRAM, so we
attach LoRA adapters (attention + MLP projections) and train only those.
Writes OOF probability json compatible with ensemble_eval.py, and can dump
the merged model for production inference.
"""
import os, sys, json, time, argparse, gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, HERE)
import gpu_finetune as gf  # noqa: E402

N3 = gf.N3
MODEL = "IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment"


class ClsDS(Dataset):
    def __init__(self, texts, labels, indices, tok, max_len):
        enc = tok(texts, truncation=True, max_length=max_len, padding=False)
        self.enc = {k: [torch.tensor(x, dtype=torch.long) for x in v] for k, v in enc.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        out = {k: v[idx] for k, v in self.enc.items()}
        out["labels"] = self.labels[idx]
        return out


def collate(batch):
    keys = [k for k in batch[0] if k != "labels"]
    out = {k: torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True, padding_value=0)
           for k in keys}
    out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


def predict_probs(model, tok, texts, max_len, device, bs=8, amp="bf16"):
    model.eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(**enc).logits.float()
            probs.append(F.softmax(logits, dim=-1).cpu())
    return torch.cat(probs).numpy()


def train_fold(texts, y, tr, va, args, n_classes, seed, tok):
    gf.set_seed(seed)
    device = torch.device("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_classes, ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16, attn_implementation=args.attn).to(device)
    lconf = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="SEQ_CLS",
        target_modules=args.target_modules.split(","))
    model = get_peft_model(model, lconf)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    ds = ClsDS(texts, y, tr, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(dl) * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    t0 = time.time()
    for _ in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(**batch).logits.float()
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.clip)
            opt.step()
            sched.step()
    train_secs = time.time() - t0

    va_texts = [texts[i] for i in va]
    probs = predict_probs(model, tok, va_texts, args.max_len, device, args.eval_bs)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return probs, train_secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--task", choices=["3class", "binary"], default="3class")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--target_modules",
                    default="query,key,value,dense,intermediate.dense,output.dense")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")

    rows = gf.load_rows("label_semantic")
    texts, y = gf.make_labels(rows, args.task)
    n_classes = int(y.max() + 1)
    print(f"# lora model={args.model} task={args.task} n={len(y)} classes={n_classes} "
          f"r={args.lora_r} a={args.lora_alpha} bs={args.batch} m={args.max_len}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    fold_metrics, train_secs = [], []
    t0 = time.time()

    for k, (tr, va) in enumerate(skf.split(texts, y), 1):
        probs, secs = train_fold(texts, y, tr, va, args, n_classes, args.seed + k, tok)
        oof[va] = probs
        m = gf.metrics_for(args.task, y[va], probs)
        fold_metrics.append({"fold": k, "acc": m["acc"], "macro_f1": m["macro_f1"],
                             "train_secs": secs})
        train_secs.append(secs)
        print(f"fold{k}: acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"train={secs:.1f}s total={time.time()-t0:.1f}s", flush=True)

    final = gf.metrics_for(args.task, y, oof)
    res = {
        "model": args.model + " [LoRA]",
        "task": args.task,
        "label_key": "label_semantic",
        "n": len(y),
        "n_classes": n_classes,
        "config": vars(args),
        "fold_metrics": fold_metrics,
        "oof": {"acc": final["acc"], "macro_f1": final["macro_f1"],
                "confusion": final["confusion"], "report": final["report"],
                "pred": final["pred"]},
        "y": y.tolist(),
        "probs": oof.tolist(),
        "train_secs": train_secs,
        "total_secs": time.time() - t0,
    }
    path = os.path.join(OUT, args.tag + ".json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nOOF acc={final['acc']:.4f} macro_f1={final['macro_f1']:.4f} "
          f"total={res['total_secs']:.1f}s saved {path}", flush=True)


if __name__ == "__main__":
    main()

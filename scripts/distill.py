#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Knowledge-distillation fine-tune: student imitates the deployed ensemble's
OOF probability distribution (soft targets) mixed with hard labels.

Outputs OOF probabilities in the same schema as gpu_finetune.py so results can
be fed to ensemble_eval.py directly.
"""
import os, json, time, random, argparse, gc, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
LAB3 = {"正面": 0, "中性": 1, "负面": 2}
N3 = ["正面", "中性", "负面"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(name="reviews_600_base.jsonl"):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", name), encoding="utf-8")]
    return [r for r in rows if r.get("label_semantic")]


def make_labels(rows, task):
    texts, ys = [], []
    for r in rows:
        lab = r["label_semantic"]
        if task == "3class":
            y = LAB3[lab]
        elif task == "binary":
            if lab == "中性":
                continue
            y = 1 if lab == "正面" else 0
        elif task == "neu":
            y = 1 if lab == "中性" else 0
        else:
            raise ValueError(task)
        texts.append(r["text"])
        ys.append(y)
    return texts, np.array(ys, dtype=np.int64)


def aug_flags(rows, task):
    flags = []
    for r in rows:
        lab = r["label_semantic"]
        if task == "binary" and lab == "中性":
            continue
        flags.append(r.get("src") == "aug")  # task=="neu" keeps all rows
    return np.array(flags, dtype=bool)


def load_teacher(task):
    """Return (probs ndarray aligned with make_labels order, teacher tag)."""
    name = {"binary": "ens_binary_allch2.json", "3class": "ens_3class_allch.json",
            "neu": "ens_neu_teacher.json"}[task]
    path = os.path.join(OUT, name)
    r = json.load(open(path, encoding="utf-8"))
    return np.array(r["probs"], dtype=np.float32), name


class TokenizedDS(Dataset):
    def __init__(self, texts, labels, soft, alpha, indices, tok, max_len):
        enc = tok(texts, truncation=True, max_length=max_len, padding=False)
        self.enc = {k: [torch.tensor(x, dtype=torch.long) for x in v] for k, v in enc.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.soft = torch.tensor(soft, dtype=torch.float32)
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        out = {k: v[idx] for k, v in self.enc.items()}
        out["labels"] = self.labels[idx]
        out["soft"] = self.soft[idx]
        out["alpha"] = self.alpha[idx]
        return out


def collate(batch):
    keys = [k for k in batch[0] if k not in ("labels", "soft", "alpha")]
    out = {k: torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True, padding_value=0) for k in keys}
    out["labels"] = torch.stack([b["labels"] for b in batch])
    out["soft"] = torch.stack([b["soft"] for b in batch])
    out["alpha"] = torch.stack([b["alpha"] for b in batch])
    return out


def class_weights(y, n_classes, device):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = len(y) / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32, device=device)


def autocast_ctx(device, amp):
    enabled = device.type == "cuda" and amp in ("bf16", "fp16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def predict_probs(model, tok, texts, max_len, device, bs=64, amp="bf16"):
    model.eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with autocast_ctx(device, amp):
                logits = model(**enc).logits.float()
            probs.append(F.softmax(logits, dim=-1).cpu())
    return torch.cat(probs).numpy()


def train_fold(model_name, tok, texts, y, soft, alpha_vec, tr, va, args, n_classes, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn = None if args.attn_implementation == "none" else args.attn_implementation
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes, ignore_mismatched_sizes=True,
        attn_implementation=attn).to(device)
    model.train()

    ds = TokenizedDS(texts, y, soft, alpha_vec, tr, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(dl) * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    cw = class_weights(y[tr], n_classes, device) if args.class_weight else None

    t0 = time.time()
    for _ in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            soft_t = batch.pop("soft")
            a = batch.pop("alpha")
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device, args.amp):
                logits = model(**batch).logits.float()
            ce = F.cross_entropy(logits, labels, weight=cw, reduction="none")
            logp = F.log_softmax(logits / args.tau, dim=-1)
            target = F.softmax(soft_t / args.tau, dim=-1)
            kl = F.kl_div(logp, target, reduction="none").sum(-1) * args.tau * args.tau
            loss = (a * ce + (1.0 - a) * kl).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
    train_secs = time.time() - t0

    va_texts = [texts[i] for i in va]
    probs = predict_probs(model, tok, va_texts, args.max_len, device, args.eval_bs, args.amp)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs, train_secs


def metrics_for(task, y_true, probs):
    if task == "3class":
        pred = probs.argmax(axis=1)
        names = N3
    else:
        pred = (probs[:, 1] >= 0.5).astype(int)
        names = ["负面", "正面"]
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "confusion": confusion_matrix(y_true, pred).tolist(),
        "pred": pred.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["3class", "binary", "neu"], default="3class")
    ap.add_argument("--teacher", default=None, help="ens_*.json in out/ (default: deployed ensemble)")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of hard-label CE")
    ap.add_argument("--aug_alpha", type=float, default=1.0, help="alpha override for src=aug rows")
    ap.add_argument("--tau", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight", type=int, default=1)
    ap.add_argument("--attn_implementation", choices=["sdpa", "eager", "none"], default="sdpa")
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    ap.add_argument("--tag", default="")
    ap.add_argument("--data", default="reviews_600_base.jsonl")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")

    rows = load_rows(args.data)
    texts, y = make_labels(rows, args.task)
    n_classes = int(y.max() + 1)

    if args.teacher:
        tr = json.load(open(os.path.join(OUT, args.teacher), encoding="utf-8"))
        teacher_probs = np.array(tr["probs"], dtype=np.float32)
        teacher_tag = args.teacher
    else:
        teacher_probs, teacher_tag = load_teacher(args.task)
    assert len(teacher_probs) == len(y), f"teacher n={len(teacher_probs)} != y n={len(y)}"
    aug = aug_flags(rows, args.task)
    alpha_vec = np.where(aug, args.aug_alpha, args.alpha).astype(np.float32)

    print(f"# distill model={args.model} task={args.task} n={len(y)} classes={n_classes} "
          f"teacher={teacher_tag} alpha={args.alpha} tau={args.tau} "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    tok = AutoTokenizer.from_pretrained(args.model)
    os.makedirs(OUT, exist_ok=True)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    fold_metrics = []
    train_secs = []
    t0 = time.time()

    for k, (tr_idx, va) in enumerate(skf.split(texts, y), 1):
        probs, secs = train_fold(args.model, tok, texts, y, teacher_probs, alpha_vec, tr_idx, va,
                                 args, n_classes, args.seed + k)
        oof[va] = probs
        m = metrics_for(args.task, y[va], probs)
        fold_metrics.append({"fold": k, "acc": m["acc"], "macro_f1": m["macro_f1"], "train_secs": secs})
        train_secs.append(secs)
        print(f"fold{k}: acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f} train={secs:.1f}s total={time.time()-t0:.1f}s")

    final = metrics_for(args.task, y, oof)
    safe_model = os.path.basename(args.model.rstrip("/"))
    tag = args.tag or f"distill_{safe_model}_{args.task}"
    res = {
        "model": args.model,
        "task": args.task,
        "n": len(y),
        "n_classes": n_classes,
        "config": vars(args),
        "teacher": teacher_tag,
        "fold_metrics": fold_metrics,
        "oof": {"acc": final["acc"], "macro_f1": final["macro_f1"],
                "confusion": final["confusion"], "pred": final["pred"]},
        "y": y.tolist(),
        "probs": oof.tolist(),
        "train_secs": train_secs,
        "total_secs": time.time() - t0,
    }
    path = os.path.join(OUT, tag + ".json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nOOF acc={final['acc']:.4f} macro_f1={final['macro_f1']:.4f} total={res['total_secs']:.1f}s")
    print(f"saved {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 SFT experiment: does adding the 1000 released-label AL rows
improve out-of-domain performance on the 1000-row random OOD set?

Conditions (train sets):
  gold      - gold-600 only (label_semantic)
  gold_al   - gold-600 + al1000 (authoritative `final` labels)
  al_only   - al1000 only

Tasks: 3class (all rows) and binary (neutral dropped).
Eval  : ood1000 `final` labels (clean held-out); al1000 reported for
        reference only (in-train for gold_al/al_only -> inflated).

Outputs: E:/GB77_data/lab2000/sft_results/{cond}_{task}_e{ep}.json
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

LAB = Path(r"E:\GB77_data\lab2000\lab2000_final.jsonl")
LAB_EVAL = LAB
OUTD = Path(r"E:\GB77_data\lab2000\sft_results")
GOLD = HERE.parent / "data" / "reviews_600_base.jsonl"
MODEL = "hfl/chinese-macbert-base"


def load_gold():
    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    return [{"text": r["text"], "label": r["label_semantic"]} for r in rows]


def load_lab(tag, path=None):
    rows = [json.loads(l) for l in open(path or LAB, encoding="utf-8")]
    return [{"text": r["text"], "label": r["final"], "set": tag, "i": r["i"]}
            for r in rows if r["set"] == tag and r["final"]]


def load_all(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [{"text": r["text"], "label": r["final"], "set": "x", "i": i}
            for i, r in enumerate(rows) if r.get("final")]


def filter_items(items, task):
    if task == "3class":
        return items
    return [it for it in items if it["label"] != "中性"]


def to_xy(items, task):
    texts, ys = [], []
    for it in filter_items(items, task):
        lab = it["label"]
        y = gf.LAB3[lab] if task == "3class" else (1 if lab == "正面" else 0)
        texts.append(it["text"])
        ys.append(y)
    return texts, np.array(ys, dtype=np.int64)


def y_true(items, task):
    return np.array([gf.LAB3[it["label"]] if task == "3class"
                     else (1 if it["label"] == "正面" else 0)
                     for it in filter_items(items, task)], dtype=np.int64)


def train_all(model_name, tok, texts, y, n_classes, args):
    gf.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes, ignore_mismatched_sizes=True,
        attn_implementation="sdpa").to(device)
    model.train()
    ds = gf.TokenizedDS(texts, y, list(range(len(y))), tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    collate_fn=gf.collate,
                    generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = max(1, len(dl) * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    cw = gf.class_weights(y, n_classes, device) if args.class_weight else None
    t0 = time.time()
    for _ in range(args.epochs):
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
    return model, device, time.time() - t0


def predict(texts, model, tok, device, max_len, bs, amp):
    model.eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with gf.autocast_ctx(device, amp):
                logits = model(**enc).logits.float()
            probs.append(F.softmax(logits, dim=-1).cpu())
    return torch.cat(probs).numpy()


def metrics_str(yt, yp):
    from sklearn.metrics import accuracy_score, f1_score
    return {"acc": round(float(accuracy_score(yt, yp)), 4),
            "macro_f1": round(float(f1_score(yt, yp, average="macro",
                                             zero_division=0)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--conditions", nargs="+",
                    default=["gold", "gold_al", "al_only"])
    ap.add_argument("--tasks", nargs="+", default=["3class", "binary"])
    ap.add_argument("--lab", default=str(LAB))
    ap.add_argument("--lab_eval", default=str(LAB_EVAL))
    ap.add_argument("--extra_lab", default=None)
    ap.add_argument("--tag_suffix", default="")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight", type=int, default=0)
    ap.add_argument("--amp", default="bf16")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    OUTD.mkdir(parents=True, exist_ok=True)

    lab_p = Path(args.lab); lab_ep = Path(args.lab_eval)
    gold = load_gold()
    al = load_lab("al1000", lab_p)
    ood = load_lab("ood1000", lab_ep)
    al_ev = load_lab("al1000", lab_ep)
    extra = load_all(args.extra_lab) if args.extra_lab else []
    train_src = {"gold": gold, "gold_al": gold + al, "al_only": al,
                 "gold_ds": gold + al, "gold_extra": gold + extra}
    eval_sets_override = {"al1000": al_ev}
    print({k: len(v) for k, v in train_src.items()}, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    for task in args.tasks:
        eval_sets = {"ood1000": ood, "al1000": eval_sets_override["al1000"], "gold600": gold}
        eval_xy = {k: to_xy(v, task) for k, v in eval_sets.items()}
        eval_yt = {k: y_true(v, task) for k, v in eval_sets.items()}
        for cond in args.conditions:
            texts, y = to_xy(train_src[cond], task)
            n_classes = int(y.max() + 1)
            for ep in args.epochs:
                ns = argparse.Namespace(**{**vars(args), "epochs": ep})
                model, device, secs = train_all(
                    MODEL, tok, texts, y, n_classes, ns)
                res = {"cond": cond, "task": task, "epochs": ep,
                       "n_train": len(y), "train_secs": round(secs, 1),
                       "seed": args.seed, "class_weight": args.class_weight}
                for ev in eval_sets:
                    etexts, _ = eval_xy[ev]
                    p = predict(etexts, model, tok, device,
                                args.max_len, 64, args.amp)
                    if task == "3class":
                        yp = p.argmax(1)
                    else:
                        yp = (p[:, 1] >= 0.5).astype(int)
                    res[ev] = {"metrics": metrics_str(eval_yt[ev], yp),
                               "probs": p.tolist()}
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                tag = f"{cond}{args.tag_suffix}_{task}_e{ep}"
                if args.class_weight:
                    tag += "_cw"
                if args.seed != 42:
                    tag += f"_s{args.seed}"
                fn = OUTD / f"{tag}.json"
                fn.write_text(json.dumps(res, ensure_ascii=False),
                              encoding="utf-8")
                print(f"{cond:8s} {task:7s} e{ep}: "
                      f"ood={res['ood1000']['metrics']} "
                      f"al={res['al1000']['metrics']} ({secs:.0f}s)",
                      flush=True)


if __name__ == "__main__":
    main()

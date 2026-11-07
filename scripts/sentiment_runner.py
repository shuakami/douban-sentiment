#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情感模型评测/微调统一入口。

两种模式：
  zero-shot  直接拿预训练情感模型推理，不训练（验证「不微调能到多少」）
  finetune   在 600 条上微调（验证「微调还能涨多少」）

三分类策略（关键）：情感预训练模型多是二分类（正/负）。
这里用「正面概率 P(pos) + 双阈值」映射到三分类：
    P >= hi        -> 正面
    P <= lo        -> 负面
    否则            -> 中性
阈值在训练集/OOF 上搜索，避免在测试集上调参造成泄漏。

用法：
  python3.11 sentiment_runner.py --model models/erlangshen-110m-sentiment --mode zero_shot
  python3.11 sentiment_runner.py --model models/erlangshen-110m-sentiment --mode finetune --folds 5
"""
import os, sys, json, gc, time, argparse, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "reviews_600_base.jsonl")
OUT = os.path.join(ROOT, "out")
LAB3 = {"正面": 0, "中性": 1, "负面": 2}
N3 = ["正面", "中性", "负面"]


# ---------------- 数据 ----------------
def load_rows(label_key="label_semantic"):
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    return [r for r in rows if r[label_key]]


def to_binary(lab):
    """三分类标签 -> 二分类正/负（中性丢弃）"""
    return None if lab == "中性" else (1 if lab == "正面" else 0)


# ---------------- 推理 ----------------
@torch.no_grad()
def predict_probs(model, tok, texts, max_len=256, bs=16):
    """返回 P(positive)，对二分类模型取 softmax 后 label=1 的概率。"""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(dev)
        logits = model(**enc).logits
        if logits.shape[-1] == 1:
            p = torch.sigmoid(logits).squeeze(-1)
        else:
            # 兼容 id2label 顺序不同的情况：按 config 找 positive 的下标
            p = torch.softmax(logits, -1)[:, pos_index(model)]
        out.extend(p.cpu().tolist())
    return np.array(out)


def pos_index(model):
    id2 = getattr(model.config, "id2label", {}) or {}
    for k, v in id2.items():
        if str(v).lower() in ("positive", "pos", "label_1", "1", "好评"):
            return int(k)
    return 1  # 默认 0=neg, 1=pos


# ---------------- 三分类阈值 ----------------
def search_thresholds(p, y3, verbose=True):
    """在给定数据上网格搜索 (lo, hi)，最大化 Macro-F1。"""
    best = (None, None, -1)
    grid = np.arange(0.05, 0.96, 0.02)
    for lo in grid:
        for hi in grid:
            if hi <= lo:
                continue
            pred = np.where(p >= hi, 0, np.where(p <= lo, 2, 1))
            f = f1_score(y3, pred, average="macro")
            if f > best[2]:
                best = (lo, hi, f)
    if verbose:
        print(f"    最优阈值 lo={best[0]:.2f} hi={best[1]:.2f} -> 该集 Macro-F1={best[2]:.4f}")
    return best[0], best[1]


def report(y3, pred, title):
    acc = accuracy_score(y3, pred); f1m = f1_score(y3, pred, average="macro")
    print(f"    {title}: Acc={acc:.4f}  Macro-F1={f1m:.4f}")
    return acc, f1m


# ---------------- 微调 ----------------
class DS(Dataset):
    def __init__(self, texts, labels, tok, ml):
        self.enc = tok(texts, truncation=True, max_length=ml, padding=False)
        self.y = labels
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        d["labels"] = torch.tensor(self.y[i]); return d


def finetune_fold(tr_texts, tr_y, va_texts, tok, args, n_cls, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_cls, ignore_mismatched_sizes=True).to(dev)
    pad = tok.pad_token_id or 0
    dl = DataLoader(DS(tr_texts, tr_y, tok, args.max_len), batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: {
                        **{k: nn.utils.rnn.pad_sequence([x[k] for x in b], batch_first=True, padding_value=pad)
                           for k in b[0] if k != "labels"},
                        "labels": torch.stack([x["labels"] for x in b])})
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = max(1, len(dl) * args.epochs)
    sch = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(args.epochs):
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            opt.zero_grad(); o = model(**b); o.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["zero_shot", "finetune"], default="zero_shot")
    ap.add_argument("--label_key", default="label_semantic")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = load_rows(args.label_key)
    texts = [r["text"] for r in rows]
    y3 = np.array([LAB3[r[args.label_key]] for r in rows])
    print(f"# 模型={os.path.basename(args.model)} | 模式={args.mode} | 样本={len(texts)} | "
          f"标签={args.label_key}")

    if args.mode == "zero_shot":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, ignore_mismatched_sizes=True).to(dev)
        print(f"  模型标签映射: {getattr(model.config,'id2label',None)}")
        p = predict_probs(model, tok, texts, args.max_len)
        del model; gc.collect()

        # 二分类（丢弃中性）
        mask = y3 != 1
        yb = (y3[mask] == 0).astype(int)
        pred_b = (p[mask] >= 0.5).astype(int)
        acc_b = accuracy_score(yb, pred_b); f1_b = f1_score(yb, pred_b, average="macro")
        print(f"\n  【二分类 正/负，丢弃中性】n={mask.sum()}")
        print(f"    Acc={acc_b:.4f}  Macro-F1={f1_b:.4f}")

        # 三分类（阈值法，全量调阈值）
        lo, hi = search_thresholds(p, y3)
        pred3 = np.where(p >= hi, 0, np.where(p <= lo, 2, 1))
        acc3, f3 = report(y3, pred3, "三分类(阈值)")
        print()
        print(classification_report(y3, pred3, target_names=N3, digits=4))
        cm = confusion_matrix(y3, pred3)
        print("  混淆矩阵(行=真实,列=预测)\n        " + "".join(f"{n:>8}" for n in N3))
        for i, n in enumerate(N3):
            print(f"  {n:>4}" + "".join(f"{v:>8}" for v in cm[i]))

        res = {"model": args.model, "mode": "zero_shot", "acc_bin": acc_b, "f1_bin": f1_b,
               "acc_3": acc3, "f1_3": f3, "lo": float(lo), "hi": float(hi),
               "secs": time.time() - t0, "p": p.tolist()}
        tag = "zs_" + os.path.basename(args.model.rstrip("/"))
        json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), ensure_ascii=False)
        print(f"\n  耗时 {time.time()-t0:.1f}s -> out/{tag}.json")
        return

    # ---------------- 微调：5 折 CV ----------------
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof_p = np.zeros(len(y3))
    for k, (tr, va) in enumerate(skf.split(texts, y3), 1):
        model = finetune_fold([texts[i] for i in tr], y3[tr], None, tok, args, 3, 42 + k)
        oof_p[va] = predict_probs(model, tok, [texts[i] for i in va], args.max_len)
        del model; gc.collect()
        print(f"  fold{k} 完成 ({time.time()-t0:.0f}s)")
    lo, hi, _ = search_thresholds(oof_p, y3)
    pred3 = np.where(oof_p >= hi, 0, np.where(oof_p <= lo, 2, 1))
    acc3, f3 = report(y3, pred3, f"\n  OOF 三分类")
    mask = y3 != 1
    acc_b = accuracy_score((y3[mask] == 0).astype(int), (oof_p[mask] >= 0.5).astype(int))
    f1_b = f1_score((y3[mask] == 0).astype(int), (oof_p[mask] >= 0.5).astype(int), average="macro")
    print(f"  OOF 二分类: Acc={acc_b:.4f} Macro-F1={f1_b:.4f}")
    res = {"model": args.model, "mode": "finetune", "acc_3": acc3, "f1_3": f3,
           "acc_bin": acc_b, "f1_bin": f1_b, "lo": float(lo), "hi": float(hi),
           "secs": time.time() - t0, "oof_p": oof_p.tolist(), "y": y3.tolist()}
    tag = "ft_" + os.path.basename(args.model.rstrip("/"))
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), ensure_ascii=False)
    print(f"\n  耗时 {time.time()-t0:.1f}s -> out/{tag}.json")


if __name__ == "__main__":
    main()

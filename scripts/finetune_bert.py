#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bert-base-chinese 微调（纯 PyTorch 训练循环，不依赖 Trainer，规避 transformers 5.x 版本差异）。

支持：
  - 分层 K 折交叉验证（600 条样本量小，单次划分方差极大，必须用 CV）
  - 两套标签视图：label_rule(星级弱监督) / label_semantic(语义标注)
  - 三分类 / 二分类（drop_neutral 或 neg_vs_rest）
  - 小样本设置 --n_per_class N：每类只取 N 条训练，模拟 few-shot 微调
  - CPU / GPU 自适应

用法示例：
  # 全量 5 折，语义标签
  python3.11 finetune_bert.py --label semantic --folds 5 --epochs 4
  # 小样本：每类 16 条，10 次重复采样
  python3.11 finetune_bert.py --label semantic --n_per_class 16 --repeats 10 --epochs 10
"""
import json, os, argparse, time, random, math, gc, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "reviews_600_base.jsonl")
MODEL_DIR = os.path.join(ROOT, "models", "bert-base-chinese")
OUT = os.path.join(ROOT, "out")
LAB2ID = {"正面": 0, "中性": 1, "负面": 2}
ID2LAB = {v: k for k, v in LAB2ID.items()}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class ReviewDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(texts, truncation=True, max_length=max_len, padding=False)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: torch.tensor(v[i]) for k, v in self.enc.items()} | {"labels": torch.tensor(self.labels[i])}


def collate(batch, pad_id):
    out = {}
    for k in batch[0]:
        if k == "labels":
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True, padding_value=pad_id)
    return out


def load_data(label_key, binary):
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    texts, ys = [], []
    for r in rows:
        lab = r[label_key]
        if lab is None:
            continue
        if binary == "drop_neutral":
            if lab == "中性":
                continue
            y = 0 if lab == "正面" else 1
        elif binary == "neg_vs_rest":
            y = 1 if lab == "负面" else 0
        else:
            y = LAB2ID[lab]
        texts.append(r["text"]); ys.append(y)
    return texts, np.array(ys)


def train_one(train_texts, train_y, val_texts, val_y, tok, args, n_classes, seed):
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=n_classes).to(dev)
    ds = ReviewDS(train_texts, train_y, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id or 0), num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = max(1, len(dl) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)
    lossf = nn.CrossEntropyLoss()

    model.train()
    for ep in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch)
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(val_texts), 64):
            enc = tok(val_texts[i:i + 64], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(dev)
            preds.extend(model(**enc).logits.argmax(-1).cpu().tolist())
    del model
    gc.collect()
    return np.array(preds), None


def summarize(yt, yp, n_classes, title, binary):
    names = (["正面", "中性", "负面"] if binary == "none"
             else ["负面", "正面"] if binary == "drop_neutral" else ["非负面", "负面"])
    acc = accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average="macro")
    print(f"  Acc={acc:.4f}  Macro-F1={f1m:.4f}")
    return acc, f1m, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["rule", "semantic"], default="semantic")
    ap.add_argument("--binary", choices=["none", "drop_neutral", "neg_vs_rest"], default="none")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--threads", type=int, default=8, help="CPU 线程数，容器内存 8GB，过高会 OOM")
    ap.add_argument("--n_per_class", type=int, default=0, help=">0 则每类只采样N条训练（few-shot）")
    ap.add_argument("--repeats", type=int, default=1, help="few-shot 下重复采样次数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    texts, y = load_data(f"label_{args.label}", args.binary)
    n_classes = int(y.max() + 1)
    print(f"# BERT 微调 | 标签={args.label} | 模式={args.binary} | 样本={len(y)} | "
          f"类别={n_classes} | device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    os.makedirs(OUT, exist_ok=True)
    tag = f"bert_{args.label}_{args.binary}" + (f"_fs{args.n_per_class}" if args.n_per_class else "_full")
    t0 = time.time()

    if args.n_per_class > 0:
        # few-shot：每类随机抽 N 条训练，剩余作测试，重复 repeats 次
        accs, f1s = [], []
        for rep in range(args.repeats):
            rng = np.random.RandomState(args.seed + rep)
            tr_idx, te_idx = [], []
            for c in range(n_classes):
                idx = np.where(y == c)[0]
                rng.shuffle(idx)
                k = min(args.n_per_class, len(idx) // 2)
                tr_idx += idx[:k].tolist(); te_idx += idx[k:].tolist()
            yp, _ = train_one([texts[i] for i in tr_idx], y[tr_idx],
                              [texts[i] for i in te_idx], y[te_idx], tok, args, n_classes, args.seed + rep)
            a, f, names = summarize(y[te_idx], yp, n_classes, f"rep{rep}", args.binary)
            accs.append(a); f1s.append(f)
        print(f"\n>> {args.repeats} 次重复：Acc {np.mean(accs):.4f}±{np.std(accs):.3f} | "
              f"Macro-F1 {np.mean(f1s):.4f}±{np.std(f1s):.3f}")
        res = {"tag": tag, "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
               "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)), "repeats": args.repeats,
               "n_per_class": args.n_per_class, "secs": time.time() - t0}
    else:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        oof = np.zeros(len(y), dtype=int)
        accs, f1s = [], []
        for k, (tr, te) in enumerate(skf.split(texts, y), 1):
            yp, _ = train_one([texts[i] for i in tr], y[tr], [texts[i] for i in te], y[te],
                              tok, args, n_classes, args.seed + k)
            oof[te] = yp
            a, f, names = summarize(y[te], yp, n_classes, f"fold{k}", args.binary)
            print(f"  fold{k}: Acc={a:.4f} Macro-F1={f:.4f}")
            accs.append(a); f1s.append(f)
        print(f"\n>> {args.folds} 折：Acc {np.mean(accs):.4f}±{np.std(accs):.3f} | "
              f"Macro-F1 {np.mean(f1s):.4f}±{np.std(f1s):.3f}")
        print("\nOOF 混淆矩阵（行=真实，列=预测）:")
        print("      " + "".join(f"{n:>8}" for n in names))
        cm = confusion_matrix(y, oof)
        for i, n in enumerate(names):
            print(f"{n:>6}" + "".join(f"{v:>8}" for v in cm[i]))
        print()
        print(classification_report(y, oof, target_names=names, digits=4))
        res = {"tag": tag, "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
               "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
               "folds": args.folds, "secs": time.time() - t0, "oof": oof.tolist(), "y": y.tolist()}

    path = os.path.join(OUT, f"{tag}.json")
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"\n耗时 {res['secs']:.1f}s | 结果已写入 {path}")


if __name__ == "__main__":
    main()

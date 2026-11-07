#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distill the deployed ensemble into a small MLP head on frozen backbone
features. Features are label-independent (pretrained backbones), so plain
5-fold CV on the head gives honest OOF. Runs on CPU — tiny data.

Usage: distill_head.py <task> <feat.npy> [feat2.npy ...] --tag <name>
                       [--hidden 512] [--epochs 300] [--alpha 0.5] [--tau 2]
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

LAB3 = {"正面": 0, "中性": 1, "负面": 2}
N3 = ["正面", "中性", "负面"]


def load_rows(name="reviews_600_base.jsonl"):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", name), encoding="utf-8")]
    return [r for r in rows if r.get("label_semantic")]


def make_labels(rows, task):
    ys = []
    for r in rows:
        lab = r["label_semantic"]
        if task == "3class":
            ys.append(LAB3[lab])
        elif task == "neu":
            ys.append(1 if lab == "中性" else 0)
        else:
            if lab == "中性":
                ys.append(None)
            else:
                ys.append(1 if lab == "正面" else 0)
    if task == "binary":
        keep = [i for i, v in enumerate(ys) if v is not None]
        return np.array([ys[i] for i in keep]), keep
    return np.array(ys, dtype=np.int64), list(range(len(ys)))


def load_teacher(task, name=None):
    if name is None:
        name = {"binary": "ens_binary_allch2.json", "3class": "ens_3class_allch.json",
                "neu": "ens_neu_teacher.json"}[task]
    r = json.load(open(os.path.join(OUT, name), encoding="utf-8"))
    return np.array(r["probs"], dtype=np.float32), name


class MLP(nn.Module):
    def __init__(self, d, n_classes, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_head(X, y, soft, alpha_vec, tr, va, n_classes, hidden, epochs, alpha, tau, seed=42, lr=1e-3):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = StandardScaler().fit(X[tr])
    Xtr = torch.tensor(sc.transform(X[tr]), dtype=torch.float32, device=device)
    Xva = torch.tensor(sc.transform(X[va]), dtype=torch.float32, device=device)
    ytr = torch.tensor(y[tr], dtype=torch.long, device=device)
    str_ = torch.tensor(soft[tr], dtype=torch.float32, device=device)
    atr = torch.tensor(alpha_vec[tr], dtype=torch.float32, device=device)
    model = MLP(X.shape[1], n_classes, hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(tr)
    bs = 64
    steps = max(1, (n + bs - 1) // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = model(Xtr[idx])
            ce = F.cross_entropy(logits, ytr[idx], reduction="none")
            logp = F.log_softmax(logits / tau, dim=-1)
            tgt = F.softmax(str_[idx] / tau, dim=-1)
            kl = F.kl_div(logp, tgt, reduction="none").sum(-1) * tau * tau
            a = atr[idx]
            loss = (a * ce + (1 - a) * kl).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        return F.softmax(model(Xva), dim=-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["binary", "3class", "neu"])
    ap.add_argument("feats", nargs="+", help=".npy feature files in out/")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--aug_alpha", type=float, default=1.0, help="alpha override for src=aug rows")
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teacher", default=None, help="ens_*.json in out/ (default: deployed ensemble)")
    ap.add_argument("--data", default="reviews_600_base.jsonl")
    args = ap.parse_args()

    rows = load_rows(args.data)
    y_full, keep = make_labels(rows, args.task)
    # features are stored for ALL rows; select the task's subset
    feats = [np.load(os.path.join(OUT, f)) for f in args.feats]
    n_rows = len(rows)
    X = np.concatenate([f[keep] if f.shape[0] == n_rows else f for f in feats], axis=1).astype(np.float32)
    teacher_probs, teacher_tag = load_teacher(args.task, args.teacher)
    y = y_full
    aug_mask = np.array([rows[i].get("src") == "aug" for i in keep])
    alpha_vec = np.where(aug_mask, args.aug_alpha, args.alpha).astype(np.float32)
    n_classes = int(y.max() + 1)
    print(f"# head-mlp task={args.task} n={len(y)} d={X.shape[1]} feats={args.feats} "
          f"teacher={teacher_tag} hidden={args.hidden} a={args.alpha} tau={args.tau}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        oof[va] = train_head(X, y, teacher_probs, alpha_vec, tr, va, n_classes,
                             args.hidden, args.epochs, args.alpha, args.tau, seed=args.seed + k)
        pred = oof[va].argmax(1) if args.task == "3class" else (oof[va][:, 1] >= 0.5).astype(int)
        print(f"fold{k}: acc={accuracy_score(y[va], pred):.4f} "
              f"f1={f1_score(y[va], pred, average='macro'):.4f}")

    pred = oof.argmax(1) if args.task == "3class" else (oof[:, 1] >= 0.5).astype(int)
    res = {
        "model": f"mlp_head[{'+'.join(args.feats)}]",
        "task": args.task,
        "n": len(y), "n_classes": n_classes,
        "config": vars(args), "teacher": teacher_tag,
        "oof": {"acc": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro")),
                "confusion": confusion_matrix(y, pred).tolist(),
                "pred": pred.tolist()},
        "y": y.tolist(), "probs": oof.tolist(),
    }
    path = os.path.join(OUT, args.tag + ".json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"OOF acc={res['oof']['acc']:.4f} macro_f1={res['oof']['macro_f1']:.4f} saved {path}")


if __name__ == "__main__":
    main()

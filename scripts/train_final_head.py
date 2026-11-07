#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ONE final feature-space MLP head on all labeled rows and save a
production checkpoint. Same recipe as distill_head.py (hard CE + soft KL from
the deployed ensemble's OOF probs on frozen backbone features), single fit on
100% of the data. Saves state_dict + StandardScaler params to models/*.pt.
"""
import os, sys, json, argparse, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import distill_head as dh  # noqa: E402


def train_full_head(X, y, soft, n_classes, hidden, epochs, alpha, tau, seed=42, lr=1e-3):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = StandardScaler().fit(X)
    Xall = torch.tensor(sc.transform(X), dtype=torch.float32, device=device)
    yall = torch.tensor(y, dtype=torch.long, device=device)
    sall = torch.tensor(soft, dtype=torch.float32, device=device)
    model = dh.MLP(X.shape[1], n_classes, hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(y)
    bs = 64
    steps = max(1, (n + bs - 1) // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = model(Xall[idx])
            ce = F.cross_entropy(logits, yall[idx])
            logp = F.log_softmax(logits / tau, dim=-1)
            tgt = F.softmax(sall[idx] / tau, dim=-1)
            kl = F.kl_div(logp, tgt, reduction="batchmean") * tau * tau
            loss = alpha * ce + (1 - alpha) * kl
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        if (ep + 1) % 50 == 0 or ep == epochs - 1:
            print(f"epoch {ep+1}/{epochs} loss={tot/max(1,(n+bs-1)//bs):.4f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(Xall), dim=-1).cpu().numpy()
    pred = probs.argmax(1) if n_classes == 3 else (probs[:, 1] >= 0.5).astype(int)
    print(f"train-fit acc={accuracy_score(y, pred):.4f} "
          f"macro_f1={f1_score(y, pred, average='macro'):.4f}", flush=True)
    return model, sc, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["binary", "3class", "neu"])
    ap.add_argument("feats", nargs="+", help=".npy feature files in out/")
    ap.add_argument("--out", required=True, help="output .pt path under models/")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teacher", default=None, help="ens_*.json in out/ (default: deployed ensemble)")
    ap.add_argument("--data", default="reviews_600_base.jsonl")
    args = ap.parse_args()

    rows = dh.load_rows(args.data)
    y, keep = dh.make_labels(rows, args.task)
    feats = [np.load(os.path.join(OUT, f)) for f in args.feats]
    n_rows = len(rows)
    X = np.concatenate([f[keep] if f.shape[0] == n_rows else f for f in feats], axis=1).astype(np.float32)
    teacher_probs, teacher_tag = dh.load_teacher(args.task, args.teacher)
    assert len(teacher_probs) == len(y), f"teacher n={len(teacher_probs)} != y n={len(y)}"
    n_classes = int(y.max() + 1)
    print(f"# final-head task={args.task} n={len(y)} d={X.shape[1]} feats={args.feats} "
          f"teacher={teacher_tag} hidden={args.hidden} a={args.alpha}", flush=True)

    model, sc, _ = train_full_head(X, y, teacher_probs, n_classes, args.hidden,
                                   args.epochs, args.alpha, args.tau, seed=args.seed)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, "models", args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "scaler_mean": sc.mean_.astype(np.float32),
        "scaler_scale": sc.scale_.astype(np.float32),
        "in_dim": int(X.shape[1]),
        "hidden": args.hidden,
        "n_classes": n_classes,
        "task": args.task,
        "feats": args.feats,
        "teacher": teacher_tag,
        "config": vars(args),
    }, out_path)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

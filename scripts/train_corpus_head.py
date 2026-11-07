#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train an MLP head on frozen MegatronBERT features extracted from the
pseudo-labeled corpus, then evaluate on the gold 600 (same backbone features).
Saves a deployable head .pt (same format as train_final_head.py)."""
import os, sys, json, argparse, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import distill_head as dh  # noqa: E402
from _paths import DATA_ROOT

LAB3 = {"正面": 0, "中性": 1, "负面": 2}
N3 = ["正面", "中性", "负面"]
FEATDIR = str(DATA_ROOT / "features")
OUTDIR = str(DATA_ROOT / "models" / "heads")


def gold_data(task):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "reviews_600_base.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r.get("label_semantic")]
    texts_y = []
    for r in rows:
        lab = r["label_semantic"]
        if task == "3class":
            texts_y.append(LAB3[lab])
        elif task == "neu":
            texts_y.append(1 if lab == "中性" else 0)
        else:
            texts_y.append(None if lab == "中性" else (1 if lab == "正面" else 0))
    return rows, texts_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="corpus feat tag under the data root (features/)")
    ap.add_argument("--task", choices=["3class", "binary", "neu"], default="3class")
    ap.add_argument("--gold-feats", default="feats_mb13b_m256.npy")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X = np.load(os.path.join(FEATDIR, args.tag + "_feats.npy"))
    meta = pd.read_pickle(os.path.join(FEATDIR, args.tag + "_meta.pkl"))
    y = meta["y"].to_numpy()
    if args.task == "binary":
        # binary = pos(0)/neg(2); remap pos->1, neg->0 (deploy convention)
        m = (y == 0) | (y == 2)
        yb = np.where(y[m] == 0, 1, 0)
        Xb, yb = X[m], yb
    elif args.task == "neu":
        yn = (y == 1).astype(int)
        Xb, yb = X, yn
    else:
        Xb, yb = X, y
    n_classes = int(yb.max() + 1)
    print(f"corpus n={len(yb)} d={Xb.shape[1]} classes={n_classes} task={args.task}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    sc = StandardScaler().fit(Xb)
    Xs = sc.transform(Xb).astype(np.float32)
    del Xb
    Xt = torch.tensor(Xs, device=device)
    yt = torch.tensor(yb, dtype=torch.long, device=device)
    del Xs
    model = dh.MLP(Xt.shape[1], n_classes, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(yt); steps = (n + args.bs - 1) // args.bs * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            logits = model(Xt[idx])
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"ep{ep+1}/{args.epochs} loss={tot/n:.4f} {time.time()-t0:.0f}s", flush=True)

    # ---- gold eval ----
    rows, yfull = gold_data(args.task)
    Xg_all = np.load(os.path.join(ROOT, "out", args.gold_feats))
    if args.task == "binary":
        keep = [i for i, v in enumerate(yfull) if v is not None]
        yg = np.array([yfull[i] for i in keep])
        Xg = Xg_all[keep]
    elif args.task == "neu":
        yg = np.array(yfull); Xg = Xg_all
    else:
        yg = np.array(yfull); Xg = Xg_all
    Xgs = torch.tensor(sc.transform(Xg).astype(np.float32), device=device)
    model.eval()
    with torch.no_grad():
        logits = model(Xgs).float()
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    if args.task == "3class":
        pred = probs.argmax(1); names = N3
    else:
        pred = (probs[:, 1] >= 0.5).astype(int); names = ["负面", "正面"] if args.task == "binary" else ["非中性", "中性"]
    acc = accuracy_score(yg, pred); f1 = f1_score(yg, pred, average="macro")
    print(f"GOLD {args.task}: acc={acc:.4f} macro_f1={f1:.4f}", flush=True)
    print(classification_report(yg, pred, target_names=names, digits=4), flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"head_{args.task}_{args.tag}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "scaler_mean": sc.mean_.astype(np.float32),
        "scaler_scale": sc.scale_.astype(np.float32),
        "in_dim": int(Xt.shape[1]), "n_classes": n_classes,
        "hidden": args.hidden,
    }, out)
    res = {"tag": args.tag, "task": args.task, "n_corpus": int(n),
           "gold": {"acc": float(acc), "macro_f1": float(f1)},
           "train_secs": time.time() - t0}
    json.dump(res, open(out + ".json", "w"), indent=1)
    np.save(out + "_goldprobs.npy", probs.astype(np.float32))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()

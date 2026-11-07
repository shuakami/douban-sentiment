#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the deployed v3 ensemble on the human-reviewed lab2000 set.

Replicates web/app.py predict() exactly (lexicon shortcut, channel ensemble,
neutral-expert blend, temperature scaling) but batched over all 2000 rows.

Input : E:/GB77_data/lab2000/lab2000_final.jsonl  (authoritative `final` labels)
Output: E:/GB77_data/lab2000/eval_v3_lab2000.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "scripts"))
import app as webapp  # noqa: E402
import short_lexicon as sl  # noqa: E402

LAB = Path(r"E:\GB77_data\lab2000\lab2000_final.jsonl")
OUT = Path(r"E:\GB77_data\lab2000\eval_v3_lab2000.json")
N3 = ["正面", "中性", "负面"]


def cls_metrics(y_true, y_pred, labels):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1c = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {"acc": round(float(acc), 4), "macro_f1": round(float(f1), 4),
            "per_class": {l: {"p": round(float(a), 4), "r": round(float(b), 4),
                              "f1": round(float(c), 4)}
                          for l, a, b, c in zip(labels, p, r, f1c)},
            "confusion": cm}


def predict_3class(texts):
    """Batched replica of app.predict() for task=3class. Returns (n,3) probs."""
    ch = webapp.channel_probs("3class", texts)
    probs = webapp.weighted_mean(ch)
    t = webapp.STATE["tasks"]["3class"]
    ne = t["neu_expert"]
    # batched neu_p_neu
    p_stu = webapp.predict_channel_probs(ne["student"], texts)[:, 1]
    feats = webapp.backbone_feats(ne["head"], texts)
    device = next(ne["head"]["bb"]["model"].parameters()).device
    x = torch.tensor((feats - ne["head"]["mean"]) / ne["head"]["scale"],
                     dtype=torch.float32, device=device)
    with torch.inference_mode():
        logits = ne["head"]["net"](x).float()
    p_head = torch.softmax(logits, dim=-1).cpu().numpy()[:, 1]
    sw, hw = ne.get("student_w", 0.5), ne.get("head_w", 0.5)
    pn = ((sw * p_stu + hw * p_head) / (sw + hw)).reshape(-1, 1)
    posneg = probs[:, [0, 2]]
    posneg = posneg / posneg.sum(1, keepdims=True)
    spec = np.concatenate([(1 - pn) * posneg[:, :1], pn,
                           (1 - pn) * posneg[:, 1:]], axis=1)
    w = ne["weight"]
    probs = (probs + w * spec) / (1 + w)
    temp = float(t.get("temperature", 0.0) or 0.0)
    if temp > 0:
        logits = np.log(np.clip(probs, 1e-9, 1.0)) / temp
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=1, keepdims=True)
    return probs


def predict_binary(texts):
    """Batched replica for task=binary. Returns (n,2) probs in model order
    [负面, 正面] (same as channel outputs before display flip)."""
    ch = webapp.channel_probs("binary", texts)
    probs = webapp.weighted_mean(ch)
    return probs


def main():
    rows = [json.loads(l) for l in open(LAB, encoding="utf-8")]
    texts = [(r["text"] or "").strip()[:4000] for r in rows]
    print(f"loaded {len(rows)} rows", flush=True)

    manifest = webapp.load_manifest()
    device = torch.device("cuda" if torch.cuda.is_available()
                          and manifest.get("use_gpu", True) else "cpu")
    webapp.STATE["device"] = str(device)
    webapp.STATE["binary_threshold"] = float(manifest.get("binary_threshold", 0.5))
    for task in ["3class", "binary"]:
        webapp.STATE["tasks"][task] = webapp.load_task(
            task, manifest["tasks"][task], device)
        print(f"loaded {task}", flush=True)

    # lexicon shortcut hits (deployed behavior: bypass GPU, fixed probs)
    lex3 = [sl.lookup(t, "3class") for t in texts]
    lexb = [sl.lookup(t, "binary") for t in texts]
    print(f"lexicon hits: 3class={sum(x is not None for x in lex3)} "
          f"binary={sum(x is not None for x in lexb)}", flush=True)

    gpu_idx = [i for i, t in enumerate(texts) if lex3[i] is None]
    gpu_texts = [texts[i] for i in gpu_idx]
    print(f"GPU inference on {len(gpu_texts)} texts...", flush=True)

    probs3 = np.zeros((len(rows), 3))
    probsb = np.zeros((len(rows), 2))
    if gpu_texts:
        p3 = predict_3class(gpu_texts)
        pb = predict_binary(gpu_texts)
        for j, i in enumerate(gpu_idx):
            probs3[i] = p3[j]
            probsb[i] = pb[j]
    # lexicon rows: replicate app.py fixed display probs -> class index
    for i, lab in enumerate(lex3):
        if lab is not None:
            probs3[i] = [0.03] * 3
            probs3[i][N3.index(lab)] = 0.94
    for i, lab in enumerate(lexb):
        if lab is not None:
            # binary display order [正面,负面]; model order [负面,正面]
            probsb[i] = [0.94, 0.03] if lab == "负面" else [0.03, 0.94]

    results = {"device": str(device), "n": len(rows), "sets": {}}
    for tag in ["ood1000", "al1000"]:
        idx = [i for i, r in enumerate(rows) if r["set"] == tag]
        # ---- 3class ----
        yt = [rows[i]["final"] for i in idx]
        yp = [N3[int(np.argmax(probs3[i]))] for i in idx]
        m3 = cls_metrics(yt, yp, N3)
        # ---- binary (drop neutral gold) ----
        bidx = [i for i in idx if rows[i]["final"] in ("正面", "负面")]
        ytb = [rows[i]["final"] for i in bidx]
        # model order [负面,正面] -> pred label
        ypb = []
        for i in bidx:
            p = probsb[i]
            ypb.append("正面" if p[1] >= p[0] else "负面")
        mb = cls_metrics(ytb, ypb, ["负面", "正面"])
        # confidence calibration (3class maxprob vs correctness)
        conf = [float(probs3[i].max()) for i in idx]
        correct = [int(a == b) for a, b in zip(yt, yp)]
        results["sets"][tag] = {
            "n": len(idx),
            "3class": m3,
            "binary": {"n": len(bidx), **mb},
            "mean_conf": round(float(np.mean(conf)), 4),
            "acc_at_conf>=0.8": round(float(np.mean(
                [c for c, cf in zip(correct, conf) if cf >= 0.8])), 4)
            if any(cf >= 0.8 for cf in conf) else None,
        }
        print(f"{tag}: 3class acc={m3['acc']} f1={m3['macro_f1']} | "
              f"binary(n={len(bidx)}) acc={mb['acc']} f1={mb['macro_f1']}",
              flush=True)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    np.savez_compressed(
        Path(r"E:\GB77_data\lab2000\v3_probs.npz"),
        probs3=probs3, probsb=probsb,
        sets=np.array([r["set"] for r in rows]),
        finals=np.array([r["final"] for r in rows]))
    print(f"-> {OUT} + v3_probs.npz", flush=True)


if __name__ == "__main__":
    main()

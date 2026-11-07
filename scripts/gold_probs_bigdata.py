"""Dump gold-600 probability predictions for a bigdata-trained checkpoint.

Usage:
  python scripts/gold_probs_bigdata.py --tag macbert_3c_pseudo_p55k_e2 --task 3class
Saves <data_root>/models/<tag>_goldprobs.npy (probs aligned with labeled rows).
"""
import argparse, json, os, sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from _paths import DATA_ROOT

GOLD = "data/reviews_600_base.jsonl"
OUTDIR = str(DATA_ROOT / "models")
LAB3 = {"正面": 0, "中性": 1, "负面": 2}


def predict_probs(model, tok, texts, max_len, device, bs=64, amp="bf16"):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(amp == "bf16" and device.type == "cuda")):
                logits = model(**enc).logits.float()
            outs.append(F.softmax(logits, -1).cpu())
    return torch.cat(outs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task", choices=["binary", "3class", "neu"], default="3class")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    rows = [r for r in rows if r.get("label_semantic")]
    texts = [r["text"] for r in rows]
    if args.task == "binary":
        keep = [i for i, r in enumerate(rows) if r["label_semantic"] != "中性"]
        texts = [texts[i] for i in keep]
    elif args.task == "neu":
        pass

    device = torch.device("cuda")
    ckpt = os.path.join(OUTDIR, args.tag)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
    probs = predict_probs(model, tok, texts, args.max_len, device, args.bs)
    np.save(os.path.join(OUTDIR, args.tag + "_goldprobs.npy"), probs)
    meta = {"tag": args.tag, "task": args.task, "n": len(texts),
            "texts_idx": list(range(len(rows))) if args.task != "binary"
            else [i for i, r in enumerate(rows) if r["label_semantic"] != "中性"]}
    json.dump(meta, open(os.path.join(OUTDIR, args.tag + "_goldmeta.json"), "w"))
    print("saved", probs.shape, "->", ckpt + "_goldprobs.npy")


if __name__ == "__main__":
    main()

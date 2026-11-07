#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark inference latency/throughput for a saved sentiment model."""
import os, sys, json, time, argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_finetune as gf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["3class", "binary"], default="binary")
    ap.add_argument("--label_key", default="label_semantic")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = gf.load_rows(args.label_key)
    texts, y = gf.make_labels(rows, args.task)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    # warmup
    gf.predict_probs(model, tok, texts[: min(32, len(texts))], args.max_len, device, args.bs, "bf16" if device.type == "cuda" else "none")
    times = []
    for _ in range(args.repeats):
        t0 = time.time()
        probs = gf.predict_probs(model, tok, texts, args.max_len, device, args.bs, "bf16" if device.type == "cuda" else "none")
        times.append(time.time() - t0)
    best = min(times)
    res = {
        "model": args.model,
        "task": args.task,
        "device": str(device),
        "n": len(texts),
        "bs": args.bs,
        "max_len": args.max_len,
        "times": times,
        "best_secs": best,
        "avg_secs": float(np.mean(times)),
        "samples_per_sec": len(texts) / best,
        "ms_per_sample": 1000 * best / len(texts),
    }
    tag = args.tag or f"bench_{os.path.basename(args.model.rstrip('/'))}_{args.task}_{device.type}"
    path = os.path.join(OUT, tag + ".json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()

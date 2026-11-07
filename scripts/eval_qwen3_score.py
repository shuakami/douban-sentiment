#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score-based evaluation of a causal LLM on the 600 Douban reviews.

Instead of parsing generated text (fragile — Qwen3-1.7B sometimes emits
label lists), teacher-force each label continuation and compare summed
log-probs. Also yields real probability distributions.

Usage:
  python scripts/eval_qwen3_score.py --model <path> --shot 3 --tag qwen3_17b_3shot_score
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_qwen3 import build_prompt, LABELS  # noqa: E402

OUT = os.path.join(ROOT, "out")


@torch.no_grad()
def score_probs(model, tok, prompts, bs=4):
    """Return (n,3) softmax probs over LABELS via teacher-forced logprob sums."""
    lab_ids = [tok.encode(l, add_special_tokens=False) for l in LABELS]
    prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    n = len(prompts)
    sumlog = np.full((n, len(LABELS)), -1e30, dtype=np.float64)

    rows = []  # (flat_idx, seq_ids, n_prompt, label_idx)
    for i, pids in enumerate(prompt_ids):
        for li, lids in enumerate(lab_ids):
            rows.append((i * len(LABELS) + li, pids + lids, len(pids), li, i))

    t0 = time.time()
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        maxlen = max(len(r[1]) for r in chunk)
        input_ids = torch.full((len(chunk), maxlen), tok.pad_token_id or tok.eos_token_id,
                               dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for b, (_, seq, _, _, _) in enumerate(chunk):
            input_ids[b, :len(seq)] = torch.tensor(seq)
            attn[b, :len(seq)] = 1
        input_ids = input_ids.to("cuda")
        attn = attn.to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
        # avoid materializing a second [B,T,V] log_softmax: gather target
        # logits and per-position logsumexp instead
        lse = torch.logsumexp(logits, dim=-1)  # [B,T]
        for b, (_, seq, npr, li, i) in enumerate(chunk):
            L = len(seq)
            total = 0.0
            for j in range(npr, L):
                total += (logits[b, j - 1, seq[j]] - lse[b, j - 1]).item()
            sumlog[i, li] = total
        del logits, lse
        torch.cuda.empty_cache()
        if (s // bs) % 20 == 0:
            print(f"  scored {min(s+bs,len(rows))}/{len(rows)} {time.time()-t0:.0f}s", flush=True)

    sumlog -= sumlog.max(axis=1, keepdims=True)
    p = np.exp(sumlog)
    return p / p.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shot", type=int, default=0, choices=[0, 3])
    ap.add_argument("--data", default="reviews_600_base.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", args.data), encoding="utf-8")]
    rows = [r for r in rows if r.get("label_semantic")]
    texts = [r["text"] for r in rows]
    y = np.array([LABELS.index(r["label_semantic"]) for r in rows])

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    shots = []
    if args.shot == 3:
        seen = set()
        for r in rows:
            if r["label_semantic"] not in seen:
                seen.add(r["label_semantic"])
                shots.append((r["text"], r["label_semantic"]))
            if len(shots) == 3:
                break

    prompts = [build_prompt(tok, t, shots, enable_thinking=False) for t in texts]
    probs = score_probs(model, tok, prompts, bs=args.bs)
    pred = probs.argmax(1)

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    print(f"\nacc={acc:.4f} macro_f1={f1:.4f}")
    print("confusion (rows=true cols=pred):")
    print(confusion_matrix(y, pred, labels=[0, 1, 2]))
    # per-class recall
    for i, n in enumerate(LABELS):
        mask = y == i
        print(f"{n} recall={ (pred[mask]==i).mean():.3f} (n={mask.sum()})")

    tag = args.tag or f"{os.path.basename(args.model)}_{args.shot}shot_score"
    res = {"model": args.model, "shot": args.shot, "n": len(y), "method": "label_logprob",
           "acc": float(acc), "macro_f1": float(f1), "y": y.tolist(),
           "pred": pred.tolist(), "probs": probs.tolist()}
    path = os.path.join(OUT, tag + ".json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print("saved", path)


if __name__ == "__main__":
    main()

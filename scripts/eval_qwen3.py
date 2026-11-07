#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a modern general-purpose LLM (Qwen3) on the 600 Douban reviews.

Zero-shot and 3-shot (one example per class) sentiment classification.
The point: compare a 2025/2026 general LLM against our ~100M distilled
specialist. Generation-based: prompt -> free text -> first label token wins.

Usage:
  python scripts/eval_qwen3.py --model Qwen/Qwen3-0.6B --shot 0
  python scripts/eval_qwen3.py --model ... --shot 3 --tag qwen3_06b_3shot
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
LABELS = ["正面", "中性", "负面"]

SYS = "你是一个严格的文本情感分类器。只输出一个词：正面、中性或负面，不要输出任何其他内容。"


def build_prompt(tok, text, shot_examples, enable_thinking=False):
    if shot_examples:
        user = "下面是几条豆瓣电影短评和它们的情绪标签。\n\n"
        for t, lab in shot_examples:
            user += f"短评：{t}\n情绪：{lab}\n\n"
        user += f"短评：{text}\n情绪："
    else:
        user = f"判断下面这条豆瓣电影短评的情绪，只输出「正面」「中性」或「负面」一个词。\n\n短评：{text}\n情绪："
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=enable_thinking)


def parse_label(gen_text):
    # first occurring label in generated text wins; strip think blocks just in case
    gen_text = re.sub(r"<think>.*?</think>", "", gen_text, flags=re.S)
    hits = [(gen_text.find(lab), lab) for lab in LABELS if lab in gen_text]
    if hits:
        return min(hits)[1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shot", type=int, default=0, choices=[0, 3])
    ap.add_argument("--data", default="reviews_600_base.jsonl")
    ap.add_argument("--max_new", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", args.data), encoding="utf-8")]
    rows = [r for r in rows if r.get("label_semantic")]
    texts = [r["text"] for r in rows]
    y = [LABELS.index(r["label_semantic"]) for r in rows]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda")
    model.eval()

    # deterministic 3-shot examples: first review of each class
    shots = []
    if args.shot == 3:
        seen = set()
        for r in rows:
            lab = r["label_semantic"]
            if lab not in seen:
                seen.add(lab)
                shots.append((r["text"], lab))
            if len(shots) == 3:
                break
        print("shot examples:", shots)

    preds, fails, t0 = [], 0, time.time()
    for i, text in enumerate(texts):
        prompt = build_prompt(tok, text, shots)
        enc = tok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        lab = parse_label(gen)
        if lab is None:
            fails += 1
            preds.append(-1)
        else:
            preds.append(LABELS.index(lab))
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(texts)} elapsed={time.time()-t0:.0f}s fails={fails}", flush=True)

    # metrics on parsed subset
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    mask = np.array([p >= 0 for p in preds])
    yp = np.array(y)[mask]
    pp = np.array(preds)[mask]
    acc = accuracy_score(yp, pp)
    f1 = f1_score(yp, pp, average="macro")
    print(f"\nparsed={mask.sum()}/{len(y)} fails={fails}")
    print(f"acc={acc:.4f} macro_f1={f1:.4f}")
    print("confusion (rows=true cols=pred):")
    print(confusion_matrix(yp, pp, labels=[0, 1, 2]))

    tag = args.tag or f"{os.path.basename(args.model)}_{args.shot}shot"
    res = {
        "model": args.model, "shot": args.shot, "n": len(y),
        "acc": float(acc), "macro_f1": float(f1), "parse_fails": int(fails),
        "y": y, "pred": preds,
    }
    path = os.path.join(OUT, tag + ".json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print("saved", path)


if __name__ == "__main__":
    main()

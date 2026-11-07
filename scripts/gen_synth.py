#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate synthetic Douban-style short reviews with Qwen3-4B for the
starved neutral/mixed class, self-filter with the deployed ensemble, and
emit candidate aug rows for distillation experiments.

Usage:
  python scripts/gen_synth.py --model <qwen3-4b path> --n-per-class 40
      --out data/synth_candidates.jsonl
"""
import os, sys, json, argparse, random, re

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

N3 = ["正面", "中性", "负面"]

PROMPT_TMPL = """你是豆瓣电影短评模拟器。请模仿下面真实豆瓣短评的语气、长度和网络用语，生成 {n} 条「{label}」情感倾向的电影短评。

要求：
- 每条 15~60 字，像真人随手写的短评，不要书面腔
- {label}类要有真实混合情绪：可以先扬后抑、先抑后扬、客观陈述里带褒贬
- 直接输出 JSON 数组，每行一条字符串，不要编号不要解释

真实示例：
{examples}

输出："""


def build_examples(rows, per=8):
    picks = {"正面": [], "中性": [], "负面": []}
    random.seed(7)
    pool = [r for r in rows if r.get("label_semantic")]
    random.shuffle(pool)
    for r in pool:
        lab = r["label_semantic"]
        if len(picks[lab]) < per:
            picks[lab].append(r["text"])
    out = []
    for lab in N3:
        for t in picks[lab]:
            out.append(f"[{lab}] {t}")
    return "\n".join(out)


@torch.no_grad()
def gen(model, tok, prompt, n=40, max_new=900):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                             temperature=0.9, top_p=0.95,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_array(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x).strip() for x in arr if isinstance(x, str) and 8 <= len(str(x).strip()) <= 120]
    except Exception:
        # fallback: line-based
        return [l.strip().strip('"').strip("'") for l in text.splitlines()
                if 8 <= len(l.strip().strip('"')) <= 120][:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/synth_candidates.jsonl")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "reviews_600_base.jsonl"), encoding="utf-8")]
    ex = build_examples(rows)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to("cuda").eval()

    results = []
    for lab in N3:
        # ask for ~1.6x to allow filtering
        n = int(args.n_per_class * 1.6)
        prompt = PROMPT_TMPL.format(n=n, label=lab, examples=ex)
        raw = gen(model, tok, prompt)
        cands = parse_array(raw)
        print(f"[{lab}] generated {len(cands)} candidates", flush=True)
        for t in cands:
            results.append({"text": t, "gen_label": lab})

    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved {len(results)} -> {args.out}")


if __name__ == "__main__":
    main()

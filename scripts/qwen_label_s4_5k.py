#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4 variant — label the two 5K arms with local Qwen3-4B (DeepSeek API
unavailable: HTTP 402 on both keys). Same rubric prompt + teacher-forced
label-logprob protocol as eval_human400_qwen.py / ood1000 Qwen baseline.

Arms (both Qwen-labelled, so the random-vs-uncertainty comparison stays
teacher-controlled):
  rand5k: E:/GB77_data/lab2000/rand5k_pool.jsonl (uniform, seed 20261113,
          all project cids excluded)
  al5k:   the --subsample 5000 --sub_seed 0 draw of extra_al17k_SWAP_ds.jsonl
          (same indices sft_human400_generic.py would use)

Writes:
  lab2000/qwen_{rand5k,al5k}_labels.jsonl  {i,cid,text,probs,label}
  lab2000/extra_{rand5k,al5k}_qwen.jsonl   {text,rating,final}  (training format)
"""
import json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _paths import DATA_ROOT  # noqa: E402
from eval_human400_qwen import RUBRIC  # noqa: E402  (same rubric, verbatim)
from eval_qwen3 import LABELS  # noqa: E402
from eval_qwen3_score import score_probs  # noqa: E402

LAB = DATA_ROOT / "lab2000"
MODEL = r"D:\Project\douban-sentiment-env\models\Qwen3-4B"
BS = 6


def prompt_for(tok, t):
    msgs = [{"role": "user", "content":
             RUBRIC + "\n现在请标注下面这条评论。\n\n短评：" + t + "\n情绪："}]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)


def al5k_indices():
    rows = [json.loads(l) for l in open(LAB / "extra_al17k_SWAP_ds.jsonl",
                                         encoding="utf-8")]
    rows = [r for r in rows if r.get("final") in LABELS]
    idx = np.random.RandomState(0).choice(len(rows), 5000, replace=False)
    return sorted(int(i) for i in idx), rows


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sets = {}
    rp = [json.loads(l) for l in open(LAB / "rand5k_pool.jsonl", encoding="utf-8")]
    sets["rand5k"] = [{"i": r["i"], "cid": r["cid"], "text": r["text"],
                       "rating": r.get("rating")} for r in rp]
    idx, rows = al5k_indices()
    sets["al5k"] = [{"i": int(k), "cid": None, "text": rows[k]["text"],
                     "rating": rows[k].get("rating")} for k in idx]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    for name, items in sets.items():
        out_lab = LAB / f"qwen_{name}_labels.jsonl"
        out_ext = LAB / f"extra_{name}_qwen.jsonl"
        done = set()
        if out_lab.exists():
            for l in open(out_lab, encoding="utf-8"):
                try:
                    done.add(int(json.loads(l)["i"]))
                except Exception:
                    pass
        todo = [it for it in items if int(it["i"]) not in done]
        print(f"[{name}] done={len(done)} todo={len(todo)}", flush=True)
        if todo:
            texts = [(it["text"] or "").strip()[:4000] for it in todo]
            prompts = [prompt_for(tok, t) for t in texts]
            t0 = time.time()
            probs = score_probs(model, tok, prompts, bs=BS)
            print(f"[{name}] scored {len(todo)} in {time.time()-t0:.0f}s",
                  flush=True)
            with open(out_lab, "a", encoding="utf-8") as f:
                for it, p in zip(todo, probs):
                    f.write(json.dumps(
                        {"i": int(it["i"]), "cid": it.get("cid"),
                         "text": it["text"],
                         "probs": [round(float(x), 4) for x in p],
                         "label": LABELS[int(p.argmax())]},
                        ensure_ascii=False) + "\n")
        # rebuild extra file from all labels (order = pool order)
        labels = {}
        for l in open(out_lab, encoding="utf-8"):
            r = json.loads(l)
            labels[int(r["i"])] = r["label"]
        with open(out_ext, "w", encoding="utf-8") as f:
            for it in items:
                lab = labels.get(int(it["i"]))
                f.write(json.dumps({"text": it["text"],
                                    "rating": it.get("rating"),
                                    "final": lab if lab in LABELS else None},
                                   ensure_ascii=False) + "\n")
        from collections import Counter
        c = Counter(labels.get(int(it["i"])) for it in items)
        print(f"[{name}] wrote {out_ext} dist={dict(c)}", flush=True)


if __name__ == "__main__":
    main()

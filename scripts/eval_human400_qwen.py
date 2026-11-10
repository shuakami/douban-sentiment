#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Label human400 with local Qwen3-4B via label-logprob scoring.

Identical protocol to the ood1000 Qwen3-4B baseline (rubric-conditioned
chat prompt + teacher-forced label logprobs, see
qwen_label_ood1000.py), applied to the blind human-annotated set so the
Table-3 teacher column can be completed on clean human labels.

Input : data/human400_labels.jsonl (`label` = human truth)
Output: <DATA_ROOT>/human400/qwen3_4b_human400_probs.npy (400,3)
        <DATA_ROOT>/human400/qwen_human400_labels.jsonl

Requires local model dir (default D:/Project/douban-sentiment-env/models/Qwen3-4B,
override with --model). Serialize with other GPU jobs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _paths import DATA_ROOT  # noqa: E402
from eval_qwen3 import LABELS  # noqa: E402
from eval_qwen3_score import score_probs  # noqa: E402

# Same 3-class rubric used for the ood1000 Qwen3-4B baseline
# (verbatim copy of lab2000/deepseek_label.py RUBRIC).
RUBRIC = """你是豆瓣电影短评情感标注员。判断评论作者对【电影本身】的情感倾向，三选一：
- 正面：表扬、推荐、肯定（含隐晦夸奖）
- 中性：客观描述剧情/演职人员、纯感慨无立场、褒贬各半难分、与电影本身无关
- 负面：批评、吐槽、失望、劝退（含反讽/阴阳怪气的贬义）

规则：
1. 只根据评论文本判断，不要推测评分
2. 关注对电影的态度，不是对剧情、演员个人或社会现象的态度
3. 反讽/玩梗要结合上下文推断真实态度
4. 没有明确褒贬、纯客观描述或纯情绪感慨 → 中性
5. 拿不准时选中性

参考示例：
评论：画面太美了，配乐也绝，一晚上看了两遍还意犹未尽 → 正面
评论：女主演技好，男主像个木头，剧情也老套 → 中性
评论：两个小时的垃圾时间，编剧是来搞笑的吧 → 负面
评论：有些人只能留在记忆里 → 中性
评论：神片子！传统手艺用特效拍，侮辱观众智商 → 负面
评论：还行吧，不功不过，但是也没啥亮点 → 中性
评论：一看名字就知道是骗钱的，果然名不虚传 → 负面
评论：本年度最佳，没有之一，感动到哭 → 正面
"""

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

IN = ROOT / "data" / "human400_labels.jsonl"
OUT = DATA_ROOT / "human400"
DEFAULT_MODEL = r"D:\Project\douban-sentiment-env\models\Qwen3-4B"
BS = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bs", type=int, default=BS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
    rows = [r for r in rows if r.get("label")]
    texts = [(r["text"] or "").strip()[:4000] for r in rows]
    gold = [r["label"] for r in rows]
    print(f"n={len(texts)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()

    def prompt_for(t):
        msgs = [{"role": "user", "content":
                 RUBRIC + "\n现在请标注下面这条评论。\n\n短评：" + t + "\n情绪："}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

    prompts = [prompt_for(t) for t in texts]
    t0 = time.time()
    probs = score_probs(model, tok, prompts, bs=args.bs)
    print(f"scored in {time.time()-t0:.0f}s", flush=True)

    np.save(OUT / "qwen3_4b_human400_probs.npy", probs)
    y = np.array([LABELS.index(g) for g in gold])
    pred = probs.argmax(1)
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc, f1 = accuracy_score(y, pred), f1_score(y, pred, average="macro")
    print(f"Qwen3-4B human400 acc={acc:.4f} f1={f1:.4f}")
    print(confusion_matrix(y, pred))
    with open(OUT / "qwen_human400_labels.jsonl", "w",
              encoding="utf-8") as f:
        for i, r in enumerate(rows):
            p = probs[i]
            rec = {"i": i, "cid": r.get("cid"), "text": r["text"],
                   "gold": r["label"],
                   "probs": [round(float(x), 4) for x in p],
                   "label": LABELS[int(p.argmax())],
                   "margin": round(float(np.sort(p)[-1]
                                         - np.sort(p)[-2]), 4)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("->", OUT, flush=True)


if __name__ == "__main__":
    main()

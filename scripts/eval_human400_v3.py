#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute deployed-v3 probs on the human-annotated 400-item held-out set.

Same protocol as the fresh500/ood1000 v3 evals: batched replica of
web/app.py predict() (post-temperature probs), lexicon shortcut applied
first. Requires the web demo models (manifest tasks) on GPU/CPU.

Input : data/human400_labels.jsonl  (repo; `label` field is human truth)
Output: <DATA_ROOT>/human400/human400_v3_probs.npz  {probs3 (400,3),
        probsb (400,2)}

Usage: python scripts/eval_human400_v3.py
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=<cache>
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "scripts"))
from _paths import DATA_ROOT  # noqa: E402
import app as webapp  # noqa: E402
import short_lexicon as sl  # noqa: E402
import eval_lab2000 as E  # noqa: E402

IN = ROOT / "data" / "human400_labels.jsonl"
OUT = DATA_ROOT / "human400"
N3 = ["正面", "中性", "负面"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
    rows = [r for r in rows if r.get("label")]
    texts = [(r["text"] or "").strip()[:4000] for r in rows]
    print(f"loaded {len(rows)} labeled rows", flush=True)

    manifest = webapp.load_manifest()
    device = torch.device("cuda" if torch.cuda.is_available()
                          and manifest.get("use_gpu", True) else "cpu")
    webapp.STATE["device"] = str(device)
    webapp.STATE["binary_threshold"] = float(
        manifest.get("binary_threshold", 0.5))
    for task in ["3class", "binary"]:
        webapp.STATE["tasks"][task] = webapp.load_task(
            task, manifest["tasks"][task], device)
        print(f"loaded {task}", flush=True)

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
        p3 = E.predict_3class(gpu_texts)
        pb = E.predict_binary(gpu_texts)
        for j, i in enumerate(gpu_idx):
            probs3[i] = p3[j]
            probsb[i] = pb[j]
    for i, lab in enumerate(lex3):
        if lab is not None:
            probs3[i] = [0.03] * 3
            probs3[i][N3.index(lab)] = 0.94
    for i, lab in enumerate(lexb):
        if lab is not None:
            probsb[i] = [0.94, 0.03] if lab == "负面" else [0.03, 0.94]

    np.savez_compressed(OUT / "human400_v3_probs.npz",
                        probs3=probs3, probsb=probsb)
    print("->", OUT / "human400_v3_probs.npz", flush=True)


if __name__ == "__main__":
    main()

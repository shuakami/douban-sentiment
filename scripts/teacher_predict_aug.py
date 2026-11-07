#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the TEACHER ensembles (manifest_teacher.json) on the augmented short-text
samples and write extended teacher-probs files for distillation:

  out/ens_3class_allch_aug.json  — 648 rows (600 base + 48 aug)
  out/ens_binary_allch2_aug.json — 540 rows (508 base + 32 non-neutral aug)

Aug rows are appended at the end of reviews_600.jsonl, so base probs keep their
order and aug probs append in file order (binary skips 中性).
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "web"))
sys.path.insert(0, HERE)
import app as A  # noqa: E402
import gpu_finetune as gf  # noqa: E402

OUT = os.path.join(ROOT, "out")
AUG_IDS_START = 601


def teacher_probs(task):
    manifest = json.load(open(os.path.join(ROOT, "web", "manifest_teacher.json"), encoding="utf-8"))
    device = torch.device("cuda")
    t = A.load_task(task, manifest["tasks"][task], device)
    A.STATE["tasks"][task] = t
    rows = gf.load_rows("label_semantic")
    aug = [r for r in rows if r.get("src") == "aug"]
    if task == "binary":
        aug = [r for r in aug if r["label_semantic"] != "中性"]
    texts = [r["text"] for r in aug]
    ch = A.channel_probs(task, texts)
    return A.weighted_mean(ch), texts, aug


def main():
    rows = gf.load_rows("label_semantic")
    aug_all = [r for r in rows if r.get("src") == "aug"]

    # ---- 3class: all 48 aug rows ----
    p3, texts3, _ = teacher_probs("3class")
    base3 = json.load(open(os.path.join(OUT, "ens_3class_allch.json"), encoding="utf-8"))
    probs3 = np.concatenate([np.array(base3["probs"], dtype=np.float32), p3], axis=0)
    y3 = np.array(base3["y"] + [ {"正面":0,"中性":1,"负面":2}[r["label_semantic"]] for r in aug_all ])
    res3 = {"model": "teacher_ens_3class+aug", "task": "3class", "n": len(y3),
            "channels": base3.get("channels", "ens_3class_allch"),
            "y": y3.tolist(), "probs": probs3.tolist(),
            "note": "base OOF probs + teacher predictions on 48 aug short texts"}
    json.dump(res3, open(os.path.join(OUT, "ens_3class_allch_aug.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"3class aug teacher probs {probs3.shape} -> ens_3class_allch_aug.json")

    # ---- binary: 32 non-neutral aug rows ----
    p2, texts2, aug2 = teacher_probs("binary")
    base2 = json.load(open(os.path.join(OUT, "ens_binary_allch2.json"), encoding="utf-8"))
    probs2 = np.concatenate([np.array(base2["probs"], dtype=np.float32), p2], axis=0)
    y2 = np.array(base2["y"] + [1 if r["label_semantic"] == "正面" else 0 for r in aug2])
    res2 = {"model": "teacher_ens_binary+aug", "task": "binary", "n": len(y2),
            "channels": base2.get("channels", "ens_binary_allch2"),
            "y": y2.tolist(), "probs": probs2.tolist(),
            "note": "base OOF probs + teacher predictions on 32 non-neutral aug texts"}
    json.dump(res2, open(os.path.join(OUT, "ens_binary_allch2_aug.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"binary aug teacher probs {probs2.shape} -> ens_binary_allch2_aug.json")

    # report teacher's own predictions on aug texts for pruning decisions
    for r, p in zip(aug_all, p3):
        pred = int(np.argmax(p))
        lab = ["正面", "中性", "负面"][pred]
        flag = "" if lab == r["label_semantic"] else "  <-- teacher disagrees"
        print(f"  {r['text']!r:12s} label={r['label_semantic']} teacher={lab} "
              f"p={np.round(p,3)}{flag}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the gold600+al1000 combined training set and aligned teacher probs
for Stage-2 distillation.

Outputs:
  data/reviews_gold_al1000.jsonl          - 1600 rows (al rows flagged src="aug")
  out/ens_lab1600_teacher_3class.json     - teacher probs (1600x3)
  out/ens_lab1600_teacher_binary.json     - teacher probs (1139x2, neutrals dropped)

Teacher = deployed v3 ensemble: gold rows use the honest OOF probs from
ens_lite_*; al1000 rows use genuine v3 predictions (v3 never trained on them).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LAB = Path(r"E:\GB77_data\lab2000")


def main():
    gold = [json.loads(l) for l in
            open(ROOT / "data" / "reviews_600_base.jsonl", encoding="utf-8")]
    lab_rows = [json.loads(l) for l in
                open(LAB / "lab2000_final.jsonl", encoding="utf-8")]
    al = [r for r in lab_rows if r["set"] == "al1000" and r["final"]]
    assert len(gold) == 600 and len(al) == 1000

    combined = []
    for r in gold:
        combined.append(r)
    for j, r in enumerate(al):
        combined.append({
            "id": 1000 + j, "page": 0, "nick": "lab_al", "star": "未知",
            "time": "", "ip": "", "votes": 0, "text": r["text"],
            "label_rule": "", "label_semantic": r["final"],
            "src": "aug", "lab_set": "al1000", "lab_i": r["i"],
            "cid": r["cid"], "rating": r["rating"],
        })
    with open(ROOT / "data" / "reviews_gold_al1000.jsonl", "w",
              encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"combined rows: {len(combined)}")

    # ---- teacher probs ----
    z = np.load(LAB / "v3_probs.npz", allow_pickle=True)
    probs3 = z["probs3"]      # (2000,3), rows = lab2000_final order
    probsb = z["probsb"]      # (2000,2) model order [负面,正面]
    sets = z["sets"]
    finals = z["finals"]
    al_idx = [i for i, s in enumerate(sets) if s == "al1000"]
    assert len(al_idx) == 1000

    ens3 = json.load(open(ROOT / "out" / "ens_lite_3class_v3.json",
                          encoding="utf-8"))
    ensb = json.load(open(ROOT / "out" / "ens_lite_binary.json",
                          encoding="utf-8"))
    g3 = np.array(ens3["probs"], dtype=np.float32)   # 600x3
    gb = np.array(ensb["probs"], dtype=np.float32)   # 508x2
    assert g3.shape == (600, 3) and gb.shape == (508, 2)

    # 3class: all 1600 rows kept by make_labels
    t3 = np.concatenate([g3, probs3[al_idx].astype(np.float32)], axis=0)
    assert t3.shape == (1600, 3)
    (ROOT / "out" / "ens_lab1600_teacher_3class.json").write_text(
        json.dumps({"tag": "v3_ood", "n": 1600, "probs": t3.tolist()},
                   ensure_ascii=False), encoding="utf-8")

    # binary: neutrals dropped -> gold non-neutral (508) + al non-neutral
    gold_nonidx = [i for i, r in enumerate(gold) if r["label_semantic"] != "中性"]
    assert len(gold_nonidx) == 508
    al_keep = [k for k, i in enumerate(al_idx)
               if finals[i] != "中性"]
    al_keep_global = [al_idx[k] for k in al_keep]
    tb = np.concatenate([gb, probsb[al_keep_global].astype(np.float32)],
                        axis=0)
    n_bin = 508 + len(al_keep)
    assert tb.shape == (n_bin, 2), tb.shape
    (ROOT / "out" / "ens_lab1600_teacher_binary.json").write_text(
        json.dumps({"tag": "v3_ood", "n": n_bin, "probs": tb.tolist()},
                   ensure_ascii=False), encoding="utf-8")
    print(f"teacher 3class: {t3.shape}  binary: {tb.shape} "
          f"(al non-neutral={len(al_keep)})")


if __name__ == "__main__":
    main()

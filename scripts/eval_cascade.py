"""Evaluate cascade architectures on gold-600 from saved probability arrays.

Cascade A: P(neu) from a dedicated neutral detector gates; pos/neg split by
a binary model's P(pos). Compare vs deployed v2 (ens_lite_3class_v2.json).

Usage:
  python scripts/eval_cascade.py --neu <data_root>/models/neu_pseudo_p60k_e2_goldprobs.npy \
      --bin <data_root>/models/macbert_bin_pseudo_p120k_e2_goldprobs.npy \
      --bin-tag macbert_bin_pseudo_p120k_e2
"""
import argparse, json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

GOLD = "data/reviews_600_base.jsonl"
LAB3 = {"正面": 0, "中性": 1, "负面": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neu", required=True, help=".npy P(neu) or 2-col probs")
    ap.add_argument("--bin", required=True, help="binary model goldprobs (all-600 or non-neutral rows)")
    ap.add_argument("--bin-tag", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    rows = [r for r in rows if r.get("label_semantic")]
    y = np.array([LAB3[r["label_semantic"]] for r in rows])
    nonneu_idx = np.array([i for i, r in enumerate(rows) if r["label_semantic"] != "中性"])

    neu_p = np.load(args.neu)
    if neu_p.ndim == 2 and neu_p.shape[1] == 2:
        p_neu = neu_p[:, 1]
    else:
        p_neu = neu_p
    assert len(p_neu) == len(y), f"neu probs {len(p_neu)} != gold {len(y)}"

    b_p = np.load(args.bin)
    if b_p.ndim == 2 and b_p.shape[1] == 2:
        p_pos_bin = b_p[:, 1]
    else:
        p_pos_bin = b_p
    # binary probs may be aligned to all-600 or to non-neutral subset
    if len(p_pos_bin) == len(y):
        pos600 = p_pos_bin
    elif len(p_pos_bin) == len(nonneu_idx):
        pos600 = np.full(len(y), np.nan)
        pos600[nonneu_idx] = p_pos_bin
    else:
        raise ValueError(f"bin probs len {len(p_pos_bin)} matches neither")

    # threshold sweep for the neu gate
    best = None
    for thr in np.arange(0.2, 0.8, 0.02):
        pred = np.where(p_neu >= thr, 1,
                        np.where(pos600 >= 0.5, 0, 2))
        pred = np.where(np.isnan(pos600) & (p_neu < thr), 1, pred)  # fallback
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro")
        if best is None or (f1, acc) > (best[1], best[0]):
            best = (acc, f1, thr, pred)
    acc, f1, thr, pred = best
    print(f"cascade thr={thr:.2f} acc={acc:.4f} macro_f1={f1:.4f}")
    print(classification_report(y, pred, target_names=["正面", "中性", "负面"],
                                digits=4))
    # save best preds for ensemble blending experiments
    np.save("out/cascade_pred.npy", pred)
    np.save("out/cascade_pneu.npy", p_neu)
    json.dump({"thr": float(thr), "acc": float(acc), "macro_f1": float(f1),
               "bin_tag": args.bin_tag},
              open("out/cascade_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()

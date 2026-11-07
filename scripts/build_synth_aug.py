#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an augmented training file (base 600 + kept synth rows with
src='aug') and an aligned teacher-probs json for distill.py.

Usage: python scripts/build_synth_aug.py --kept data/synth_kept.jsonl
       --teacher out/ens_lite_3class_v2.json --tag synth1
"""
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", default="data/synth_kept.jsonl")
    ap.add_argument("--teacher", default="ens_lite_3class_v2.json")
    ap.add_argument("--tag", default="synth1")
    ap.add_argument("--max-per-class", type=int, default=60)
    args = ap.parse_args()

    base = [json.loads(l) for l in open(os.path.join(ROOT, "data", "reviews_600_base.jsonl"), encoding="utf-8")]
    kept = [json.loads(l) for l in open(os.path.join(ROOT, args.kept), encoding="utf-8")]

    # cap per class, prefer higher gen_conf
    by = {}
    for r in kept:
        by.setdefault(r["label_semantic"], []).append(r)
    sel = []
    for lab, rs in by.items():
        rs.sort(key=lambda r: -r["gen_conf"])
        sel.extend(rs[: args.max_per_class])
    print(f"kept total={len(kept)} selected={len(sel)} " +
          " ".join(f"{k}:{min(len(v), args.max_per_class)}" for k, v in by.items()))

    max_id = max(r["id"] for r in base)
    aug_rows = []
    for i, r in enumerate(sel):
        aug_rows.append({
            "id": max_id + 1 + i, "page": 0, "nick": "synth", "star": "未知",
            "time": "", "ip": "", "votes": 0, "text": r["text"],
            "label_rule": r["label_semantic"], "label_semantic": r["label_semantic"],
            "src": "aug", "gen_conf": r["gen_conf"],
        })

    data_name = f"reviews_600_{args.tag}.jsonl"
    with open(os.path.join(ROOT, "data", data_name), "w", encoding="utf-8") as f:
        for r in base + aug_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote data/{data_name} n={len(base) + len(aug_rows)}")

    # teacher probs: deployed ensemble for base rows + uniform for aug rows
    tr = json.load(open(os.path.join(ROOT, "out", args.teacher), encoding="utf-8"))
    probs = [list(map(float, p)) for p in tr["probs"]]
    assert len(probs) == len(base), f"teacher rows {len(probs)} != base {len(base)}"
    uni = [1 / 3, 1 / 3, 1 / 3]
    for _ in aug_rows:
        probs.append(uni)
    tag = f"ens_lite_3class_v2_{args.tag}teacher"
    json.dump({"probs": probs, "tag": tag},
              open(os.path.join(ROOT, "out", tag + ".json"), "w", encoding="utf-8"))
    print(f"wrote out/{tag}.json rows={len(probs)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4 — sample 5,000 corpus reviews for the random-vs-uncertainty control.

Uniform random sample (seed 20261113) from the 910,998-row cleaned corpus
(E:/GB77_data/clean/comments.pkl) after excluding every cid used in any
project pool — same enumeration as scripts/audit_human400_overlap.py —
plus a normalized-text exclusion for extra safety.

Writes E:/GB77_data/lab2000/rand5k_pool.jsonl  {"i","cid","text","rating"}
"""
import json, pickle, re, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_ROOT  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LAB = DATA_ROOT / "lab2000"
H4 = REPO / "data" / "human400_labels.jsonl"
SEED = 20261113
N = 5000

excluded_cids = set()
excluded_texts = set()

def add_text(t):
    excluded_texts.add(re.sub(r"\s+", "", (t or "").strip()))

# human400 itself
for l in open(H4, encoding="utf-8"):
    r = json.loads(l)
    excluded_cids.add(str(r["cid"]))
    add_text(r.get("text"))

# gold-600
for l in open(REPO / "data/reviews_600_base.jsonl", encoding="utf-8"):
    r = json.loads(l)
    excluded_cids.add(str(r["id"]))
    add_text(r.get("text"))

# all lab2000 jsonl (cid field) + text exclusion for the final label file
for f in LAB.glob("*.jsonl"):
    for l in open(f, encoding="utf-8"):
        try:
            d = json.loads(l)
        except Exception:
            continue
        c = d.get("cid")
        if c is not None:
            excluded_cids.add(str(c))
        if f.name == "lab2000_final.jsonl":
            add_text(d.get("text"))

# repo data jsonl (id/cid), excluding the eval file itself
for f in (REPO / "data").rglob("*.jsonl"):
    if f.name == "human400_labels.jsonl":
        continue
    for l in open(f, encoding="utf-8"):
        try:
            d = json.loads(l)
        except Exception:
            continue
        for k in ("cid", "id"):
            if d.get(k) is not None:
                excluded_cids.add(str(d[k]))
                break

# corpus-side pool manifests — skip full-corpus coverage maps (>500K rows:
# stu3c_{,bin,r2}_meta.pkl contain every corpus cid by construction and would
# vacuously exclude the whole corpus)
for p in sorted(list((DATA_ROOT / "pseudo").glob("*_meta.pkl")) +
                list((DATA_ROOT / "features").glob("*_meta.pkl"))):
    try:
        m = pickle.load(open(p, "rb"))
    except Exception:
        continue
    if hasattr(m, "columns") and "cid" in m.columns and len(m) <= 500000:
        excluded_cids.update(m["cid"].astype(str))

print(f"excluded_cids={len(excluded_cids)} excluded_texts={len(excluded_texts)}")

df = pickle.load(open(DATA_ROOT / "clean" / "comments.pkl", "rb"))
mask = ~df["cid"].astype(str).isin(excluded_cids)
mask &= ~df["content"].map(lambda t: re.sub(r"\s+", "", str(t).strip()) in excluded_texts)
pool = df[mask]
print(f"corpus={len(df)} eligible={len(pool)}")

samp = pool.sample(n=N, random_state=SEED).reset_index(drop=True)
out = LAB / "rand5k_pool.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for i, r in samp.iterrows():
        f.write(json.dumps({"i": int(i), "cid": int(r["cid"]),
                            "text": str(r["content"]),
                            "rating": None if pd.isna(r["rating"]) else float(r["rating"])},
                           ensure_ascii=False) + "\n")
print(f"wrote {out} n={len(samp)}")

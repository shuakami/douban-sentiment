#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify human400 has zero cid overlap with every training pool of the
evaluated systems (deployed v3 + al17k SFT mix).

Pools checked:
  - gold-600 (data/reviews_600_base.jsonl, id field)
  - lab2000 all jsonl under <DATA_ROOT>/lab2000 (cid field) -> covers
    AL1000/3k/5k/17k SFT pools and fresh500
  - repo data/**/*.jsonl (id/cid fields; lab500 is text-matched below)
  - corpus-side pool manifests: <DATA_ROOT>/pseudo/*_meta.pkl and
    <DATA_ROOT>/features/*_meta.pkl (actual training pools; the three
    910,998-row metas are full-corpus metadata, not training subsets)

Deployed v3 channels (web/manifest.json): macbert_3c_distill,
macbert_bin_distill, mlp_3c/mlp_bin/mlp_neu mb13b heads (all gold-trained),
macbert_neu_corpus (= neu_pseudo_p60k_e2, stu3c conf>=0.7 pool subset).
macbert_bin_strict_p60k_e2 (rating-pool experiment) is NOT a deployed
channel and NOT evaluated on human400.

Usage: python scripts/audit_human400_overlap.py
Requires DOUBAN_DATA_ROOT (default E:/GB77_data) for corpus-side metas.
"""
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_ROOT  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LAB = DATA_ROOT / "lab2000"
H4 = REPO / "data" / "human400_labels.jsonl"

rows = [json.loads(l) for l in open(H4, encoding="utf-8")]
rows = [r for r in rows if r.get("label")]
hcids = {str(r["cid"]) for r in rows}
htexts = {re.sub(r"\s+", "", r["text"].strip()) for r in rows}
print(f"human400: n={len(rows)} unique_cids={len(hcids)}")

results = []

# 1. gold600
s = set()
for l in open(REPO / "data/reviews_600_base.jsonl", encoding="utf-8"):
    s.add(str(json.loads(l)["id"]))
results.append(("gold600", len(s), len(hcids & s)))

# 2. lab2000 all jsonl (cid field)
s = set()
for f in LAB.glob("*.jsonl"):
    for l in open(f, encoding="utf-8"):
        try:
            c = json.loads(l).get("cid")
        except Exception:
            continue
        if c is not None:
            s.add(str(c))
results.append(("lab2000_all_jsonl", len(s), len(hcids & s)))

# 3. repo data jsonl (id/cid fields; exclude the eval file itself)
s = set()
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
                s.add(str(d[k]))
                break
results.append(("repo_data_jsonl", len(s), len(hcids & s)))

# 4. corpus-side pool manifests
for p in sorted(list((DATA_ROOT / "pseudo").glob("*_meta.pkl")) +
                list((DATA_ROOT / "features").glob("*_meta.pkl"))):
    try:
        m = pickle.load(open(p, "rb"))
    except Exception:
        continue
    if hasattr(m, "columns") and "cid" in m.columns:
        cs = set(m["cid"].astype(str))
        inter = len(hcids & cs)
        tag = p.parent.name + "/" + p.name
        if len(cs) > 500000:
            tag += " [full-corpus meta]"
        results.append((tag, len(cs), inter))

# 5. text-level dup vs gold600 + lab2000
gtexts = set()
for l in open(REPO / "data/reviews_600_base.jsonl", encoding="utf-8"):
    gtexts.add(re.sub(r"\s+", "", json.loads(l)["text"].strip()))
lt = set()
lf = LAB / "lab2000_final.jsonl"
if lf.exists():
    for l in open(lf, encoding="utf-8"):
        lt.add(re.sub(r"\s+", "", json.loads(l)["text"].strip()))
print("text∩gold600:", len(htexts & gtexts),
      " text∩lab2000:", len(htexts & lt))

print("\n=== cid overlap audit ===")
bad = 0
for name, n, inter in results:
    full = "full-corpus" in name
    flag = "" if (inter == 0 or full) else "  <-- OVERLAP!"
    if inter and not full:
        bad += 1
    print(f"{name:55s} pool={n:>7} overlap={inter}{flag}")
print("\nVERDICT:", "CLEAN" if bad == 0 else f"{bad} POOLS CONTAMINATED")
sys.exit(1 if bad else 0)

"""Deduplicate gold-600 by exact review text -> reviews_600_dedup592.jsonl.

Keeps the first occurrence of each stripped text (600 -> 592 rows),
writes a dropped->kept mapping, and subsets the row-aligned frozen
feature / corpus-detector arrays used by the OOF pipeline.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")

SRC = "reviews_600_base.jsonl"
DST = "reviews_600_dedup592.jsonl"
MAP = "dedup_gold600_map.json"

rows = [json.loads(l) for l in open(os.path.join(DATA, SRC), encoding="utf-8")]
assert len(rows) == 600

seen = {}
keep_idx, drop = [], []
for i, r in enumerate(rows):
    key = r["text"].strip()
    if key in seen:
        drop.append({"row": i, "id": r.get("id"), "dup_of_row": seen[key],
                     "dup_of_id": rows[seen[key]].get("id"),
                     "label": r.get("label_semantic"),
                     "text": r["text"][:80]})
    else:
        seen[key] = i
        keep_idx.append(i)

assert len(keep_idx) == 592 and len(drop) == 8, (len(keep_idx), len(drop))

kept = [rows[i] for i in keep_idx]
with open(os.path.join(DATA, DST), "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
print("kept label dist:", Counter(r["label_semantic"] for r in kept))
print("dropped ids:", [d["id"] for d in drop])

with open(os.path.join(DATA, MAP), "w", encoding="utf-8") as f:
    json.dump({"source": SRC, "n_source": 600, "n_kept": 592,
               "keep_rows": keep_idx, "dropped": drop}, f,
              ensure_ascii=False, indent=2)

# subset row-aligned arrays (row order == file order)
keep_idx = np.array(keep_idx)
for name in ["feats_mb13b_m256.npy", "cascade_pneu.npy"]:
    src = os.path.join(OUT, name)
    arr = np.load(src)
    assert arr.shape[0] == 600, (name, arr.shape)
    dst_name = name.replace(".npy", "_dedup592.npy")
    np.save(os.path.join(OUT, dst_name), arr[keep_idx])
    print("wrote", dst_name, arr[keep_idx].shape)

# sanity: no remaining duplicate texts
texts = [r["text"].strip() for r in kept]
assert len(set(texts)) == 592
print("OK: 592 unique texts")

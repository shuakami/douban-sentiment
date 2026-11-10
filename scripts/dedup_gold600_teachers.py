"""Build dedup-aligned teacher-probability files for the gold-600 dedup rerun.

ens_3class_allch / ens_neu_teacher store probs for all 600 rows;
ens_binary_allch2 stores probs for the 508 non-neutral rows only.
Each output keeps the same JSON envelope but with probs subset to the
592-row dedup file (and the 503-row binary subset) so distill.py's
len(teacher_probs)==len(y) assert and distill_head.py's positional
indexing stay aligned.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")

mp = json.load(open(os.path.join(DATA, "dedup_gold600_map.json"), encoding="utf-8"))
keep_idx = mp["keep_rows"]  # 592 positions in original file order

rows = [json.loads(l) for l in open(os.path.join(DATA, "reviews_600_base.jsonl"), encoding="utf-8")]


def subset(src_name, dst_name, idx):
    d = json.load(open(os.path.join(OUT, src_name), encoding="utf-8"))
    d["probs"] = [d["probs"][i] for i in idx]
    d["n"] = len(idx)
    d["dedup_of"] = src_name
    with open(os.path.join(OUT, dst_name), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    print(dst_name, len(idx), "x", len(d["probs"][0]))


# 3class + neu teachers: row-aligned with all 600 rows -> subset by keep_idx
subset("ens_3class_allch.json", "ens_3class_allch_dedup592.json", keep_idx)
subset("ens_neu_teacher.json", "ens_neu_teacher_dedup592.json", keep_idx)

# binary teacher: aligned with non-中性 rows of the original 600.
non_neu_pos = [i for i, r in enumerate(rows) if r["label_semantic"] != "中性"]
rank = {pos: k for k, pos in enumerate(non_neu_pos)}
dedup_non_neu = [p for p in keep_idx if rows[p]["label_semantic"] != "中性"]
bin_idx = [rank[p] for p in dedup_non_neu]
assert len(bin_idx) == 503, len(bin_idx)
subset("ens_binary_allch2.json", "ens_binary_allch2_dedup592.json", bin_idx)
print("OK")

"""De-identify publicly released Douban review texts.

Moviedata-10M terms and Douban's terms do not permit us to redistribute
raw comment text.  This script rewrites the tracked data artifacts so that
they contain labels, identifiers and SHA-256 text hashes only:

  * ``"text"``     -> ``"text_sha256"``   (JSONL / JSON dicts, recursive)
  * ``"content"``  -> ``"content_sha256"`` (Aliyun request dumps)
  * TSV ``text`` column -> ``text_sha256`` column
  * annotation.html embedded ``DATA`` -> ``text_sha256`` (``cid`` kept)

Original full-text files are preserved once under ``data/_fulltext/`` and
``annotation/_fulltext/`` (git-ignored) so local experiments can restore
them with ``scripts/restore_texts.py``.  Run this script again before
committing if the working copies were restored.

Synthetic texts in ``data/synth_*.jsonl`` are model-generated and are NOT
touched.
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "_fulltext"

JSONL_TARGETS = [
    "data/reviews_600.jsonl",
    "data/reviews_600_base.jsonl",
    "data/reviews_600_dedup592.jsonl",
    "data/reviews_600_synth1.jsonl",
    "data/label_semantic.jsonl",
    "data/label_rule.jsonl",
    "data/reviews_gold_al1000.jsonl",
    "data/human400_labels.jsonl",
    "data/lab2000/lab2000_final.jsonl",
    "data/lab2000/fresh500.jsonl",
    "data/lab2000/friend_IAA_200.jsonl",
    "data/lab500_consensus.jsonl",
    "data/lab500_llm.jsonl",
]
JSON_TARGETS = [
    "data/dedup_gold600_map.json",
    "data/aliyun/aliyun_3class.json",
    "data/aliyun/aliyun_binary.json",
    "data/aliyun/aliyun_merged.json",
]
TSV_TARGETS = ["data/suspect_labels.tsv"]
HTML_TARGETS = ["annotation/annotation.html"]

TEXT_KEYS = {"text": "text_sha256", "content": "content_sha256"}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def backup(rel: str) -> None:
    src = ROOT / rel
    dst = FULL / rel
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def strip_obj(obj):
    """Recursively replace text/content string values with sha256 fields."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in TEXT_KEYS and isinstance(v, str):
                out[TEXT_KEYS[k]] = sha(v)
            else:
                out[k] = strip_obj(v)
        return out
    if isinstance(obj, list):
        return [strip_obj(x) for x in obj]
    return obj


def process_jsonl(rel: str) -> int:
    path = ROOT / rel
    lines, changed = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            lines.append(line)
            continue
        row = json.loads(line)
        new = strip_obj(row)
        if new != row:
            changed += 1
        lines.append(json.dumps(new, ensure_ascii=False))
    if changed:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def process_json(rel: str) -> bool:
    path = ROOT / rel
    obj = json.loads(path.read_text(encoding="utf-8"))
    new = strip_obj(obj)
    if new != obj:
        path.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        return True
    return False


def process_tsv(rel: str) -> bool:
    path = ROOT / rel
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return False
    header = lines[0].split("\t")
    if "text" not in header:
        return False
    ti = header.index("text")
    header[ti] = "text_sha256"
    out = ["\t".join(header)]
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) > ti and cols[ti]:
            cols[ti] = sha(cols[ti])
        out.append("\t".join(cols))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True


def process_html(rel: str) -> bool:
    path = ROOT / rel
    html = path.read_text(encoding="utf-8")
    marker = "const DATA = "
    i = html.find(marker)
    if i < 0:
        return False
    start = i + len(marker)
    while html[start] in " \t":
        start += 1
    if html[start] != "[":
        return False
    data, end = json.JSONDecoder().raw_decode(html[start:])
    if not isinstance(data, list):
        return False
    new = strip_obj(data)
    if new == data:
        return False
    html = html[:start] + json.dumps(new, ensure_ascii=False) + html[start + end:]
    path.write_text(html, encoding="utf-8")
    return True


def main() -> None:
    total = 0
    for rel in JSONL_TARGETS + JSON_TARGETS + TSV_TARGETS + HTML_TARGETS:
        p = ROOT / rel
        if not p.exists():
            print(f"skip (missing): {rel}")
            continue
        backup(rel)
        if rel.endswith(".jsonl"):
            n = process_jsonl(rel)
        elif rel.endswith(".tsv"):
            n = int(process_tsv(rel))
        elif rel.endswith(".html"):
            n = int(process_html(rel))
        else:
            n = int(process_json(rel))
        total += n
        print(f"{rel}: {'stripped' if n else 'already clean'}")
    print(f"done ({total} rows/objects updated)")


if __name__ == "__main__":
    sys.exit(main())

# Data License and Provenance

## What is released

The files under `data/` contain **labels, identifiers, and SHA-256 text
hashes only**. Raw Douban review texts are **not redistributed** in this
repository.

- `reviews_600*.jsonl`, `label_*.jsonl` — the `text` field has been replaced
  by `text_sha256` (SHA-256 of the UTF-8 string). Row `id`, `star`,
  `label_semantic`, `label_rule` and all other fields are unchanged.
- `human400_labels.jsonl`, `lab2000/*.jsonl`, `lab500_*.jsonl`,
  `reviews_gold_al1000.jsonl` — same treatment; `cid` / pool index `i` /
  `n` fields are preserved so rows can be re-joined to source texts.
- `data/aliyun/*.json` — `content` replaced by `content_sha256`.
- `data/suspect_labels.tsv` — `text` column replaced by `text_sha256`.
- `data/dedup_gold600_map.json` — `dropped[].text` replaced by
  `text_sha256`.
- `annotation/annotation.html` — embedded `DATA` keeps `cid`, `len` and
  `text_sha256`.
- `data/synth_*.jsonl` — **model-generated synthetic reviews** created for
  this project; these texts are our own output and remain in full.
- `data/lab2000/ds_*.jsonl`, `qwen_*.jsonl`, `*.npy` — labels, probabilities
  and aggregate statistics; they never contained review text.

## Why texts are withheld

Two upstream constraints apply:

1. **Moviedata-10M** (the 910,998-review corpus, csuldw/CSU 2019,
   https://github.com/csuldw/AntSpider) is released for research use with
   attribution required, and its terms state that users may not
   redistribute the data without separate permission. Corpus-derived
   subsets in this repository therefore ship as labels + `cid`/hashes.
2. **Douban user comments** are platform content collected without
   platform authorization (see the paper's Ethical Considerations /
   Appendix H). Deleting usernames and comment IDs does not make raw
   comment text non-identifiable, so we do not republish it.

## How to obtain the texts

- **Corpus-derived sets** (human400, lab2000 pools, fresh500, al-pools):
  obtain Moviedata-10M from the official site
  (http://moviedata.csuldw.com, password on request) and join on `cid`
  (human400) or pool index `i`/`n` (lab2000, see `data/lab2000/README.md`).
  `text_sha256` lets you verify you recovered the exact string.
- **Gold-600** (`reviews_600*.jsonl`, `label_*.jsonl`): these are reviews
  from a single Douban film page. Re-collect that page yourself, or join
  by `text_sha256` against any Douban comment dump you legitimately hold.
- **Local mirror**: if you are the authors, the pre-strip originals are in
  the git-ignored `data/_fulltext/` and `annotation/_fulltext/` directories.
  Restore them for local runs with `python scripts/restore_texts.py`, and
  re-run `python scripts/deidentify_public_texts.py` before committing.

## Terms

1. Use for research and non-commercial purposes only.
2. Do not redistribute review texts; if you obtain source data yourself,
   comply with Moviedata-10M / Douban terms.
3. Attribute Moviedata-10M to csuldw/CSU when using corpus-derived data.
4. The authors give no warranty regarding accuracy or fitness; labels —
   especially LLM-consensus and pseudo-labels — are noisy by construction
   (see the paper's provenance analysis).

# Data License and Provenance

The files under `data/` are released for **non-commercial research purposes
only**. They are NOT covered by the MIT license in `LICENSE`, which applies
to source code only.

## Provenance

- `reviews_600*.jsonl`, `label_*.jsonl` — Douban movie short reviews
  collected by the authors for this research. Douban user comments are
  platform content used **without authorization**; we release a minimized
  subset (text + semantic labels, user identifiers stripped) consistent with
  research-use fair-dealing norms.
- `lab2000/`, `lab500_*.jsonl`, `fresh500`-related artifacts — derived from
  the same Douban review pool; LLM-generated labels are marked as such in the
  accompanying metadata.
- `human400_labels.jsonl` — 400 corpus reviews with blind single-annotator
  semantic labels (no star ratings shown at annotation time); the
  contamination-free final test set.
- The 910,998-row training corpus derives from **Moviedata-10M** by
  csuldw (CSU, 2019; https://github.com/csuldw/AntSpider), released for
  research use with attribution required. **The raw corpus is not
  redistributed** in this repository; only derived evaluation/benchmark
  subsets (~5.0K reviews, including human400) are included.

## Terms

1. Use for research and non-commercial purposes only.
2. Do not redistribute the review texts outside research contexts.
3. Attribute Moviedata-10M to csuldw/CSU when using corpus-derived data.
4. The authors give no warranty regarding accuracy or fitness; labels —
   especially LLM-consensus and pseudo-labels — are noisy by construction
   (see the paper's provenance analysis).

# human400 blind annotation kit

Materials used to produce `data/human400_labels.jsonl`, the
contamination-free human test set reported in the paper
(§ "The clean human test: human400").

- `ANNOTATION_GUIDELINES.md` — written labeling rules (positive /
  neutral / negative + optional flags).
- `annotation.html` — self-contained offline annotation page; the 400
  items are embedded, star ratings are hidden, progress persists in
  `localStorage` (`ood_annot_v1`), and results export as
  `ood_human_labels.jsonl`.

The item set was sampled from the 911K-row corpus with length
stratification after excluding every known training/evaluation pool
by comment ID; a post-hoc audit verified zero identifier overlap.
Labels are single-annotator (no human–human IAA available).

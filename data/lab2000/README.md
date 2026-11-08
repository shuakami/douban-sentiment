# lab2000 benchmark + Stage-2 artifacts

Data for the OOD / active-learning study in `PAPER.md` and `thesis/`.
Review texts are Douban content (research use only, per the upstream
Moviedata-10M license); labels are our own annotation work.

## Files

| File | Contents |
|---|---|
| `lab2000_final.jsonl` | **The benchmark.** 2,000 reviews: `ood1000` (1,000 uniform random corpus rows, held-out test) + `al1000` (1,000 margin/entropy-selected rows, training). Fields: `text`, `rating`, `label_a`/`label_b` (two blind LLM passes), `agree`, `final` (authoritative label), `final_source` (`user` = human-adjudicated, `consensus` = LLM two-pass agreement). 479 rows are human-adjudicated. |
| `fresh500.jsonl` | 500 uniformly-sampled corpus reviews disjoint from all training/eval texts; labels come from the DeepSeek teacher (see `ds_deepseek_flash_fresh500_labels.jsonl`). |
| `ds_deepseek_flash_labels.jsonl` | DeepSeek-V4.1-Flash zero-shot 3-class labels for ood1000 (992/1000 parseable). |
| `ds_deepseek_v4_pro_labels.jsonl` | DeepSeek-V4-Pro zero-shot 3-class labels for ood1000 (950/1000 parseable). |
| `ds_deepseek_flash_bin_labels.jsonl` | DeepSeek-V4.1-Flash binary (pos/neg) labels for the 673 non-neutral ood1000 rows (661 parseable). |
| `ds_deepseek_flash_al1000_labels.jsonl` | DeepSeek labels for al1000 (987/1000 parseable) — the LLM-teacher substitute for human adjudication. |
| `ds_deepseek_flash_al5000_labels.jsonl` | DeepSeek labels for al5000 (next 5,000 AL rows; 4,906 parseable). |
| `ds_deepseek_flash_al5000b_r2_labels.jsonl` | DeepSeek labels for the second 5,000-row AL batch (re-run with `reasoning_effort=low`, 5,000/5,000 parseable). Supersedes the failed first pass. |
| `ds_deepseek_flash_al3k_r1_labels.jsonl` / `_r2_labels.jsonl` | DeepSeek labels for two 3,000-row disagreement-mined AL batches (al14kds / al17kds rounds). |
| `ds_deepseek_flash_fresh500_labels.jsonl` | DeepSeek labels for fresh500. |
| `qwen_ood1000_labels.jsonl` / `qwen_al5000b_labels.jsonl` | Local Qwen3-4B teacher labels (label-logprob scoring) for ood1000 and the second AL batch — the open-teacher ablation. |
| `qwen3_4b_ood1000_probs.npy` / `qwen3_4b_al5000b_probs.npy` | Raw 3-class probability arrays for the Qwen runs. |
| `friend_IAA_200.jsonl` / `friend_IAA_map.json` | The 200-row second-human IAA subset: texts given to the second annotator and the `n → (set, i)` mapping back to `lab2000_final.jsonl`. |
| `friend_IAA_README.md` | The annotation instructions given to the second human annotator. |
| `iaa_friend_user.json` | Aggregate IAA statistics: second-human vs released labels 71.5% / κ=0.568 (95% CI [0.47, 0.66]); passA-vs-passB 83.3% / κ=0.74. Per-row second-human labels were collected via the README protocol; aggregates are released here. |

## Annotation protocol detail

`label_a`/`label_b` are two independent passes of the **same** LLM
annotator agent with **different few-shot example sets**
(`fewshot_A.json` / `fewshot_B.json`, built from gold-600 semantic
labels), blind to star ratings and to each other's output — equivalent
to two annotators with different rubric views, not two different
models. All 335 pass-disagreements plus 144 spot-checked agreements
were adjudicated by the author (`final_source == "user"`, 479 rows);
the remaining 1,521 rows use two-pass consensus.

## Notes

- `final` in `lab2000_final.jsonl` is the authoritative label;
  `label_a`/`label_b` are pre-annotations, not ground truth.
- `ood1000` is a held-out test set: no model in the paper trains on it.
  `al1000` and the `al*` files are training data.
- The 479 human-adjudicated rows (`final_source == "user"`) are
  disagreement-enriched; do not treat them as a representative test
  distribution.
- LLM labels carry the teachers' biases; they are pseudo-labels, not
  ground truth (see the teacher-quality analysis in the paper).

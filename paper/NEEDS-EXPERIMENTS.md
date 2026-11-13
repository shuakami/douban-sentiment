# NEEDS-EXPERIMENTS — prioritized plan for the reframed SRW paper

Companion to `VERIFICATION-REPORT.md`. Every verdict below is grounded in
artifacts that were actually inspected; nothing here assumes data that does
not exist. Verdicts: `FEASIBLE-NOW` (existing artifacts or local compute
suffice), `NEEDS-HUMAN` (new human annotation required), `NEEDS-API` (new
LLM-API labeling required), `IMPOSSIBLE` (cannot be done with current
data, regardless of effort).

Legend for "who": **HUMAN** = requires a person; **AGENT** = an agent can
execute it now (GPU work on the local RX 9070 GRE ROCm box, serialized —
never two jobs at once; kill the demo first, it holds ~5 GB VRAM; set
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HOME=D:/Project/douban-sentiment-env/hf`);
**BOTH** = human + agent.

---

## P0 — must be done by a human (cannot be substituted)

### A. Second independent annotator on human400 — `NEEDS-HUMAN`

- **What:** all 400 rows of `data/human400_labels.jsonl` re-labelled by a
  second annotator, blind (no model predictions, no existing labels, no
  star rating shown), under the same written guidelines already released
  under `annotation/` in both `douban-sentiment/` and `srw-wt/`.
- **Why needed:** human400 is the paper's only identifier-disjoint test
  and is single-annotator; the paper itself lists "a second annotator on
  human400" as a conclusion-changing experiment (`draft_6.tex:355-357`).
- **Spec:** preserve `i`, `cid`, `text_sha256` per row; store labels in
  the same schema (`label`, optional `flags`); do NOT show ratings or any
  model output. After collection: raw agreement + Cohen's κ overall and
  per class, confusion matrix vs. annotator-1, then adjudicate
  disagreements (third pass or author consensus) and report both
  pre- and post-adjudication agreement.
- **Effort:** ~2–3 h for the annotator (400 short reviews, median ~24
  chars); minutes of agent compute for IAA statistics.
- **Caveat:** `friend_IAA_200` does **not** cover this — it is a 200-row
  ood1000 subset and its per-row second-human labels were never
  returned (only aggregates in `iaa_friend_user.json`).

### B. Clean paired human-vs-LLM sample on ood1000 — `NEEDS-HUMAN`

- **What:** 300–400 ood1000 rows blind-labelled by a human **with LLM
  outputs hidden**, so human labels can be paired row-by-row against the
  already-stored DeepSeek-Flash and Qwen3-4B predictions.
- **Why needed:** the professor-requested "paired same-text human vs LLM"
  comparison. Existing coverage is insufficient: `friend_IAA_200` (200
  rows) has no released per-row labels; the 479 adjudicated rows were
  labelled with student predictions visible (anchoring, disclosed in the
  paper) and are disagreement-enriched.
- **Spec:** stratified sample across LLM label × confidence (use stored
  `ds_deepseek_flash_labels.jsonl`, `qwen_ood1000_labels.jsonl`,
  `lab2000_final.jsonl` fields `label_a/label_b/conf_a/conf_b`); same
  guidelines as human400; record `i`/`cid`/`text_sha256`; LLM arms are
  reused from existing artifacts — **no new API calls needed** unless a
  different teacher/prompt is wanted (then the LLM arm becomes
  `NEEDS-API`).
- **Effort:** ~2–4 h annotator time for 300–400 rows; agent computes
  human-vs-DS, human-vs-Qwen agreement and re-scores the benchmark split.
- **Note:** this also produces the first clean human–human IAA number for
  the corpus domain and directly tests the "consensus inflation" claim
  (currently ~26 pp on adjudicated rows) on an unbiased sample.

### D. Multi-movie rating–sentiment study — `NEEDS-HUMAN` (prerequisite for F)

- **What:** new semantic labels on reviews drawn from **multiple films**,
  with star ratings preserved — e.g. 400–800 rows across 8–15 films,
  film-stratified, length-stratified within film, blind single-annotator
  pass under the same guidelines.
- **Why needed:** gold-600 is one film page; `mid` exists for every
  corpus row but there is **no multi-movie semantic ground truth**
  anywhere in `data/` or `E:/GB77_data`. Corpus `mid`/`rating` fields
  make sampling trivial (`sample_lab2000.py`/`sample_fresh500.py` are
  directly adaptable).
- **Effort:** ~4–8 h annotator time depending on size; agent builds the
  sampler and eval harness.
- **Until this exists:** the paper must keep saying gold-600 is
  single-film and must not claim cross-movie generalization (it already
  does — `srw.tex:83-85`, `draft_4_5.tex:78-81`).

---

## P1 — agent-executable now (no human, no API)

### C-small. Random-vs-uncertainty AL sampling, small scale — `FEASIBLE-NOW` — **DONE (2026-11-10)**

- **Status:** COMPLETE. Arms: `E:/GB77_data/lab2000/ctrl500_random_ds.jsonl`
  (fresh500 uniform sample, DeepSeek labels) vs `ctrl500_al_ds.jsonl` (first
  500 margin/entropy-selected rows of `extra_al17k_ds`, same teacher family).
  Verified: 500 rows each, 0 text overlap with human400, all 500 AL rows ⊂
  al17k head. Trained with the al17k SFT recipe (gold-600 + 500,
  `scripts/sft_human400_al500ctrl.py`; 3class e4 × seeds 42/123/456/789/999,
  binary e8 × 42/123/789/999); probs in
  `E:/GB77_data/human400/human400_sft_probs_ctrl500_{random,al}/`, metrics in
  `E:/GB77_data/human400/ctrl500_eval.json`.
- **Result (human400, mean-seed probs, argmax):**
  - 3-class: random acc 60.50 / macroF1 56.89; AL acc 61.25 / macroF1 55.31.
    McNemar p=0.83; bootstrap 95% CI on acc diff (AL−random) [−3.8, +5.3]pp.
    Per-class: AL arm trades negative recall (30.8% vs 41.5%) for positive
    recall (78.2% vs 55.8%).
  - Binary (n=221 non-neutral): random 79.19/76.14; AL 77.38/71.40.
- **Interpretation:** at n=500 the sampling strategy makes no reliable
  difference — the +9pp human400 gain at 17K is consistent with a *scale*
  effect of teacher-labelled data, not with uncertainty selection per se.
  Paper wording should say exactly this and not overclaim AL causality.
- **Caveats:** teacher labels both arms (no human ground truth on the 500);
  AL arm oversamples uncertain/neutral-ish rows (label dist differs);
  fresh500 partially exposed (disclosed).

- **What the professor asked:** random vs uncertainty AL sampling at
  equal budget.
- **What exists:** every AL pool (al1000, al5000, al5000b, al3k rounds,
  al17k) is margin/entropy/disagreement-sampled; the only uniform-random
  teacher-labelled pool is **fresh500** (500 rows, DeepSeek-Flash
  labels, `sample_fresh500.py` confirms uniform sampling).
- **Feasible version (n=500 per arm):**
  1. Random arm = fresh500's 500 DS-Flash labels (already stored in
     `ds_deepseek_flash_fresh500_labels.jsonl`; SFT probs already exist
     in `fresh500_sft_probs_al17k/`).
  2. Uncertainty arm = a 500-row subset of `extra_al17k_ds.jsonl` (pure
     DS-labelled, margin/entropy/disagreement-sampled), matched on
     length/rating if desired.
  3. Train two identical SFT students (`sft_human400_al17k.py` recipe,
     gold+500), evaluate on **human400** — fresh500 texts are
     identifier-disjoint from human400, so training on its labels does
     not contaminate the test.
  4. Cost: 2 student trainings (~minutes each on the local GPU) + eval.
- **Caveats to state:** n=500 is far below the 17K operating point; the
  random arm's labels come from the same teacher family (convention
  alignment affects both arms equally, which is what we want); using
  fresh500 labels for training consumes it as an eval set — acceptable
  because it is already partially exposed and LLM-labelled.
- **Full-scale version (17K random vs 17K uncertainty): `NEEDS-API`** —
  would require ~17K new DeepSeek labels at the original cost.

### E. Neutral specialist vs class-weighting / focal / oversampling / threshold — `FEASIBLE-NOW` — **DONE (2026-11-13)**

**New compute completed (agent, GPU-serialized):** focal loss
($\gamma{=}2$ on the hard-CE term) and deterministic class-balanced
over-sampling (minority classes repeated round-robin to the majority
count inside each train fold) were added as optional flags to
`scripts/distill.py` and `scripts/train_final_distill.py`
(`--focal_gamma`, `--balance_classes`; defaults off, deployed recipe
unchanged: $\alpha{=}0.5$, $\tau{=}2$, e8, m256, seed 42,
class\_weight=1). 3-class student only, gold-600.

Results (gold-600 5-fold OOF / human400 3-class acc, gold-only
student):

| variant | OOF acc | OOF macroF1 | human400 acc | human400 macroF1 |
|---|---|---|---|---|
| base (deployed recipe) | 0.8017 | 0.7418 | 0.5275 | 0.4928 |
| focal $\gamma{=}2$ | 0.8100 | 0.7512 | 0.5000 | 0.4705 |
| balance\_classes | 0.8083 | 0.7457 | 0.4875 | 0.4676 |

Marginalized binary on human400 (pos/(pos+neg) on 221 non-neutral
rows): base 0.6923, focal 0.6787, bal 0.7014 acc.

Artifacts: `out/distill_macbert_3c_a50_t2_e8_m256_{focal2,bal}.json`;
`E:/GB77_data/human400/human400_sft_probs_eexp/{base,focal,bal}.npy`;
models `models/eexp_{base,focal,bal}`; driver
`scripts/e_focal_oversample.sh`; log
`E:/GB77_data/human400/e_focal_oversample.log`.

**Verdict:** both variants improve in-domain OOF (+0.4--0.9pp) but
degrade identifier-disjoint human400 (−1.4 to −4.0pp acc) — another
instance of the paper's central theme that in-domain gains need not
transfer. Written into `appendix.tex` negative-results list and the
`draft_4_5.tex` rejection sentence.

Already-run evidence (no new compute):
- **Class weighting:** `distill_macbert_3c_a50_t2_e8_m256_cw` and the
  `swp_*_cwbal` / `*_cw` OOF JSONs in `out/`; rejected in the round-3
  sweep (AGENTS.md; appendix negative-results list covers "class
  weights" inside the 3-class student grid). Claim in
  `draft_4_5.tex:40-42` is supported indirectly.
- **Threshold / cascade:** `scripts/tune_threshold.py` +
  `eval_cascade.py` split-half collapse already documented; binary
  threshold grid runnable on stored OOF probs in minutes.
- **Neutral specialist:** extensively tested and rejected —
  `eval_neu_decomp.py`, corpus detector, TF-IDF specialist, frozen-MacBERT
  neu head, all in the appendix negative list.

### G. Full fold-isolated pipeline re-run — `FEASIBLE-NOW` (compute-bound)

- **What exists:** `scripts/dedup_retrain.sh` already completed the
  dedup-592 α∈{0.5,1.0} pipeline; `eval_a100_pipeline.py` /
  `eval_dedup_pipeline.py` rebuild ensemble metrics from stored OOF
  probs in seconds; `crossfit_gold600.py` implements fold-held-out SFT.
- **What is missing for a *fully* fold-isolated rerun:**
  1. **Fold-isolated teachers (Leak B fix):** teacher and student
     currently share the same 5-fold partition, so teacher soft targets
     on a student's val fold come from models trained on that fold.
     Fix = retrain teacher channels under a shifted/second partition or
     leave-one-fold-out teachers — roughly 5× teacher-side cost.
  2. **Corpus neutral detector (Leak A fix):** `train_bigdata.py
     --task neu` on 115K pseudo rows is the heavy job (hours on this
     GPU); a fold-isolated version must exclude gold-val-fold-derived
     pseudo rows — feasible but the pseudo-label provenance chain makes
     "clean" subtle (pseudo labels descend from stu3c fit on all 600).
  3. **Inner-fold fusion-weight selection** instead of reusing deployed
     weights.
- **Estimate:** serialized GPU, plausibly 1–2 days wall-clock including
  the corpus-detector retrain; a cheaper scoped version (α=1.0 + dedup +
  shifted-partition teachers for student channels only, corpus channel
  reported as a bound) is ~hours and matches what the paper already
  labels "conservative lower bounds".
- **Hardware confirmed:** `torch 2.11.0+rocm7.13.0`,
  `torch.cuda.is_available()=True`, AMD Radeon RX 9070 GRE.

---

## P2 — impossible or not worth it

### F. Cross-movie generalization split — `IMPOSSIBLE` (with current labels)

- gold-600 is a single film page; no multi-movie semantic labels exist.
  Corpus `mid` fields are not labels. This becomes possible only after
  experiment D produces multi-movie annotations — and even then it is a
  *new* study, not a re-analysis.
- **Recommended paper phrasing (already largely present):** "gold-600
  is drawn from a single film page, so in-domain numbers are not
  cross-domain estimates" (`srw.tex:83-85`); optionally add "cross-movie
  generalization is not estimable from our labels and is left to future
  work" near the limitations paragraph.

### Full-scale random-vs-uncertainty AL — `NEEDS-API` (see C)

- ~17K fresh random-sample teacher labels at original DeepSeek cost.
  Only worth it if the professor specifically wants the causal AL claim
  at scale; the n=500 version answers the directional question cheaply.

---

## Suggested execution order

1. **Now (agent):** C-small random-vs-uncertainty 500-row comparison;
   E focal/oversampling student runs; optionally start G's cheap scoped
   version. All GPU work serialized.
2. **Human (parallel):** A (second annotator, human400) and B (paired
   ood1000 sample) can be annotated by the same person in one sitting
   (~5–7 h total) since guidelines and the offline annotation page
   already exist.
3. **Human (bigger):** D multi-movie annotation — only if the professor
   wants the cross-movie claim; otherwise keep the single-film
   limitation wording.
4. **Paper updates after results:** whichever of A/B/C/E/G produce
   numbers should replace the corresponding "future work" sentences in
   `srw.tex:75-78` and `draft_6.tex:355-357`; F stays as limitation
   wording unless D is funded.

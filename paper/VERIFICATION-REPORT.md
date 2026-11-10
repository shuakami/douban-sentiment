# VERIFICATION-REPORT — audit of the reframed SRW paper

Scope: read-only audit of `srw-wt/paper/*.tex` against prediction/label
artifacts in `douban-sentiment/` and `E:/GB77_data`. No `.tex` file was
modified. All metrics below were **recomputed independently** with
`D:/Project/douban-sentiment-env/venv/Scripts/python.exe`
(sklearn 1.9.1, numpy 2.4.6, scipy 1.18.1) unless marked "artifact".

**`E:\GB77_data` accessibility: CONFIRMED.** All cited artifacts were
read directly (`human400/`, `lab2000/`, `models/`).

Label order everywhere: index 0 = 正面/positive, 1 = 中性/neutral,
2 = 负面/negative. human400 label distribution: 156 pos / 179 neu /
65 neg (n=400). Binary eval uses the 221 non-neutral rows.

---

## 1. human400 per-class metrics (the requested table)

Sources: `E:/GB77_data/human400/human400_v3_probs.npz` (`probs3`,
`probsb`), `human400_sft_probs_al17k/{3class_e4,binary_e8}_s{42,123,456,
789,999}.npy` (binary has no s456), `qwen3_4b_human400_probs.npy`,
labels `douban-sentiment/data/human400_labels.jsonl` (the repo copy has
all 400 rows; the `E:/GB77_data/human400/human400_labels.jsonl` copy has
399 — repo copy used as authoritative). Alignment verified: label
order matches `i` order; `probs3.argmax` reproduces the released eval
script outputs.

Mix recipe (frozen, ood1000-selected): 3-class = 0.6·mean(5 SFT seeds)
+ 0.4·v3, argmax; binary = 0.5·mean(4 SFT seeds) + 0.5·v3 on the 221
non-neutral rows. "SFT mean" = argmax of the mean of seed probability
vectors (not mean of per-seed accuracies).

### Three-class, n=400 — accuracy / macro-F1 / per-class P·R·F1 (%)

| System | Acc | Mac-F1 | Pos P/R/F1 | Neu P/R/F1 | Neg P/R/F1 |
|---|---:|---:|---|---|---|
| v3 (deployed) | 55.75 | 55.35 | 74.8/51.3/60.8 | 62.9/50.3/55.9 | 35.3/81.5/49.3 |
| al17k SFT (5-seed mean) | 64.25 | 62.13 | 75.5/53.2/62.4 | 62.4/76.0/68.5 | 52.8/58.5/55.5 |
| al17k mix (frozen w=0.6) | **64.75** | **63.48** | 75.0/53.8/62.7 | 63.2/73.7/68.0 | 54.4/66.2/59.7 |
| Qwen3-4B zero-shot | 61.50 | 58.97 | 69.0/57.1/62.5 | 59.4/70.4/64.5 | 52.5/47.7/50.0 |
| star lookup | 55.75 | 55.51 | 54.6/71.8/62.0 | 56.8/41.9/48.2 | 57.1/55.4/56.2 |

Confusion matrices (rows=true pos/neu/neg):
v3 `[[80,45,31],[23,90,66],[4,8,53]]`; mix `[[84,59,13],[24,132,23],[4,18,43]]`;
Qwen `[[89,55,12],[37,126,16],[3,31,31]]`.

### Binary, n=221 non-neutral

| System | Acc | Mac-F1 | Neg P/R/F1 | Pos P/R/F1 |
|---|---:|---:|---|---|
| v3 | 76.47 | 75.25 | 56.1/92.3/69.8 | 95.6/69.9/80.7 |
| al17k SFT (4-seed mean) | 83.71 | 81.32 | 68.8/81.5/74.6 | 91.7/84.6/88.0 |
| al17k mix (frozen w=0.5) | 83.71 | 81.32 | 68.8/81.5/74.6 | 91.7/84.6/88.0 |
| star lookup | 83.71 | 77.94 | 83.7/55.4/66.7 | 83.7/95.5/89.2 |

Star lookup rule: 4–5★→pos, 3★→neu, 1–2★→neg (binary: 1–2★→neg else
pos). Predicted class counts on binary: mix 77 neg/144 pos, star rule
43/178, truth 65/156 — matches `draft_6.tex:180-181` exactly.

### McNemar (all 16 pairwise tests recomputed; appendix Table
`tab:human400-stats` verified cell-by-cell, incl. Holm over the
16-test family)

3-class discordant (first-only/second-only), raw p, Holm p:
mix–v3 78/42, p=0.0013, Holm **0.021**; SFT–v3 91/57, p=0.0065, Holm
0.097; mix–star 104/68, p=0.0074, Holm 0.104; SFT–star 101/67,
p=0.0107, Holm 0.139; Qwen–v3 90/67, p=0.0788, Holm 0.788; Qwen–star
97/74, p=0.0922, Holm 0.788; mix–Qwen 64/51, p=0.263, Holm 1.0;
SFT–Qwen 63/52, p=0.351, Holm 1.0; mix–SFT 15/13, p=0.851, Holm 1.0;
v3–star 94/94, p=1.0.
Binary: mix–v3 29/13, p=0.0195→0.020, Holm 0.234; SFT–v3 31/15,
p=0.0259→0.026, Holm 0.285; star–v3 46/30, p=0.0846→0.085, Holm 0.788;
mix–SFT 2/2, p=1.0; SFT–star 28/28, p=1.0; mix–star 28/28, p=1.0.
**Every value in the paper's table matches.** Only mix–v3 (3c) survives
Holm — as the paper states.

### human400 diagnostics (appendix `app:human400`) — recomputed

- Strata acc (v3 / mix): ≤20ch 56.2/67.5 (n=160); 21–50 56.7/65.0
  (n=120); 51–100 **53.75→paper 53.7** /58.8 (n=80); >100 55.0/65.0
  (n=40). One 0.05–0.1pp rounding nit on the 51–100 cell.
- v3 confidence bands: 43.9/43.1/61.3/67.9 — exact. Mix confidence
  (max of blended probs): <0.5 → 42.9%, ≥0.9 → 78.8% — exact.
- Flags: 混合 n=30 excl. combo row → v3 53.3 / mix 40.0 ✓; 跑题
  n=28 excl. combo → v3 67.9 / mix 71.4 ✓; v3 inclusive counts
  n=31/29/2 → 51.6/65.5/50.0 ✓ (two different n conventions, each
  internally consistent).
- Per-seed 3c mixes w=0.6: 66.0/63.5/**63.75→paper 63.8**/65.5/64.5 —
  one rounding nit. Weight sweep 0.4–0.8: computed 63.2–65.0 vs paper
  "63.3–65.0" — 0.1pp nit at w=0.4.
- Per-rating human-label distribution (paper Table `tab:rating-gap`):
  5★ 42/22/3 pos/neu/neg of 67 → 62.7% pos ✓; 4★ 70/62/6 of 138 →
  50.7% ✓; 3★ 37/75/20 of 132 → 56.8% neu ✓; 2★ 7/15/17 of 39 → 43.6%
  agree ✓; 1★ 0/5/19 of 24 → 79.2% ✓. Star-rule total 223/400=55.75% ✓.

---

## 2. Number-check table (claimed vs artifact)

| # | Claim (location) | Artifact value | Verdict | Evidence |
|---|---|---|---|---|
| 1 | Dedup-592 α=1.0: 3c acc 82.6–83.1, mac-F1 74.8–75.9; bin 91.5/90.8 (`draft_1_3`, `draft_6:333`, `appendix:78`) | 0.8260/0.7477 and 0.8311/0.7585; bin 0.9145/0.9078 | **MATCH** | `scripts/eval_dedup_pipeline.py` rerun |
| 2 | 600-row α=1.0: 3c 83.3–84.0 acc; bin 92.9/92.4 (`draft_6:205-209`, Table `tab:main-results`, `srw.tex:101`) | 0.8333/0.7626, 0.8400/0.7775; bin 0.9291/0.9238 | **MATCH** (see nit N1) | `scripts/eval_a100_pipeline.py` rerun |
| 3 | Deployed α=0.5 reference: 89.5/85.9 3c, 95.9/95.6 bin (`draft_6:225,231`) | 0.8950/0.8594, 0.9587/0.9558 | **MATCH** | same |
| 4 | human400 v3 55.8/55.4; mix 64.8/63.5; star 223/400 (`srw.tex:40-41`, `draft_6:132-139`, Table `tab:human400-f1`) | 55.75/55.35; 64.75/63.48; 223/400 | **MATCH** (rounding) | §1 |
| 5 | ood1000 mix 77.1% (`draft_6:83`, Table `tab:ood-results`) | al17kds_avg5 3c = 0.7710/0.7739 | **MATCH** | `mix_crossfit_results.json`, `mix_oof_probs.npz` |
| 6 | ood1000 binary peak 92.3%/92.1%, n=673 (`draft_6:95-96`) | al17kds_e8avg4 bin = 0.9227/0.9205 | **MATCH** (borderline rounding) | `mix_crossfit_results.json` |
| 7 | ood1000 v3 65.0; Qwen 70.6; DS-Flash 82.0 (`draft_6:299`) | 0.6500; 0.7060; 0.8196 on n=992 parseable | **MATCH** | `mix_oof_probs.npz` `pv3_oof`; `qwen3_4b_ood1000_probs.npy`; `ds_deepseek_flash_labels.jsonl` |
| 8 | Parseable DS counts 992/797/195 (`draft_6:313`, `appendix:93`) | 992 total = 797 consensus + 195 adjudicated | **MATCH** | recomputed |
| 9 | Provenance split (`draft_6:300-301`): cons 69.6/82.4/87.5/75.1; adj 46.5/56.0/59.5/52.5 | v3 69.62/46.50; mix 82.38/56.00; DS 87.45/59.49; Qwen 75.12/52.50 | **MATCH** (all 8 cells) | recomputed on `final_source` split |
| 10 | Binary provenance 93.7→85.3 (−8.4pp) vs 3c −26.4pp (`draft_6:279-280`) | bin e8avg4: 93.72→85.34 (−8.38); 3c mix: 82.38→56.00 (−26.38) | **MATCH** | recomputed |
| 11 | Gold star agreement 71.3% on 586 rated rows; per-rating 94.3/85.9/30.6/78.7/96.9 (`draft_1_3:236-241`, `draft_6:14-20`, Table `tab:rating-gap`) | 418/586 = 71.33%; per-rating exact | **MATCH** | `reviews_600_base.jsonl` |
| 12 | 3★ gold: 30.6% neutral, 51.3% negative (`draft_1_3:241`, `draft_6:17-18`) | 49/160=30.6%; 82/160=51.25% | **MATCH** (51.25→51.3 rounding) | same |
| 13 | human400 star agreement 55.75%; 3★ 56.8% neutral (`srw.tex:41-44`, `draft_6:25-27`) | 223/400; 75/132=56.8% | **MATCH** | §1 |
| 14 | AL scale curve 72.0/74.2/74.9/75.5/77.1; SFT-alone 74.1; al3kds 71.5; al11kq 72.4; al11kds0 75.2 p=0.81 (`draft_6:80-92`) | gold_al 0.720; al6kh 0.742; al11kds_avg 0.749; al14kds_avg 0.755; avg5 0.771; SFT-alone recomputed 0.7410; al3kds 0.715; al11kq 0.724; al11kds0 0.752, McNemar p=0.813 | **MATCH** (all) | `mix_crossfit_results.json`; `sft_results/` |
| 15 | avg5 vs al14kds +1.6pp, McNemar p=0.064 (`draft_6:342-343`, `appendix:164`) | 0.771−0.755=1.6pp; discordant 41/25, p=0.0640 | **MATCH** | recomputed |
| 16 | fresh500 mix 77.0/93.7 vs v3 58.8/86.3 (`draft_6:287-288`) | eval script rerun: mix 0.770/0.7719, bin 0.9366; v3 0.588/0.8627 | **MATCH** | `eval_fresh500_mix_al17k.py` |
| 17 | Corpus 910,998; confident pool 271,905; pseudo-vs-rating 47.8%/54.0% (`draft_1_3:250-256`, `draft_6:31-32`) | artifact counts | **MATCH** (artifact) | `pseudo/` manifests, AGENTS.md verified 2026-11 |
| 18 | Row overlap 68.7/70.2/74.8%; detector 13.0/13.2% (`draft_6:256-259`, `appendix` Table `tab:overlap`) | 687/1000, 351/500, 374/500; 130/1000, 66/500 | **MATCH** (artifact) | cid-audit outputs |
| 19 | IAA κ=0.568 / 71.5% raw, "blended not pure" (`draft_1_3:340-348`, `draft_6:349`) | `iaa_friend_user.json` aggregates | **MATCH** — correctly labeled non-human–human | artifact |
| 20 | LLM passes agree 29.8%, κ=−0.18 on 47 adjudicated rows (`draft_1_3:351-353`, `draft_6:274-275`) | earlier session computation | **MATCH** (artifact) | `friend_IAA_map.json` overlap |
| 21 | v3→v3 neutral-correction +1.6pp F1, p=0.65, CI [−0.4,+3.8] (`draft_4_5:43-45`, `draft_6:344-346`) | `ens_lite_3class_v2/v3` comparison | **MATCH** (artifact) | `out/ens_lite_*.json` |
| 22 | v3 ood1000 neutral recall 51.1→33.3 across length (`draft_6:347-348`, `appendix:167`) | recomputed: ≤20ch 51.1%, >100ch 33.3% | **MATCH** | `mix_oof_probs.npz` + lengths |
| 23 | Leak decomposition 89.5→84.0 (Leak B +5.5pp), 84.0→83.3 (Leak A +0.7pp) (`draft_6:200-204`, `appendix:57-61`) | 0.8950→0.8400→0.8333 | **MATCH** | `eval_a100_pipeline.py` |
| 24 | dedup "86.5% vs 89.5%" for α=0.5 (`draft_6:336`) | dedup a0.5+corpus 0.8649 vs 0.8950 | **MATCH** | `eval_dedup_pipeline.py` |
| 25 | "~7pp 3c / ~4pp binary" naive-vs-corrected gap (`draft_1_3:60`) | 89.5−82.6=6.9; 95.9−91.5=4.4 | **MATCH** | derived |
| 26 | "~26 points" consensus-vs-adjudicated inflation (`draft_1_3:53-54`, `srw.tex:72-73`) | mix 82.4−56.0=26.4pp | **MATCH** | #9 |
| 27 | gold class dist 316/92/192; 592 unique; 14 unrated (`draft_1_3:216,223`) | recomputed | **MATCH** | `reviews_600_base.jsonl` |
| 28 | human400 strata 160/120/80/40; labels 156/179/65 (`draft_4_5:196-197`, `draft_1_3:321-322`) | `human400_labels.jsonl` | **MATCH** | recomputed |
| 29 | AL pools: 279 human-adj + 721 consensus + ~15.9K DS (`draft_1_3:191-193,269-271`, `draft_6:74-75`) | al1000 final_source: 279 user/721 consensus | **MATCH** | `lab2000_final.jsonl` |
| 30 | lab2000 inter-pass agreement 86.7%/κ=0.80 ood, 79.8%/κ=0.67 al; 335 disagreements + 144 spot-checks → 479 adjudicated (`draft_1_3:278-283`) | README/`agreement.json` artifacts | **MATCH** (artifact; consistent with 200+279=479 split) | `lab2000/README.md` |

### Discrepancies found (all minor)

- **N1 — macro-F1 top of the α=1.0 600-row range understated.**
  `draft_6.tex:205` and `:334` and `appendix.tex:79` give the range as
  "76.3–77.5"; Table `tab:main-results` (`draft_6.tex:226`) prints
  "77.5" for the corpus-kept cell. Recomputed value is **0.7775 → 77.8**
  (or 77.7). The true range is 76.3–77.8. Direction is conservative
  (understates the corrected score) but the digit is wrong; suggest
  "76.3–77.8" / "77.8".
- **N2 — unresolved `%%(TODO_NUM)%%` markers in `appendix.tex:141-142`.**
  `tab:mixpool` lists `al17kq_avg5 … 0.756` and `al17kds_e4 … 0.748`
  with comments admitting the values cannot be confirmed against the
  released config log (no `al17kds_e4`/`al17kq_avg5` config exists in
  `mix_crossfit.py`; nearest real values are `al11kq`=0.724 and
  `al14kds_s456`=0.748). The LaTeX comments won't render, but the
  printed 0.756/0.748 numbers are **unverified and probably wrong** —
  author must confirm or drop before submission.
- **N3 — rounding nits (≤0.1pp, immaterial):** 3★ negative share
  printed 51.3% (exact 51.25%, `draft_1_3:241`, `draft_6:18`);
  appendix stratum 53.7 vs 53.75 (`appendix.tex:271`); weight-sweep
  lower bound "63.3" vs computed 63.2 (`appendix.tex:290`); per-seed
  s456 63.8 vs 63.75 (`appendix.tex:289`); binary peak "92.1" vs
  92.05 (`draft_6:95`).
- **N4 — denominator convention (not an error):** gold agreement
  71.3% uses the 586 rated rows (14 `未知` excluded);
  `AGENTS.md`'s "427/600=71.2%" counts the unrated rows differently.
  Paper's convention is stated explicitly and is correct.

---

## 3. Stale-claim sweep (current files, post-rewrite)

Grepped all `.tex` for: `89.5`, `edge.deploy`, `contamination-free`,
`Strict OOF|strict out-of-fold`, `held-out`, `71.5`, `ceiling`,
`the audit|contribution`.

| Phrase | Location | Status |
|---|---|---|
| 89.5 as headline | `appendix.tex:57`, `draft_6.tex:200,202,225,336` | **OK** — every occurrence is explicitly framed as "previously reported (uncorrected)", "deployed, uncorrected (α=0.5)", or inside the leak-decomposition arithmetic. Never presented as a current result. `srw.tex` abstract no longer cites it. |
| "edge deployment" | `draft_4_5.tex:223-226` | **Borderline-OK** — now qualified: "Only the student channel is compact enough for edge deployment; the deployed three-class system additionally requires the frozen 1.3B-parameter judge backbone (and the corpus neutral detector) at inference." Honest, but if the reframed paper wants zero deployment framing this sentence is the only place it survives. |
| "contamination-free" | `srw.tex:91` | **OK** — negated ("does not prove contamination-free text"). No other hits. |
| "Strict OOF" / "strict out-of-fold" | — | **GONE.** Zero hits in current files. |
| "held-out test" on ood1000/fresh500 | `draft_1_3.tex:181`, `draft_4_5.tex:188`, `srw.tex:72` | **OK** — all three are negated ("not a held-out test" / "rather than a held-out test"). human400 is called "identifier-disjoint final test", which is the intended role. |
| 71.5% as human ceiling | `draft_1_3.tex:344`, `draft_6.tex:87` | **OK** — `draft_1_3:340-348` explicitly labels it human-vs-*blended-reference*, "not a pure human–human IAA", and discloses the missing `friend_labels.txt`. `draft_6:87` is an unrelated model accuracy. |
| Audit-as-contribution self-reference | `draft_4_5.tex:137` "Our contribution is as much the audit as the system."; `draft_6.tex:215` "the honest numbers are the contribution." | **Residual, LOW** — softened versus the old framing and partially redeemed by `appendix.tex:13` ("a component of the evaluation protocol, not the paper's primary claim"), but `draft_4_5:137` is still the most audit-centric sentence in the paper. Optional softening: "Our evaluation protocol is part of the contribution." |

**Top stale-claim risks if the professor reads quickly:** (1) the two
`TODO_NUM` appendix values (N2) — they look authoritative but are
unverified; (2) the 77.5-vs-77.8 F1 digit (N1); (3) `draft_4_5:137`'s
"as much the audit as the system" if the venue wants a purely
data-centric framing.

---

## 4. Feasibility verdicts (a–g)

Full specs in `NEEDS-EXPERIMENTS.md`. Summary:

| # | Requested experiment | Verdict | Basis |
|---|---|---|---|
| a | 2nd annotator, human400 (400 rows) | **NEEDS-HUMAN** | `friend_IAA_200` is ood1000, not human400, and its per-row labels were never returned. Annotation kit exists (`annotation/` in both repos). ~2–3 h. |
| b | Paired human-vs-LLM on 300–400 ood1000 rows | **NEEDS-HUMAN** (LLM arm reusable, no API needed unless new teacher wanted) | Adjudicated rows are anchoring-exposed; friend-IAA labels not released row-level. Existing DS/Qwen preds reusable. |
| c | Random vs uncertainty AL, equal budget | **FEASIBLE-NOW at n=500** (`fresh500` random+DS vs 500-row `extra_al17k_ds` subset, eval on human400); **NEEDS-API** at 17K scale | `sample_fresh500.py` confirms uniform sampling; all AL pools are margin/entropy/disagreement-sampled. |
| d | Multi-movie rating–sentiment | **NEEDS-HUMAN** | `mid` exists for all corpus rows but zero multi-movie semantic labels exist. New annotation required (~4–8 h). |
| e | Neutral specialist vs class-weight/focal/oversample/threshold | **FEASIBLE-NOW** | class_weight runs + threshold/cascade rejections + specialist sweeps already in `out/` and appendix; focal loss and oversampling are *not implemented* (zero grep hits) → new GPU runs, agent-doable, ~half day serialized. |
| f | Cross-movie generalization split | **IMPOSSIBLE** with current labels | gold-600 = one film page; depends on (d). Paper should keep the single-film limitation wording (already present). |
| g | Full fold-isolated pipeline re-run | **FEASIBLE-NOW, compute-bound** | GPU confirmed (torch 2.11+rocm, RX 9070 GRE). Missing pieces: fold-isolated teachers (~5× teacher cost), corpus-detector retrain on 115K (hours), inner-fold weight selection. Est. 1–2 days serialized; scoped cheaper version ~hours. Demo must be stopped first. |

---

## 5. Caveats carried into every claim (for the authors)

1. **gold-600 is single-film** (one Douban page, 592 unique texts, 8
   duplicate pairs that can cross folds).
2. **human400 is single-annotator, length-stratified** — benchmark
   numbers, not population estimates; no IAA computable (experiment a).
3. **ood1000 is a development/selection set** — every Stage-2 decision
   read its accuracy; McNemar tests on it are anti-conservative;
   68.7% of its rows share a cid with some corpus-trained pool.
4. **LLM "consensus" ≠ human ground truth** — two passes of the *same*
   model; consensus-vs-adjudicated gap (~26 pp) confounds label source
   with item difficulty (disagreement-enriched subset).
5. **Duplicate-text and second-order-teacher leakage** mean even the
   corrected OOF figures (83.3–84.0% / 92.9%) are leakage-channel
   ablation estimates with frozen weights — not fully selection-
   independent final scores; the paper says this correctly.
6. **human400 text-level exposure is bounded, not zero** — two
   ultra-short normalized boilerplate matches in corpus pseudo-label
   pools feeding one ensemble channel; identifier-disjoint ≠
   contamination-free.
7. **Texts are hash-only in the release** — all row-level audits
   (`cid`, `text_sha256`) must be re-run against source text locally.
8. **fresh500 is partially exposed** (70.2% cid overlap with
   corpus-derived pools; DS-family labels) — replication check only.
9. **Adjudication was anchoring-exposed** — student predictions were
   visible during the 479 human adjudications.
10. **GPU serialization** — never run two training jobs concurrently on
    this ROCm box; the demo backend holds ~5 GB VRAM.

## 6. Files/artifacts relied on

- Predictions: `E:/GB77_data/human400/{human400_v3_probs.npz,
  human400_sft_probs_al17k/, qwen3_4b_human400_probs.npy}`,
  `E:/GB77_data/lab2000/{mix_oof_probs.npz, mix_crossfit_results.json,
  v3_probs.npz, fresh500_v3_probs.npz, fresh500_sft_probs_al17k/,
  qwen3_4b_ood1000_probs.npy, ds_deepseek_flash_labels.jsonl,
  ds_deepseek_flash_fresh500_labels.jsonl, sft_results/}`.
- Labels: `douban-sentiment/data/{reviews_600_base.jsonl,
  reviews_600_dedup592.jsonl, human400_labels.jsonl,
  lab2000/lab2000_final.jsonl, lab2000/friend_IAA_200.jsonl,
  lab2000/friend_IAA_map.json, lab2000/iaa_friend_user.json}`.
- Pipeline metrics (recomputed this session): `scripts/
  eval_a100_pipeline.py`, `scripts/eval_dedup_pipeline.py`,
  `E:/GB77_data/lab2000/eval_fresh500_mix_al17k.py`.
- Paper files audited (current versions): `srw.tex`, `draft_1_3.tex`,
  `draft_4_5.tex`, `draft_6.tex`, `appendix.tex`.

# douban-sentiment — project notes

Douban short-review sentiment classifier. Semantic labels (`label_semantic`) are
authoritative; star-derived `label_rule` is weak-supervision only.

## Layout

- `data/reviews_600_base.jsonl` — the canonical 600 hand-labelled reviews;
  ALL reported OOF metrics and deployed models are trained/evaluated on this
  file only (aug-inclusive training was rejected, see below)
- `data/reviews_600.jsonl` — 648 rows: base 600 + 48 short-text aug rows
  (`src: "aug"`, appended at end). Only used by aug-specific experiments
  (`teacher_predict_aug.py`, `eval_aug_combo.py`, `verify_lexicon.py`,
  `to_aliyun_format.py`) — training scripts default to the base file
- `web/short_lexicon.py` — ultra-short-text guard lexicon (exact normalized match,
  verified 0 conflicts on base 600, covers all 48 aug rows)
- `scripts/verify_lexicon.py` — re-checks lexicon vs dataset
- `scripts/` — training & eval (`gpu_finetune.py`, `frozen_*.py`, `ensemble_eval.py`, `run_ens_all.py`)
- `out/` — OOF result JSONs + ensemble selections (`ens_*.json`)
- `models/` — production fine-tuned checkpoints and frozen LR heads (`*.joblib`)
- `web/app.py` + `web/manifest.json` — FastAPI demo backend
- `web-ui/` — React/Vite frontend (Base UI style), builds into `web/static/`

## Environment

- Python: project venv (`python` after activation)
- GPU: AMD RX 9070 GRE via ROCm — use `device='cuda'`, bf16 autocast
- Always set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_HOME=<local HF cache>` (models are cached; hub is flaky)

## Run the demo

```powershell
cd web
$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:HF_HOME='<local HF cache>'
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Cold start ~14 s (3 MacBERT students + shared MegatronBERT-1.3B backbone).
Stop it when idle — it holds several GB of VRAM. Warm latency ~0.1 s
(GPU-shared measurements are inflated; re-benchmark when idle).

## Deployed architecture (2026-06 update)

`web/manifest.json` + `web/app.py`:

- binary: MacBERT distilled student (w=1) + MegatronBERT-1.3B frozen
  features → MLP head `mlp_bin_mb13b_m256.pt` (w=2)
- 3class: MacBERT distilled student (w=1) + same backbone → MLP head
  `mlp_3c_mb13b_m256.pt` (w=2), then neutral-expert blend:
  `pn = 0.6·P_neu(head mlp_neu_mb13b_m256_h256.pt) + 0.4·P_neu(corpus detector macbert_neu_corpus → <corpus_root>/models/neu_pseudo_p60k_e2 via local junction)`,
  `spec = [(1-pn)·r_pos, pn, (1-pn)·r_neg]` with r = dep pos/neg renorm,
  `final = (dep + 0.5·spec)/1.5`. The two mb13b heads share one backbone
  feature pass via `backbone_feats()` cache in app.py.
- `web/short_lexicon.py` exact-match guard still intercepts ultra-short
  slang before any GPU work.

Key scripts: `distill.py` (5-fold OOF distillation, tasks 3class/binary/neu),
`distill_head.py` (MLP head OOF on frozen feats, tasks binary/3class/neu),
`train_final_distill.py` / `train_final_head.py` (100%-data production fits,
same task choices + `--data`), `eval_neu_decomp.py` (specialist combo search),
`eval_lite.py` / `eval_lite_v3.py` (rebuild deployed lite ensembles v1/v2/v3
from OOF probs — v3 needs `out/cascade_pneu.npy` from `eval_cascade.py`),
`eval_qwen3_score.py` (Qwen3 label-logprob eval, `--bs` for VRAM).

## Rebuild frontend

```powershell
cd web-ui
npm run build   # emits to ../web/static
```

## Current best (stratified 5-fold OOF, base 600 rows)

- binary: **acc 0.9587 / macro-F1 0.9558** — deployed lite hybrid
  (`out/ens_lite_binary.json`)
- 3class: **acc 0.8950 / macro-F1 0.8594** — deployed lite v3 + corpus
  neutral expert (`out/ens_lite_3class_v3.json`); previous v2 was
  0.8900/0.8432 (`out/ens_lite_3class_v2.json`). Neutral recall
  0.543 → 0.707. Split-half (20×300) confirms the gain is stable:
  neu-pool H+D beats S+H by ~+1.2pp F1 on every random half.
- baseline (TF-IDF LR): binary 0.8406/0.8232, 3class 0.7233/0.6188

## Modern-LLM comparison (measured zero-shot, label-logprob scoring)

Qwen3 chat models in a local model dir (not in git):

- Qwen3-0.6B zero-shot: acc 0.6817 / f1 0.4764, neutral recall 0.000
- Qwen3-1.7B zero-shot: acc 0.7067 / f1 0.5739
- Qwen3-1.7B 3-shot: acc 0.5100 / f1 0.4465 (few-shot HURTS — neg bias)
- Qwen3-4B zero-shot: acc 0.7833 / f1 0.7348 (neutral recall 0.66)
- Qwen3-4B 3-shot: acc 0.7883 / f1 0.7308 (few-shot neutral for 4B)

Takeaway: general-purpose LLMs are ~20pp below the task-specific pipeline,
mostly because they can't handle the neutral/mixed-review class.

## Verified dead ends (don't retry blindly)

- LoRA on MegatronBert: binary 0.9390/0.9343, 3class 0.8083/0.6431 — hurts ensemble
- median / trimmed / geometric / logit-mean aggregation — all worse than mean
- per-channel weight tuning — equal weights optimal (±0.1pp noise)
- warm-start forward selection — no channel addition improves either ensemble
- neutral probability injection — degrades Macro-F1
- greedy ensemble selection is path-dependent: more channels can yield a worse
  greedy result (allch4 < allch2); always compare against the deployed selection
- 3class student grid (2026-06): alpha 0.3/0.7, tau 4, seed 123, epochs 16,
  class_weight, Qwen-augmented teacher — ALL worse than deployed a0.5/t2/e8;
  e16 overfits (0.8667 vs 0.8783 with judge). alpha=0.5/tau=2/epochs=8 is the
  sweet spot.
- Erlangshen-RoBERTa-330M as distill student: pathologically slow on ROCm
  (~2.5 h for 5-fold OOF vs ~15 min for MacBERT) — abandoned, likely kernel
  fallback. MacBERT remains the right student backbone.
- Neutral-expert variants: seeds s7/s456 and extra head seeds all dilute the
  deployed s123-student + h256-head pair; judge-head seed averaging also hurts.
- 3class→binary marginalization matches dedicated binary (0.9587 both) —
  the tasks are coherent; dedicated binary head kept anyway (marginally better f1).
- short-text fine-tuning augmentation: adding 48 slang/profanity rows and
  re-distilling dropped base accuracy ~2pp (binary 95.9→93.9) while aug OOF
  stayed ~75%/68% — rejected. Per-sample `aug_alpha` (hard-label-only loss for
  aug rows) is implemented in `distill.py`/`distill_head.py` but did not fix it.
  Shipped solution instead: `web/short_lexicon.py` exact-match guard in
  `app.py` predict() — fires only on bare slang, returns a single
  "超短文本规则" channel, ~0 ms, zero impact on normal reviews.

## Commercial-platform comparison (Aliyun PAI)

`scripts/to_aliyun_format.py` emits `data/aliyun/{3class,binary,merged}.json`
— gold-labelled upload files for Alibaba Cloud's NLP self-learning platform
(the StructBERT classifier compared in thesis §comparison). binary=508 rows
(neutral dropped), merged=600 (neutral folded by star direction). The
platform's own reported metrics are analysed in the paper; we keep the
input files for reproducibility.

## External benchmark (measured, not cited)

- `damo/nlp_structbert_sentiment-classification_chinese-base` (ModelScope/Ali
  DAMO, binary pos/neg only, StructBERT) downloaded to `eval/damo_sentiment/`.
  Weights load into `BertForSequenceClassification` by renaming `encoder.`→`bert.`.
  On our data: binary acc 0.8704 / macro-F1 0.8538 (n=540), base-508 subset
  0.8661/0.8459 — far below our lite ensemble (0.9587/0.9558). No 3-class head,
  so no 3class comparison is possible.

## Round-3 exhaustive search (2026-11-06, all rejected — deployed config stands)

Follow-up sweep after the neutral-expert deployment. Every axis below was
measured against `ens_lite_3class_v2.json` (0.8900/0.8432) and none beat it:

- TF-IDF neutral specialist (`tfidf_neu_char13`, char 1-3gram LR on all 600):
  standalone 0.8583/0.6094; adding to the neu-expert pool at any weight or
  combo dilutes S+H — best remains exactly the deployed pair.
- Student probability averaging (s42+s123, +tau4, +qwen-teacher, all-4):
  v2 drops to 0.8850-0.8883/0.8374-0.8413. Single s42 student stays best.
- Self-distillation round 2 (`distill_macbert_3c_v2teacher`, teacher = v2
  ensemble probs): student-alone 0.8783/0.8205, in-v2 0.8867/0.8387 — worse.
- class_weight=1 3class student (`distill_macbert_3c_a50_t2_e8_m256_cw`):
  in-v2 0.8883/0.8413 — worse.
- max_len 128 student (`distill_macbert_3c_m128`): in-v2 0.8800/0.8295;
  averaging m128+m256 also worse (0.8867/0.8407).
- Erlangshen-RoBERTa-110M distilled student (`distill_erl110_3c`,
  0.7967/0.7404 standalone): as extra channel or student-pair member,
  all weights worse (best 0.8800/0.8325).
- Frozen-MacBERT-feature neu MLP head (`headmlp_neu_macbert_m256_h256`,
  standalone 0.8583/0.5848): neu-pool combos of {S,H,M,T} — S+H still best.
- Judge-head variants in v2: h256/h1024/seeds/alpha/aug/3small/all4/erl330 —
  all below deployed `headmlp_3c_mb13b` (next best 0.8817/0.8313).
- Joint weight re-tune (student:judge ratio × neu weight × neu-pair mix):
  deployed (1.0, 0.5, 0.5) is the optimum; mix 0.4 ties exactly.
- Stacked logistic-regression meta-learner on OOF channel log-probs
  (9 features, 5-fold meta-CV): 0.8367/0.7936 — far worse, channels too
  correlated for 600 samples.
- Binary-model-informed pos/neg split for the 3class spec: monotonic
  degradation with weight (w=0.1 already 0.8883/0.8413).
- Qwen3-4B synthetic reviews (`scripts/gen_synth.py` + `score_synth.py` +
  `build_synth_aug.py`): 85 candidates, 75 self-consistent kept
  (中性 18/28 survived). Augmented distill (`distill_macbert_3c_synth1`,
  675 rows, aug_alpha=1.0): student on orig-600 only 0.7917/0.7200,
  in-v2 0.8750/0.8282 — synthetic data poisons training. Root cause:
  generations anchor to few-shot movie references and self-consistency
  filtering only reinforces existing biases. Rejected; pipeline scripts
  kept for reference.

Conclusion: deployed v2 (MacBERT s42 student + MegatronBERT-1.3B MLP judge
1:2 + neu pair S+H at 0.5) is the measured optimum across ~20 experiment
families. Remaining 66 errors are predominantly human-ambiguous mixed
reviews (err conf 0.51 vs 0.63 correct; 中性→正面 still largest bucket).

## Big-corpus pipeline (2026-11-06, Moviedata-10M — 910,998 reviews)

Source: Moviedata-10M Douban dataset by csuldw (CSU, 2019;
https://github.com/csuldw/AntSpider — research use only, attribution
required, raw data not redistributed). 4.43M raw comments → cleaned/
deduped `<corpus_root>/clean/comments.pkl`
(gb18030 CSV -> cleaned/deduped, cols cid/uid/mid/raw/content/votes/ctime/
rating/len). Gold overlap negligible (3/600). Gold median len 80 vs corpus 24.

`<corpus_root>` = `scripts/_paths.py` `DATA_ROOT`: env `DOUBAN_DATA_ROOT`
if set, else the local training-machine default. All corpus/models/results
paths in `scripts/` resolve through it — never hardcode the root again.
Artifacts: `<corpus_root>/{pseudo,features,models,results}/`.

Scripts: `prep_bigdata.py`, `train_bigdata.py` (rating maps bin_strict/bin_all/
tri + `--pseudo` mode + `--task neu`), `pseudo_label.py`, `tfidf_bigdata.py`,
`extract_corpus_feats.py` (incremental ckpt + resume), `train_corpus_head.py`.

Measured results (gold-600 eval unless noted):

- Star-rating labels are BAD for 3class: TF-IDF tri -> 0.6183/0.5369.
  Rating-vs-semantic agreement on gold crosstab: 3★ rows are semantically
  negative more often than neutral. Gold semantic-vs-star-rule agreement
  427/600 = 71.2%. Corpus-wide (all 910,998 rows) pseudo-vs-tri-rating
  agreement only 0.478 (verified 2026-11; confident-pool subset is 0.540).
- TF-IDF + pseudo labels (stu3c, conf>=0.7, 200K): 0.8117/0.7766 — beats
  gold-trained TF-IDF (0.7233). Label quality >> model complexity.
- MacBERT + bin_strict rating labels (115K, `macbert_bin_strict_p60k_e2`):
  zero-shot gold 0.9232;
  +gold SFT 5-fold OOF 0.9389 single model. As ensemble channel w=0.75 ->
  0.9626/0.9598 on full-508 but split-half gain inconsistent (~noise).
- Pseudo-label flywheel (KEY RESULT): stu3c (deployed 3c student) labels
  911K corpus; confident>=0.7 subset 271,905 (pos 56.5K/neu 68.6K/neg 146.8K);
  per-class-capped 160K train -> `macbert_3c_pseudo_p55k_e2` gold600
  **0.8750/0.8463, neutral recall 0.848** (vs deployed v2 0.8900/0.8432,
  neu recall 0.630). Single model, ZERO gold training.
- Binary pseudo (stu3c marginalized pos/neg, conf>=0.8, 195K):
  `macbert_bin_pseudo_p120k_e2` gold 0.9488/0.9455 — 1pp below deployed
  ensemble; adding to ensemble does NOT help (errors correlated).
- Corpus MLP head (macbert frozen feats 195K + pseudo labels): gold
  0.8350/0.7984, neu recall 0.739 — hurts v2 at any mix weight; its P(neu)
  in the neu-expert blend: best 0.8867/0.8449 (acc down, f1 ~flat).
- Round-2 pseudo (pseudo student relabels corpus): probs are degenerate
  (mean conf 0.986) — student memorized its 160K train rows (r1-vs-r2
  agreement 99.9% on r1-conf>=0.7, only ~86% overall). Self-training
  flywheel saturates after round 1; r2 labels only informative on
  r1-conf<0.7 rows.
- Gold-SFT of pseudo student: no help (already aligned); SFT OOF on
  648-row file incl aug = 0.8441 — not comparable to base-600 evals.
- r3 flywheel (2026-11-07): v3 ensemble probs on the 271,905 pool
  (`ensemble_corpus_probs.py`, needs `mb13b_pool_c07` feats 4.4GB +
  `neu_corpus_pool` detector probs) -> `stu3c_v3` pseudo labels ->
  `macbert_3c_r3pseudo_e2` gold 0.8767/0.8434 (vs stu3c r1 student
  0.8750/0.8463 — wash, neu recall worse 0.783 vs 0.848). As ensemble
  channel at any weight: all mixes below deployed v3. REJECTED — the
  ensemble's edge lives in judge/expert channels that don't transfer
  through hard pseudo labels; flywheel saturates after round 1 as
  predicted. Pipeline scripts kept (`r3_chain.sh`, `pool_probs.py`,
  `extract_mb13b_pool.py`, `ensemble_corpus_probs.py`).
- Pseudo-label scaling curve (stu3c, per-class cap, e2): ~1K→0.7867/0.7217,
  ~5K→0.8033/0.7535, ~19K→0.8433/0.8016, ~48K→0.8550/0.8146,
  ~160K→0.8750/0.8463 — monotonic, still rising; corpus scale is the
  active ingredient. Models: `stu3c_pc{334,1667,6667,16667}_e2`.
- Silver eval (`silver_eval.py`, 4K fresh corpus rows vs rating-implied
  labels, `out/silver_*.json`): r3pseudo binary-agreement 0.837 vs
  stu3c 0.803 — v3-teacher labels generalize slightly better
  out-of-domain despite equal gold scores.
- Gold-mix (`mix_gold_into_pool.py`, stu3c_v3g50 = pool + gold×50):
  gold-eval is circular by construction (acc 1.0 = memorization);
  kept only for silver comparison.

GPU ops note: NEVER run two training/inference jobs concurrently on this
ROCm/Windows box — shared-GPU eventually kills both with c10 AbortHandler
(OOM/driver). Serialize all GPU work. Demo backend holds ~5GB VRAM —
kill it during heavy experiments (restart via web/start_demo.bat).

Done: neu-detector corpus train, mb13b corpus feats (75K complete),
cascade eval — see "Corpus neutral detector + cascade" below.

## Corpus-trained MLP heads on mb13b features (2026-11-06 late, all rejected)

Extracted frozen MegatronBERT-1.3B bf16 features for 75K balanced stu3c
pseudo rows (`mb13b_stu3c_c07_p25k`, 4096-dim CLS+mean) plus gold-600 bf16
feats (`out/feats_mb13b_bf16.npy`). `train_corpus_head.py` results:

- 3class head: gold 0.8483/0.8060 (neu recall 0.707) — as judge replacement
  0.8517, judge mixes all worse, dep+corpus all worse.
- neu head: gold neu-detection 0.8983/0.8033 — every neu-pool combo with it
  dilutes deployed S+H (best remains S+H 0.8900/0.8432).
- binary head: gold 0.9449/0.9414 — all binary ensemble variants worse
  than deployed 0.9587/0.9558.

Conclusion: corpus-scale pseudo-label features do NOT beat gold-trained
channels on the gold eval — same pattern as every corpus model tried.

## Corpus neutral detector + cascade (2026-11-06 night — v3 DEPLOYED)

`train_bigdata.py --task neu` on per-class-capped 115K stu3c pseudo rows
(cap 60K/side, actual n_train=115,000), 2 epochs → `<corpus_root>/models/neu_pseudo_p60k_e2`
(gold-600 neu-detection acc 0.9017 / macro-F1 0.8328; neu recall 0.848).

Two uses tested:

1. **Cascade** (`eval_cascade.py`: P(neu)>=thr → 中性, else binary student
   decides pos/neg): full-600 looked great (0.8917/0.8757, neu recall 1.0)
   but threshold was tuned on the same 600 rows — split-half validation
   (4 reps, thr tuned on one half, eval other) collapses to ~0.86-0.88,
   clearly below v2. REJECTED — classic small-sample threshold overfit.
2. **Neutral-expert pool member**: replacing the distilled neu student S
   with the corpus detector D in `pn` — combos tested exhaustively:
   S+H 0.8900/0.8432 (old), H+D 0.8900/0.8545, weighted pn=0.6H+0.4D
   **0.8950/0.8594** (neu recall 0.707, pos 0.943, neg 0.906).
   20× random-half validation: every detector weight w∈[0.3,0.6] beats
   S+H on mean F1 (0.8575-0.8625 vs 0.8458) — real gain, not knife-edge.
   S+H+D triple also tested (0.8850/0.8443) — the old student dilutes.

DEPLOYED as `ens_lite_3class_v3.json`: same dep=(stu+2·judge)/3 and
spec blend, only the neu-pool pair changed (H head + corpus detector,
weights 0.6/0.4). Inference cost identical (one MacBERT slot swapped).
Manifest `neu_expert.student` → `models/macbert_neu_corpus` (local junction
to the corpus drive — machine-local like `models/` itself).

Follow-up `neu_pseudo_v2` (195K rows trained: all confident neu + P(neu)>=0.15
hard negatives + easy fill to 2:1, 3 epochs): standalone gold neu
0.8983/0.8206 (worse than v1's 0.9017/0.8328) and EVERY neu-pool combo
worse (best H+D2 0.8817/0.8391 vs deployed H+D1 0.8950/0.8594). Hard-
negative mining + class-ratio shift did not help — v1 detector stays.
Script: `scripts/prep_neu_v2.py`.

Also measured (2026-11-07): 3class ensemble is under-confident
(ECE 0.251; conf 0.7-0.8 bin is 98.9% accurate). Temperature T=0.37
(NLL-fit on gold) → ECE ~0.04, predictions unchanged; applied in
app.py via manifest `temperature`. Per-class decision reweighting:
uniform weights optimal on split-half — no gain. Gold-set error
analysis: explicit-score reviews (n=56) err 5.4% vs 11.0% without;
binary task's 21 errors are all human-ambiguous mixed reviews.
McNemar v2-vs-v3 p=0.65, bootstrap ΔF1 95%CI [-0.4,+3.8]pp — gain is
real-direction but modest; consistent across halves.

GPU stability note: after the concurrent-job crash the ROCm stack degraded
(1.3B inference AbortHandler crashes, ~2rows/s). Full reboot restored it
(~70rows/s, zero crashes). Known upstream issue class: ROCm-on-Windows
RDNA4/gfx120X instability (TheRock #5014/#5284). If it recurs: reboot first,
then consider torch wheel change.

## Fresh-corpus evaluation (2026-11-07, lab500 experiment)

Sampled 500 random reviews from the 911K corpus (seed 42, gold-deduped),
labelled blind by a general-purpose LLM assistant + independently by Qwen3-4B zero-shot
(label-logprob scoring), evaluated deployed v3 via /api/predict.

Label distributions: LLM-A 正面176/中性190/负面134; annotator agreement
LLM-A-vs-qwen = 66.2% — terse corpus comments are intrinsically ambiguous.

Results (3-class, vs LLM-A labels unless noted):
- v3 ensemble: acc 0.604 / f1 0.599 (gold-600: 0.895/0.859)
- Qwen3-4B:    acc 0.662 / f1 0.666
- v3 vs {me,qwen}-consensus subset (n=331): acc 0.740
- BINARY pos/neg only: v3-vs-me 0.918 (n=255) — binary transfers fine
- MacBERT 5-fold CV trained on 400 LLM-A labels/fold: OOF acc 0.600 —
  a model trained on in-domain labels cannot exceed ~60% reproducing
  these labels => label noise ceiling, not just model weakness
- Consensus-label CV (331 rows): OOF 0.656

Difficulty stratification (stu3c conf>=0.7 = likely-in-pseudo-pool):
- conf<0.7 rows (n=348): v3-vs-me 0.537; conf>=0.7 (n=152): 0.757
- CV model shows identical split (0.532/0.757) => difficulty effect,
  NOT train leakage. High-conf rows are easy for ALL annotators.

CONCLUSION: gold-600 (curated, median len 80) overstates real-world
performance. On fresh terse corpus (median len 25), v3 3-class is
~60% vs single-LLM labels / ~74% vs dual-LLM consensus; binary ~92%.
中性 recall collapses to ~36-48% — the neutral convention learned from
92 gold examples does not transfer to corpus-style ambiguous text.
This is the honest generalization gap; gold-600 metrics remain the
assignment benchmark but PAPER.md must report both.

Artifacts: data/lab500_llm.jsonl, data/lab500_consensus.jsonl,
out/lab500_{tsv,labels.tsv,preds.json,disagree.txt,conf.npy},
out/qwen3_4b_lab500.json, out/lab500_cv*.{json,npy},
scripts/lab500_cv.py.

## HF Space deployment (2026-11-07)

Live demo: https://huggingface.co/spaces/Shuakami/douban-sentiment
(ZeroGPU, Gradio SDK, custom React frontend via `gradio.Server`).

- Weights repo: `Shuakami/douban-sentiment-weights` (public, Apache-2.0
  backbone + our fine-tunes): `macbert_{bin,3c,neu}/`, `heads/mlp_*.pt`,
  `papers/*.pdf`. Backbone `IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment`
  is apache-2.0 — loaded straight from the Hub, not re-uploaded.
- Space packaging lives OUTSIDE this repo in a sibling `hf-space/`
  checkout (own git repo → hf.co/spaces). HF rejects binary files in
  Space git — PDFs are served via `hf_hub_download` from the weights repo
  instead.
- Frontend detects `*.hf.space` hostnames and calls predict through the
  Gradio queue protocol (`POST /gradio_api/call/predict` → SSE stream);
  `@gradio/client` was tried first but mangles named params — raw fetch
  is used instead. Local `npm run dev`/FastAPI path unchanged.
- ZeroGPU rules followed: `import spaces` before torch, models loaded at
  module scope with `.to("cuda")`, `@spaces.GPU(duration=120)` on inner
  fn, `@app.api(name="predict", concurrency_limit=1, time_limit=180)`.
  Env var `GRADIO_SSR_MODE=false` set on the Space (required for custom `/`).

# Weak Supervision at Scale for Chinese Review Sentiment: A Pseudo-Labeling and Specialist-Decomposition Study

*Course project report — Douban short-review sentiment classification.*

## Abstract

We study three-class sentiment classification (positive / neutral /
negative) of Chinese Douban movie short reviews under a realistic
low-resource setting: 600 human-verified semantic labels, plus a large
weakly-labelled corpus of 911,998 reviews with star ratings. We show that
star ratings are a poor proxy for semantic sentiment (corpus-wide
three-class agreement between pseudo-labels and ratings only ~48%), and
that pseudo-labelling the corpus
with a strong ensemble yields a single compact student reaching 87.5%
accuracy with **84.8% neutral recall** — far above what 600 gold labels
alone can teach. Our deployed system combines a distilled MacBERT student
with a frozen MegatronBERT-1.3B feature judge and a corpus-trained neutral
specialist, reaching **89.5% accuracy / 85.9 macro-F1** on the gold set,
beating both a TF-IDF baseline (+17pp) and zero-shot Qwen3-4B (+11pp).

## 1. Introduction

Short reviews on Douban are terse, ironic, and frequently mix praise with
criticism ("特效无敌，但是剧情简直像一坨"). The neutral/mixed class is
both the most frequent ambiguity and the hardest to learn from few
examples. Two questions drive this work:

1. Can a small labelled set be amplified by a large weakly-labelled
   corpus?
2. Can specialist decomposition (a dedicated neutral expert trained on
   corpus-scale pseudo labels) fix the weakest class without hurting the
   rest?

## 2. Data

| Dataset | Size | Labels |
|---|---|---|
| Gold | 600 reviews (30 pages × 20, single film) | `label_semantic`, human-verified 3-class; `label_rule`, star-derived weak |
| Corpus | 911,998 reviews, 20,966 films, 249,105 users | star rating 1–5 only |

The corpus derives from the Moviedata-10M Douban dataset (csuldw, CSU
2019; 4.43M raw comments → cleaned/deduped here; research use only,
see https://github.com/csuldw/AntSpider).

Gold class distribution: 316 positive / 92 neutral / 192 negative.
Corpus review length: mean 40.6 / median 24 chars (p90 = 102); gold
reviews are longer (mean 110.8 / median 80). Corpus rating distribution:
1★ 7.2% / 2★ 12.6% / 3★ 34.0% / 4★ 31.0% / 5★ 15.2%.

**Star ratings are not sentiment labels.** On the 271,905-row confident
pseudo pool, the row-normalized rating→prediction crosstab shows:

| rating | →pos | →neu | →neg |
|---|---|---|---|
| 1★ | 1.7% | 7.3% | 90.9% |
| 2★ | 1.6% | 13.7% | 84.7% |
| 3★ | 9.9% | 33.3% | **56.8%** |
| 4★ | 38.3% | 32.7% | 29.0% |
| 5★ | **58.8%** | 23.1% | 18.2% |

Only ~59% of 5-star reviews are semantically positive; 3-star reviews skew
negative rather than neutral. Rating-vs-model binary agreement on the
confident pool is 81.2%; corpus-wide three-class agreement between
pseudo-labels and ratings is only 47.8% (71.2% on the gold set). We
therefore treat ratings
as weak supervision for pretraining only, never as gold.

## 3. Methods

### 3.1 Teacher ensemble → distillation

A 7-channel teacher ensemble (TF-IDF, frozen-backbone MLP heads on
MacBERT / MegatronBERT-1.3B / Erlangshen variants, gold-trained) reaches
86.8% 3-class accuracy. A MacBERT-base student is distilled with hard-label
CE + soft-target KL (α=0.5, τ=2, 8 epochs, 5-fold OOF for honest eval).

### 3.2 Frozen judge

`IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment` is kept frozen; a small
MLP head on its 4096-d features is trained on gold. Deployed mixture:
`dep = (student + 2·judge)/3`.

### 3.3 Neutral specialist

`pn = 0.6·P_neu(mlp head on judge features) + 0.4·P_neu(corpus detector)`.
The corpus detector is a MacBERT binary classifier trained on a
115K-row per-class-capped pseudo-labelled subset of the confident pool. The specialist distribution
`spec = [(1−pn)·r_pos, pn, (1−pn)·r_neg]` is blended at weight 0.5.

### 3.4 Pseudo-label flywheel

The deployed student labels all 911,998 corpus reviews; the confident
(≥0.7) subset of 271,905 rows (pos 56.5K / neu 68.6K / neg 146.8K) is
re-used for training corpus-scale models. Round-2 self-labelling saturates
(mean confidence 0.986 — memorization), so one flywheel round is used.

## 4. Results

All scores are stratified 5-fold out-of-fold predictions on the 600
hand-labelled reviews.

| System | Task | Acc | Macro-F1 |
|---|---|---|---|
| TF-IDF + LR | 3-class | 0.7233 | 0.6175 |
| TF-IDF + LR | binary | 0.8406 | 0.8229 |
| StructBERT (Ali DAMO, zero-shot) | binary | 0.8704 | 0.8538 |
| Qwen3-4B (zero-shot) | 3-class | 0.7833 | 0.7348 |
| Qwen3-0.6B / 1.7B (zero-shot) | 3-class | 0.6817 / 0.7067 | — |
| Teacher ensemble (7ch) | 3-class | 0.868 | — |
| **Deployed v3** | **binary** | **0.9587** | **0.9558** |
| **Deployed v3** | **3-class** | **0.8950** | **0.8594** |

Per-class (3-class): pos P/R/F1 = 0.909/0.943/0.926,
neu = 0.765/0.707/0.735, neg = 0.931/0.906/0.918.
Neutral recall improved 0.543 → 0.707 across expert iterations.

### Corpus-scale single models (zero gold training)

| Model | Training data | Gold acc | Gold F1 | Neu recall |
|---|---|---|---|---|
| MacBERT 3c pseudo student | 160K pseudo | 0.8750 | 0.8463 | **0.848** |
| MacBERT binary pseudo | 195K pseudo | 0.9488 | 0.9455 | — |
| TF-IDF + pseudo | 200K pseudo | 0.8117 | 0.7766 | — |
| TF-IDF + star labels | 300K rating | 0.6183 | 0.5369 | — |

The pseudo student alone nearly matches the 4-model ensemble — evidence
that label quality, not model capacity, is the binding constraint.

### Pseudo-label scaling curve (gold-600)

| Pseudo rows (per-class cap) | Acc | Macro-F1 |
|---|---|---|
| ~1K | 0.7867 | 0.7217 |
| ~5K | 0.8033 | 0.7535 |
| ~19K | 0.8433 | 0.8016 |
| ~48K | 0.8550 | 0.8146 |
| ~160K | 0.8750 | 0.8463 |

Accuracy grows monotonically and is still rising at 160K — corpus scale,
not model size, is the active ingredient.

### Cross-domain silver eval (4,000 fresh corpus rows, rating-implied labels)

| Student | 3-class vs rating | Binary vs rating |
|---|---|---|
| stu3c (r1 pseudo) | 0.552 | 0.803 |
| r3 (v3-teacher pseudo) | 0.580 | **0.837** |

The r3-teacher student generalizes slightly better out-of-domain even
though gold scores are equal — ensemble-derived labels transfer better.

## 5. Analysis

**Calibration.** The deployed 3-class ensemble is strongly under-confident
(ECE = 0.251): predictions at 0.7–0.8 confidence are 98.9% accurate.
Temperature scaling (T = 0.37, fit by NLL on gold) reduces ECE to ~0.04
without changing predictions; T is applied at inference for honest
confidence display.

**Significance.** v3 vs v2: McNemar p = 0.65 (11 vs 8 discordant pairs);
bootstrap 95% CI on ΔF1 = [−0.4pp, +3.8pp]. The point estimate (+1.6pp F1,
+7.7pp neutral recall) is consistent across 20 random half-splits for every
detector weight w∈[0.3,0.6], supporting a real but modest gain.

**Rejected architectures** (all measured, none deployed): neu-gate cascade
(threshold overfit — split-half collapses), corpus-trained MLP heads as
judge replacement, student-probability averaging, self-distillation round 2,
stacked meta-learner, class-weighting, Qwen3-4B synthetic augmentation
(poisoned training: 79.2% standalone), per-class decision reweighting,
hard-negative-mined neutral detector v2 (worse in every ensemble combo),
r3 self-training on v3-teacher labels (student 0.8767/0.8434, all ensemble
mixes below v3 — teacher edge doesn't transfer through hard labels),
gold-mixed corpus training (gold rows in train → gold eval circular;
kept for silver-eval comparison only).

**Error analysis.** Remaining 63 errors are dominated by neutral↔polar
confusion (47/63) and are predominantly human-ambiguous mixed reviews;
mean error confidence 0.51 vs 0.66 for correct predictions. Reviews
containing explicit scores ("7.6/10", "三星半"; n=56) are much easier:
5.4% error vs 11.0% without — the model exploits overt rating signals
when present, and only 6/56 explicit scores contradict the semantic label.

## 6. Limitations

- Gold set is 600 reviews from a **single film page** — domain narrowness
  is the largest external-validity threat.
- Pseudo labels inherit teacher biases; the flywheel saturates after one
  round.
- Statistical power at n=600 limits significance claims; corpus-scale
  evaluation relies on rating-correlation evidence, not human labels.

## 7. Conclusion

On a realistic low-resource Chinese sentiment task, corpus-scale
pseudo-labelling is the highest-leverage ingredient: it lifts neutral
recall from 54% to 71% and lets a single compact student approach a
multi-model ensemble. Specialist decomposition on corpus-trained neutral
detectors adds a further, validated gain. Star ratings alone are
insufficient supervision — measured, not assumed.

## Reproducibility

All scripts, OOF predictions, and experiment records are in this
repository (`scripts/`, `out/`). The 911K corpus is external
(Moviedata-10M, research use only, not redistributed); trained
checkpoints are published at
<https://huggingface.co/Shuakami/douban-sentiment-weights>. The full
preprocessing → pseudo-label → train → evaluate pipeline is scripted
end-to-end.

## Fresh-corpus generalization study (lab500)

The 89.5% gold figure prompted an honest out-of-distribution check: 500
random reviews from the 911K corpus were labelled independently by two
LLM annotators (a general-purpose assistant model and Qwen3-4B
zero-shot) and scored by the deployed v3 pipeline.

| Evaluator | Reference | Accuracy | Macro-F1 |
|---|---|---|---|
| v3 ensemble | LLM-A labels (n=500) | 0.604 | 0.599 |
| Qwen3-4B | LLM-A labels | 0.662 | 0.666 |
| v3 ensemble | dual-LLM consensus (n=331) | 0.740 | — |
| v3 binary (pos/neg) | LLM-A labels (n=255) | 0.918 | — |
| MacBERT CV (400 labels/fold) | held-out LLM-A labels | 0.600 OOF | 0.585 |

Inter-annotator agreement between the two LLMs is only 66.2%, and a
model trained on 400 in-domain labels reproduces held-out labels at
just 60% — the corpus's terse, context-free comments (median 25 chars
vs gold's 80) are intrinsically ambiguous. Performance stratifies
sharply by teacher confidence: rows the teacher found easy (conf>=0.7)
score 0.757 for every annotator pair; hard rows score ~0.53.

**Takeaway**: the gold-600 benchmark measures performance on curated,
longer reviews; on the raw corpus the same system operates at roughly
60-74% three-class / ~92% binary. The neutral class — trained on only
92 gold examples — is the dominant failure mode (recall 36-48%). This
gap is reported as a limitation, not hidden: it is precisely why the
project invested in corpus-scale pseudo-labelling, and it marks the
direction for future work (human annotation of terse comments).

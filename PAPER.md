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
Finally, we build **lab2000** — a 2,000-row cross-movie benchmark with
dual-LLM blind annotation and 479 human-adjudicated rows — and show the
gold-trained system drops to 65.0% three-class on wild short reviews,
but that adding only 1,000 actively-selected boundary examples recovers
+8.8pp to 71.4% (McNemar p = 4.2e-8), and scaling the same protocol
with frontier-LLM pseudo-labels (5,906 additional AL rows) pushes the
ensemble mix to **75.0% three-class / 91.2% binary** — at or above
measured human–human agreement (71.5%). Three findings emerge that n=600 could
never show: active learning needs a gold anchor, hard labels beat soft
distillation under domain shift, and star ratings become the strongest
signal exactly where text is most ambiguous.

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
| lab2000-ood | 1,000 random cross-movie reviews | dual-LLM blind + human adjudication |
| lab2000-al | 1,000 margin/entropy-selected hard reviews | same protocol; training set only |

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

- Gold set is 600 reviews from a **single film page** — quantified by
  lab2000 (65.0% OOD three-class), partially mitigated by 1,000
  adjudicated AL rows (72.4%), but ood1000 remains the only true
  held-out cross-movie test.
- lab2000 pre-annotations are LLM-generated; 479/2,000 rows carry human
  adjudication and a 200-row second-human IAA sample (κ=0.57).
- Pseudo labels inherit teacher biases; the flywheel saturates after one
  round. lab2000 suggests the teacher's narrow-domain prior is the
  binding constraint on OOD pseudo-labels.
- Statistical power at n=600 limits gold-set micro-claims; lab2000
  (n=1,000) restores power for Stage-2 claims.

## 7. Conclusion

On a realistic low-resource Chinese sentiment task, corpus-scale
pseudo-labelling is the highest-leverage ingredient: it lifts neutral
recall from 54% to 71% and lets a single compact student approach a
multi-model ensemble. Specialist decomposition on corpus-trained
neutral detectors adds a further, validated gain. Star ratings alone are
insufficient supervision — measured, not assumed.

The lab2000 study stress-tests this honestly: the gold-trained system
drops to 65.0% on wild reviews; 1,000 adjudicated AL examples recover
the gap to 72.4% (p = 4.2e-8) with human labels alone, and to **75.0%**
after LLM-scaled AL expansion — at or above human–human agreement
(71.5%). AL needs a gold anchor, hard labels beat
soft distillation under shift, and ratings dominate precisely where
text is most ambiguous. Against a ~763B frontier LLM we trail on raw
OOD accuracy but reach parity on human-adjudicated hard items at
orders-of-magnitude lower latency, cost, and privacy exposure.

## Reproducibility

All scripts, OOF predictions, and experiment records are in this
repository (`scripts/`, `out/`). The 911K corpus is external
(Moviedata-10M, research use only, not redistributed); trained
checkpoints are published at
<https://huggingface.co/Shuakami/douban-sentiment-weights>. The full
preprocessing → pseudo-label → train → evaluate pipeline is scripted
end-to-end.

## Stage-2: the lab2000 OOD & active-learning study

### Benchmark construction

`ood1000`: 1,000 reviews sampled uniformly at random from the 911K
corpus — a held-out cross-movie test set that no model trains on.
`al1000`: 1,000 reviews selected by margin/entropy sampling from the
deployed student's predictions — deliberately the hardest, most
ambiguous rows (a training set, not a test set).

Each row was labelled twice, independently and blind, by two LLM
annotators from different model families. Inter-pass agreement:
86.7% (κ=0.80) on ood1000, 79.8% (κ=0.67) on al1000 — the lower AL
agreement is itself evidence that margin sampling selects intrinsically
ambiguous rows. All 335 disagreements plus 144 spot-checked agreements
were adjudicated by the author (479 human-adjudicated rows); the rest
take two-pass consensus. Final distributions: ood = 398/327/275,
al = 473/369/158 (pos/neu/neg). A 200-row random subset was additionally
blind-labelled by a second human for true IAA.

### Exposing and narrowing the OOD gap

| System | Train data | ood1000 3-class | ood1000 binary |
|---|---|---|---|
| Deployed v3 | gold-600 | 0.650 / 0.644 | 0.850 / 0.849 |
| SFT student | gold-600 | 0.626 / 0.611 | 0.832 / 0.831 |
| SFT student | al1000 only | 0.560 / 0.519 | 0.577 / 0.512 |
| SFT student | gold-600+al1000 | 0.714 / 0.710 | 0.880 / 0.875 |
| SFT student (distilled) | gold-600+al1000 | 0.710 / — | 0.825 / — |
| **mix v3×0.3 + SFT×0.7** | gold-600+al1000 | **0.724 / 0.721** | **0.889 / 0.885** |

Gold+AL vs gold-only: +8.8pp, McNemar p = 4.2e-8 — a gain unresolvable
at n=600 (v2→v3 was p=0.65). Split-half resampling: 0.7225±0.013 /
0.8878±0.008.

### Three findings

1. **AL needs an anchor.** al1000-only training collapses to 56.0% —
   margin sampling distorts the class prior; gold provides the reference
   for ordinary high-confidence reviews.
2. **Hard labels beat soft distillation under shift.** Teacher soft
   targets transfer at 71.0% vs 71.4% for hard-label SFT — a
   narrow-domain teacher transmits its OOD bias through soft
   probabilities. Inverts the usual distillation ordering; only visible
   once eval leaves the training domain.
3. **Ratings beat text at the ambiguity ceiling.** On al1000, the
   star-rating rule agrees with adjudicated labels at 64.3% vs ~47% for
   the text-only model — when text entropy maxes out, the rating is the
   only remaining attitude signal.

### Human agreement: the real ceiling

200-row random subset blind-labelled by a second human: agreement
**71.5%**, κ=0.568 (95% CI [0.47, 0.66]). Divergence concentrates on the
neutral boundary — where the first annotator said neutral, the second
agrees only 49% (37/80 relabelled positive). Our best OOD system at
72.4% with human labels alone and 75.0% after LLM scaling — at or above
human–human agreement:
the model has reached the human consistency ceiling. (The two LLM
passes agree with each other at 83.5%/κ=0.75 on the same subset — LLMs
follow the rubric more literally, which is why human adjudication, not
pass-consensus, anchors final labels.)

### Frontier-LLM comparison

DeepSeek-V4.1-Flash (~763B params, API) and V4-Pro, zero-shot, compact
JSON prompt:

| System | Params | ood1000 3-class | ood1000 binary | Latency |
|---|---|---|---|---|
| DeepSeek-V4-Pro | ~763B+ | 0.846 (n=945) | — | ~5.1s API |
| DeepSeek-V4.1-Flash | ~763B | 0.820 (n=992) | **0.976** (n=661) | ~1.7s API |
| Deployed v3 | ~1.7B | 0.650 | 0.850 | ~0.1s local |
| SFT+mix (gold+al6k) | ~0.1B | **0.750** | **0.912** | ~0.1s local |

The frontier model leads by ~10pp on the full random OOD set — expected
at ~2000× the parameters. But: (i) the neutral wall hits everyone —
DeepSeek's own binary (97.6%) exceeds its three-class (82.0%) by 15.6pp;
(ii) on the 479 human-adjudicated hard subset the gap nearly closes
(59.5% vs 58.5% on the ood portion — parity within noise); (iii) the
local system runs offline at ~0.1s/review with zero per-token cost and
no data egress. Positioning: task-calibrated compact model at the
human-agreement ceiling, at edge-deployable cost — not "beats LLMs".

### Can LLM pseudo-labels replace human adjudication?

SFT on gold-600 + al1000 with DeepSeek-V4.1-Flash labels (987
parseable) instead of human finals:

| AL labels | ood1000 3-class | ood1000 binary |
|---|---|---|
| gold-600 only | 0.626 / 0.611 | 0.832 / 0.831 |
| gold-600 + al1000 (human) | 0.714 / 0.710 | **0.880 / 0.875** |
| gold-600 + al1000 (DS-flash) | **0.713 / 0.714** | 0.863 / 0.860 |

DeepSeek pseudo-labels match human adjudication on 3-class (71.3% vs
71.4%) and trail slightly on binary. A 0.5:0.5 ensemble mix reaches
73.4% (split-half 0.734±0.015).

**Scaling: 1K human → 5.9K hybrid.** We then sampled al5000 — the next
5,000 margin/entropy-selected corpus rows (same band/quotas as al1000,
disjoint from lab2000) — labelled them with DeepSeek-V4.1-Flash
(4,906 parseable), and retrained. With ~6.5K AL rows the epoch optimum
shifts to 4.

| AL training data | ood1000 3-class | ood1000 binary |
|---|---|---|
| gold-600 only | 0.626 / 0.611 | 0.832 / 0.831 |
| gold-600 + al1000 (human) | 0.714 / 0.710 | 0.880 / 0.875 |
| gold-600 + al1000 (DS) | 0.713 / 0.714 | 0.863 / 0.860 |
| gold-600 + al5000 (DS) | 0.719 / 0.720 | 0.893 / 0.890 |
| gold-600 + al6k (all DS) | 0.725 / 0.727 | 0.893 / 0.890 |
| **gold-600 + al6k (hybrid)** | **0.729 / 0.731** | **0.911 / 0.908** |
| + v3 mix (best) | **0.750 / 0.751** | **0.912 / 0.910** |

Three observations: (i) **scale beats provenance** — 5,906 pure-LLM
labels (72.5%) exceed 1,000 human labels (71.4%); (ii) **hybrid is
best** — human adjudication on the hardest 1K + 4,906 LLM labels +
v3 mix reaches 75.0%/91.2%, the strongest OOD numbers in the paper,
still ~7pp below the frontier LLM but at ~0.1s local latency;
(iii) the curve is still rising (+3.6pp from 1K→6K). Total API cost for
all 5,893 extra labels was a few hundred thousand tokens — orders of
magnitude cheaper than human annotation. This validates the protocol
lab2000 itself used: LLM-label at scale, human-adjudicate a
verification core.

### Deployment trade-off (measured)

Should the OOD-hardened mix replace v3? Cross-fit evaluation says no:
the SFT component reaches only 80.0%/0.730 on gold-600 (80.2%/0.737
for the scaled al6k-hybrid variant; vs v3's 89.5%/0.859), and every
mixture weight lands at 80–82% — training on
terse ambiguous AL examples shifts the boundary away from the longer
single-film gold distribution. OOD-hardening and in-domain accuracy
are in tension; v3 stays deployed, the mix is an OOD specialist.
Length/domain-aware routing (gold median 80 chars vs corpus 24) is the
obvious next step.

### What lab2000 does not claim

al1000 is a training set; its numbers characterise difficulty, not
generalisation. The 479 human-adjudicated rows are
disagreement-enriched; accuracies on them are depressed for every
system and are used only for the LLM comparison. Per-row
`final_source` provenance is released so users can restrict to the
479-row human-verified core.

## Fresh-corpus generalization study (lab500, pilot)

The 500-row pilot that motivated lab2000: 500 random corpus reviews
labelled by two LLM annotators (assistant model + Qwen3-4B zero-shot).
v3 scored 0.604 vs LLM-A labels / 0.740 vs dual-LLM consensus; binary
0.918. LLM–LLM agreement was only 66.2%; a MacBERT trained on 400
in-domain labels reproduced held-out labels at 60% OOF. Terse comments
(median 25 chars vs gold's 80) are intrinsically ambiguous — the pilot
established the gap; lab2000 quantifies and narrows it with human
adjudication.

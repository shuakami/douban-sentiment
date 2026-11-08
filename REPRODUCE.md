# Reproducibility map

Every headline number in `PAPER.md` / `thesis/` traces to a concrete
artifact and script. This file is the index; `data/lab2000/README.md`
documents the benchmark files themselves.

## 0. Environment

```powershell
# project venv (Python + torch + transformers + sklearn + scipy)
cd douban-sentiment
$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'
$env:HF_HOME='<local HF cache>'          # models are cached locally
$env:DOUBAN_DATA_ROOT='E:/GB77_data'     # corpus + experiment artifacts (not in git)
```

- GPU jobs must be **serialized** on the ROCm/Windows box — never run
  two training/inference jobs concurrently (driver AbortHandler).
- The 911K-review corpus (`Moviedata-10M`, research-use only) is **not**
  in git; it lives under `$DOUBAN_DATA_ROOT/clean/comments.pkl`.

## 1. Gold-600 results (deployed system)

| Paper claim | Artifact | Regenerate |
|---|---|---|
| 3-class 89.5% acc / 85.9 macro-F1, neutral recall 70.7% | `out/ens_lite_3class_v3.json` | `python scripts/eval_lite_v3.py` |
| Binary 95.9% / 95.6 | `out/ens_lite_binary.json` | `python scripts/eval_lite.py` |
| TF-IDF baseline 72.3/61.9, 84.1/82.3 | `out/tfidf_*.json` | `python scripts/tfidf_oof.py` |
| Qwen3 zero-shot 78.3% (4B) | `out/qwen3_*_score.json` | `python scripts/eval_qwen3_score.py` |
| Per-channel OOF probs | `out/distill_*.json`, `out/headmlp_*.json` | `scripts/distill.py`, `scripts/distill_head.py` |

All gold metrics are stratified 5-fold OOF on
`data/reviews_600_base.jsonl` (600 rows; the 648-row file adds 48 aug
rows used only by aug-specific experiments).

## 2. Corpus-scale pseudo-label pipeline

| Paper claim | Artifact | Regenerate |
|---|---|---|
| Confident pool 271,905 rows (56.5K/68.6K/146.8K) | `$DATA_ROOT/pseudo/` | `scripts/pseudo_label.py` |
| Corpus student 87.5%/84.8% (zero gold) | `$DATA_ROOT/results/` | `scripts/train_bigdata.py` |
| Neutral detector v1 (115K, cap 60K/side) | `$DATA_ROOT/models/neu_pseudo_p60k_e2` | `scripts/train_bigdata.py --task neu` |
| Scaling curve ~1K→160K | `$DATA_ROOT/models/stu3c_pc*_e2` | `scripts/train_bigdata.py --pseudo` |
| Silver eval 0.837 binary agreement | `out/silver_*.json` | `scripts/silver_eval.py` |
| Star-rating crosstab + 81.2% binary agreement on confident pool | `out/corpus_rating_crosstab.json` | `scripts/corpus_rating_analysis.py` |
| Corpus-wide 3-class pseudo-vs-rating agreement 47.8% (54.0% on confident pool) | `out/corpus_tri_agreement.json` | `scripts/corpus_tri_agreement.py` |

## 3. lab2000 OOD / active-learning study

Working dir: `$DOUBAN_DATA_ROOT/lab2000/` (released subset in
`data/lab2000/`).

| Paper claim | Artifact | Regenerate |
|---|---|---|
| v3 OOD 65.0% 3-class / 85.0% binary | `sft_results/*ood*` + `eval_lab2000.py` | `python scripts/eval_lab2000.py` |
| gold+al1000 SFT 71.4% (+8.8pp, p=4.0e-8) | `sft_results/gold_al_*.json` | `python scripts/sft_lab2000.py` |
| al6k hybrid 74.2% mix | `sft_results/gold_extra_al6kh_*.json` | `scripts/sft_lab2000.py` |
| al11k/al14k/al17k mixes 74.9–77.1% | `sft_results/gold_extra_al*b*_*.json` | `scripts/sft_lab2000.py` |
| **Cross-fit mix weights + all mix metrics** | `mix_crossfit_results.json`, `mix_oof_probs.npz` | `python scripts/mix_crossfit.py` |
| McNemar p-values (0.0015, 1.7e-3, 0.041, …) | `mix_oof_probs.npz` | `python /e/GB77_data/lab2000/mix_metrics_recompute.py` |
| Length/provenance stratification (this paper's §stratification) | `length_strat_results.json`, `stratification_extra_results.json` | `python /e/GB77_data/lab2000/length_strat_eval.py` |
| Second-human IAA 71.5% / κ=0.568 | `data/lab2000/iaa_friend_user.json`, `friend_IAA_map.json` | aggregates released; protocol in `friend_IAA_README.md` |
| LLM pass agreement 83.5% / κ=0.75 (on the 200-row IAA subset; 83.3%/0.74 on all 2000) | `lab2000_final.jsonl` (`label_a`/`label_b`) + `friend_IAA_map.json` | recompute from file |
| DeepSeek-V4.1-Flash / V4-Pro OOD | `data/lab2000/ds_deepseek_*.jsonl` | labels released; scoring via `scripts/eval_lab2000.py` |
| Qwen3-4B teacher ablation | `data/lab2000/qwen_*` | labels + probs released |
| fresh500 al14k 74.6%/92.6% vs v3 58.8%/86.3% | `fresh500*.jsonl/.npy`, `fresh500_sft_probs/` | `/e/GB77_data/lab2000/eval_fresh500_mix.py` |
| fresh500 al17kds 77.0%/93.7% vs v3 (headline mix) | `fresh500_sft_probs_al17k/` | `/e/GB77_data/lab2000/sft_fresh500_al17k.py`, `eval_fresh500_mix_al17k.py` |
| gold-600 cross-fit deployment numbers | `crossfit_gold600*.json` | `scripts/crossfit_gold600.py` |

## 4. Comparisons & audits

| Paper claim | Artifact |
|---|---|
| StructBERT (Aliyun DAMO) 86.6/84.6 on 508-subset | `scripts/to_aliyun_format.py` inputs in `data/aliyun/`; weights `damo/nlp_structbert_sentiment-classification_chinese-base` (ModelScope) |
| Aliyun PAI page audit (43-row effective val) | analysis in `thesis/`; input files `data/aliyun/` |
| lab500 pilot (Appendix) | `scripts/lab500_cv.py`, appendix table |
| >20 rejected architectures | `out/*.json` experiment records + `AGENTS.md` notes |

## 5. Figures & documents

| Item | Source |
|---|---|
| All paper figures | `thesis/make_figures.py` → `thesis/fig_*.pdf` |
| EN/ZH PDFs | `thesis/thesis.tex`, `thesis/thesis_zh.tex` (xelatex, 2 passes) |
| HF weights | `Shuakami/douban-sentiment-weights` (sha256-verified vs `models/`) |
| Live demo | `web/app.py` + `web/manifest.json`; Space `Shuakami/douban-sentiment` |

## 6. What is NOT reproducible from the repo alone

- Raw Moviedata-10M corpus (license: research use, not redistributed)
- SFT experiment checkpoints on ood1000/al* (GPU-trained; prob
  outputs are released in `sft_results/` + `mix_oof_probs.npz`)
- DeepSeek API labels (released as jsonl; regenerating costs API tokens)
- The second-human IAA labels per-row (aggregates released; protocol
  documented for replication)

# SynthID-Text Watermarking in Chinese-to-English Machine Translation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

Data, analysis code, and generated figures for:

> **Linguistic Patterns and Keyed Watermarks in Chinese-to-English Machine Translation: A Paired Study**
> Liu Chang — ENG2001, CUHK-Shenzhen

## Overview

The study asks two questions with one paired design. Sixty Chinese expository passages (400–600 Han
characters, selected from LCMC category J) were translated twice by **Qwen2.5-3B-Instruct** under an
identical prompt, identical sampling settings and an identical seed — once with **SynthID-Text**
enabled and once disabled. Every source text is therefore its own control, and the only manipulated
factor is whether the watermarking processor was attached during decoding.

| Outcome | Watermark disabled | Watermark enabled | Paired difference | Test |
|---|---|---|---|---|
| Correct-key mean g-score (primary) | 0.5003 | 0.5403 | +0.0399 [+0.0372, +0.0427] | t(59) = 28.97, p < .001, d_z = 3.74 |
| Score gap (secondary) | 0.00014 | 0.04035 | +0.0402 [+0.0372, +0.0433] | t(59) = 26.41, p < .001, d_z = 3.41 |
| MATTR (window 50) | 0.8094 | 0.8099 | +0.0005 [−0.0056, +0.0066] | t(59) = 0.16, p = .876, d_z = 0.02 |
| Repeated trigram ratio | 0.0446 | 0.0466 | +0.0020 [−0.0053, +0.0094] | t(59) = 0.55, p = .583, d_z = 0.07 |

Detection evidence separates completely — 60/60 pairs increase, and the two conditions do not overlap
at the item level (disabled max 0.5144 < enabled min 0.5246) — while both lexical measures stay inside
their null intervals. The watermark is detectable; the translation's lexical profile is unchanged.

## Repository structure

```
.
├── data/
│   ├── translation_results.csv   # 60 pairs: zh_source, en_watermark_off, en_watermark_on, tokens, seed
│   ├── metrics_per_item.csv      # per-item MATTR, trigram ratio, g-score for both conditions
│   └── score_gap_results.csv     # per-item correct-key and incorrect-key scores, gap before/after
├── scripts/
│   ├── run_experiment_synthid.py # generation: paired translations, fixed seed, watermark on/off
│   ├── compute_metrics.py        # MATTR, repeated trigram ratio, correct-key mean g-score, paired stats
│   ├── compute_score_gap.py      # score gap against five incorrect keys (+ cross-check vs compute_metrics)
│   ├── score_gap_analysis.py     # independent re-implementation, used to cross-check the gap file
│   ├── validate_metrics.py       # negative control (shifted / random keys) and trigram-definition checks
│   ├── normality_check.py        # Shapiro–Wilk and Wilcoxon signed-rank robustness checks
│   ├── compare_defs.py           # four operationalisations of the repeated trigram ratio
│   ├── report_extras.py          # aligned word counts and per-item rise/fall counts
│   ├── final_stats.py            # statistics table used in the manuscript
│   ├── subset_stats.py           # the 50-passage main-only subset, reported as a sensitivity check
│   ├── make_paper_assets.py      # Table 1, Figures 1–2, LaTeX tables, figure captions
│   └── appendix_tables.py        # Table A1 (configuration) and Table A2 (corpus composition)
├── results/
│   ├── fig1_paired_slope.png
│   ├── fig2_forest.png
│   ├── tables.md                 # Table 1 (revised) and Table 2 (descriptive statistics)
│   ├── tables.tex                # booktabs version
│   ├── appendix_tables.md        # Tables A1–A2
│   ├── figure_captions.md        # captions plus sentences to cite the figures
│   └── metrics_summary.md        # full report: definitions, sensitivity checks, negative control
├── manuscript/
│   └── Results_RQ1_detection.txt # RQ1 results paragraph as submitted
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
└── .gitignore
```

Every script resolves paths from its own location (`scripts/` → repo root → `data/`, `results/`), so
they can be run from any working directory.

## Requirements

Python 3.13 was used for all reported numbers (`transformers` 5.22.2, `torch` 2.11.0 + CUDA 12.8,
`lexicalrichness` 0.5.1, `scipy` 1.18.1, `matplotlib` 3.11.2).

```bash
# CPU-only or a different CUDA build: install torch your usual way, then
pip install -r requirements.txt

# the CUDA 12.8 build used here
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

A GPU is only needed to *regenerate* the translations. All analysis scripts load the tokenizer alone,
so the metric pipeline runs on CPU in a few minutes.

## Getting the model

Weights are not distributed here (6.17 GB). Fetch them once and point the scripts at the directory —
the three scripts that need it read `QWEN_MODEL_DIR`, and fall back to `../experiment/Qwen2.5-3B-Instruct`
when the variable is unset:

```bash
# option A: ModelScope
pip install modelscope
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./Qwen2.5-3B-Instruct

# option B: Hugging Face
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./Qwen2.5-3B-Instruct

export QWEN_MODEL_DIR=$(pwd)/Qwen2.5-3B-Instruct    # Windows: set QWEN_MODEL_DIR=%cd%\Qwen2.5-3B-Instruct
```

## Quick start

Reproduce the statistics and the paper's tables and figures from the shipped CSV (no GPU, no weights
download beyond the tokenizer):

```bash
python scripts/compute_metrics.py      # -> data/metrics_per_item.csv, results/metrics_summary.md
python scripts/compute_score_gap.py    # -> data/score_gap_results.csv (cross-checks its g-score)
python scripts/normality_check.py      # -> Shapiro–Wilk and Wilcoxon results
python scripts/make_paper_assets.py    # -> results/fig1_paired_slope.png, fig2_forest.png, tables.md, tables.tex
python scripts/appendix_tables.py      # -> results/appendix_tables.md
```

Reproduce the translations themselves (GPU, ~45 min for 60 pairs on one consumer card):

```bash
python scripts/run_experiment_synthid.py                    # full run from data/translation_results.csv
python scripts/run_experiment_synthid.py --limit 3          # smoke test
python scripts/run_experiment_synthid.py --out data/regenerated.csv   # annotate-only rerun elsewhere
```

Reruns resume automatically: any `id` already present in the `--out` file is skipped, so an interrupted
run can be restarted with the same command.

## Method notes

**Length matching.** The longer translation of each pair is truncated to the shorter one's
whitespace-token count before any lexical measure or g-score is computed. Both conditions are then
scored on texts of identical length; 35 pairs kept the enabled side intact, 24 kept the disabled side,
1 was equal.

**g-score.** Detection uses the *generation-side* `SynthIDTextWatermarkLogitsProcessor` with the exact
key set, `ngram_len = 5`, `sampling_table_size = 2^16`, `sampling_table_seed = 0` used at generation
time. `compute_g_values()` is averaged over all valid cells after multiplying by
`compute_context_repetition_mask()` and `compute_eos_token_mask()[:, ngram_len-1:]`. This is the
mean-score statistic from the SynthID-Text paper; the Bayesian detector head is **not** used, because
its pretrained weights are not distributed with `transformers` — a caveat stated in the paper's
limitations.

**Negative control.** A *permuted* key set is not a valid control: `accumulate_hash()` is a linear
congruential update, so permuting keys only adds a position-independent constant that disappears when
the per-depth values are averaged — a permuted key reproduces the correct-key score exactly. The control
therefore uses keys with *different values*: shifting every key by +37 gives ΔM = +0.0041 (t = 3.93) and
a fully random key set gives ΔM = −0.0015 (t = −1.20, p = .235). Un-watermarked text scores 0.5003 under
the correct key, with only 28 of 60 items above 0.5 — the behaviour expected of random text.

**Trigram ratio definitions.** Four definitions were computed (token-based vs type-based, with and
without punctuation); none of them yields a significant difference (t = 0.36–1.16, p = .42–.72),
so the null result does not depend on the operationalisation. See `metrics_summary.md`.

**Robustness.** Paired differences are close to normal (Shapiro–Wilk p = .070 / .071 / .116 / .044) and
Wilcoxon signed-rank tests agree with the t tests in every case (detection p = 1.6 × 10⁻¹¹ for both;
MATTR p = .918; trigram p = .244).

## Data notes

- `translation_results.csv` holds the 60 source passages together with both translations. The source
  passages come from **LCMC category J** and remain under the corpus's own terms; they are included here
  for verification of the reported numbers, not for redistribution. If you need the full corpus, obtain
  it from Lancaster — the file IDs (`J03`, `J20`, …) let you rebuild the exact sample.
- The 60 items are 50 main + 10 reserve passages drawn from 49 LCMC files; the paper reports the pooled
  n = 60 (numbers for the 50-passage subset are in `metrics_summary.md`).
- The watermark keys are published deliberately: they are this study's own test keys, and reproduction
  of the detection scores requires them.
- `results/metrics_summary.md` is the analysis log written while the numbers were produced (Chinese
  prose, kept verbatim); the English summary is the results table above.

## Limitations

One model, one translation direction, one text type, 60 source texts. `skip_special_tokens=True` during
decoding means the detected token sequence is a re-tokenisation of the output rather than a bit-exact
replay; the 60/60 separation suggests this does not affect the conclusion. MATTR and the repeated
trigram ratio do not exhaust stylistic space, and no human evaluation of quality or authorship was run.

## Citation

```bibtex
@misc{liu2026synthidmt,
  title  = {Linguistic Patterns and Keyed Watermarks in Chinese-to-English Machine Translation: A Paired Study},
  author = {Liu, Chang},
  year   = {2026},
  note   = {Course paper, ENG2001, CUHK-Shenzhen},
  url    = {https://github.com/stefanokratzdisteln-hash/synthid-mt-watermark}
}
```

## License

Code is released under the MIT License — see [LICENSE](LICENSE). The Chinese source passages retain the
terms of the LCMC corpus and are not relicensed here.

# Phase T5 — LSTM vs Transformer: the formal comparison

Test = PIE set03, N=2094 windows. 10,000 paired percentile bootstrap resamples of Delta-AUC (`np.random.default_rng(42)`, same resample indices applied to both models' probabilities each iteration -- ROC-AUC via tie-corrected Mann-Whitney). Headline = seed-averaged probability vectors (mean of the 5 seeds' sigmoid outputs, elementwise). **PARITY GATE PASSED** before any Delta below was computed: every recomputed LSTM per-seed test AUC matched its stored `final.json` value EXACTLY (delta 0.00e+00, all 5 seeds; the gate's tolerance is 1e-6, the observed drift was zero). (The transformer's own recomputed probabilities show a ~1e-6 drift from its stored `final.json` values — expected CPU/GPU floating-point device drift, PLAN.md §10, since Phase T4 trained on Kaggle T4 GPU and this script runs locally on CPU; confirmed far below the 1e-4 threshold that would indicate a real bug, and orders of magnitude too small to affect any number reported below.)

## Verdict

**WIN.** The paired-bootstrap 95% CI of Delta-AUC excludes 0 ([+0.0097, +0.0174], Delta=+0.0135) and the paired t-test is significant (p<0.05). `transformer_searched` measurably beats the frozen BiLSTM on the identical 2094 test windows, at matched protocol and a >=2x search budget. Adopted as the headline model for this AUC-first comparison, with param-count and latency caveats reported alongside (see `04_final_summary.md`, `06_latency_report.md`); the BiLSTM is retained as the efficient, lower-latency baseline. METRIC SCOPE: this WIN is specific to AUC -- under the supervisor's later F1-first hierarchy, identical F1-first optimization of both families ends in a statistical TIE on F1 (dF1 +0.0008, 95% CI [-0.0124, +0.0142]; `f1_optimization/06_comparison_report.md`), so model-choice claims must carry the metric qualifier.

---

## Headline: absolute test AUC (seed-averaged probabilities, Issue-4-format 95% CI)

| model | params | test AUC | 95% CI (ROC) | PR-AUC | 95% CI (PR) |
|---|---|---|---|---|---|
| BiLSTM baseline (frozen) | 594,561 | **0.9423** | [0.9306, 0.9533] | 0.8891 | [0.8607, 0.9161] |
| transformer_searched (winner) | 794,241 | **0.9558** | [0.9453, 0.9656] | 0.9104 | [0.8832, 0.9363] |
| transformer_default | 268,417 | **0.9428** | [0.9312, 0.9538] | 0.8957 | [0.8712, 0.9185] |

**Note on the BiLSTM's 0.9423 above vs. the 0.9324 ± 0.0114 you'll see everywhere else in this repo (journal_prep, the manuscript, this project's own other tables) — these are two different, both-correct statistics over the exact same zero-leakage `journal_prep/issue2_clean_protocol` checkpoints, not a data or correctness discrepancy:**
- **0.9324 ± 0.0114** = the plain average of the 5 seeds' own independent test AUCs. This is the canonical, citable BiLSTM number used everywhere outside this specific report.
- **0.9423** (this table) = the AUC obtained by averaging the 5 seeds' *predicted probabilities* together first (an implicit small ensemble), then scoring that one combined probability vector. It exists only because the paired bootstrap below needs a single probability vector per model to compare — it is not a replacement headline figure, and the 0.9324 number remains the one to cite as "the BiLSTM's AUC."

## Primary comparison — paired bootstrap of Delta-AUC (transformer_searched - lstm)

Delta = **+0.0135**, 95% CI **[+0.0097, +0.0174]** (excludes 0).

Secondary evidence — paired t-test over the 5 training seeds: t=3.498, p=0.0249 (significant at 0.05). **n=5 is low statistical power** (PLAN.md §6) — the window-paired bootstrap above (n=2094 windows) is the primary evidence, this is a secondary check, not the deciding one on its own.

### Per-seed-pair Delta (transformer_searched_auc - lstm_auc)

| seed | lstm AUC | transformer_searched AUC | Delta |
|---|---|---|---|
| 42 | 0.9131 | 0.9489 | +0.0357 |
| 0 | 0.9334 | 0.9474 | +0.0140 |
| 1 | 0.9432 | 0.9499 | +0.0067 |
| 2 | 0.9363 | 0.9484 | +0.0121 |
| 3 | 0.9358 | 0.9538 | +0.0180 |

## Secondary comparison — transformer_default vs lstm (context, not the verdict)

Delta = **+0.0005**, 95% CI **[-0.0034, +0.0043]**, paired t-test p=0.8274. Reported for completeness ("you never tuned the transformer" preempt) — the pre-registered verdict is driven by `transformer_searched` (the actual staged-search winner), not this un-searched default.

## Per-seed absolute test AUC

| seed | lstm | transformer_searched | transformer_default |
|---|---|---|---|
| 42 | 0.9131 | 0.9489 | 0.9362 |
| 0 | 0.9334 | 0.9474 | 0.9294 |
| 1 | 0.9432 | 0.9499 | 0.9409 |
| 2 | 0.9363 | 0.9484 | 0.9264 |
| 3 | 0.9358 | 0.9538 | 0.9356 |

---

*Pre-registered decision rule and verdict templates: `transformer/PLAN.md` §6. This report implements that rule verbatim; the verdict above was not known before this script ran.*

# Phase T5 — Final 5-seed results (Stage D)

Aggregated from `../phase4_kaggle_final/runs_final/*/seed<k>/final.json` (downloaded Kaggle T4×2 output, independently re-verified in `PROGRESS_LOG.md`). This is pure aggregation — the formal comparison against the frozen LSTM (paired bootstrap + verdict) is `05_compare_vs_lstm.py`.

**Determinism-of-record check: PASS** — `transformer_searched` seed42 rerun reproduces val_auc=0.982463, test_auc=0.948863 exactly (|delta| = 0.0e+00).

## 5-seed mean ± std (test set03, N=2094)

| config | params | val AUC | test AUC | test PR-AUC | test F1 | test Acc |
|---|---|---|---|---|---|---|
| `transformer_searched` | 794,241 | 0.9789 ± 0.0038 | **0.9497 ± 0.0025** | 0.9010 ± 0.0102 | 0.8446 ± 0.0129 | 0.8942 ± 0.0089 |
| `transformer_default` | 268,417 | 0.9629 ± 0.0056 | **0.9337 ± 0.0058** | 0.8733 ± 0.0073 | 0.8159 ± 0.0145 | 0.8778 ± 0.0136 |
| BiLSTM baseline (frozen, Issue 4) | 594,561 | 0.9644 ± 0.0043 | **0.9324 ± 0.0114** | 0.876 | 0.8275 ± 0.0123 | 0.8827 ± 0.0091 |

## Per-seed detail

| config | seed | best_epoch | val AUC | test AUC | test F1 | test Acc |
|---|---|---|---|---|---|---|
| `transformer_searched` | 42 | 19 | 0.9825 | 0.9489 | 0.8498 | 0.8992 |
| `transformer_searched` | 0 | 12 | 0.9727 | 0.9474 | 0.8307 | 0.8811 |
| `transformer_searched` | 1 | 32 | 0.9808 | 0.9499 | 0.8410 | 0.8926 |
| `transformer_searched` | 2 | 28 | 0.9785 | 0.9484 | 0.8374 | 0.8930 |
| `transformer_searched` | 3 | 22 | 0.9803 | 0.9538 | 0.8642 | 0.9050 |
| `transformer_default` | 42 | 12 | 0.9594 | 0.9362 | 0.8229 | 0.8849 |
| `transformer_default` | 0 | 21 | 0.9651 | 0.9294 | 0.8252 | 0.8849 |
| `transformer_default` | 1 | 34 | 0.9698 | 0.9409 | 0.8304 | 0.8911 |
| `transformer_default` | 2 | 4 | 0.9553 | 0.9264 | 0.8042 | 0.8577 |
| `transformer_default` | 3 | 8 | 0.9650 | 0.9356 | 0.7970 | 0.8706 |

**This table alone is not the verdict** — raw mean-vs-mean comparisons don't account for sampling noise on the shared 2094 test windows. See `05_comparison_report.md` for the pre-registered paired-bootstrap comparison and Verdict (PLAN.md §6).

---

**Metric-scope note (2026-07-13):** the F1/acc columns above are raw @0.5 values from
the AUC-first protocol and invite an untested "transformer is F1-better" reading
(0.8446 vs 0.8275). That comparison was subsequently run properly, with identical
F1-first optimization of both families (`f1_optimization/`): the LSTM improves to
F1 0.8444 ± 0.0078 and the families **TIE on F1** (ΔF1 +0.0008, CI includes 0). The
AUC WIN is unaffected; F1 claims must cite the F1-first program, not this table.

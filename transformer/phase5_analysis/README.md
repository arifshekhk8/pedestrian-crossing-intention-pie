# phase5_analysis/ — DONE, verdict: WIN on AUC

Phase T5 (see `../PLAN.md` §7): the local analysis that turns Phase T4's checkpoints into
the actual answer — does the transformer beat the LSTM? **On AUC, yes: WIN** (on F1, after identical F1-first optimization of both families, they TIE — `f1_optimization/`).**

Contains:
- `04_final_report.py` → `04_final_results.csv`, `04_final_summary.md` — aggregates
  `../phase4_kaggle_final/runs_final/` into 5-seed mean±std tables (winner + default vs
  the frozen LSTM row), plus the determinism-check cross-reference.
- `05_compare_vs_lstm.py` → `05_comparison_{results.csv,results.json,report.md,figure.png}`
  — **the formal comparison.** Regenerates probabilities from the frozen LSTM checkpoints
  (`journal_prep/issue2_clean_protocol/runs_clean/multiseed/`) and the transformer
  checkpoints, runs the LSTM parity gate, a 10k paired bootstrap of ΔAUC, and a paired
  t-test, then reports one of the three pre-registered verdict templates (`../PLAN.md` §6).
- `06_latency_transformer.py` → `06_latency_{results.json,report.md}` — Issue-9 protocol
  applied to `transformer_searched` on this M4, compared against the BiLSTM's
  0.575 ms/window.
- `07_loso_report.py` → `07_loso_{results.csv,report.md}` — aggregates
  `../phase4_kaggle_final/runs_loso/` against the LSTM's 6-fold LOSO
  (`journal_prep/issue5_loso_cv/05_loso_results.csv`, read directly rather than
  hand-transcribed).

## Result

**Verdict: WIN.** Paired bootstrap (10k resamples, same 2094 test windows, same
resample indices applied to both models each iteration — `np.random.default_rng(42)`)
of ΔAUC (`transformer_searched` − BiLSTM), on seed-averaged probability vectors:

**Δ = +0.0135, 95% CI [+0.0097, +0.0174]** — excludes 0. Paired t-test over the 5
training seeds: t=3.498, p=0.0249 — significant at 0.05. Both pre-registered
conditions for WIN are met.

| model | params | test AUC (seed-avg) | 95% CI (ROC) |
|---|---|---|---|
| BiLSTM baseline (frozen) | 594,561 | 0.9423 | [0.9306, 0.9533] |
| transformer_default | 268,417 | 0.9428 | [0.9312, 0.9538] |
| **transformer_searched (winner)** | 794,241 | **0.9558** | [0.9453, 0.9656] |

**Note:** the BiLSTM's 0.9423 above is a different statistic from the canonical
`0.9324 ± 0.0114` cited everywhere else in the repo — this one is the AUC of the 5
seeds' averaged probabilities (needed for the paired bootstrap below to have a single
probability vector per model), not the plain mean of the 5 seeds' own AUCs. Both are
computed from the identical zero-leakage checkpoints; `0.9324 ± 0.0114` remains the
number to cite as "the BiLSTM's AUC." Full explanation: `05_comparison_report.md`.

**The secondary comparison is the tell:** `transformer_default` vs BiLSTM is a clean
**TIE** (Δ=+0.0005, CI [-0.0034, +0.0043], p=0.827) — an un-searched transformer with
the LSTM's own recipe does *not* beat it. The win came from the 78-config staged
search finding a genuinely better architecture (deeper, last-token pooling, sinusoidal
PE), not from switching architecture families alone. This is exactly the story
Issue 8 told for the LSTM's own grid search, now told for the transformer too.

**Parity gate passed exactly:** every recomputed LSTM per-seed test AUC matched its
stored `final.json` value to `0.00e+00`. A non-mandated sanity check on the
transformer's own checkpoints found a consistent ~1e-6 to 8e-6 drift across all 10
runs between locally-recomputed and Kaggle-GPU-stored test AUC — this is the CPU/GPU
floating-point device drift `PLAN.md` §10 explicitly pre-registered as an expected
risk (Phase T4 trained on Kaggle T4 GPU; this recomputation runs on local CPU), not a
bug — confirmed via a direct test that ruled out batch-size as the cause, and the
magnitude is 2-3 orders of magnitude too small to affect any reported number.

**LOSO (winner, 6-fold, recomputed with `ddof=1` for consistency — see `07_loso_report.md`
for why this differs slightly from Issue 5's own published `ddof=0` number):**
0.939 ± 0.044 vs the BiLSTM's 0.928 ± 0.045. Directionally consistent with the
fixed-split win, though 6 folds is far too few for a hypothesis test — reported as a
generalization sanity check, not primary evidence.

**Latency (M4, this deployment hardware):** the searched transformer is *faster* than
the BiLSTM per window at CPU batch-1 (0.459 ms vs 0.575 ms, a 1.25× factor) despite
~1.3× the parameters (794,241 vs 594,561) — the fully parallel self-attention forward
pass over T=16 tokens apparently outruns the BiLSTM's inherently sequential
recurrence on this hardware/batch size. Both are ~2 orders of magnitude inside a
30 fps frame budget regardless; latency is not a deployment concern either way.

Prerequisite (met): Phase T4's checkpoints exist (`../phase4_kaggle_final/runs_final/`,
`runs_loso/`).

**Next: Phase T6** — close remaining `PLAN.md` DONE blocks, slot these numbers into
`journal_prep/issue3_baseline_comparison/` and the manuscript, update root docs.

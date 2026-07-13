# Phase T5 — LOSO 6-fold CV (transformer_searched, seed 42)

Issue-5 protocol applied to the Phase-T3 winner (`d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05`): per-fold test = one PIE set, train+val = the other 5 (85/15 pedestrian-grouped split), per-fold pos_weight + train-only norm. Identical architecture/recipe across folds. Aggregated from `../phase4_kaggle_final/runs_loso/<fold>.json` (downloaded Kaggle output, independently re-verified in `PROGRESS_LOG.md`).

| Fold (test set) | test N | pos % | AUC | PR-AUC | F1 | Acc | pos_w | best ep | LSTM AUC (ref, Issue 5) |
|---|---|---|---|---|---|---|---|---|---|
| set01 | 258 | 54.7 | 0.9050 | 0.8259 | 0.9046 | 0.8953 | 2.084 | 21 | 0.9055 |
| set02 | 310 | 57.4 | 0.9212 | 0.9189 | 0.8725 | 0.8548 | 2.103 | 31 | 0.8822 |
| set03 | 2094 | 32.5 | 0.9496 | 0.8914 | 0.8457 | 0.8949 | 1.940 | 29 | 0.9307 |
| set04 | 1610 | 30.6 | 0.8847 | 0.7401 | 0.7140 | 0.7876 | 1.928 | 55 | 0.8916 |
| set05 | 47 | 31.9 | 1.0000 | 1.0000 | 0.9677 | 0.9787 | 2.104 | 22 | 0.9979 |
| set06 | 587 | 23.9 | 0.9746 | 0.9069 | 0.8516 | 0.9216 | 1.935 | 35 | 0.9628 |

**Mean ± std over 6 folds (unweighted): AUC 0.9392 ± 0.0436, PR-AUC 0.8806 ± 0.0886, F1 0.8594 ± 0.0841.**

Excluding the tiny, uninterpretable set05 fold (N=47), the 5 folds with N≥100: AUC 0.9270 ± 0.0357.

**vs BiLSTM (Issue 5):** 6-fold 0.928 ± 0.045, excl. set05 0.915 ± 0.033 — recomputed here from Issue 5's raw `05_loso_results.csv` with `ddof=1` (sample std) for consistency with this report's own convention; Issue 5's originally-published text quotes `0.928 ± 0.041` / `0.915 ± 0.029` using `ddof=0` (population std) — same underlying per-fold numbers, not a data discrepancy. Transformer LOSO mean is above the BiLSTM's by 0.0107 (unweighted 6-fold), above by 0.0124 excl. set05.

**This is a descriptive fold table, not a hypothesis test** — 6 folds is too few for a paired test to have any power; it's reported as a generalization sanity check (does the win/loss on the fixed test03 split hold up when every set gets a turn as the held-out fold), not as the primary evidence. The primary evidence is `05_comparison_report.md`'s window-paired bootstrap on the fixed test03 split.

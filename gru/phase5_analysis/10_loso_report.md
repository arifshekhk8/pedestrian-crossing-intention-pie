# GRU study — Phase G5 Leave-One-Set-Out CV (generalization check)

GRU F1-winner (h256), 6 folds, seed 42, Issue-5 protocol (per-fold pedestrian-grouped 85/15 val split, per-fold pos_weight + train-only norm, AUC-selected so folds are comparable to the BiLSTM/Transformer LOSO). 6 folds is descriptive, not a hypothesis test — the fixed-split paired + cluster bootstrap (07/08) is the actual evidence.

| Fold (test set) | test N | pos % | AUC | PR-AUC | F1 | Acc | pos_w | best ep |
|---|---|---|---|---|---|---|---|---|
| set01 | 258 | 54.7 | 0.945 | 0.946 | 0.874 | 0.864 | 2.08 | 34 |
| set02 | 310 | 57.4 | 0.931 | 0.914 | 0.870 | 0.855 | 2.10 | 14 |
| set03 | 2094 | 32.5 | 0.928 | 0.879 | 0.842 | 0.895 | 1.94 | 26 |
| set04 | 1610 | 30.6 | 0.897 | 0.746 | 0.733 | 0.816 | 1.93 | 22 |
| set05 | 47 | 31.9 | 0.998 | 0.996 | 0.968 | 0.979 | 2.10 | 15 |
| set06 | 587 | 23.9 | 0.975 | 0.890 | 0.856 | 0.928 | 1.94 | 24 |

**GRU LOSO (6-fold, unweighted): AUC 0.946 ± 0.036, F1 0.857.** Excluding the tiny set05 fold (N=47, near-perfect and uninterpretable): AUC 0.935 ± 0.028.

## Comparison (LOSO mean AUC, all AUC-selected, ddof=1)

| model | 6-fold AUC | excl. set05 | source |
|---|---|---|---|
| **GRU (this study)** | **0.946 ± 0.036** | 0.935 | this folder |
| BiLSTM | 0.928 ± 0.041 | 0.915 | Issue 5 |
| Transformer | 0.939 ± 0.044 | 0.927 | transformer/phase5 |

The GRU's fold-average AUC (0.946) sits in the same band as the BiLSTM (0.928) and Transformer (0.939) — the cross-set generalization is consistent with the fixed-split finding, and set03 is not an unusually easy fold for the GRU (its set03 AUC 0.928 ≈ its fixed-split number). Individual folds vary (set04 is the hardest for all three families); 6 folds is too few for a significance test, so this is reported as a generalization sanity check only.

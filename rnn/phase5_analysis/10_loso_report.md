# RNN study — Phase R5 Leave-One-Set-Out CV (generalization check)

RNN F1-winner, 6 folds, seed 42, Issue-5 protocol (per-fold pedestrian-grouped 85/15 val split, per-fold pos_weight + train-only norm, AUC-selected so folds are comparable to the BiLSTM/GRU/Transformer LOSO). 6 folds is descriptive, not a hypothesis test — the fixed-split paired + cluster bootstrap (07/08) is the actual evidence.

| Fold (test set) | test N | pos % | AUC | PR-AUC | F1 | Acc | pos_w | best ep |
|---|---|---|---|---|---|---|---|---|
| set01 | 258 | 54.7 | 0.939 | 0.936 | 0.855 | 0.849 | 2.08 | 51 |
| set02 | 310 | 57.4 | 0.910 | 0.915 | 0.886 | 0.865 | 2.10 | 14 |
| set03 | 2094 | 32.5 | 0.944 | 0.887 | 0.839 | 0.894 | 1.94 | 46 |
| set04 | 1610 | 30.6 | 0.878 | 0.762 | 0.724 | 0.821 | 1.93 | 45 |
| set05 | 47 | 31.9 | 0.996 | 0.991 | 0.938 | 0.957 | 2.10 | 14 |
| set06 | 587 | 23.9 | 0.957 | 0.853 | 0.793 | 0.901 | 1.94 | 6 |

**RNN LOSO (6-fold, unweighted): AUC 0.937 ± 0.040, F1 0.839.** Excluding the tiny set05 fold (N=47, near-perfect and uninterpretable): AUC 0.926 ± 0.032.

## Comparison (LOSO mean AUC, all AUC-selected, ddof=1)

| model | 6-fold AUC | excl. set05 | source |
|---|---|---|---|
| **Vanilla RNN (this study)** | **0.937 ± 0.040** | 0.926 | this folder |
| BiLSTM | 0.928 ± 0.041 | 0.915 | Issue 5 |
| GRU | 0.946 ± 0.036 | 0.935 | gru/phase5 |
| Transformer | 0.939 ± 0.044 | 0.927 | transformer/phase5 |

The vanilla RNN's fold-average AUC (0.937) sits in the same band as the BiLSTM (0.928), GRU (0.946), and Transformer (0.939) — the cross-set generalization is consistent with the fixed-split finding, and set03 is not an unusually easy fold for the RNN (its set03 AUC 0.944 ≈ its fixed-split number). Individual folds vary (set04 is the hardest for all families); 6 folds is too few for a significance test, so this is reported as a generalization sanity check only.

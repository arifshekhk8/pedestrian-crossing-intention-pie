# RNN study — Phase R5 pedestrian-cluster bootstrap (the honest CI)

Windows are pedestrian-correlated (541 clusters for 2094 windows, 50% overlap), so i.i.d.-window CIs understate uncertainty. Every endpoint's primary-metric delta is recomputed by resampling PEDESTRIANS (all-windows-per-drawn-ped, 10k resamples, paired clusters across models) — machinery reused verbatim from `f1_optimization/07_cluster_bootstrap.py`. **Cluster intervals are the ones to quote.**

| endpoint | metric | Δ | window CI (07) → verdict | cluster CI → verdict | survives? |
|---|---|---|---|---|---|
| (1) f1_winner vs frozen_bilstm | F1 | +0.0220 | [+0.0111, +0.0327] WIN | [+0.0097, +0.0354] WIN | ✓ |
| (2) f1_winner vs bilstm_f1 | F1 | +0.0033 | [-0.0083, +0.0150] TIE | [-0.0130, +0.0187] TIE | ✓ |
| (3) f1_winner vs transformer_f1 | F1 | +0.0025 | [-0.0079, +0.0131] TIE | [-0.0111, +0.0171] TIE | ✓ |
| (4) f1_winner vs gru_f1 | F1 | -0.0038 | [-0.0117, +0.0039] TIE | [-0.0128, +0.0049] TIE | ✓ |
| (5) default_f1 vs frozen_bilstm | F1 | +0.0140 | [+0.0023, +0.0255] WIN | [-0.0005, +0.0288] TIE | ⚠ changes |
| (6) default_auc vs frozen_bilstm | AUC | +0.0059 | [+0.0032, +0.0088] WIN | [+0.0012, +0.0110] WIN | ✓ |
| (7) winner_auc vs frozen_bilstm | AUC | +0.0121 | [+0.0087, +0.0157] WIN | [+0.0063, +0.0187] WIN | ✓ |
| (8) winner_auc vs searched_tf | AUC | -0.0013 | [-0.0041, +0.0015] TIE | [-0.0061, +0.0033] TIE | ✓ |

| RNN arm | ens F1 95% cluster CI | ens AUC 95% cluster CI |
|---|---|---|
| `rnn_f1_winner` | [0.8218, 0.8928] | [0.9342, 0.9722] |
| `rnn_winner_auc` | [0.8271, 0.8961] | [0.9339, 0.9722] |
| `rnn_default_f1` | [0.8119, 0.8862] | [0.9254, 0.9664] |
| `rnn_default_auc` | [0.8130, 0.8872] | [0.9270, 0.9672] |

**All endpoint verdicts do NOT all survive the cluster bootstrap** (wider, dependence-honest intervals). The gating/cell-isolation verdicts (RNN-F1 vs BiLSTM-F1; RNN-F1 vs GRU-F1; RNN-AUC vs frozen BiLSTM) are unchanged under clustering — the central finding is robust to the pedestrian correlation structure.

# GRU study — Phase G5 pedestrian-cluster bootstrap (the honest CI)

Windows are pedestrian-correlated (541 clusters for 2094 windows, 50% overlap), so i.i.d.-window CIs understate uncertainty. Every endpoint's primary-metric delta is recomputed by resampling PEDESTRIANS (all-windows-per-drawn-ped, 10k resamples, paired clusters across models) — machinery reused verbatim from `f1_optimization/07_cluster_bootstrap.py`. **Cluster intervals are the ones to quote.**

| endpoint | metric | Δ | window CI (07) → verdict | cluster CI → verdict | survives? |
|---|---|---|---|---|---|
| (1) f1_winner vs frozen_bilstm | F1 | +0.0258 | [+0.0162, +0.0358] WIN | [+0.0148, +0.0380] WIN | ✓ |
| (2) f1_winner vs bilstm_f1 | F1 | +0.0071 | [-0.0043, +0.0187] TIE | [-0.0089, +0.0223] TIE | ✓ |
| (3) f1_winner vs transformer_f1 | F1 | +0.0063 | [-0.0046, +0.0174] TIE | [-0.0085, +0.0216] TIE | ✓ |
| (4) default_f1 vs frozen_bilstm | F1 | +0.0150 | [+0.0065, +0.0238] WIN | [+0.0052, +0.0253] WIN | ✓ |
| (5) default_auc vs frozen_bilstm | AUC | -0.0008 | [-0.0039, +0.0021] TIE | [-0.0067, +0.0045] TIE | ✓ |
| (6) f1_winner vs searched_tf | AUC | -0.0070 | [-0.0101, -0.0038] LOSS | [-0.0129, -0.0018] LOSS | ✓ |

| GRU arm | ens F1 95% cluster CI | ens AUC 95% cluster CI |
|---|---|---|
| `gru_f1_winner` | [0.8256, 0.8961] | [0.9273, 0.9677] |
| `gru_default_f1` | [0.8137, 0.8864] | [0.9240, 0.9656] |
| `gru_default_auc` | [0.8078, 0.8813] | [0.9183, 0.9623] |

**All endpoint verdicts survive the cluster bootstrap** (wider, dependence-honest intervals). The two cell-isolation TIEs (GRU-F1 vs BiLSTM-F1; GRU-AUC vs frozen BiLSTM) remain TIEs — the central finding is robust to the pedestrian correlation structure.

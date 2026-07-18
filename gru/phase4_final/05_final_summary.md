# GRU study — Phase G4 final test evaluation (set03, touched ONCE)

Single test pass through `05_gru_test_eval.py` after the G3 sign-off. **Parity gate PASS** (frozen BiLSTM per-seed test AUC recomputed from checkpoints matches stored `final.json`, max |Δ| = 0.00e+00). All probabilities regenerated on CPU; τ\* selected on each seed's own **validation** probs (never test); ensemble = the 5 seeds' averaged probabilities.

Per-seed = mean ± std (ddof=1) over seeds [42,0,1,2,3] at each seed's own τ\*. `ens` = deployable 5-seed probability ensemble (a different statistic — always labeled). AUC is threshold-free.

| arm | stat | F1@0.5 | F1@τ\* | Acc@τ\* | AUC |
|---|---|---|---|---|---|
| `gru_f1_winner` | per-seed | 0.8499 ± 0.0077 | 0.8488 ± 0.0111 | 0.9013 | 0.9408 ± 0.0066 |
| `gru_f1_winner` | ensemble | 0.8565 | 0.8628 | 0.9107 | 0.9489 |
| `gru_default_f1` | per-seed | 0.8443 ± 0.0185 | 0.8443 ± 0.0198 | 0.8983 | 0.9386 ± 0.0066 |
| `gru_default_f1` | ensemble | 0.8520 | 0.8520 | 0.9021 | 0.9460 |
| `gru_default_auc` | per-seed | 0.8352 ± 0.0119 | 0.8403 ± 0.0104 | 0.8980 | 0.9327 ± 0.0099 |
| `gru_default_auc` | ensemble | 0.8474 | 0.8466 | 0.8988 | 0.9415 |

Reference (frozen, for orientation — formal deltas in Phase G5 `07_compare.py`): BiLSTM F1 0.828 / AUC 0.9324; BiLSTM-F1 0.844 / 0.940; Transformer-F1 0.847 / 0.947; searched Transformer 0.845 / 0.9497.

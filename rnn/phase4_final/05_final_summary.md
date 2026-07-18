# RNN study — Phase R4 final test evaluation (set03, touched ONCE)

Single test pass through `05_rnn_test_eval.py` after the R3 sign-off. **Parity gate PASS** (frozen BiLSTM per-seed test AUC recomputed from checkpoints matches stored `final.json`, max |Δ| = 0.00e+00). All probabilities regenerated on CPU; τ\* selected on each seed's own **validation** probs (never test); ensemble = the 5 seeds' averaged probabilities.

Per-seed = mean ± std (ddof=1) over seeds [42,0,1,2,3] at each seed's own τ\*. `ens` = deployable 5-seed probability ensemble (a different statistic — always labeled). AUC is threshold-free.

| arm | stat | F1@0.5 | F1@τ\* | Acc@τ\* | AUC |
|---|---|---|---|---|---|
| `rnn_f1_winner` | per-seed | 0.8543 ± 0.0119 | 0.8518 ± 0.0120 | 0.9018 | 0.9480 ± 0.0015 |
| `rnn_f1_winner` | ensemble | 0.8602 | 0.8590 | 0.9078 | 0.9546 |
| `rnn_winner_auc` | per-seed | 0.8490 ± 0.0152 | 0.8450 ± 0.0222 | 0.8937 | 0.9481 ± 0.0058 |
| `rnn_winner_auc` | ensemble | 0.8640 | 0.8634 | 0.9102 | 0.9545 |
| `rnn_default_f1` | per-seed | 0.8430 ± 0.0133 | 0.8441 ± 0.0125 | 0.8968 | 0.9415 ± 0.0072 |
| `rnn_default_f1` | ensemble | 0.8543 | 0.8510 | 0.9045 | 0.9470 |
| `rnn_default_auc` | per-seed | 0.8413 ± 0.0176 | 0.8360 ± 0.0208 | 0.8893 | 0.9421 ± 0.0085 |
| `rnn_default_auc` | ensemble | 0.8492 | 0.8519 | 0.9045 | 0.9483 |

Reference (frozen, for orientation — formal deltas in Phase R5 `07_compare.py`): BiLSTM F1 0.828 / AUC 0.9324; BiLSTM-F1 0.844 / 0.940; Transformer-F1 0.847 / 0.947; searched Transformer 0.845 / 0.9497; GRU-F1 0.849 / 0.941.

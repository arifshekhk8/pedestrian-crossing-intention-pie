# 04 — F1-protocol training + pos_weight sweep (val-only)

Hybrid rule per PLAN.md §3.3: stop/schedule on val AUC (frozen dynamics), checkpoint = best val F1 (tie acc, then AUC). `f1 (auc-ckpt)` shows what the frozen AUC-checkpoint rule would have scored on the same trajectory (gate G1).

| cell | pw | val F1 (5-seed) | val F1 auc-ckpt | val acc | val AUC |
|---|---|---|---|---|---|
| `lstm_lr1e-03_do0.3_h128_nl2` | 1.682 | 0.8371 ± 0.0256 | 0.8267 | 0.9174 | 0.9642 |
| `lstm_lr1e-03_do0.3_h256_nl2` | 1 | 0.8361 ± 0.0248 | 0.7981 | 0.9192 | 0.9638 |
| `lstm_lr1e-03_do0.3_h256_nl2` | 1.3 | 0.8437 ± 0.0205 | 0.8220 | 0.9202 | 0.9600 |
| `lstm_lr1e-03_do0.3_h256_nl2` | 1.682 | 0.8508 ± 0.0193 | 0.8309 | 0.9252 | 0.9679 |
| `lstm_lr1e-03_do0.3_h256_nl2` | 2.1 | 0.8415 ± 0.0097 | 0.8294 | 0.9174 | 0.9669 |
| `lstm_lr1e-03_do0.3_h256_nl2` | 2.5 | 0.8507 ± 0.0116 | 0.8503 | 0.9237 | 0.9696 |
| `lstm_lr1e-03_do0.5_h128_nl2` | 1.682 | 0.8420 ± 0.0167 | 0.8103 | 0.9177 | 0.9617 |
| `transformer_default` | 1.682 | 0.8251 ± 0.0177 | 0.8010 | 0.9098 | 0.9567 |
| `transformer_searched` | 1 | 0.8579 ± 0.0219 | 0.8412 | 0.9293 | 0.9711 |
| `transformer_searched` | 1.3 | 0.8565 ± 0.0106 | 0.8504 | 0.9281 | 0.9726 |
| `transformer_searched` | 1.682 | 0.8612 ± 0.0098 | 0.8426 | 0.9297 | 0.9732 |
| `transformer_searched` | 2.1 | 0.8486 ± 0.0113 | 0.8302 | 0.9227 | 0.9680 |
| `transformer_searched` | 2.5 | 0.8632 ± 0.0140 | 0.8506 | 0.9278 | 0.9774 |

## Selections + gates (04_selection.json)

```json
{
  "lstm": {
    "headline_cfg_id": "lr1e-03_do0.3_h256_nl2",
    "headline_cfg": {
      "lr": 0.001,
      "dropout": 0.3,
      "hidden": 256,
      "num_layers": 2
    },
    "confirm_winner": "lr1e-03_do0.3_h256_nl2",
    "pw": 1.682,
    "baseline_cfg_id": "lr1e-03_do0.3_h128_nl2",
    "gates": {
      "G1": {
        "wins": 2,
        "n": 5,
        "passed": false
      },
      "G2": {
        "best_pw": 1.682,
        "passed": false,
        "mean_f1": {
          "pw1": 0.8360771902138913,
          "pw1.3": 0.8436784462653864,
          "pw1.682": 0.8507705115510594,
          "pw2.1": 0.8414540238409243,
          "pw2.5": 0.8507496513883327
        }
      },
      "G3": {
        "winner": "lr1e-03_do0.3_h256_nl2",
        "baseline": "lr1e-03_do0.3_h128_nl2",
        "winner_f1": 0.8507705115510594,
        "baseline_f1": 0.8370874751737221,
        "passed": true
      }
    }
  },
  "transformer": {
    "pw": 2.5,
    "gates": {
      "G1": {
        "wins": 4,
        "n": 5,
        "passed": true
      },
      "G2": {
        "best_pw": 2.5,
        "passed": true,
        "mean_f1": {
          "pw1": 0.8578581102179234,
          "pw1.3": 0.8564539128069114,
          "pw1.682": 0.86116835519795,
          "pw2.1": 0.8486496510636587,
          "pw2.5": 0.8632297352230989
        }
      },
      "G1_default": {
        "wins": 4,
        "n": 5,
        "passed": true
      }
    }
  }
}
```

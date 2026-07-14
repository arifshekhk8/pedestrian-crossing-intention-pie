# 01 — Threshold audit on the frozen models (VAL ONLY; test untouched)

tau* rule (pre-registered, PLAN.md §3.1): argmax val F1 over achievable cutoffs,
tie-break higher val acc then |tau-0.5|, bounded [0.05, 0.95].

## Per-seed: val metrics at 0.5 vs at tau*

| model | seed | tau* | val F1 @0.5 | val F1 @tau* | val acc @0.5 | val acc @tau* |
|---|---|---|---|---|---|---|
| lstm_baseline | 42 | 0.5251 | 0.8411 | 0.8438 | 0.9196 | 0.9211 |
| lstm_baseline | 0 | 0.5467 | 0.8443 | 0.8519 | 0.9180 | 0.9243 |
| lstm_baseline | 1 | 0.5654 | 0.8221 | 0.8302 | 0.9085 | 0.9148 |
| lstm_baseline | 2 | 0.5916 | 0.8657 | 0.8780 | 0.9290 | 0.9369 |
| lstm_baseline | 3 | 0.4101 | 0.8333 | 0.8415 | 0.9117 | 0.9132 |
| transformer_searched | 42 | 0.5728 | 0.8916 | 0.9022 | 0.9448 | 0.9511 |
| transformer_searched | 0 | 0.7035 | 0.8150 | 0.8635 | 0.8991 | 0.9322 |
| transformer_searched | 1 | 0.3595 | 0.8642 | 0.8673 | 0.9306 | 0.9290 |
| transformer_searched | 2 | 0.3163 | 0.8454 | 0.8690 | 0.9227 | 0.9306 |
| transformer_searched | 3 | 0.8993 | 0.8364 | 0.8590 | 0.9148 | 0.9322 |
| transformer_default | 42 | 0.3492 | 0.8135 | 0.8195 | 0.9038 | 0.9006 |
| transformer_default | 0 | 0.4413 | 0.8469 | 0.8608 | 0.9259 | 0.9306 |
| transformer_default | 1 | 0.4349 | 0.8296 | 0.8411 | 0.9164 | 0.9196 |
| transformer_default | 2 | 0.5621 | 0.8000 | 0.8125 | 0.8864 | 0.8959 |
| transformer_default | 3 | 0.3771 | 0.7935 | 0.8462 | 0.8991 | 0.9180 |

## 5-seed averaged-probability ensemble (the deployable form)

| model | tau*_ens | val F1 @0.5 | val F1 @tau* | val acc @0.5 | val acc @tau* |
|---|---|---|---|---|---|
| lstm_baseline | 0.5417 | 0.8606 | 0.8634 | 0.9274 | 0.9306 |
| transformer_searched | 0.6254 | 0.8746 | 0.8750 | 0.9353 | 0.9369 |
| transformer_default | 0.6127 | 0.8580 | 0.8704 | 0.9274 | 0.9385 |

## Mean per-seed gain from tau* alone (no retraining)

| model | mean val F1 @0.5 | mean val F1 @tau* | mean gain |
|---|---|---|---|
| lstm_baseline | 0.8413 | 0.8491 | +0.0078 |
| transformer_searched | 0.8505 | 0.8722 | +0.0217 |
| transformer_default | 0.8167 | 0.8360 | +0.0193 |

## Checkpoint headroom (history.json): val F1 at AUC-best epoch vs trajectory max

| model | seed | AUC-best ep | val F1 there | max val F1 (ep) | headroom |
|---|---|---|---|---|---|
| lstm_baseline | 42 | 17 | 0.8411 | 0.8442 (5) | +0.0030 |
| lstm_baseline | 0 | 5 | 0.8443 | 0.8443 (5) | +0.0000 |
| lstm_baseline | 1 | 6 | 0.8221 | 0.8221 (6) | +0.0000 |
| lstm_baseline | 2 | 8 | 0.8657 | 0.8657 (8) | +0.0000 |
| lstm_baseline | 3 | 16 | 0.8333 | 0.8520 (17) | +0.0186 |
| transformer_searched | 42 | 19 | 0.8916 | 0.8916 (19) | +0.0000 |
| transformer_searched | 0 | 12 | 0.8150 | 0.8634 (11) | +0.0483 |
| transformer_searched | 1 | 32 | 0.8642 | 0.8671 (42) | +0.0029 |
| transformer_searched | 2 | 28 | 0.8454 | 0.8701 (24) | +0.0247 |
| transformer_searched | 3 | 22 | 0.8364 | 0.8696 (29) | +0.0332 |
| transformer_default | 42 | 12 | 0.8135 | 0.8659 (7) | +0.0524 |
| transformer_default | 0 | 21 | 0.8469 | 0.8469 (21) | +0.0000 |
| transformer_default | 1 | 34 | 0.8296 | 0.8431 (23) | +0.0135 |
| transformer_default | 2 | 4 | 0.8000 | 0.8199 (10) | +0.0199 |
| transformer_default | 3 | 8 | 0.7935 | 0.8350 (10) | +0.0414 |

Mean headroom across the 15 frozen runs: **+0.0172 val F1** — the expected gain of the hybrid F1-checkpoint rule (PLAN.md §3.3), measured before any retraining.

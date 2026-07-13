# Phase T1 — Sanity report

Overall: **ALL GATES PASS**

## Gate 0 — protocol asserts

- [PASS] X shape: got (4906, 16, 5)
- [PASS] train N: got 2178
- [PASS] val N: got 634
- [PASS] test N: got 2094
- [PASS] test positives: got 681
- [PASS] pos_weight: 1366/812 = 1.682 (expect 1.682)
- [PASS] norm_mean shape: got (5,)
- [PASS] norm_std shape: got (5,)

## Gate 1 — linear-probe floor

- L=0 transformer wrapper (mean-pool, no encoder), 30 epochs: val AUC 0.8986
- sklearn LogisticRegression (flat 80-D input, balanced): val AUC 0.9388
- L=2 transformer (d64/ff128/cls/learned), 30 epochs: val AUC 0.9306
- [PASS] L>=1 transformer clearly beats the linear floor (0.9306 vs floor max 0.9388)

## Gate 2 — overfit a tiny batch

- [PASS] 64-window overfit, 200 epochs, dropout=0: train acc 1.0000 (expect 1.0)

## Gate 3 — determinism probe (CPU) + parameter table

- [PASS] default config, seed 42 twice on CPU, 5 epochs: val AUC 0.934379 vs 0.934379 (|delta|=0.00e+00)

Parameter ladder (Stage-A sizes; brackets the 594,561-param BiLSTM):

| (d_model, ff) | L | params |
|---|---|---|
| (64, 128) | 2 | 68,673 |
| (64, 128) | 4 | 135,617 |
| (128, 256) | 2 | 268,417 |
| (128, 256) | 4 | 533,377 |
| (128, 512) | 2 | 400,001 |
| (128, 512) | 4 | 796,545 |

`transformer_default` (d128/L2/ff256/cls/learned): **268,417 params** (BiLSTM baseline: 594,561).

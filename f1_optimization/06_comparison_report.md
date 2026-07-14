# 06 — F1-first comparison: endpoints and verdicts

Test = set03, N=2094 (touched once per arm, in 05 only). 10,000 paired percentile bootstrap resamples (`np.random.default_rng(42)`, same indices both sides). Metric hierarchy per the supervisor: **F1 -> acc -> AUC**. All thresholds are val-fitted (per-seed and ensemble tau*), fixed before 05 touched test.

## Arms (test)

The two right-hand columns are DIFFERENT statistics of the same 5 runs (the repo's 0.932-vs-0.942 reconciliation discipline): the per-seed column is the citable model-quality number; the ensemble column combines the 5 seeds' probabilities into one deployable predictor before scoring.

| arm | description | test F1 @tau* (5-seed mean ± std) | ens F1 @tau* | ens F1 @0.5 | acc (5-seed) | ens AUC |
|---|---|---|---|---|---|---|
| A0 | LSTM lr1e-03_do0.3_h128_nl2 frozen @0.5 (restated) | 0.8275 ± 0.0123 | 0.8370 | 0.8370 | 0.8827 | 0.9423 |
| A1 | LSTM lr1e-03_do0.3_h128_nl2 frozen @tau* | 0.8343 ± 0.0183 | 0.8452 | 0.8370 | 0.8884 | 0.9423 |
| A2 | LSTM lr1e-03_do0.3_h128_nl2 F1-protocol pw1.682 @tau* | 0.8392 ± 0.0140 | 0.8421 | 0.8387 | 0.8934 | 0.9398 |
| A3 | LSTM lr1e-03_do0.3_h256_nl2 F1-protocol pw1.682 @tau* (headline) | 0.8444 ± 0.0078 | 0.8557 | 0.8536 | 0.8990 | 0.9467 |
| B0 | Transformer searched frozen @0.5 (restated) | 0.8446 ± 0.0129 | 0.8490 | 0.8490 | 0.8942 | 0.9558 |
| B1 | Transformer searched frozen @tau* | 0.8487 ± 0.0180 | 0.8617 | 0.8490 | 0.8987 | 0.9558 |
| B2 | Transformer searched F1-protocol pw1.682 @tau* | 0.8463 ± 0.0060 | 0.8617 | 0.8497 | 0.8977 | 0.9539 |
| B3 | Transformer searched F1-protocol pw2.5 @tau* (headline) | 0.8470 ± 0.0178 | 0.8565 | 0.8495 | 0.8962 | 0.9550 |
| B4 | Transformer default F1-protocol pw1.682 @tau* (architecture control) | 0.8213 ± 0.0059 | 0.8378 | 0.8378 | 0.8777 | 0.9421 |

## Pre-registered endpoints (ensemble vectors, paired bootstrap)

### (i) A3 vs A0 — what F1-first optimization bought the LSTM

**Delta-F1 = +0.0187**, 95% CI [+0.0073, +0.0300] (excludes 0). Delta-acc = +0.0148 CI [+0.0067, +0.0229]; Delta-AUC = +0.0044 CI [+0.0019, +0.0069]. Paired t over 5 seeds: t=2.568, p=0.0621 (secondary, n=5).

**Verdict: IMPROVED.**

Per-seed-pair Delta-F1: +0.0109, +0.0026, +0.0064, +0.0272, +0.0371

### (ii) B3 vs B0 — what it bought the transformer

**Delta-F1 = +0.0075**, 95% CI [-0.0021, +0.0173] (includes 0). Delta-acc = +0.0086 CI [+0.0019, +0.0153]; Delta-AUC = -0.0008 CI [-0.0029, +0.0013]. Paired t over 5 seeds: t=0.289, p=0.7870 (secondary, n=5).

**Verdict: NO SIGNIFICANT CHANGE.**

Per-seed-pair Delta-F1: -0.0109, +0.0306, -0.0164, +0.0043, +0.0043

### (iii) B3 vs A3 — family verdict under F1-first

**Delta-F1 = +0.0008**, 95% CI [-0.0124, +0.0142] (includes 0). Delta-acc = +0.0014 CI [-0.0076, +0.0105]; Delta-AUC = +0.0083 CI [+0.0039, +0.0126]. Paired t over 5 seeds: t=0.324, p=0.7624 (secondary, n=5).

**Verdict: TIE.**

Per-seed-pair Delta-F1: +0.0052, +0.0211, -0.0253, -0.0029, +0.0151

## Gates (04_selection.json)

```json
{
  "lstm": {
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
  },
  "transformer": {
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
```

## Honesty notes

- Every arm's F1@0.5 is reported alongside F1@tau* (arms table) — literature numbers are typically at each paper's own operating point.
- tau* is fitted on val (N=634, 155 pos) and applied unchanged to test; the val->test transfer is visible per arm in `05_final_arms.json` (`val_at_tau` vs `test`).
- A0/B0 restate the frozen models exactly (parity-gated in 05).

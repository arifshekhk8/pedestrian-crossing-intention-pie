# Documentation — class weighting, splits, overfitting & leakage control

This note answers three questions a reviewer (or the thesis committee) will ask about the shared
training protocol used by **all four families** (BiLSTM · Transformer · GRU · vanilla RNN):

1. Is `pos_weight = 1.682` correct?
2. Is it computed before or after the split, and where is it actually applied?
3. How are overfitting and leakage handled?

Everything below traces to the single model-agnostic engine every family runs through:
[`journal_prep/issue12_unified_pipeline/12_unified_engine.py`](../issue12_unified_pipeline/12_unified_engine.py).

---

## 1. Is `pos_weight = 1.682` correct? Yes.

It is the exact class ratio of the **training split only** (`n_neg / n_pos`):

| Split | sets | n | neg | pos | neg/pos | pos frac |
|---|---|---|---|---|---|---|
| **train** | set01/02/04 | 2178 | 1366 | 812 | **1.6823 → 1.682** ✓ | 0.373 |
| val | set05/06 | 634 | 479 | 155 | 3.090 | 0.244 |
| test | set03 | 2094 | 1413 | 681 | 2.075 | 0.325 |
| all | — | 4906 | 3258 | 1648 | 1.977 | 0.336 |

`1366 / 812 = 1.6823`, which is the hardcoded `POS_WEIGHT = 1.682` in
[`12_unified_engine.py:63`](../issue12_unified_pipeline/12_unified_engine.py#L63). Correct to the
rounding.

It was not merely assumed — the GRU and vanilla-RNN studies **swept**
`pos_weight ∈ {1.0, 1.3, 1.682, 2.1, 2.5}` (5-seed val-F1) on the search winner and confirmed
1.682 as the operating point (`gru/phase2_search/02_gru_search.py`, `rnn/phase2_search/02_rnn_search.py`).
So it is both correct *and* validated.

> Note: the **legacy** leaky-era pipeline used `pos_weight = 1.44` (819 neg / 570 pos on the old
> `sequences/`). That is a retracted historical artifact — the clean, journal-bound protocol uses
> **1.682**.

---

## 2. Computed *after* the split, from **train only** — and where it is used

**After splitting, from the training set only.** This is the point that matters: `1.682` is the
**train** balance (1366/812), *not* the pooled-data balance. Had it been computed before the split
(on all 4906 windows) it would be **1.977** — a value that would have peeked at the val+test class
distributions. The fact that the number is 1.682 proves it was derived correctly: post-split,
train-only.

**Where it is applied** — a single line in the loss, `train_run` in
[`12_unified_engine.py:234`](../issue12_unified_pipeline/12_unified_engine.py#L234):

```python
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
```

- It scales the positive ("will cross") class up by 1.682× in the **training-loss gradient only**.
- It does **not** touch evaluation: val/test metrics use a plain threshold of 0.5 on
  `sigmoid(logit)` (`metrics_at`, [`:174`](../issue12_unified_pipeline/12_unified_engine.py#L174)).
  So it is purely a training-time class-imbalance correction, not a decision-threshold change.
- All four families route through this one `train_run`, so all four share the identical 1.682.
- The **LOSO** robustness runs recompute `pos_weight` **per fold** from each fold's own training
  set (`gru/phase4_final/06_gru_loso.py`, `rnn/phase4_final/*`), which is the leak-free thing to do
  when the fold boundaries change.

---

## 3. Overfitting and leakage control

### Overfitting — five mechanisms (all in `train_run`)

| Mechanism | Setting | Location |
|---|---|---|
| Early stopping | patience 15 on val AUC, max 100 epochs | [`:276`](../issue12_unified_pipeline/12_unified_engine.py#L276) |
| Best-checkpoint selection | saves the best-val-metric epoch (F1-first rule: F1 → acc → AUC), never the last epoch | [`:279`](../issue12_unified_pipeline/12_unified_engine.py#L279) |
| LR decay | `ReduceLROnPlateau(mode=max, factor 0.5, patience 5)` on val AUC | [`:237`](../issue12_unified_pipeline/12_unified_engine.py#L237) |
| Dropout | 0.3 in the recurrent/encoder stack | model builders |
| Weight decay (L2) | 1e-5 in the optimizer | [`:197`](../issue12_unified_pipeline/12_unified_engine.py#L197) |

Plus a structural safeguard: `train_run` has **no test code path at all** — model selection is
val-only, and test (set03) is evaluated by exactly one designated final script per study. The model
that ships is therefore never chosen against the test set.

### Leakage — three layers

1. **Split by recording set, not random** — train = set01/02/04, val = set05/06, **test = set03**
   ([`:60–62`](../issue12_unified_pipeline/12_unified_engine.py#L60)). Because recording sets
   partition the videos, no pedestrian / video / scene can appear in more than one split. This kills
   identity- and scene-level leakage that a random per-window split would introduce.
2. **Train-only normalization** — the per-feature z-score mean/std are computed from `Xtr` only and
   then applied to val/test ([`:228–230`](../issue12_unified_pipeline/12_unified_engine.py#L228)),
   so no val/test statistics leak into training. (Same principle as the `pos_weight` point above.)
3. **Event-anchored windows (the important one)** — the original leaky build observed **67.9 % of
   crossers mid-crossing**. The clean protocol re-anchors every window at the `crossing_point` event
   and requires **TTE ≥ 30 frames** of look-ahead
   ([`issue2_clean_protocol/02_build_sequences_clean.py`](../issue2_clean_protocol/02_build_sequences_clean.py)).
   The re-audit confirms **0 % window leakage**: 0 of 4906 windows contain a `crossing` frame, and
   anchor-frame bbox geometry alone cannot separate the classes (all |rank-biserial| < 0.3), so the
   task is genuine ahead-of-time prediction, not disguised detection
   ([`issue2_clean_protocol/02_leakage_report_clean.md`](../issue2_clean_protocol/02_leakage_report_clean.md)).

---

## Bottom line

`pos_weight = 1.682` is correct, computed **after** the split from the **training set only** (not
the pooled data), and applied only in the training BCE loss (evaluation stays at threshold 0.5).
Overfitting is controlled by early stopping + best-val-checkpoint + dropout / weight-decay / LR-decay,
and leakage by set-level splits, train-only normalization, and event-anchored windows verified
leak-free. The protocol is identical for all four families by construction, so the cross-family
comparison in [`README.md`](README.md) is fair.

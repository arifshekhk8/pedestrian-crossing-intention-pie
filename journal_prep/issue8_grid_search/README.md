# Issue 8 — Hyperparameter grid search ✅

*(Supervisor specifically requested this section.)* The hyperparameters (lr,
dropout, hidden, num_layers) were hand-set with no documented search, so a reviewer
asks "why these values?" This runs a transparent grid search with a **leakage-proof
selection protocol** and reports the full grid — so the chosen config is justified
by a documented val-AUC ranking rather than asserted.

> **Metric note (F1-first).** This grid selected on **val AUC** (the metric in force
> when it was run), so it is an **AUC-conditional** result — see the metric-conditional
> notes in `08_grid_search_summary.md`. The symmetric **F1-first** hyperparameter/
> operating-point optimization of both model families is
> [`../../f1_optimization/`](../../f1_optimization/) (metric hierarchy F1 → acc → AUC).

## How to run

```bash
source .venv/bin/activate
python journal_prep/issue8_grid_search/08_grid_search.py   # MPS; ~25 min full run
# per-run JSONs are cached, so re-running only regenerates the summary/figure
```

## Protocol (the part that has to be airtight)

Search space, fixed a priori: **lr ∈ {1e-3, 5e-4, 1e-4} · dropout ∈ {0.2, 0.3, 0.5}
· hidden ∈ {64, 128, 256} · num_layers ∈ {1, 2}** (batch=32 fixed). The locked
architecture uses *inter-layer* LSTM dropout, which is **inert at num_layers=1**, so
those cells are merged → **36 distinct configs** (27 two-layer + 9 one-layer). Not
running 18 duplicate one-layer cells is honesty about the model, not a shortcut.

1. **Stage 1** — all 36 configs at seed 42, ranked by **validation AUC** (set05/06).
   *Test set is never evaluated here.*
2. **Stage 2** — the top-5 + baseline are re-run across **5 seeds**; the winner is the
   highest **mean** val AUC (guards against a single-seed val fluke). *Test still
   untouched.*
3. **Stage 3** — winner + baseline trained ×5 seeds; the **test set (set03) is touched
   exactly once.** Test never informs any selection decision.

The val-only stages physically do not call test evaluation (`eval_test=False`), so
selection-on-test leakage is ruled out by construction. Everything not searched is
locked to the baseline (`sequences_clean/`, train-only norm, pos_weight=1.682,
weight_decay=1e-5, patience 15, best-on-val-AUC). Baseline = `lr1e-03_do0.3_h128_nl2`
(reproduces 0.932 ± 0.011).

## Result

| | config | val AUC (5-seed) | **test AUC (5-seed, once)** | params |
|---|---|---|---|---|
| **winner** | `lr1e-04_do0.2_h256_nl2` | 0.969 ± 0.006 | **0.930 ± 0.005** | 2.24 M |
| baseline | `lr1e-03_do0.3_h128_nl2` | 0.964 ± 0.004 | 0.929 ± 0.012 | 595 k |

**Verdict — the documented search confirms the hand-set baseline.** The grid's
val-winner beats the baseline on the held-out test by **Δ +0.0006 AUC, paired-t
p=0.914 — not significant**, while costing **3.8× the parameters** and a 10× lower
learning rate (slower to train). So the baseline is **statistically as good** as the
best config the search found, and is retained for efficiency/continuity. The
hyperparameters are now justified by a search, not asserted.

**Two points that make this defensible:**
- *Selection-noise control mattered.* The single-seed (seed 42) val leader was
  `lr1e-04_do0.3_h128` (0.9702), but on 5-seed mean val AUC the winner became
  `lr1e-04_do0.2_h256` (the lowest-variance candidate) — a single-seed grid would
  have chosen differently. Multi-seeding the candidates is what made the selection
  robust.
- *Val/test gap is informative.* The highest *validation* AUCs cluster on lr=1e-4,
  but that edge does not carry to test — a sign the small, class-skewed val split
  (set05/06) mildly favours the lower lr without better generalisation, which
  supports the baseline's lr=1e-3.

This reuses no Issue-7 runs (self-contained), but its hidden=256/lr=1e-3 cells
reproduce Issue 7's hidden-size finding (256 ≈ 128 on test).

## Files

```
08_grid_search.py              harness (Stage 1 grid → Stage 2 multiseed → Stage 3 test)
08_grid_full.csv               all 36 configs, seed-42 val AUC (the full grid, sorted)
08_candidates_multiseed.csv    top-5 + baseline, 5-seed val AUC
08_grid_search_summary.md      protocol + full grid + winner-vs-baseline + verdict
08_grid_search_figure.png      grid val-AUC landscape + winner-vs-baseline test bars
runs_grid/<cfgid>/seed<k>.json   Stage 1–2 runs (val only, no test)
runs_final/<cfgid>/seed<k>.json  Stage 3 runs (winner + baseline, with test)
```

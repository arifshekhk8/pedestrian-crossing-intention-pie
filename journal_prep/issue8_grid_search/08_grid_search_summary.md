# Issue 8 — Hyperparameter grid search

## Protocol (selection on validation only; test touched once)

Search space (a priori): **lr ∈ {1e-3, 5e-4, 1e-4}**, **dropout ∈ {0.2, 0.3, 0.5}**, **hidden ∈ {64, 128, 256}**, **num_layers ∈ {1, 2}** (batch=32 fixed). The locked architecture uses inter-layer LSTM dropout, inert at num_layers=1, so those cells are merged → **36 distinct configs** (27 two-layer + 9 one-layer).

1. **Stage 1** — all 36 configs at seed 42, ranked by **validation AUC** (set05/06). *The test set is not evaluated in this stage.*
2. **Stage 2** — the top-5 configs + the baseline are re-run across 5 seeds; the winner is the highest **mean val AUC** (guards against a single-seed val fluke). *Test still untouched.*
3. **Stage 3** — the winner and the baseline are trained ×5 seeds and the **test set (set03) is touched once**. Test never informs selection.

Everything not searched is locked to the baseline (`sequences_clean/`, train-only norm, pos_weight=1.682, weight_decay=1e-05, patience 15, 100 max epochs, best-on-val-AUC). Baseline config = `lr1e-03_do0.3_h128_nl2` (reproduces 0.932 ± 0.011).

## Stage 2 — candidate ranking by mean validation AUC

| config | val AUC (5-seed) |
|---|---|
| `lr1e-04_do0.2_h256_nl2` **winner** | 0.9692 ± 0.0058 |
| `lr1e-04_do0.2_h128_nl2`  | 0.9679 ± 0.0101 |
| `lr1e-03_do0.5_h128_nl2`  | 0.9660 ± 0.0071 |
| `lr1e-04_do0.3_h128_nl2`  | 0.9649 ± 0.0090 |
| `lr1e-03_do0.3_h128_nl2` baseline | 0.9644 ± 0.0043 |

## Final — winner vs baseline on the held-out test set (touched once)

| config | val AUC (5-seed) | **test AUC (5-seed)** |
|---|---|---|
| winner `lr1e-04_do0.2_h256_nl2` | 0.9692 ± 0.0058 | **0.9299 ± 0.0053** |
| baseline `lr1e-03_do0.3_h128_nl2` | 0.9644 ± 0.0043 | 0.9293 ± 0.0119 |

## Verdict

**The search selected `lr1e-04_do0.2_h256_nl2`** (highest mean val AUC), vs the baseline `lr1e-03_do0.3_h128_nl2`. On the held-out test set the winner scores 0.930 ± 0.005 vs the baseline 0.929 ± 0.012 (Δ +0.0006, paired-t p=0.914). The difference is **not significant** (p=0.914): the winner is within seed noise of the baseline on test, while costing **3.8× the parameters** (2,237,313 vs 594,561) and a 10× lower learning rate (slower to train). So the documented search **confirms the hand-set baseline is statistically as good** as the best config it found, and we retain the baseline for efficiency and continuity — the hyperparameters are now justified by a search, not asserted. The selection-noise control mattered concretely: the single-seed (seed 42) val leader was `lr1e-04_do0.3_h128_nl2` (0.9702), but on 5-seed *mean* val AUC the winner is `lr1e-04_do0.2_h256_nl2` (the most stable candidate, lowest val std) — a single-seed grid would have selected a different config.

*Observation:* the highest *validation* AUCs cluster on lr=1e-4 (top of `08_grid_full.csv`), but that edge does not carry to the test set — a sign the small, class-skewed val split (set05/06) slightly favours the lower lr without better generalisation, which in turn supports the baseline's lr=1e-3.

Full 36-config grid (seed-42 val AUC) in `08_grid_full.csv`; candidate multiseed in `08_candidates_multiseed.csv`. **Test set was used exactly once, on the final config(s).**
---

## Post-audit notes (2026-07-13)

1. **This verdict is AUC-conditional.** Selection, ranking, and the
   "search confirms the baseline" conclusion all use val/test AUC. Under the
   supervisor's later F1-first hierarchy (`f1_optimization/`), re-ranking this same
   grid by val F1 selects a different config (`lr1e-03_do0.3_h256_nl2`), which became
   the F1-first headline LSTM (test F1 0.844 vs the baseline's 0.828). The AUC-based
   conclusion above stands *for AUC*; it does not transfer to F1. Note also the
   AUC-winner (`lr1e-04_do0.2_h256_nl2`) is F1-unstable on test (seed0 F1 0.67).
2. **Reproducibility caveat (measured):** these cached MPS runs are only partially
   reproducible today — nn.LSTM training on Apple MPS is process-history-dependent
   (dropout-0.5 cells reproduce bit-exactly; dropout-0.2/0.3 and lr1e-4 cells drift
   ~6e-3 val AUC), while CPU training is fully context-free. Stage-1's top-5 are
   separated by ≤0.006 val AUC — within that drift. The grid's *relative ranking*
   should therefore be treated as indicative; any config cited in the manuscript is
   backed by fresh multi-seed runs under the unified engine
   (`journal_prep/issue12_unified_pipeline/`), not by this cache alone. Measurements:
   `12_equivalence_report.md`, `f1_optimization/03_shortlist_results.md`.

# Phase T3 — Transformer staged search report (local re-derivation)

**Independently recomputed from the raw `runs_search/**/seed*.json` files** — every number below was recalculated from scratch and cross-checked against `_stage_summary.json`; the script asserts on any mismatch. See `transformer/PLAN.md` §4 for the full protocol.

## Protocol reminder (val-only; test set03 has not been touched anywhere yet)

1. **Stage A** — 36 architecture configs at seed 42, default recipe (Adam, lr=1e-3, plateau, dropout=0.1, wd=1e-5), ranked by validation AUC.
2. **Stage B** — 36 training-recipe configs on Stage-A's #1 architecture, seed 42, val AUC.
3. **Transfer check** — the top-3 recipes re-tried on Stage-A architectures #2 and #3 (6 configs), to confirm the recipe ranking isn't architecture-specific.
4. **Stage C** — top-5 (architecture × recipe) combos from the pooled 78, plus `transformer_default` (always carried), each × 5 seeds [42,0,1,2,3]. **Winner = highest mean val AUC** (guards against a single-seed fluke). Test set still untouched anywhere in this pipeline.

## Stage C — candidate ranking by mean validation AUC (5 seeds each)

| rank | config | val AUC (5-seed) | tag |
|---|---|---|---|
| 1 | `d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` | 0.9789 ± 0.0038 | **winner** |
| 2 | `d128_ff256_L4_last_spe__adamw_lr1e-03_warmup_cosine_do0.1_wd1e-02` | 0.9788 ± 0.0037 |  |
| 3 | `d128_ff256_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` | 0.9773 ± 0.0042 |  |
| 4 | `d128_ff256_L4_last_spe__adamw_lr1e-03_plateau_do0.1_wd1e-02` | 0.9746 ± 0.0051 |  |
| 5 | `d128_ff256_L4_last_spe__adamw_lr1e-03_plateau_do0.1_wd1e-05` | 0.9745 ± 0.0069 |  |
| 6 | `d128_ff256_L2_cls_lpe__adam_lr1e-03_plateau_do0.1_wd1e-05` | 0.9629 ± 0.0056 | transformer_default |

## Winner vs default (validation set — test not yet touched)

| | config | val AUC (5-seed) |
|---|---|---|
| **winner** | `d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` | **0.9789 ± 0.0038** |
| transformer_default | `d128_ff256_L2_cls_lpe__adam_lr1e-03_plateau_do0.1_wd1e-05` | 0.9629 ± 0.0056 |

`transformer_default` ranks **66/78** in the full seed-42 architecture pool — the search meaningfully outperformed the un-searched default.

## Verdict

**The search selected `d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05`** over `transformer_default` — a deeper (num_layers=4 vs 2), differently-pooled (pool='last' vs 'cls') architecture with sinusoidal rather than learned positional encoding. Notably, `pool="last"` mirrors the BiLSTM's own readout (its last timestep), which is a reassuring sign the search converged on something sensible rather than an arbitrary config. The selection-noise control mattered concretely, exactly as it did for the LSTM in Issue 8: the single-seed (seed 42) leader across all 78 configs was `d128_ff256_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05`, but its 5-seed **mean** val AUC (0.9773) ends up lower than the actual winner's (0.9789) — a single-seed search would have picked the wrong config.

**This is a validation-set result only.** For context (not a formal comparison — protocols differ and only a test-set comparison counts): the LSTM's own 5-seed validation AUC (Issue 8) is 0.9644 ± 0.0043, and its test AUC — **the number the transformer must actually beat** — is 0.9324 ± 0.0114. Validation AUC is not directly comparable across the two searches (different validation-selection histories can inflate a val number without a matching test gain — this is exactly why Phase T4 touches test exactly once and Phase T5 uses a paired bootstrap rather than comparing val numbers directly).

**Next step: Phase T4.** Paste `d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` into `phase4_kaggle_final/04_final_loso_kaggle.ipynb`'s CONFIG cell, run it — that notebook trains the winner + default × 5 seeds and touches test set03 exactly once, plus the 6-fold LOSO CV.
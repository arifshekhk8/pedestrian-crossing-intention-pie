# RNN study — Phase R3 search review (val-only; test UNTOUCHED)

Independently re-derived from the raw `phase2_search/runs_search/*.json` files by `03_search_report.py`; every ranking recomputed from scratch and cross-checked against the search's own `_stage_summary.json` (exact agreement, incl. the instability-ledger count). **93 run files scanned — zero carry a `test` key** (val-only by construction).

All numbers are **validation** metrics (set05/06, N=634), 5-seed mean ± std (ddof=1). `val F1` = best-val-F1-epoch F1; `val AUC` = max val AUC over the trajectory (`val_at_auc_best`). Everything through the unified engine, `--family birnn`, CPU.

## Winners (val-only selection)

- **F1-winner (primary, F1-first hierarchy): `lr1e-04_do0.2_h256_nl2`** — val F1 **0.8554 ± 0.0141**, val acc 0.9265, val AUC 0.9721.
- **AUC-winner: same config (`lr1e-04_do0.2_h256_nl2`)** — the F1 and AUC rankings agree.

- **pos_weight:** swept {1.0, 1.3, 1.682, 2.1, 2.5} on the F1-winner; best mean val F1 at pw 1.682 (0.8554), anchor 1.682 = 0.8554 → **chosen pw 1.682** (anchor retained).

## Instability ledger (vanilla-RNN divergence watch)

**0 runs diverged.** Every config across the grid, multiseed, and pos_weight sweep trained cleanly (all val AUC ≥ 0.7) — the vanilla tanh RNN is stable over the 16-step window at every searched setting.

## Candidate multiseed (5-seed val, sorted by val F1)

| config | val F1 | val acc | val AUC | note |
|---|---|---|---|---|
| `lr1e-04_do0.2_h256_nl2` | 0.8554 ± 0.0141 | 0.9265 | 0.9721 ± 0.0051 | **F1-winner** |
| `lr1e-03_doNA_h256_nl1` | 0.8509 ± 0.0130 | 0.9249 | 0.9714 ± 0.0054 |  |
| `lr1e-03_do0.5_h256_nl2` | 0.8498 ± 0.0221 | 0.9246 | 0.9677 ± 0.0052 |  |
| `lr5e-04_do0.2_h128_nl2` | 0.8491 ± 0.0194 | 0.9233 | 0.9670 ± 0.0072 |  |
| `lr5e-04_doNA_h256_nl1` | 0.8473 ± 0.0065 | 0.9233 | 0.9689 ± 0.0039 |  |
| `lr1e-03_do0.3_h128_nl2` | 0.8429 ± 0.0141 | 0.9202 | 0.9671 ± 0.0030 | rnn_default |
| `lr1e-03_do0.5_h128_nl2` | 0.8428 ± 0.0070 | 0.9196 | 0.9686 ± 0.0048 |  |
| `lr1e-03_do0.3_h64_nl2` | 0.8417 ± 0.0120 | 0.9186 | 0.9670 ± 0.0023 |  |

## pos_weight sweep (F1-winner, 5-seed val F1)

| pos_weight | val F1 | chosen |
|---|---|---|
| 1 | 0.8511 ± 0.0214 |  |
| 1.3 | 0.8498 ± 0.0204 |  |
| 1.682 | 0.8554 ± 0.0141 | ✅ |
| 2.1 | 0.8510 ± 0.0239 |  |
| 2.5 | 0.8551 ± 0.0187 |  |

## Top-5 grid rankings (seed 42)

- top-5 by val F1: `lr5e-04_doNA_h256_nl1`, `lr1e-03_do0.5_h128_nl2`, `lr1e-03_do0.5_h256_nl2`, `lr1e-03_doNA_h256_nl1`, `lr5e-04_do0.2_h128_nl2`
- top-5 by val AUC: `lr1e-03_doNA_h256_nl1`, `lr5e-04_doNA_h256_nl1`, `lr1e-03_do0.5_h128_nl2`, `lr1e-03_do0.3_h64_nl2`, `lr1e-04_do0.2_h256_nl2`

Full 36-config grid in `03_arch_grid.csv` (ranked by seed-42 val F1).

---

## ⏸ HUMAN CHECKPOINT

Test set03 is still **untouched**. The next phase (R4) trains these winners × 5 seeds and evaluates test **exactly once**. **Confirm the F1-winner (`lr1e-04_do0.2_h256_nl2`) and pos_weight 1.682, plus the R4 arm set (see PLAN.md §5), before R4 runs.**
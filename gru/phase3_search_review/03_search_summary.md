# GRU study — Phase G3 search review (val-only; test UNTOUCHED)

Independently re-derived from the raw `phase2_search/runs_search/*.json` files by `03_search_report.py`; every ranking recomputed from scratch and cross-checked against the search's own `_stage_summary.json` (exact agreement). **89 run files scanned — zero carry a `test` key** (val-only by construction).

All numbers are **validation** metrics (set05/06, N=634), 5-seed mean ± std (ddof=1). `val F1` = best-val-F1-epoch F1; `val AUC` = max val AUC over the trajectory (`val_at_auc_best`). Everything through the unified engine, `--family gru`, CPU.

## Winners (val-only selection)

- **F1-winner (primary, F1-first hierarchy): `lr5e-04_do0.3_h256_nl2`** — val F1 **0.8683 ± 0.0241**, val acc 0.9331, val AUC 0.9747.
- **AUC-winner (secondary): `lr1e-03_do0.2_h256_nl2`** — val AUC **0.9760 ± 0.0017**, val F1 0.8608. *Differs from the F1-winner — both carried to G4.*

- **pos_weight:** swept {1.0, 1.3, 1.682, 2.1, 2.5} on the F1-winner; best mean val F1 at pw 1.682 (0.8683), anchor 1.682 = 0.8683 → **chosen pw 1.682** (anchor retained).

## Candidate multiseed (5-seed val, sorted by val F1)

| config | val F1 | val acc | val AUC | note |
|---|---|---|---|---|
| `lr5e-04_do0.3_h256_nl2` | 0.8683 ± 0.0241 | 0.9331 | 0.9747 ± 0.0026 | **F1-winner** |
| `lr5e-04_do0.2_h256_nl2` | 0.8622 ± 0.0305 | 0.9309 | 0.9755 ± 0.0027 |  |
| `lr1e-03_do0.3_h256_nl2` | 0.8616 ± 0.0158 | 0.9281 | 0.9737 ± 0.0039 |  |
| `lr5e-04_do0.5_h256_nl2` | 0.8613 ± 0.0240 | 0.9297 | 0.9743 ± 0.0053 |  |
| `lr1e-03_do0.2_h256_nl2` | 0.8608 ± 0.0095 | 0.9300 | 0.9760 ± 0.0017 | **AUC-winner** |
| `lr1e-03_doNA_h256_nl1` | 0.8573 ± 0.0192 | 0.9281 | 0.9705 ± 0.0047 |  |
| `lr1e-03_do0.3_h128_nl2` | 0.8558 ± 0.0138 | 0.9268 | 0.9709 ± 0.0038 | gru_default |

## pos_weight sweep (F1-winner, 5-seed val F1)

| pos_weight | val F1 | chosen |
|---|---|---|
| 1 | 0.8547 ± 0.0245 |  |
| 1.3 | 0.8612 ± 0.0195 |  |
| 1.682 | 0.8683 ± 0.0241 | ✅ |
| 2.1 | 0.8667 ± 0.0183 |  |
| 2.5 | 0.8621 ± 0.0232 |  |

## Top-5 grid rankings (seed 42)

- top-5 by val F1: `lr5e-04_do0.5_h256_nl2`, `lr5e-04_do0.3_h256_nl2`, `lr1e-03_do0.3_h256_nl2`, `lr1e-03_doNA_h256_nl1`, `lr5e-04_do0.2_h256_nl2`
- top-5 by val AUC: `lr1e-03_do0.2_h256_nl2`, `lr5e-04_do0.5_h256_nl2`, `lr5e-04_do0.3_h256_nl2`, `lr1e-03_do0.3_h256_nl2`, `lr5e-04_do0.2_h256_nl2`

Full 36-config grid in `03_arch_grid.csv` (ranked by seed-42 val F1).

---

## ⏸ HUMAN CHECKPOINT

Test set03 is still **untouched**. The next phase (G4) trains these winners × 5 seeds and evaluates test **exactly once**. **Confirm the F1-winner and AUC-winner (`lr5e-04_do0.3_h256_nl2` / `lr1e-03_do0.2_h256_nl2`) and pos_weight 1.682 before G4 runs.**
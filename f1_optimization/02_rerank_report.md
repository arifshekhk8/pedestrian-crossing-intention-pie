# 02 — F1 re-ranking of both cached searches (val-only JSONs; no training)

Ranking key: **val F1 -> val acc -> val AUC** (supervisor hierarchy).

## Transformer (78 configs, seed-42) — top 10

| F1 rank | cfg | val F1 | val acc | val AUC |
|---|---|---|---|---|
| 1 | `d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` | 0.8916 | 0.9448 | 0.9825 |
| 2 | `d128_ff256_L4_last_spe__adamw_lr1e-04_plateau_do0.1_wd1e-02` | 0.8882 | 0.9432 | 0.9795 |
| 3 | `d128_ff256_L4_last_spe__adamw_lr1e-04_plateau_do0.1_wd1e-05` | 0.8882 | 0.9432 | 0.9795 |
| 4 | `d128_ff256_L4_last_spe__adamw_lr3e-04_warmup_cosine_do0.1_wd1e-02` | 0.8847 | 0.9416 | 0.9823 |
| 5 | `d128_ff256_L4_last_spe__adamw_lr3e-04_warmup_cosine_do0.1_wd1e-05` | 0.8847 | 0.9416 | 0.9823 |
| 6 | `d128_ff256_L4_last_spe__adamw_lr1e-03_warmup_cosine_do0.1_wd1e-05` | 0.8779 | 0.9416 | 0.9822 |
| 7 | `d128_ff512_L4_last_spe__adamw_lr1e-03_plateau_do0.1_wd1e-05` | 0.8773 | 0.9369 | 0.9801 |
| 8 | `d64_ff128_L4_mean_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` | 0.8742 | 0.9369 | 0.9732 |
| 9 | `d128_ff512_L4_last_spe__adamw_lr1e-03_plateau_do0.1_wd1e-02` | 0.8715 | 0.9353 | 0.9813 |
| 10 | `d128_ff256_L2_cls_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` | 0.8701 | 0.9322 | 0.9792 |

5-seed candidates by mean val F1:

| cfg | val F1 (5-seed) | val AUC mean |
|---|---|---|
| `d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` | 0.8505 ± 0.0290 | 0.9789 |
| `d128_ff256_L4_last_spe__adamw_lr1e-03_warmup_cosine_do0.1_wd1e-02` | 0.8484 ± 0.0129 | 0.9788 |
| `d128_ff256_L4_last_spe__adamw_lr1e-03_plateau_do0.1_wd1e-05` | 0.8403 ± 0.0203 | 0.9745 |
| `d128_ff256_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` | 0.8393 ± 0.0223 | 0.9773 |
| `d128_ff256_L4_last_spe__adamw_lr1e-03_plateau_do0.1_wd1e-02` | 0.8332 ± 0.0147 | 0.9746 |
| `d128_ff256_L2_cls_lpe__adam_lr1e-03_plateau_do0.1_wd1e-05` | 0.8167 ± 0.0218 | 0.9629 |

**Transformer config decision: UNCHANGED.** The frozen AUC winner `d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` is seed-42 F1 rank 1 and ALSO the 5-seed val-F1 leader (`d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05`). The searched architecture+recipe carries over to the F1-first program as-is; only pos_weight and the checkpoint rule are re-optimized (PLAN.md §3.3-3.4).

## LSTM (36 configs, seed-42) — top 8

| F1 rank | cfg | val F1 | val acc | val AUC |
|---|---|---|---|---|
| 1 | `lr1e-04_do0.3_h128_nl2` | 0.8483 | 0.9227 | 0.9702 |
| 2 | `lr1e-04_do0.2_h256_nl2` | 0.8477 | 0.9274 | 0.9660 |
| 3 | `lr1e-03_do0.5_h64_nl2` | 0.8464 | 0.9227 | 0.9547 |
| 4 | `lr1e-03_do0.3_h256_nl2` | 0.8440 | 0.9196 | 0.9642 |
| 5 | `lr1e-03_do0.2_h128_nl2` | 0.8411 | 0.9196 | 0.9629 |
| 6 | `lr1e-04_doNA_h128_nl1` | 0.8318 | 0.9132 | 0.9549 |
| 7 | `lr1e-03_do0.3_h128_nl2` | 0.8288 | 0.9101 | 0.9645 |
| 8 | `lr1e-04_do0.2_h128_nl2` | 0.8286 | 0.9243 | 0.9697 |

(baseline `lr1e-03_do0.3_h128_nl2` sits at F1 rank 7.)

Existing 5-seed candidates by mean val F1:

| cfg | val F1 (5-seed) | val AUC mean |
|---|---|---|
| `lr1e-03_do0.5_h128_nl2` | 0.8368 ± 0.0170 | 0.9660 |
| `lr1e-04_do0.3_h128_nl2` | 0.8198 ± 0.0495 | 0.9649 |
| `lr1e-03_do0.3_h128_nl2` | 0.8165 ± 0.0144 | 0.9644 |
| `lr1e-04_do0.2_h256_nl2` | 0.8043 ± 0.0656 | 0.9692 |
| `lr1e-04_do0.2_h128_nl2` | 0.7490 ± 0.1626 | 0.9679 |

## LSTM shortlist (PLAN.md §5 step 2)

Shortlist = seed-42 F1 top-5 UNION existing 5-seed configs (8 configs). Missing seeds to complete (AUC protocol, val-only): 

- `lr1e-04_do0.3_h128_nl2` — 5-seed cache present
- `lr1e-04_do0.2_h256_nl2` — 5-seed cache present
- `lr1e-03_do0.5_h64_nl2` — **needs completion runs**
- `lr1e-03_do0.3_h256_nl2` — **needs completion runs**
- `lr1e-03_do0.2_h128_nl2` — **needs completion runs**
- `lr1e-03_do0.3_h128_nl2` — 5-seed cache present
- `lr1e-04_do0.2_h128_nl2` — 5-seed cache present
- `lr1e-03_do0.5_h128_nl2` — 5-seed cache present

Machine-readable: `02_shortlist.json` (consumed by `03_lstm_shortlist.py`).

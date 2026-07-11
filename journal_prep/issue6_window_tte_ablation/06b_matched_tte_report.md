# Issue 6 — Matched-cohort TTE control (horizon isolated from track length)

All three horizons restricted to the **common tte60-eligible cohort** (L ≥ 76 frames), one window per pedestrian at T = 30/45/60. The three cells share an **identical pedestrian population and labels** in both train (N=519) and test (set03, N=493) — only the observed 16-frame window moves. Locked baseline contract (5-D BiLSTM, train=set01/02/04, val=set05/06, train-only norm, pos_weight=1.682, lr=0.001, patience 15); 5 seeds [42, 0, 1, 2, 3]. matched-tte60 reuses the existing `runs/tte60`.

| TTE (horizon) | N train | N test | best ep | AUC | PR-AUC | F1 | Acc |
|---|---|---|---|---|---|---|---|
| 30 (1.00 s) | 519 | 493 | 22 | 0.961 ± 0.007 | 0.915 ± 0.012 | 0.855 ± 0.009 | 0.902 ± 0.008 |
| 45 (1.50 s) | 519 | 493 | 13 | 0.946 ± 0.004 | 0.880 ± 0.009 | 0.856 ± 0.023 | 0.907 ± 0.012 |
| 60 (2.00 s) | 519 | 493 | 12 | 0.919 ± 0.009 | 0.848 ± 0.002 | 0.786 ± 0.028 | 0.868 ± 0.008 |

**Max between-horizon mean-AUC spread = 0.0419**, vs average within-horizon seed std = ±0.0065.

Pairwise (paired t-test, matched seeds; Mann-Whitney U):

| pair | ΔAUC | paired-t p | Mann-Whitney p |
|---|---|---|---|
| tte30 vs tte45 | +0.0144 | 0.004 | 0.016 |
| tte30 vs tte60 | +0.0419 | 0.000 | 0.008 |
| tte45 vs tte60 | +0.0275 | 0.000 | 0.008 |

Kruskal–Wallis across the three horizons: p = 0.002.

## Matched cohort vs all-eligible single-point

| TTE | all-eligible AUC | matched-cohort AUC | Δ (sample effect) |
|---|---|---|---|
| 30 | 0.960 | 0.961 | +0.000 |
| 45 | 0.948 | 0.946 | -0.001 |
| 60 | 0.919 | 0.919 | +0.000 |

## Verdict

**The horizon effect is real, not a sampling artifact.** On a *fixed cohort* (identical 493-pedestrian test set at all three horizons), AUC still declines monotonically 0.961 (1.0 s) → 0.946 (1.5 s) → 0.919 (2.0 s); spread 0.0419 exceeds seed noise (±0.0065), every pairwise paired-t is significant (all p ≤ 0.004), Kruskal–Wallis p = 0.002. Removing the nested-sample confound leaves the decline intact, so the single-point result in `06_` is confirmed: **prediction AUC degrades significantly with horizon** on leak-free data — the intuitive behaviour the leaky single-seed run had masked.

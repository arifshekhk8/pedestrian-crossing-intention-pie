# Issue 6 — Multi-seed window + TTE ablations (clean protocol)

5-D baseline BiLSTM on the clean leak-free sequences (Issue 2), 5 seeds [42, 0, 1, 2, 3], MPS. Everything locked to the baseline (train=set01/02/04, val=set05/06, test=set03; train-only norm; pos_weight=1.682 fixed across all cells; lr=0.001, dropout 0.3, hidden 128, 2 layers, patience 15). Test (set03) touched once per (config, seed). The ablated factor is the only variable.

## Observation-window sweep (obs_len ∈ {8,16,30}, TTE band [30,60])

| obs_len | band (TTE) | N train | N test | best ep | AUC | PR-AUC | F1 | Acc |
|---|---|---|---|---|---|---|---|---|
| 8 (0.27s) | [30,60] | 4428 | 4256 | 3 | 0.931 ± 0.008 | 0.873 ± 0.006 | 0.831 ± 0.016 | 0.890 ± 0.008 |
| 16 (0.53s) | [30,60] | 2178 | 2094 | 8 | 0.933 ± 0.007 | 0.870 ± 0.012 | 0.828 ± 0.014 | 0.883 ± 0.012 |
| 30 (1.00s) | [30,60] | 1567 | 1481 | 12 | 0.937 ± 0.007 | 0.867 ± 0.006 | 0.834 ± 0.017 | 0.888 ± 0.016 |

**Max between-condition mean-AUC spread = 0.0058**, vs the average within-condition seed std = ±0.0073. The spread is below seed noise.

Pairwise significance (AUC, n=5 seeds): paired t-test (matched seeds) + Mann-Whitney U:

| pair | ΔAUC | paired-t p | Mann-Whitney p |
|---|---|---|---|
| obs8 vs obs16 | -0.0019 | 0.777 | 0.841 |
| obs8 vs obs30 | -0.0058 | 0.212 | 0.222 |
| obs16 vs obs30 | -0.0039 | 0.512 | 1.000 |

Kruskal–Wallis omnibus across the three obs_len conditions: p = 0.566.

## Prediction-horizon sweep (TTE ∈ {30,45,60}, single-point, obs_len 16)

| TTE | band (TTE) | N train | N test | best ep | AUC | PR-AUC | F1 | Acc |
|---|---|---|---|---|---|---|---|---|
| 30 (1.00s) | [30,30] (1.00s) | 562 | 541 | 23 | 0.960 ± 0.004 | 0.920 ± 0.006 | 0.862 ± 0.012 | 0.907 ± 0.010 |
| 45 (1.50s) | [45,45] (1.50s) | 541 | 521 | 17 | 0.948 ± 0.004 | 0.882 ± 0.011 | 0.842 ± 0.023 | 0.895 ± 0.020 |
| 60 (2.00s) | [60,60] (2.00s) | 519 | 493 | 12 | 0.919 ± 0.009 | 0.848 ± 0.002 | 0.786 ± 0.028 | 0.868 ± 0.008 |

**Max between-condition mean-AUC spread = 0.0417**, vs the average within-condition seed std = ±0.0055. The spread is above seed noise.

Pairwise significance (AUC, n=5 seeds): paired t-test (matched seeds) + Mann-Whitney U:

| pair | ΔAUC | paired-t p | Mann-Whitney p |
|---|---|---|---|
| tte30 vs tte45 | +0.0128 | 0.008 | 0.008 |
| tte30 vs tte60 | +0.0417 | 0.000 | 0.008 |
| tte45 vs tte60 | +0.0289 | 0.004 | 0.008 |

Kruskal–Wallis omnibus across the three TTE conditions: p = 0.002.

## Cross-check: obs16/[30,60] reproduces the existing baseline

This MPS run of the shared centre cell (obs16, band [30,60]) gives **AUC 0.933 ± 0.007**, reproducing the existing CPU multiseed baseline (0.932 ± 0.011, issue2_clean_protocol/04_multiseed_summary.md) within seed noise — MPS backend and reused data are consistent.

## Verdict

**Observation window — insensitive (old claim confirmed).** The obs_len ∈ {8,16,30} mean-AUC spread (0.0058) is *smaller than the within-condition seed std* (±0.007) — the three settings are statistically equivalent: the between-setting difference is within run-to-run noise. We lead with this effect-size / equivalence argument rather than the non-significant paired-t (smallest p 0.212, Kruskal–Wallis 0.566), because failing to reject at n=5 seeds is weak evidence of a null on its own. The single-seed 'insensitive to window length' claim survives multi-seed scrutiny on clean data; **obs_len=16 is a safe choice.**

**Prediction horizon — significant, monotonic decline (single-seed claim OVERTURNED).** AUC falls 0.960 (1.0 s) → 0.948 (1.5 s) → 0.919 (2.0 s) as the horizon lengthens. The spread (0.0417) exceeds seed noise (±0.005), every pairwise paired-t is significant (all p ≤ 0.008), and Kruskal–Wallis p = 0.002. This corrects the old leaky single-seed 'insensitive to TTE' conclusion: on leak-free, crossing-point-anchored data the model degrades gracefully and significantly with horizon — the intuitive, expected behaviour (further-ahead prediction is harder). The old flat TTE curve was a leakage artifact (the model was detecting in-progress crossings regardless of nominal horizon, Issues 1–2). Caveat: single-point TTE cells use a smaller, single-horizon test set (N≈500), so their absolute AUCs (0.92–0.96) are not directly comparable to the band-based headline 0.932 — the result is the relative trend, not three new headline numbers.

This decline is **confirmed on a matched cohort** (see `06b_matched_tte_report.md` / `06b_matched_tte_figure.png`): restricting all three horizons to the *same* pedestrians — those eligible for the longest horizon (TTE=60) — removes the nested-sample confound (under single-point sampling, TTE=30 admits 48 extra short/harder tracks that TTE=60 cannot). On that fixed cohort the decline is essentially unchanged (sample effect ≤0.002 AUC, every pairwise p≤0.004), so the horizon effect is genuine, not an artifact of differing track-length eligibility.

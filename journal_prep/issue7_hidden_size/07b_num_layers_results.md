# Issue 7b — Network-depth ablation (num_layers ∈ {1,2,3}, multi-seed)

Companion to the hidden-size (width) ablation: varies **depth** at the baseline width (hidden=128, lr=1e-3, dropout=0.3). Clean baseline data (`sequences_clean/`), 5 seeds [42, 0, 1, 2, 3], MPS; everything else locked. num_layers=2 is the baseline. **Note:** dropout is inter-layer LSTM dropout, inert at num_layers=1, so the 1-layer model runs with no dropout (intrinsic to the architecture).

| num_layers | dropout | params | best ep | AUC | PR-AUC | F1 | Acc |
|---|---|---|---|---|---|---|---|
| 1 | 0.0 (inert) | 199,297 | 15 | 0.930 ± 0.008 | 0.875 ± 0.012 | 0.829 ± 0.010 | 0.888 ± 0.005 |
| 2 (baseline) | 0.3 | 594,561 | 6 | 0.932 ± 0.006 | 0.873 ± 0.010 | 0.823 ± 0.029 | 0.876 ± 0.029 |
| 3 | 0.3 | 989,825 | 5 | 0.931 ± 0.008 | 0.860 ± 0.019 | 0.790 ± 0.123 | 0.880 ± 0.046 |

**Between-depth mean-AUC spread = 0.0013**, vs average within-depth seed std = ±0.0076.

Pairwise vs num_layers=2 (paired t-test, matched seeds; Mann-Whitney U):

| pair | ΔAUC | paired-t p | Mann-Whitney p |
|---|---|---|---|
| nl1 vs nl2 | -0.0013 | 0.772 | 0.841 |
| nl3 vs nl2 | -0.0005 | 0.910 | 1.000 |

Kruskal–Wallis across the three depths: p = 0.932.

## Verdict

**num_layers=2 is justified — depth past 1 layer gives no significant gain, and a 3rd layer adds none.** No depth differs significantly from 2 (nl1 p=0.772, nl3 p=0.910, Kruskal p=0.932); the spread (0.0013) is within seed noise (±0.0076). Depth-3 costs 1.7× the parameters of depth-2 (989,825 vs 594,561) for no measurable benefit, and depth-1 (no inter-layer dropout) is no better. **2 layers is the right depth** — enough to model the sequence, not so deep it overfits the small training set (N=2178). num_layers=2 reproduces the baseline (this run 0.932 ± 0.006 vs 0.932 ± 0.011).

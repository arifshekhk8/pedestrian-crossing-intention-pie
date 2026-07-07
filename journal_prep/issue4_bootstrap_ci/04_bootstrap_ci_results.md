# Issue 4 — Bootstrap 95% CIs on test AUC (clean protocol)

Test = PIE set03, N=2094 windows (32.5% positive). 10,000 percentile bootstrap resamples of (probs, labels); fixed RNG (seed 42) so resamples are paired across models. ROC-AUC via tie-corrected Mann-Whitney; PR-AUC = average precision.

## What this shows (for the paper)

1. **Test-sampling uncertainty ≈ training noise.** The baseline's bootstrap 95% CI
   half-width (±0.013) is about the same size as its seed-to-seed std (±0.010). So
   the headline **AUC 0.932 ± 0.011 already reflects uncertainty of the right
   magnitude**; the 95% CI is ≈ **[0.92, 0.95]**. Neither source is hidden.
2. **The ego-speed gap is statistically unambiguous.** bbox-only CIs top out at
   ~0.80 and never come near the baseline CIs (≥0.90); the ~0.15–0.18 separation
   dwarfs the ~0.025 CI widths. The "+0.18 AUC from ego-speed" (Issue 2) is **not**
   sampling noise.
3. **Attention ≈ baseline, confirmed.** Attention CIs overlap the baseline CIs
   almost entirely (e.g. attention 0.925 [0.912, 0.938] vs baseline 0.932
   [0.920, 0.945]) — **no significant difference**, consistent with Issue 2.
4. **PR-AUC corroborates** under 32.5% prevalence (baseline ~0.86–0.90 vs a 0.325
   base rate), so the ranking quality isn't a ROC-only optimism artifact.

Reporting convention for the paper: e.g. *"test ROC-AUC 0.932 (95% CI [0.92, 0.95]),
PR-AUC 0.876"*.

## Per-model summary (5 training seeds)

| Model | ROC-AUC (seed mean ± std) | typical 95% CI width | PR-AUC (mean ± std) |
|---|---|---|---|
| baseline | 0.932 ± 0.010 | ±0.013 (≈[0.920, 0.945]) | 0.876 ± 0.016 |
| bbox_only | 0.753 ± 0.018 | ±0.023 (≈[0.731, 0.776]) | 0.610 ± 0.013 |
| attention | 0.925 ± 0.009 | ±0.013 (≈[0.912, 0.938]) | 0.865 ± 0.007 |

## Per-seed detail (point estimate + 95% bootstrap CI)

| Model | Seed | ROC-AUC | ROC 95% CI | PR-AUC | PR 95% CI |
|---|---|---|---|---|---|
| baseline | 42 | 0.913 | [0.897, 0.928] | 0.856 | [0.826, 0.886] |
| baseline | 0 | 0.933 | [0.921, 0.946] | 0.866 | [0.835, 0.898] |
| baseline | 1 | 0.943 | [0.931, 0.954] | 0.892 | [0.864, 0.919] |
| baseline | 2 | 0.936 | [0.924, 0.948] | 0.896 | [0.872, 0.917] |
| baseline | 3 | 0.936 | [0.924, 0.947] | 0.867 | [0.838, 0.898] |
| bbox_only | 42 | 0.732 | [0.708, 0.756] | 0.609 | [0.571, 0.649] |
| bbox_only | 0 | 0.777 | [0.756, 0.798] | 0.624 | [0.586, 0.663] |
| bbox_only | 1 | 0.769 | [0.748, 0.790] | 0.590 | [0.551, 0.630] |
| bbox_only | 2 | 0.732 | [0.709, 0.756] | 0.604 | [0.565, 0.643] |
| bbox_only | 3 | 0.755 | [0.732, 0.778] | 0.624 | [0.586, 0.663] |
| attention | 42 | 0.923 | [0.909, 0.936] | 0.868 | [0.840, 0.894] |
| attention | 0 | 0.930 | [0.917, 0.942] | 0.863 | [0.833, 0.894] |
| attention | 1 | 0.931 | [0.918, 0.943] | 0.869 | [0.839, 0.900] |
| attention | 2 | 0.932 | [0.920, 0.944] | 0.872 | [0.843, 0.900] |
| attention | 3 | 0.909 | [0.894, 0.923] | 0.851 | [0.822, 0.879] |

# Issue 4 — Bootstrap CIs on Test AUC

Most PIE papers report a point AUC with no confidence interval; with only 2,094
test windows (set03) the sampling uncertainty matters. We percentile-bootstrap
(10k) each checkpoint's raw test predictions → 95% CI on ROC-AUC and PR-AUC, for
all three model variants × 5 seeds. Full plan: [`../PLAN.md`](../PLAN.md) (Issue 4).

## Files

| File | What it is |
|---|---|
| `04_bootstrap_ci.py` | regenerates test probs from saved checkpoints, 10k bootstrap → 95% CIs |
| `04_bootstrap_ci_results.md` | results table + interpretation (paste-ready) |
| `04_bootstrap_ci_results.csv` | raw per-model/per-seed CIs |

## Reproduce (local, CPU)

```bash
source .venv/bin/activate
python journal_prep/issue4_bootstrap_ci/04_bootstrap_ci.py --B 10000
```

Reuses checkpoints from Issue 2 (`runs_clean/multiseed/` for the 5-D baseline,
`kaggle_result/runs_multiseed_clean/` for bbox-only + attention) and the model
classes from `06b_local_verify_seed42.py`. ~80 s on M4 CPU.

## Headline

Baseline **ROC-AUC 0.932, 95% CI ≈ [0.92, 0.95]**, PR-AUC 0.876. Test-sampling
uncertainty (±0.013) ≈ seed std (±0.010). The ego-speed gap (baseline vs
bbox-only) is statistically unambiguous; attention ≈ baseline (CIs overlap).
Addresses the "point estimate, no CI" gap common in prior PIE work.

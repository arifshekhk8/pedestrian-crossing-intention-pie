# Issue 5 — Leave-One-Set-Out Cross-Validation

The main result uses one fixed split (test=set03). PIE has 6 recording sets;
rotating the held-out set across all 6 and averaging is a much stronger
generalization claim. Full plan: [`../PLAN.md`](../PLAN.md) (Issue 5).

## Files

| File | What it is |
|---|---|
| `05_loso_cv.py` | trains the baseline 6× (each set held out as test), reports per-fold + mean±std |
| `05_loso_results.md` | results table + interpretation (paste-ready) |
| `05_loso_results.csv` | raw per-fold metrics |

## Protocol

Per fold: test = one PIE set; train+val = the other 5, split **85/15 by pedestrian**
(grouped — no ped's windows span train and val); early-stop on val AUC; per-fold
train-only normalization + per-fold `pos_weight = n_neg/n_pos`. Identical
architecture/hyperparameters to the baseline.

## Reproduce (local, M4 GPU/MPS, ~80 s)

```bash
source .venv/bin/activate
python journal_prep/issue5_loso_cv/05_loso_cv.py --device mps --seed 42
# --device cpu also works (~4 min); per-fold ±0.02–0.03 backend variation, mean stable
```

## Headline

**AUC 0.928 ± 0.041 over 6 folds** (0.915 ± 0.029 excluding the 47-window set05
fold). **set03 (our fixed-split fold) is representative at 0.931**, not an easy
fold — answers "what if set03 is easy?". Softest large fold: **set04 (0.892,
N=1610)** — honest Limitations note. Addresses the "single fixed split" gap
(positioning matrix, Issue 3).

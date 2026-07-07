# Issue 2 — Clean-protocol model comparison (all three variants, multi-seed)

All three architectures on the leak-free `sequences_clean/` (N=4,906,
crossing_point-anchored, TTE∈[30,60], 0% leakage), identical splits
(train set01/02/04, val set05/06, test set03 = **2,094 windows / 681 positives**),
`pos_weight=1.682`, threshold 0.5, early stop on val AUC, test touched once per run.
**Multi-seeded over 5 seeds [42, 0, 1, 2, 3]** — baseline locally
(`04_multiseed_baseline.py`), the two variants on **Kaggle T4** via
`06_multiseed_variants_kaggle.ipynb` (outputs in `kaggle_result/`). Old numbers are
the original leaky `runs/` results.

| Model | Inputs | OLD AUC (leaky) | **NEW AUC (clean, 5-seed)** | Δ AUC | NEW F1 | NEW Acc |
|---|---|---|---|---|---|---|
| BiLSTM baseline | bbox + ego-speed (5-D) | 0.931 | **0.932 ± 0.011** | +0.001 | 0.828 ± 0.012 | 0.883 ± 0.009 |
| BiLSTM bbox-only | bbox (4-D) | 0.889 | **0.753 ± 0.020** | **−0.136** | 0.551 ± 0.028 | 0.744 ± 0.007 |
| BiLSTM + attention | bbox + ego-speed (5-D) | 0.933 | **0.925 ± 0.010** | −0.008 | 0.821 ± 0.009 | 0.879 ± 0.010 |

Validated two independent ways: the Kaggle clean numbers match a local CPU
cross-check (`06b_local_verify_seed42.py`: bbox-only 0.770 ± 0.016, attention
0.933 ± 0.004), and each saved checkpoint reproduces its `final.json` AUC on the
local clean test set (bbox seed42 0.7325, attn seed42 0.9228 — exact).

## Finding 1 — ego-vehicle speed is the dominant predictor (+0.18 AUC)

The clean gap between the 5-D baseline (0.932) and bbox-only (0.753) is **+0.179
AUC**, attributable to the single ego-speed channel. bbox-only **collapses −0.136**
(0.889 → 0.753) once observation windows end strictly before crossing onset and the
static-geometry shortcut is gone (Issue 1: crossers are large/low/close in frame —
bbox-area rank-biserial +0.65 → +0.25). The leaky data had hidden this: it made
bbox-only look near-baseline (0.889) and ego-speed worth only ~0.04. The honest
number is ~0.18. This is the single clearest demonstration that the old evaluation
measured *detection of an in-progress crossing*, not prediction — and it pins down
*which* of the two inputs carries the signal. It also supports the "bbox + ego
velocity is a legitimate minimal modality" framing (cf. Occlusion-Aware Diffusion,
bbox+ego only, 0.93–0.95).

## Finding 2 — attention gives no measurable benefit on clean data

Clean attention (0.925 ± 0.010) is within seed noise of the plain BiLSTM
(0.932 ± 0.011), if anything marginally lower; the local cross-check agrees
(attention 0.933 ≈ baseline 0.932). The apparent attention edge on leaky data
(0.945 / 0.933) was an artifact of the shortcut — on genuine prediction it adds
nothing. Report attention as "no significant improvement over the baseline," not
as a positive result.

## The leak fix is methodological, not a deflation

Removing 100% of the leakage left the baseline essentially unchanged
(0.931 → 0.932). The contribution of Issues 1–2 is correctness: genuine pre-onset
prediction, N grew 3.5× (1,389 → 4,906), the static shortcut weakened, and
convergence moved from a suspicious epoch 3 to a believable epoch ~17 — not a
smaller headline. That 0.932 is the number to carry into the Issue-3 baseline table.

## Honest limitation to state in the paper

Because ego-speed is the dominant input, the model partly rides on the
**ego-driver's own anticipation**: in PIE the instrumented vehicle slows when its
(human) driver expects a crossing, so vehicle speed at observation time correlates
with the driver's judgment of intent. This is a legitimate inference-time signal
(not temporal leakage of the label), but it means the 5-D result is not purely
vision-based. Worth a sentence in Limitations + a candidate "speed-perturbation"
robustness check.

## Provenance & data-hygiene notes

- Baseline 5-D multi-seed (0.932 ± 0.011) from `04_multiseed_baseline.py` (local).
  Variants multi-seeded on Kaggle T4 (clean) — `kaggle_result/`
  (`multiseed_clean_results.csv`, `multiseed_clean_summary.md`).
- ⚠ The **first** Kaggle variant run was discarded: it silently trained on the OLD
  LEAKY `sequences/` (test N=587, giving bbox 0.883 / attn 0.945) because the
  notebook's `find_seq_dir()` grabbed the leaky `X.npy` from the attached
  `pie-bilstm` dataset. Caught via the confusion-matrix size (587 vs clean 2,094)
  and a checkpoint that scored 0.879 on leaky but only 0.680 on clean test. The
  notebook now hard-errors unless it loads the clean N=4,906 data. (A brief
  "backend-fragility" theory that the first run inspired was just this clean-vs-leaky
  mixup — there is no backend effect; CPU/MPS/CUDA agree on clean data.)
- ⚠ Two cosmetic bugs remain in the downloaded `kaggle_result/` (it was re-run with
  the pre-fix notebook): `multiseed_clean_summary.csv` has its two model labels
  swapped, and `multiseed_clean_summary.md`'s header text says `POS_WEIGHT=1.44`
  (training actually used 1.682). The numbers in `.md` / `results.csv` are correct —
  use those.

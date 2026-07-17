# Issue 6 — Multi-seed window + TTE ablations ✅

Re-runs the observation-window and prediction-horizon ablations on the **clean,
leak-free protocol** (Issue 2) across **5 seeds**, replacing the old single-seed
runs (root `08_ablation_window.py` / `09_ablation_tte.py`, which ran on the
**leaky** `sequences/`). The reviewer concern: the old "AUC insensitive to
obs_len/TTE" rested on a single-seed spread (~0.005 AUC) that sat *below* the
seed-to-seed std (~0.011) — undefendable as written.

> **Metric note (F1-first).** This issue is an **AUC-scoped** ablation (window/TTE
> sensitivity is measured on ranking AUC). Under the project's **F1 → acc → AUC**
> hierarchy the F1-first headline lives in [`../../f1_optimization/`](../../f1_optimization/)
> and [`../issue3_baseline_comparison/`](../issue3_baseline_comparison/); AUC is
> reported here as threshold-free corroboration.

## How to run

```bash
source .venv/bin/activate
python journal_prep/issue6_window_tte_ablation/06_multiseed_ablations.py --device mps
# --build-only builds the per-config sequences without training
# per-run JSONs are cached, so re-running only regenerates the summary/figure
```

Local on the M4 GPU (MPS), ~11 s/training, 30 trainings (6 configs × 5 seeds).

## Design (TTE-band mapping decided 2026-06-26)

The clean builder anchors windows on PIE's `crossing_point` and takes a TTE
**band** (`--tte-min/--tte-max`), not a single `--tte`. The sweeps map onto it as:

- **Window sweep:** `obs_len ∈ {8,16,30}`, TTE band fixed at the canonical `[30,60]`.
  `obs16/[30,60]` reuses `../issue2_clean_protocol/sequences_clean/` — it is the
  window-sweep centre **and** the shared baseline.
- **TTE sweep:** `obs_len` fixed at 16, **single-point** band `[T,T]` for
  `T ∈ {30,45,60}` ("predict exactly T frames ahead" — the natural reading of
  TTE=T and faithful to the old single-point `09_`; 0-width band so only the
  horizon moves between cells). `obs16/[30,60]` is **not** a TTE cell.

Everything else is **locked** to the baseline (`04_train_bilstm.py` / `06b`):
5-D `BiLSTMIntentPredictor`, split train=set01/02/04 · val=set05/06 · test=set03,
train-only norm, **pos_weight=1.682 fixed across all cells** (CLAUDE.md convention:
only the ablated factor moves), lr=1e-3, hidden 128, 2 layers, dropout 0.3,
patience 15, threshold 0.5, best-on-val-AUC checkpoint, test touched once per
(config, seed).

## Headline result — two different answers

| Axis | Result | Spread vs seed std | Significance | Verdict |
|---|---|---|---|---|
| **Observation window** (8/16/30) | 0.931 / 0.933 / 0.937 | 0.0058 < ±0.007 | all pairwise p > 0.21; Kruskal p 0.566 | **Insensitive — old claim confirmed** |
| **Prediction horizon** (TTE 30/45/60) | 0.960 / 0.948 / 0.919 | 0.0417 ≫ ±0.005 | every pairwise p ≤ 0.008; Kruskal p 0.002 | **Significant monotonic decline — old claim OVERTURNED** |

- **Window length doesn't matter** — the between-setting spread (0.006) is *smaller
  than the seed-to-seed noise* (0.007), so the three settings are statistically
  **equivalent** (argued from effect size, not the underpowered n=5 p-value).
  obs_len=16 is a safe choice; the old single-seed conclusion survives.
- **Prediction horizon DOES matter** — AUC declines significantly and
  monotonically as the model predicts further ahead (1.0 s → 2.0 s). The old
  *leaky* single-seed "insensitive to TTE" was an artifact: on leaky data the
  model detected in-progress crossings regardless of nominal horizon. On
  leak-free, `crossing_point`-anchored data it degrades gracefully — the
  intuitive, expected behaviour, and a point in favour of the model's validity.
- **Cross-check:** the obs16/[30,60] cell on MPS gives 0.933 ± 0.007, reproducing
  the existing CPU multiseed baseline (0.932 ± 0.011) within seed noise.

> ⚠ The single-point TTE cells use a smaller, single-horizon test set (N≈500), so
> their absolute AUCs (0.92–0.96) are **not** directly comparable to the band-based
> headline 0.932 — read the TTE result as the *relative trend*, not as three new
> headline numbers.

### Matched-cohort TTE control (`06b_`) — the horizon effect is not a sample artifact

Under single-point sampling the three TTE cells use **nested, different** pedestrian
sets (TTE=30 needs track length L≥46, TTE=60 needs L≥76), so TTE=30 carries 48 extra
*short, harder* tracks (Issue-2 parity: 46–75f AUC 0.863 vs ≥76f 0.919) — a reviewer
can argue the decline is partly a change of sample. `06b_matched_track_tte.py` removes
that confound: it restricts all three horizons to the **common cohort eligible for
TTE=60** (identical pedestrians *and* labels in train and test; only the observed
16-frame window slides). The decline survives **essentially unchanged**:

| TTE | all-eligible (06_) | matched cohort (06b_) | sample effect |
|---|---|---|---|
| 30 (1.0s) | 0.960 | 0.961 | +0.001 |
| 45 (1.5s) | 0.948 | 0.946 | −0.002 |
| 60 (2.0s) | 0.919 | 0.919 | 0.000 |

Matched spread 0.0419 > seed std ±0.0065; every pairwise paired-t p ≤ 0.004; Kruskal
p 0.002. **The horizon effect is genuine, not an artifact of track-length
eligibility** — this is the publication-defensible version of the TTE result.

## Files

```
06_multiseed_ablations.py          harness (build + 30 trainings + analysis)
06_window_multiseed.csv            per (obs_len, seed) test metrics
06_tte_multiseed.csv               per (TTE, seed) test metrics
06_multiseed_ablation_summary.md   mean±std tables + significance + verdict
06_ablation_figure.png             window (flat) vs TTE (declining) AUC, seed scatter
06b_matched_track_tte.py           matched-cohort TTE control (removes nested-sample confound)
06b_matched_tte_results.csv        per (horizon, seed) on the fixed cohort
06b_matched_tte_report.md          matched table + paired tests + verdict
06b_matched_tte_figure.png         all-eligible vs matched-cohort AUC-vs-horizon (curves overlap)
sequences/<config>/                per-config clean X/y/meta
sequences_matched/<cfg>/           matched-cohort X/y/meta (tte30, tte45)
runs/<config>/seed<k>.json         per-run metrics
runs_matched/<cfg>/seed<k>.json    matched-cohort runs (tte30, tte45; tte60 reused)
```

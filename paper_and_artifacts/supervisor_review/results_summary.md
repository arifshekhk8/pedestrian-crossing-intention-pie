# Results Summary — all numbers in one place

> **⚠ SUPERSEDED SNAPSHOT (2026-06-16, banner added 2026-07-13).** All numbers below are
> **leaky-era** (test = 587 sequences, N = 1,389, `pos_weight` 1.44, AUC-first). They
> predate the leakage fix and the F1-first directive — **do not quote them.** Current
> leak-free, F1-first headline: **F1 0.844 / 0.847**, Acc 0.897 / 0.896, AUC 0.940 / 0.947
> (BiLSTM / Transformer) on clean test set03 (2,094 windows). See
> `../../journal_prep/issue3_baseline_comparison/` and `../../f1_optimization/README.md`.

All figures are in `figures/`. All numbers are read directly from the run outputs
(`runs/*/final.json`) and the demo prediction CSVs (`data/`). Test set = PIE
**set03** (held out; never seen in training/validation).

---

## A. Main model — BiLSTM 5-D baseline (the headline result)

Test set: set03, 587 sequences. Single run, seed 42, best-val-AUC checkpoint.

| Metric | Value |
|---|---|
| **AUC-ROC** | **0.931** |
| F1 | 0.844 |
| Accuracy | 0.874 |
| Precision | 0.820 |
| Recall | 0.870 |
| Confusion `[[TN, FP], [FN, TP]]` | `[[313, 44], [30, 200]]` |

Figure: `figures/01_baseline_training.png`

---

## B. Model comparison (one factor changed at a time)

| Model | Input | Params | AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| **BiLSTM baseline** | 5-D (bbox + speed) | 594,561 | **0.931** | 0.844 | 0.820 | 0.870 |
| BiLSTM bbox-only | 4-D (bbox) | ~594K | 0.889 | 0.797 | 0.712 | 0.904 |
| BiLSTM + attention | 5-D | ~611,457 | 0.933 | 0.845 | 0.779 | 0.922 |

**Deltas vs baseline**
- bbox-only: **AUC −0.042**, precision −0.108 → ego-speed is a real, useful signal.
- attention: AUC +0.002 (noise), **recall +0.052**, precision −0.041 → trades
  precision for recall (the AV-safe direction).

Figures: `figures/02_all_models_comparison.png`,
`figures/03_precision_recall_tradeoff.png`

---

## C. Ablation 1 — observation window length (TTE fixed = 45)

| obs_len | seconds | AUC | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|
| 8 | 0.27 s | 0.936 | 0.862 | 0.885 | 0.809 | 0.922 |
| 16 | 0.53 s | 0.931 | 0.844 | 0.874 | 0.820 | 0.870 |
| 30 | 1.00 s | 0.935 | 0.839 | 0.857 | 0.770 | 0.921 |

**Finding:** AUC spread ≤ 0.005 — insensitive to history length. Even 0.27 s
suffices (good for low-latency deployment). Figure: `figures/04_window_ablation.png`

---

## D. Ablation 2 — prediction horizon TTE (obs fixed = 16)

| TTE | seconds | AUC | F1 | Accuracy | Precision | Recall | N_test |
|---|---|---|---|---|---|---|---|
| 30 | 1.0 s | 0.959 | 0.863 | 0.893 | 0.849 | 0.878 | 596 |
| 45 | 1.5 s | 0.931 | 0.844 | 0.874 | 0.820 | 0.870 | 587 |
| 60 | 2.0 s | 0.944 | 0.846 | 0.866 | 0.789 | 0.913 | 566 |

**Finding:** AUC stays high (0.93–0.96) out to a 2 s horizon; longer horizon →
higher recall, lower precision. Footnote: N_test shrinks slightly with longer TTE
(end-of-video tracks drop out). Figure: `figures/05_tte_ablation.png`

---

## E. Live demo — qualitative + quantitative check

Pipeline: raw video → **YOLO26-M** → **ByteTrack** → 16-frame `[bbox, ego-speed]`
buffer → **BiLSTM 5-D baseline** → `P(cross)`. Run on MacBook Air M4 (MPS),
~18 fps on 1080p, ~50 s per 900-frame (30 s) clip.

### Clip A — `demo_video_0016.mp4` (set03/video_0016, frames 1916–2815)
- Scenario: car approaches an intersection (ego-speed 0→27→0) and **stops**
  while a crowd crosses.
- 69 tracked persons, 3,051 window predictions. Crossing crowd boxed **red
  (P ≈ 0.75–0.88)**, sidewalk people low/green.
- GT note: in this window PIE only annotates crossers (10, all label 1); their
  matched mean P(cross) = **0.655** (correct direction). No negatives here → no
  contrastive number; this clip is the **qualitative crowd demo**.
- Figures: `figures/06_demo_crossing_crowd.png`, `figures/07_demo_approach.png`

### Clip B — `demo_video_0012.mp4` (set03/video_0012, frames 7676–8576)
- Scenario: **moving vehicle** (ego-speed mean ≈ 21, up to 34), **mixed** GT:
  5 crossing + 5 not-crossing pedestrians.
- 2,752 window predictions. Per-pedestrian (matched to PIE GT by IoU):

  | GT class | n | mean P(cross) | range |
  |---|---|---|---|
  | crossing (1) | 5 | **0.814** | 0.61 – 0.89 |
  | not-crossing (0) | 5 | **0.013** | 0.01 – 0.02 |

  **Ped-level AUC = 1.000, accuracy@0.5 = 1.000** (perfect separation).
- Figure: `figures/08_intent_over_time.png` — P(cross) over time: every crosser
  (red) sits above 0.5, every non-crosser (green) near 0. One crosser's curve
  dips late (the model briefly doubts a pedestrian finishing their crossing) —
  shown honestly.

> **Caveat (state it):** clip B's perfect 1.000 is on **10 pedestrians** — a
> small, illustrative sample, not a benchmark. The benchmark number is the
> **test-set AUC 0.931** (Section A). The demo's job is to show the *deployed
> pipeline* produces sensible, correctly-separated probabilities on raw pixels.

---

## F. What a reviewer should take away

1. A compact BiLSTM predicts crossing intent at **0.93 AUC** on held-out PIE
   scenes, 1.5 s ahead.
2. **Ego-speed contributes** (+0.042 AUC); attention trades precision for recall.
3. Results are **stable** across observation windows (8–30) and horizons (1–2 s).
4. The model **deploys end-to-end** on raw video (YOLO26 + ByteTrack) and gives
   clean, well-separated probabilities in real scenes.

## G. Multi-seed robustness (5 seeds: 42, 0, 1, 7, 123)

Contract identical to single-seed runs. Mean ± std (sample std, ddof=1).

| Model | AUC | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| **BiLSTM 5-D (baseline)** | **0.948 ± 0.013** | 0.853 ± 0.008 | 0.878 ± 0.007 | 0.808 ± 0.017 | 0.903 ± 0.021 |
| BiLSTM 4-D (bbox-only) | 0.887 ± 0.011 | 0.801 ± 0.018 | 0.832 ± 0.020 | 0.750 ± 0.043 | 0.863 ± 0.041 |
| BiLSTM 5-D + attention | 0.942 ± 0.007 | 0.848 ± 0.006 | 0.871 ± 0.007 | 0.787 ± 0.017 | 0.920 ± 0.010 |

Raw per-seed data: `data/multiseed_results.csv`. Summary CSV: `data/multiseed_summary.csv`.

**Key findings:**
- Baseline mean AUC (0.948) exceeds the single-seed headline (0.931) because seed 42
  happened to be a below-average run — the model is consistently stronger than reported.
- Ego-speed gap confirmed across seeds: baseline 0.948 vs bbox-only 0.887 = **+0.061 AUC**
  (larger than the single-seed delta of +0.042).
- Attention model has the **lowest variance** (AUC std 0.007 vs 0.013 for baseline) —
  the most stable architecture to train.
- All stds are small relative to the inter-model gaps; the ranking
  (baseline ≥ attention >> bbox-only) is not a seed-42 artifact.

---

## H. Open items before final submission (see README §8)
- ~~Multi-seed runs (mean ± std)~~ — **DONE** (Section G above).
- Tabulate vs published PIE intention baselines (the `PIEPredict` repo is local).
- (Optional) attention-weight visualization figure from the attention variant.

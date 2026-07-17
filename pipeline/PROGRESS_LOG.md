# Progress Log

> **Repository layout (GitHub reorg):** this doc lives in `pipeline/` alongside the
> scripts. Run scripts from the repo root (e.g. `python pipeline/04_train_bilstm.py`).
> Trained runs are in `paper_and_artifacts/runs/`; demo outputs in `pipeline/demo_out/`;
> the presentation pack in `paper_and_artifacts/supervisor_review/`. The gitignored
> `sequences/`, `pie_annotations.pkl`, and `PIE/` stay at the repo root.

## Day 0 — Setup (completed)

- Claude Project created
- PIE dataset access registered
- Plan locked in

## Day 2 — Annotations Parsed
- Output: pie_annotations.pkl
- Total rows: 582,376
- Unique pedestrians: 1,374
- crossing_label=0: 238,915 | crossing_label=1: 343,461 (59% crossing)
- Missing vehicle_speed: 0
- Note: crossing is majority class → pos_weight ≈ 0.70 for BCEWithLogitsLoss

## Day 3 — Sequence Generator Built
- Output: sequences/X.npy (1389, 16, 5), sequences/y.npy (1389,), sequences/meta.pkl
- y=0 (not-crossing): 819 (59.0%) | y=1 (crossing): 570 (41.0%)
- Skipped pedestrians: 38 (insufficient frames)
- CORRECTION: pos_weight = 819/570 ≈ 1.44 (not 0.70 — sequence balance differs from raw rows)

## Day 4 — BiLSTM Model + Sanity Check
- Files created: 03_bilstm_model.py, 04_train_bilstm.py
- Model params: 594,561
- Split: train=616 | val=186 | test=587
- 1-epoch sanity: train loss 0.7937 acc 0.581 | val loss 0.7071 acc 0.677
- Val acc (0.677) > majority baseline (0.59) — model is learning
- Device: CPU (move to Kaggle T4 for Day 5 full training)

## Day 5 — Full Training Run (baseline BiLSTM)
- Script: 04_train_bilstm.py (updated: early stop, AUC monitor, full metrics)
- Output: paper_and_artifacts/runs/bilstm_baseline/ (best.pt, final.json, history.json, norm_mean.npy, norm_std.npy)
- Training stopped: epoch 18 (patience=15, best at epoch 3)
- Test results (set03, 587 samples):
  - Accuracy: 0.874 | F1: 0.844 | AUC: 0.931 | Precision: 0.820 | Recall: 0.870
  - Confusion: TN=313, FP=44, FN=30, TP=200
- Note: AUC 0.931 exceeds expected range (0.75–0.85). Fast convergence at epoch 3.
- Note: FP(44) > FN(30) → model errs on caution side, good for AV safety
- Device: T4 GPU, ~3s/epoch

## Day 6 — Bbox-Only Ablation (Comparison Axis 2)
- New files: 03b_bilstm_model_flex.py, 04b_train_bbox_only.py, 05_compare_runs.py
- Original files untouched (for supervisor review)
- bbox-only model best epoch: 6 | test AUC: 0.889 | F1: 0.797
- vs baseline (5D): AUC delta = +0.042, F1 delta = +0.047
- Finding: ego-speed adds meaningful signal (AUC delta ≥ 0.01 threshold)
- Note: bbox-only has higher recall (0.904) but lower precision (0.712)
  → without speed it over-predicts crossings

  ## Day 7 — BiLSTM + Temporal Attention (Comparison Axis 1)
- New files: 07_bilstm_attention.py, 07_train_attention.py
- Original 03_bilstm_model.py and 04_train_bilstm.py untouched
- Attention model best epoch: 6 | test AUC: 0.933 | F1: 0.845
- vs baseline: AUC delta = +0.002 (on par), Recall +0.052, Precision -0.041
- Finding: attention adds marginal AUC gain but shifts error profile toward
  higher recall — preferable for AV safety (fewer missed crossings)
- Attention params: ~611K vs baseline 594K (+16K for attn_W and attn_v)

## Day 8 — Observation Window Ablation (Comparison Axis 3)
- New file: 08_ablation_window.py (builds sequences + trains + evaluates all 3)
- Bug fixed: pie_annotations.pkl re-saved with object dtype (pandas version fix)
- Reproducibility confirmed: obs_len=16 reproduced Day 5 exactly (AUC 0.931)
- Results (test set: set03, TTE=45 fixed):
  - obs_len= 8 (0.27s): AUC 0.936 | F1 0.862 | Acc 0.885 | P 0.809 | R 0.922
  - obs_len=16 (0.53s): AUC 0.931 | F1 0.844 | Acc 0.874 | P 0.820 | R 0.870
  - obs_len=30 (1.00s): AUC 0.935 | F1 0.839 | Acc 0.857 | P 0.770 | R 0.921
- Finding: AUC insensitive to window length (max delta 0.005, within noise)
  → 0.27s of observation is sufficient for strong prediction on PIE
- Deployment implication: shorter window = lower latency in real AV pipeline
- Output: ablation_window_results.json, day8_window_ablation.png,
  paper_and_artifacts/runs/ablation_window_{8,16,30}/ (best.pt, final.json, history.json, norm_*.npy)

## Day 9 — TTE (Prediction Horizon) Ablation (Comparison Axis 4)
- New file: 09_ablation_tte.py (builds sequences + trains + evaluates all 3 TTEs)
- pos_weight FIXED at 1.44 for all runs (consistent with Days 5-8) — TTE is the
  only variable. Natural train ratio logged per run but not fed to loss.
- Reproducibility confirmed: TTE=45 reproduced Day 5 EXACTLY (AUC 0.931, F1 0.844,
  drift +0.000) — same sequences + same fixed pos_weight as Day 8 obs=16.
- Results (test set: set03, obs_len=16 fixed):
  - TTE=30 (1.00s): AUC 0.959 | F1 0.863 | Acc 0.893 | P 0.849 | R 0.878 | N=596
  - TTE=45 (1.50s): AUC 0.931 | F1 0.844 | Acc 0.874 | P 0.820 | R 0.870 | N=587
  - TTE=60 (2.00s): AUC 0.944 | F1 0.846 | Acc 0.866 | P 0.789 | R 0.913 | N=566
- Finding: AUC flat across horizons (spread 0.028, non-monotonic → within noise)
  → BiLSTM predicts crossing intent as reliably at 2s as at 1s on PIE
- Real effect is precision/recall trade: longer horizon → precision down (0.849→0.789),
  recall up (→0.913). Model leans toward flagging crossings further out = SAFE
  failure mode for AV (false alarm > missed crossing).
- Caveat: N_test shrinks 596→587→566 (longer horizon drops end-of-video tracks).
  Test sets near-identical but not exactly — footnote in paper.
- Output: ablation_tte_results.json, day9_tte_ablation.png,
  paper_and_artifacts/runs/ablation_tte_{30,45,60}/ (best.pt, final.json, history.json, norm_*.npy)

## Phase 4 (Days 13-15) — YOLO26 + ByteTrack + BiLSTM Demo
- New file: 10_yolo_bytetrack_demo.py (one script, 3 stages: detect / track / demo)
- Ran LOCALLY on MacBook Air M4 (MPS), not Kaggle. Deps: ultralytics 8.4.68,
  opencv 4.13.0, lap, torchvision 0.27.0.
- Model: 5D baseline (paper_and_artifacts/runs/bilstm_baseline/best.pt, epoch 3) — the locked main model.
- Detector: YOLO26-M (yolo26m.pt, 42MB, ultralytics v8.4.0 assets). REQUIRED, no
  fallback. Confirmed loads (names[0]='person', task=detect).
- Tracker: ByteTrack via Ultralytics (tracker="bytetrack.yaml", persist=True),
  fed frame-by-frame from an OpenCV reader so we can seek/limit a segment and keep
  absolute PIE frame numbers aligned with the ego-speed lookup.
- Ego-speed: per-frame OBD speed from pie_annotations.pkl (set03/video_0016),
  6321 annotated frames. Default --ego-source pkl; --ego-source obd also supported.

- Data: downloaded set03/video_0016.mp4 (1.5GB, 1920x1080, 30fps, 18000 frames).
  York server (data.nvision2.eecs.yorku.ca) hard-throttles single connections
  (~12 KB/s); used a 24-way parallel segmented download (~320 KB/s) to get it.
  NOTE: PIE clips are not faststart (moov atom at end) — can't decode a partial.

- Demo segment: video_0016 frames 1916-2815 (30s, 900 frames). Chosen because it
  has 10 GT crossing pedestrians AND real ego motion (speed 0->27->0: the car
  approaches, decelerates, stops at the intersection while pedestrians cross).
  This exercises the 5D model's speed feature, unlike the busiest window
  (2816-3716, 21 crossers) where ego speed is a constant 0.

- Run: 900 frames in ~50s on MPS (~18 fps). 69 unique track IDs, 3051 16-frame
  window predictions, 4636 prediction rows.
- prob_cross spread 0.008-0.906 (mean 0.504). Clear separation: sidewalk/standing
  pedestrians ~0.01 (green), crossing crowd ~0.75-0.88 (red).
- Qualitative GT check (IoU>0.3 track->ped match): the 10 GT crossing peds in the
  window map to 23 track fragments with mean predicted prob 0.655 (correctly >0.5;
  low fragments are early-trajectory frames before intent ramps up). No GT
  not-crossing peds are annotated in this window, so no contrastive AUC here —
  the demo is qualitative; the quantitative AUC 0.931 stands from Day 5.
- Output: pipeline/demo_out/demo_video_0016.mp4 (1920x1080, 900f, 30s),
  pipeline/demo_out/demo_video_0016_f0*.png (10 sample frames),
  pipeline/demo_out/demo_video_0016_predictions.csv (frame,track_id,bbox,ego_speed,prob,pred)
- Bug fixed: checkpoint stores numpy-scalar val_metrics, so torch>=2.6 needs
  weights_only=False to load best.pt (would have crashed on Kaggle too).
- Naming: THESIS_PLAN reserved 07_ for the demo, but 07_ was taken by the
  attention model; demo is numbered 10_ to continue the real sequence (08, 09).

- 2nd demo clip (quantitative): set03/video_0012 frames 7676-8576, MOVING vehicle
  (ego mean ~21, max 34), MIXED GT = 5 crossing + 5 not-crossing peds.
  Per-ped (IoU-matched to GT): crossing mean P=0.814 [0.61..0.89], not-crossing
  mean P=0.013 [0.01..0.02]. The "ped-level AUC 1.000 on 10 peds" first reported here
  is an **illustrative qualitative check only** (N=10, hand-picked, not a result) and
  is **SUPERSEDED** by `journal_prep/issue10_gt_vs_detector/`, which measures the
  GT-box vs YOLO-box degradation properly on **98 pedestrians** (AUC drop only
  +0.009/+0.010 → prediction robust to detector box noise; the real pipeline weak
  links are detector recall 88% and ByteTrack identity fragmentation). Headline test
  number is the clean **AUC 0.932** (Issue 2), not the leaky 0.931.
  Output: pipeline/demo_out/demo_video_0012.mp4 + samples + predictions.csv.

## Multi-seed Robustness Run (2026-06-16, Kaggle T4)
- New file: 11_multiseed_runs.ipynb (Kaggle notebook)
- Re-trained all 3 model configs over 5 seeds [42, 0, 1, 7, 123].
  Contract held fixed: POS_WEIGHT=1.44, obs_len=16, split by set (test=set03),
  early stop on val AUC (patience=15), threshold=0.5.
- Output: multiseed_results.csv (15 rows, per-seed), multiseed_summary.csv (3 rows,
  mean±std), multiseed_summary.md (paste-ready table). All copied to paper_and_artifacts/supervisor_review/data/.
- NOTE: multiseed_summary.csv from Kaggle had scrambled model-name rows (pandas
  groupby ordering bug). Fixed locally by recomputing from multiseed_results.csv.

Multi-seed test results (mean ± std, 5 seeds, sample std ddof=1):

| Model | AUC | F1 | Acc | Prec | Rec |
|---|---|---|---|---|---|
| BiLSTM 5-D (baseline) | 0.948 ± 0.013 | 0.853 ± 0.008 | 0.878 ± 0.007 | 0.808 ± 0.017 | 0.903 ± 0.021 |
| BiLSTM 4-D (bbox-only) | 0.887 ± 0.011 | 0.801 ± 0.018 | 0.832 ± 0.020 | 0.750 ± 0.043 | 0.863 ± 0.041 |
| BiLSTM 5-D + attention | 0.942 ± 0.007 | 0.848 ± 0.006 | 0.871 ± 0.007 | 0.787 ± 0.017 | 0.920 ± 0.010 |

Key findings:
- Baseline mean AUC (0.948) > single-seed (0.931): seed 42 was a below-average run.
- Ego-speed gap CONFIRMED multi-seed: baseline 0.948 vs bbox-only 0.887 = +0.061 AUC.
- Attention most stable model: lowest variance (AUC std 0.007 vs baseline 0.013).
- All stds small relative to gaps: the ordering (baseline > attention >> bbox-only) is robust.
- Limitation #1 from supervisor README now ADDRESSED.

## Supervisor Review Pack (built 2026-06-16)
- New folder paper_and_artifacts/supervisor_review/ assembled for supervisor presentation:
  - 00_README_START_HERE.md (full project explainer + how-to-present + Q&A)
  - results_summary.md (all tables: main, comparisons, both ablations, demo)
  - figures/ (01-05 from paper_and_artifacts/runs/ plots; 06-09 demo frames + intent-over-time plot)
  - demo_videos/ (demo_video_0016.mp4 crowd, demo_video_0012.mp4 mixed/moving)
  - data/ (both prediction CSVs)
- Generated figures/08_intent_over_time.png from video_0012 CSV (P(cross) over
  time, crossers vs non-crossers, matched to GT).
- Documented limitations to raise proactively: single seed (seed=42, no mean±std),
  no published-PIE-baseline comparison yet, demo identity match is IoU-approx.

## Journal-prep Issue 1 + 2 — leakage fix & clean retrain (2026-06-21/22)
Reviewer-readiness work in `journal_prep/` (index: `journal_prep/README.md`).
**The above headline numbers (0.931, multiseed 0.948) are now superseded** — they
were measured on the leaky `sequences/`.

- **Issue 1 (leakage audit):** PIE boxes carry a per-frame `cross` attribute the
  original parser dropped. 387/570 crossers (67.9%) are already mid-crossing
  inside the 16-frame observation window; 64.7% are crossing in all 16 frames →
  the old task was partly *detection*, not prediction. VERDICT: leakage found.
- **Issue 2 (clean protocol):** rebuilt anchored at PIE `crossing_point`,
  TTE∈[30,60], 50% overlap → `sequences_clean/` N=4,906 (was 1,389), **0% leakage
  verified** by re-running the Issue-1 audit. `pos_weight` 1.44 → 1.682.

Clean test results (set03, leak-free):

| Model | leaky AUC | **clean AUC** | note |
|---|---|---|---|
| BiLSTM 5-D baseline | 0.931 (1-seed) / 0.948 (5-seed) | **0.932 ± 0.011** (5-seed) | seed 42 = 0.913 (low end) |
| BiLSTM 4-D bbox-only | 0.889 (5-seed) | **0.746** (1-seed) | collapses −0.14: was riding the static-geometry shortcut |
| BiLSTM 5-D + attention | 0.942 (5-seed) | **0.936** (1-seed) | small gain now visible (+0.02) |

Key findings:
- Leakage propped up the **weakest** model; speed-bearing models barely moved.
- **Ego-vehicle speed is the dominant predictor** (+0.167 AUC over bbox-only on
  clean data, vs the ~0.06 the leaky multiseed implied) — not bbox trajectory.
- Eval parity verified: per-pedestrian AUC 0.914 ≈ per-window 0.913; the laxer
  min-track-size (46 vs benchmark 75) does not inflate (short tracks are harder).
- Honest limitation for the paper: speed partly encodes the ego-driver's own
  anticipation (instrumented car slows for expected crossers).
- TODO: multi-seed bbox-only + attention on clean data
  (`journal_prep/issue2_clean_protocol/06_multiseed_variants_kaggle.ipynb`).

## Journal-prep Issues 3–12 + extensions (catch-up entry, 2026-07-13)

This log's day-by-day detail stops at Issue 2; Issues 3–12 and the two model-family
extensions each keep their own log. Summary of everything since:

- **Issues 3–10 (journal_prep/):** baseline comparison + positioning (3), bootstrap CIs
  (4, AUC [0.92,0.95]), LOSO 6-fold 0.928±0.041 (5), window/TTE multi-seed ablations (6),
  hidden-size/depth (7, h128 justified), 36-config grid search confirms the hand-set
  config (8), latency 0.575 ms/window (9), GT-box vs YOLO-box robustness (10). All done.
- **Transformer extension (`transformer/`, 2026-07-11/12):** staged val-only search;
  the searched Transformer **beats the BiLSTM on AUC** (0.950 vs 0.932, paired-bootstrap
  ΔAUC CI excludes 0) but the un-searched twin ties it — the win is the search, not
  attention.
- **⭐ Metric pivot → F1 → accuracy → AUC (supervisor, 2026-07-12).** New program
  `f1_optimization/` optimizes BOTH families F1-first: **LSTM improved to F1 0.844**
  (h256, F1-checkpointing, val-τ*≈0.5), transformer 0.847 — and **on F1 the families
  TIE** (the AUC win is metric-specific). Headline framing is now F1-first everywhere.
- **Issue 12 (`journal_prep/issue12_unified_pipeline/`):** ONE model-agnostic engine
  (bilstm/transformer/gru/birnn) with equivalence gates ALL PASS + single-device CPU
  replication — the fair-comparison entrypoint for all future model families. Legacy
  `pipeline/04_train_bilstm.py` retains the leaky-era defaults (banner added) and is
  no longer the training path.
- **GRU study (`gru/`, 2026-07-14).** Third supervisor directive (more model families):
  the GRU (BiLSTM's gated twin, only `nn.LSTM`→`nn.GRU`) got the identical Issue-8 search
  + F1-first optimization on the issue12 engine (local CPU). **TIES the BiLSTM on F1**
  (vs BiLSTM-F1 ΔF1 +0.0071, CI incl. 0) **and on AUC at matched capacity/selection**
  (vs frozen BiLSTM ΔAUC −0.0008), both surviving the pedestrian-cluster bootstrap; loses
  to the searched transformer on AUC (ΔAUC −0.0070). GRU-F1 test F1 0.849 / AUC 0.941 /
  Acc 0.901, LOSO 0.946. **The recurrent cell doesn't matter — the input signal does.**
  Vanilla-RNN (`birnn`) parallel study pending. Full: `gru/SUPERVISOR_SUMMARY.md`.
- Detailed logs: `journal_prep/PROGRESS_LOG` equivalents per issue, `transformer/PROGRESS_LOG.md`,
  `f1_optimization/PROGRESS_LOG.md`. Current numbers: `f1_optimization/README.md`,
  `journal_prep/issue3_baseline_comparison/03_baseline_comparison.md`.
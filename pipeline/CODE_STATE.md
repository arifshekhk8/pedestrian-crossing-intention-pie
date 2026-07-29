# Code State

> **Repository layout (GitHub reorg):** this doc lives in `pipeline/` alongside the
> scripts. Run scripts from the repo root (e.g. `python pipeline/04_train_bilstm.py`).
> Trained runs are in `paper_and_artifacts/runs/`; demo outputs in `pipeline/demo_out/`;
> the presentation pack in `paper_and_artifacts/supervisor_review/`. The gitignored
> `sequences/`, `pie_annotations.pkl`, and `PIE/` stay at the repo root. The
> Transformer-vs-BiLSTM extension is a separate top-level folder, `transformer/`,
> with its own docs (`transformer/PLAN.md` / `README.md` / `PROGRESS_LOG.md`) — not
> tracked in this file. Likewise the F1-first program (`f1_optimization/`, 2026-07-12:
> LSTM F1 0.828→0.844, families TIE on F1), the GRU-vs-BiLSTM recurrent-cell study
> (`gru/`, 2026-07-14: GRU ties the BiLSTM on F1 and AUC — cell doesn't matter, input
> does; own docs), the vanilla-RNN gating-isolation study (`rnn/`, 2026-07-14: the
> un-gated `birnn` ties the LSTM/GRU on F1 and ties the searched transformer on AUC —
> gating buys nothing over 16 steps; smallest/fastest family; own docs), and the
> unified model-agnostic training
> engine (`journal_prep/issue12_unified_pipeline/`, 2026-07-13 — THE entrypoint for
> new training; `04_train_bilstm.py` below is a historical artifact with leaky-era
> defaults, see its docstring banner).

## 01_parse_annotations.py — DONE
Parses PIE XML annotations into a single pandas DataFrame.
- Input: PIE dataset root
- Output: pie_annotations.pkl
- Columns: set_id, video_id, ped_id, frame, x1, y1, x2, y2, vehicle_speed, action, crossing_label

## pie_annotations.pkl — DONE
- 582,376 frame-level rows, 1,374 unique pedestrians
- crossing 0: 238,915 | crossing 1: 343,461
- No missing vehicle_speed values

## 02_build_sequences.py — DONE
Builds fixed-length observation windows from pie_annotations.pkl.
- Config: obs_len=16, TTE=45, input_dim=5 (x1,y1,x2,y2,vehicle_speed)
- Output: sequences/X.npy, sequences/y.npy, sequences/meta.pkl

## sequences/ — DONE
- X.npy shape: (1389, 16, 5)
- y.npy shape: (1389,) — 0: 819 (59%) | 1: 570 (41%)
- pos_weight for BCEWithLogitsLoss = 1.44

## 03_bilstm_model.py — DONE
2-layer stacked BiLSTM. Input projection 5->64, BiLSTM hidden=128,
num_layers=2, dropout=0.3, bidirectional=True. Output head Linear(256->1).
Takes last timestep. Returns raw logits.

## 04_train_bilstm.py — DONE
Training script with PIE set splits, normalization, BCEWithLogitsLoss pos_weight=1.44.
Adam lr=1e-3, weight_decay=1e-5. Accepts --epochs arg.
- 1-epoch sanity passed: val acc 0.677

## 04_train_bilstm.py — DONE (updated Day 5, confirmed Day 6)
Full training script for bbox + ego-speed (5D) baseline. Early stopping
(patience=15) on val AUC. Saves best.pt, history.json, final.json, norm_*.npy.
- Run: paper_and_artifacts/runs/bilstm_baseline/ | Best epoch: 3
- Test: AUC 0.931 | F1 0.844 | Acc 0.874 | P 0.820 | R 0.870
- NOTE: this is the 5D baseline. For bbox-only see 04b_train_bbox_only.py

## 03b_bilstm_model_flex.py — DONE (Day 6)
Same architecture as 03_bilstm_model.py but input_dim is a constructor parameter.
- input_dim=5 → bbox + ego-speed (default, matches baseline)
- input_dim=4 → bbox-only (used by 04b_train_bbox_only.py)
- Class name: BiLSTMIntentPredictorFlex

## 04b_train_bbox_only.py — DONE (Day 6)
Bbox-only ablation training script. Drops column index 4 (vehicle_speed) after
loading, passes input_dim=4 to BiLSTMIntentPredictorFlex. All other settings
identical to 04_train_bilstm.py (same seed, patience, lr, pos_weight).
- Output: paper_and_artifacts/runs/bilstm_bbox_only/ (best.pt, final.json, history.json, norm_*.npy)
- Best epoch: 6 | test AUC: 0.889 | F1: 0.797 | Acc: 0.819

## 05_compare_runs.py — DONE (Day 6)
Reads final.json from multiple run directories and prints a side-by-side
comparison table with delta row. Used for supervisor presentation.
- Currently compares: bilstm_baseline vs bilstm_bbox_only
- Will be extended in Day 12 to include all ablation runs

## 07_bilstm_attention.py — DONE (Day 7)
BiLSTM + additive temporal attention model. Identical backbone to
03_bilstm_model.py but replaces last-timestep pooling with Bahdanau-style
attention over all T=16 timesteps. return_attn=True flag returns per-frame
weights (B, T) for visualization.
- input_dim=5 (bbox + ego-speed), attn_dim=64
- Params: ~611,457 (+16K over baseline)

## 07_train_attention.py — DONE (Day 7)
Training script for attention model. Identical to 04_train_bilstm.py except
imports BiLSTMAttentionIntentPredictor from 07_bilstm_attention.py.
- Output: paper_and_artifacts/runs/bilstm_attention/ (best.pt, final.json, history.json, norm_*.npy)
- Best epoch: 6 | test AUC: 0.933 | F1: 0.845 | Acc: 0.867 | P: 0.779 | R: 0.922

## 08_ablation_window.py — DONE (Day 8)
Observation window ablation script. Loops over obs_len = [8, 16, 30].
For each: builds sequences from pie_annotations.pkl via subprocess call to
02_build_sequences.py, trains baseline BiLSTM (identical hyperparams),
evaluates on test set. Saves per-run results + comparison plot.
- All else fixed: TTE=45, input_dim=5, same seed/lr/patience/pos_weight
- Output: /kaggle/working/ablation_window_results.json
         /kaggle/working/day8_window_ablation.png
         /kaggle/working/ablation_window_{8,16,30}/
- Finding: AUC within 0.005 across all windows → insensitive to obs length

## pie_annotations.pkl — UPDATED (Day 8)
Re-saved with plain object dtype for string columns. Required because original
was saved with pandas 2.1+ StringDtype which older Kaggle pandas cannot load.
Fix applied locally: df[col].astype(object) for all string columns.

## 09_ablation_tte.py — DONE (Day 9)
TTE (prediction horizon) ablation script. Loops over TTE = [30, 45, 60].
For each: builds sequences from pie_annotations.pkl via subprocess call to
02_build_sequences.py (--tte arg), trains baseline BiLSTM, evaluates on test.
Saves per-run results + comparison plot.
- pos_weight FIXED at 1.44 (NOT recomputed per run) — keeps TTE the only
  variable and reproduces Day 5 baseline at TTE=45. Natural class ratio is
  logged per run for the record (final.json: n_train_pos, n_train_neg).
- All else fixed: obs_len=16, input_dim=5, same seed/lr/patience.
- Output: /kaggle/working/ablation_tte_results.json
         /kaggle/working/day9_tte_ablation.png
         /kaggle/working/ablation_tte_{30,45,60}/
- Finding: AUC insensitive to horizon out to 2s; real effect is P/R trade
  (longer horizon → lower precision, higher recall).

## 10_yolo_bytetrack_demo.py — DONE (Phase 4, Days 13-15)
End-to-end demo: raw video → YOLO26-M → ByteTrack → per-track 16-frame buffer of
[x1,y1,x2,y2,ego_speed] → BiLSTM (5D baseline) → P(cross) overlay → annotated mp4.
- Stages: --stage {detect,track,demo}. Frames read via OpenCV with --start-frame
  / --max-frames (seek + limit a segment); each frame fed to yolo.track(persist=True).
- Device auto: cuda → mps (Apple Silicon) → cpu; override with --device.
- Loads BiLSTMIntentPredictor (03_bilstm_model.py) + best.pt with weights_only=False
  (ckpt holds numpy-scalar val_metrics). Normalization = (x-mean)/std from
  norm_mean/std.npy, threshold 0.5 — identical to 04_train_bilstm.py.
- Ego-speed: build_speed_map() from pie_annotations.pkl (default) or *_obd.xml
  (--ego-source obd, reuses parse_obd from 01_parse_annotations.py).
- YOLO26-M REQUIRED, no fallback (raises if yolo26m.pt won't load).
- Naming: 10_ (not the reserved 07_, which the attention model took).
- Verified run: set03/video_0016 frames 1916-2815 on M4 MPS, ~50s/900 frames.
  Output: pipeline/demo_out/demo_video_0016.mp4 + sample PNGs + predictions.csv.

## 12_supervisor_demo.py — DONE (2026-07-29)
Presentation demo. Same model, front end and threshold as 11_ (imports them, so the
two cannot drift); the difference is entirely in the presentation and in what it
measures.
- Overlay: per-pedestrian verdict ("WILL CROSS 0.78" / "not crossing" / "buffering
  n/16"), a probability bar with a tick at tau, header (model, threshold, frame, ego
  speed, live throughput) and a legend. Colliding labels are nudged apart and joined
  to their box by a leader line.
- `--scene {anticipation,bystander,driving,busy,uncertainty}`: presets picked from the
  ranked candidate table. `uncertainty` is included on purpose so the reel is not all
  successes.
- `--live` plays in a window while computing (space pauses, q quits); `--write-video`
  writes H.264 via an ffmpeg transcode, because OpenCV can only emit mp4v and mp4v
  will not play in Keynote, PowerPoint or a browser.
- Timing is split into the system (detector + tracker, intention ensemble) and demo
  scaffolding (privacy blur, overlay, encode), which are excluded from the headline
  FPS. Measured on the M4 Air at 1920x1080: detector ~38 ms/frame, ensemble ~2 ms per
  window (~0.4 ms per single model), together ~22 FPS = 0.7x real time, detector 80-90%
  of it. Detection-bound, matching Section 4.9.
- Ego speed is flagged `(held)` on frames PIE does not annotate, so a carried-forward
  value is never read as a measurement.
- Instructions: pipeline/HOW_TO_RUN_THE_DEMO.md. Outputs to pipeline/demo_videos/
  (gitignored) with a .json of provenance beside each mp4.

## PIE_clips/set03/video_0016.mp4, video_0012.mp4 — DOWNLOADED (Phase 4)
Two raw demo clips (each ~1.5GB, 1920x1080, 30fps, 18000 frames). Fetched from
the York server via 24-way parallel segmented download (server throttles single
conns). video_0016 = crowd crossing at a stop (qualitative); video_0012 = moving
vehicle + mixed crossing/not-crossing GT (quantitative, ped-level AUC 1.000 on 10
peds). Other set03 clips not downloaded.

## paper_and_artifacts/supervisor_review/ — DONE (Phase 4 deliverable)
Self-contained pack for supervisor presentation. 00_README_START_HERE.md (full
explainer + presentation script + Q&A + honest limitations), results_summary.md
(all tables), figures/ (01-05 results plots, 06-09 demo frames + intent-over-time),
demo_videos/ (both mp4s), data/ (both prediction CSVs). All numbers sourced from
paper_and_artifacts/runs/*/final.json and the demo CSVs.

## 11_multiseed_runs.ipynb — DONE (2026-06-16, Kaggle T4)
Kaggle notebook: re-trains baseline / bbox-only / attention over 5 seeds
[42,0,1,7,123] and reports mean±std for all test metrics (addressed the
single-seed limitation). Self-contained — embeds the 3 model classes (from 03,
03b, 07) and the exact train/eval/split/normalize logic from 04_train_bilstm.py.
Contract held fixed: POS_WEIGHT=1.44, obs_len=16, split by set (test=set03),
early stop on val AUC (patience 15), threshold 0.5; seed 42 reproduces Day-5.
- Inputs: sequences (X.npy/y.npy/meta.pkl) attached as Kaggle dataset.
- Outputs: multiseed_results.csv, multiseed_summary.csv, multiseed_summary.md.
  All output files at project root + paper_and_artifacts/supervisor_review/data/.
- CAUTION: Kaggle output multiseed_summary.csv had scrambled model-name rows
  (pandas groupby ordering bug). Fixed locally by recomputing from raw results CSV.
- Results: BiLSTM baseline 0.948±0.013 | bbox-only 0.887±0.011 | attention 0.942±0.007
  (mean AUC over 5 seeds; see PROGRESS_LOG.md for full table)

## yolo26m.pt — DOWNLOADED (Phase 4)
YOLO26-M weights (42MB), auto-fetched by ultralytics 8.4.68 on first load.

## .venv — UPDATED (Phase 4; sklearn added 2026-06-21)
Added ultralytics 8.4.68, opencv-python 4.13.0, lap, torchvision 0.27.0 for the
demo (alongside existing torch 2.12 / numpy). **scikit-learn 1.9.0 was installed
2026-06-21** so the journal_prep clean retrains could run locally on the M4 —
earlier notes that "sklearn is NOT installed locally" are now out of date.

## journal_prep/ — reviewer-readiness (started 2026-06-21; Issues 1,2,4–10 DONE · 3 parked · 11 in progress)
Self-contained folder fixing the 11 issues a journal reviewer would raise, kept
**separate from the root pipeline above** (which stays as the historical record).
Index + status: `journal_prep/README.md`; master plan: `journal_prep/PLAN.md`.

- **issue1_leakage_audit/** — DONE. Proved the root AUC 0.931 is inflated: PIE
  boxes carry a per-frame `cross` attribute the original `01_parse_annotations.py`
  dropped; 67.9% of crossers are already mid-crossing inside the 16-frame window.
  `01_leakage_audit.py` (gained `--seq-dir`/`--out-dir` flags, defaults unchanged),
  `01_leakage_report.md`, `cross_state_map.pkl`.
- **issue2_clean_protocol/** — DONE (see its `README.md`). Leak-free rebuild
  anchored at PIE `crossing_point`, TTE∈[30,60], 50% overlap → `sequences_clean/`
  (N=4,906, **0% leakage verified**). Retrained all 3 variants into `runs_clean/`.
  - 5-D baseline: **AUC 0.932 ± 0.011** (5-seed); seed 42 = 0.913.
  - bbox-only: **0.753 ± 0.020** (5-seed) — collapses from leaky 0.889; **ego-speed,
    not bbox, is the dominant signal (+0.18 AUC)**. attention: **0.925 ± 0.010**
    (5-seed) — no measurable benefit on clean data.
  - eval parity verified (per-ped ≈ per-window; min-track-size not a confound).
  - Variant multi-seeding DONE on Kaggle T4 (`kaggle_result/`), cross-checked locally.
- **Contract change for clean data:** `pos_weight` recomputed 1.44 → **1.682**
  (clean train split 1366 neg / 812 pos). Root scripts `04_train_bilstm.py`,
  `04b_train_bbox_only.py`, `07_train_attention.py` gained a `--pos_weight` flag
  (default 1.44 → old commands reproduce byte-for-byte).
- **issue3_baseline_comparison/** — PARKED (finalize last). Published-baseline table
  (PCPA / GTransPDM / PIP-Net / Occlusion-Diffusion) + the "their-limitation → our-
  response" positioning matrix (`04_positioning_vs_prior_work.md`).
- **issue4_bootstrap_ci/** — DONE. 10k bootstrap on clean test (N=2,094): baseline
  ROC-AUC 0.932, **95% CI ≈ [0.92, 0.95]**, PR-AUC 0.876; ego-speed gap unambiguous.
- **issue5_loso_cv/** — DONE. Leave-one-set-out: **6-fold AUC 0.928 ± 0.041**; set03
  fold 0.931 ≈ fixed-split → set03 is representative, not an easy fold.
- **issue6_window_tte_ablation/** — DONE (multi-seed, MPS). Window length insensitive
  (0.931/0.933/0.937, equivalent); **TTE declines significantly** 0.960→0.948→0.919
  @1.0/1.5/2.0s (overturns old leaky "insensitive to TTE"); matched-cohort control
  (`06b_`) confirms it's not a sample artifact.
- **issue7_hidden_size/** — DONE. hidden 64/128/256 → 0.927/0.933/0.938; **128 kept**
  (256 n.s. at 3.8× params). Depth companion `07b_`: layers 1/2/3 ≈ 0.930/0.932/0.931
  (depth insensitive). Model is small-data-limited, not capacity-limited.
- **issue8_grid_search/** — DONE (supervisor-requested). 36-config grid, **val-only
  selection + test touched once**; search **confirms the hand-set baseline** (val-
  winner beats it on test by Δ+0.0006, p=0.91 n.s.). Hyperparameters now documented.
- **issue9_latency/** — DONE (M4, inference only). **Isolated BiLSTM = 0.575 ms/window**
  (CPU, ~58× inside 30 fps); CPU beats MPS at batch 1 (GPU dispatch overhead);
  pipeline **detection-bound** (YOLO26-M 93%, BiLSTM 4.5% → 27.5 fps).
- **issue10_gt_vs_detector/** — DONE (inference only). GT-box vs YOLO-box on demo
  clips (98 peds): **prediction robust to box noise** (AUC drop +0.009/+0.010, 3%
  decision flips); weak links are perception — detector recall 88%, ByteTrack
  fragmentation severe (track purity 39%). Replaces old qualitative "N=10, AUC 1.000".
- **Issue 11** (this doc cleanup): in progress.


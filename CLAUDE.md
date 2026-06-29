# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A master's-thesis project: predicting **pedestrian crossing intention** from the
PIE dataset with a bidirectional LSTM over short bbox + ego-speed sequences, plus
a YOLO26 + ByteTrack live-video demo. It is a linear, numbered research pipeline —
not an application. There is no build system, no test suite, no linter; "running
the code" means executing the numbered scripts in order.

Authoritative project state lives in three hand-maintained docs (in `pipeline/`) —
**read these first**, they are kept current with real numbers:
- `pipeline/THESIS_PLAN.md` — locked architecture, dataset splits, day-by-day plan.
- `pipeline/PROGRESS_LOG.md` — chronological results log (every run's numbers).
- `pipeline/CODE_STATE.md` — per-file status and what each script produces.

`paper_and_artifacts/supervisor_review/` is a self-contained presentation pack
(explainer + figures + demo videos); regenerate its figures/CSVs from the run
outputs, don't hand-edit.

## Repository layout (after the GitHub reorg)

Three top-level folders. **Run scripts from the repo root** (e.g.
`python pipeline/04_train_bilstm.py`) so the relative paths resolve.
- `pipeline/` — all numbered scripts, the three project docs, the multi-seed
  result tables, and the live-demo outputs (`pipeline/demo_out/`).
- `journal_prep/` — the 11-issue journal-readiness program (one folder per issue).
- `paper_and_artifacts/` — `Journal_writing/` (manuscript), `runs/` (trained
  checkpoints + norm stats), and `supervisor_review/` (presentation pack).

Gitignored data lives at the repo root and is NOT tracked: `PIE/`, `PIE_clips/`,
`PIEPredict/`, `sequences/`, `pie_annotations.pkl`, `yolo26m.pt`, `.venv/`, `venv/`.

## Pipeline (scripts run in numeric order)

```
01_parse_annotations.py   PIE XML -> pie_annotations.pkl (one row per ped per frame)
02_build_sequences.py     pkl -> sequences/{X.npy (N,16,5), y.npy, meta.pkl}
03_bilstm_model.py        BiLSTMIntentPredictor (the locked baseline architecture)
04_train_bilstm.py        train 5-D baseline -> paper_and_artifacts/runs/bilstm_baseline/
03b + 04b                 bbox-only (4-D) ablation -> paper_and_artifacts/runs/bilstm_bbox_only/
07_bilstm_attention.py + 07_train_attention.py   attention variant -> paper_and_artifacts/runs/bilstm_attention/
08_ablation_window.py     obs_len {8,16,30} sweep
09_ablation_tte.py        TTE {30,45,60} sweep
10_yolo_bytetrack_demo.py Phase 4 live demo (YOLO26 -> ByteTrack -> BiLSTM -> overlay)
05_compare_runs.py        side-by-side table from paper_and_artifacts/runs/*/final.json
```

## Critical conventions (get these wrong and results silently break)

- **Inference contract** (must match `04_train_bilstm.py` exactly anywhere the
  model is used): feature order `[x1, y1, x2, y2, vehicle_speed]` as **raw PIE
  pixel coords** (1920×1080), NOT normalized to image size; standardize with
  `(x - mean) / std` using the per-feature `norm_mean.npy`/`norm_std.npy` saved in
  each run dir; window is exactly **obs_len=16** timesteps; decision threshold
  **0.5** on `sigmoid(logit)`.
- **Checkpoints need `weights_only=False`.** `best.pt` stores numpy-scalar
  `val_metrics` next to the state_dict, so `torch.load(..., weights_only=False)`
  is required on torch ≥ 2.6 (the default True crashes).
- **Module imports use importlib** because filenames start with digits, e.g.
  `import_module("03_bilstm_model").BiLSTMIntentPredictor`. Do not rename scripts
  to "fix" this.
- **Fixed data splits by recording set** (no random split — prevents leakage):
  train = set01/02/04, val = set05/06, **test = set03**. Defined in
  `04_train_bilstm.py` (`TRAIN_SETS`/`VAL_SETS`/`TEST_SETS`); reuse, don't redefine.
- **`POS_WEIGHT = 1.44`** (819 neg / 570 pos) is held fixed across all runs,
  including the TTE ablation, so the only variable is the one being ablated.
- **Reproducibility:** `set_seed(42)` + cuDNN deterministic. Re-running a config
  must reproduce the logged AUC; the ablations explicitly verify this against Day 5.
- **File-numbering quirk:** `THESIS_PLAN.md` reserved `07_` for the demo, but `07_`
  was taken by the attention model, so the demo is `10_`. New scripts continue the
  real sequence; don't reuse a taken number.

## Commands

Always activate the venv first (it holds torch/ultralytics/etc.):
```bash
source .venv/bin/activate
```

Build data and train the baseline (run from the repo root):
```bash
python pipeline/01_parse_annotations.py --pie-root PIE       # -> pie_annotations.pkl
python pipeline/02_build_sequences.py --obs-len 16 --tte 45  # -> sequences/
python pipeline/04_train_bilstm.py --epochs 100              # -> paper_and_artifacts/runs/bilstm_baseline/
python pipeline/05_compare_runs.py                           # results table
```

Run the live demo (Phase 4). Reads frames via OpenCV so you can seek/limit a
segment; device auto-selects cuda → mps → cpu:
```bash
python pipeline/10_yolo_bytetrack_demo.py --stage demo \
  --video PIE_clips/set03/video_0012.mp4 --video-id video_0012 \
  --start-frame 7676 --max-frames 900 \
  --weights-dir paper_and_artifacts/runs/bilstm_baseline --dump-csv --out-dir pipeline/demo_out
# --stage detect / track run just that sub-step; --ego-source obd reads *_obd.xml instead of the pkl
```

## Execution environments (this matters)

- **Local (this machine):** MacBook Air M4. The demo (`10_`) runs here on **MPS**;
  raw PIE clips (repo root) and `paper_and_artifacts/runs/` weights are present. **`scikit-learn` is NOT installed
  locally** (training/metrics ran on Kaggle) — compute metrics manually (e.g.
  Mann-Whitney AUC) rather than importing sklearn in local scripts.
- **Kaggle (T4 GPU):** training and the ablation sweeps (`04`, `08`, `09`) were
  run there. Those scripts hard-code `/kaggle/working/` and `/kaggle/input/`
  output paths — adjust paths when running them locally.

## Data acquisition

`PIE/annotations*` are committed, but **raw video clips are not** (~1.5 GB each).
The York host throttles single connections hard (~12 KB/s); use a parallel
segmented download (HTTP range requests are supported) — see
`PROGRESS_LOG.md` Phase 4. PIE clips are NOT faststart (moov atom at end), so a
partially-downloaded file is undecodable — finish + assemble before reading.
`PIE/` and `PIEPredict/` are vendored upstream repos (dataset tooling + the
original paper's baseline), kept for reference/comparison.

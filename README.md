# Pedestrian Crossing-Intention Prediction on PIE

Predicting whether a pedestrian is about to cross the road, from a short window of
their bounding-box motion plus the ego-vehicle's speed. The model is a two-stream
bidirectional LSTM trained on the [PIE dataset](https://data.nvision2.eecs.yorku.ca/PIE_dataset/),
and there is a live-video demo built on YOLO26 detection and ByteTrack tracking.

This is a research pipeline for a master's thesis and an in-progress journal paper,
not an application. The code runs as a set of numbered scripts executed in order.

## Headline result

Reporting hierarchy is **F1 → accuracy → AUC** (supervisor directive). On a
leakage-free, crossing-point-anchored PIE protocol, the two-stream (bounding box +
ego-speed) models, optimized for F1, reach:

| Metric (test set03, 5 seeds) | BiLSTM (F1-first) | Transformer (F1-first) |
|---|---|---|
| **F1** | **0.844 ± 0.008** | **0.847 ± 0.017** |
| Accuracy | 0.897 | 0.896 |
| ROC-AUC | 0.940 | 0.947 |
| Inference latency | 0.575 ms/window | 0.459 ms/window |

Two cheap input streams reach **F1 0.844–0.847** — within 0.02–0.03 of the multimodal
F1 ceiling (PedFormer 0.87) — while holding the **highest AUC in the standard-protocol
table** (0.94–0.95), at a fraction of the feature-extraction cost of the 3–7-stream
models above them. Under identical F1-first optimization the BiLSTM and Transformer
**tie on F1**, so the parsimony finding is about the input signal, not the architecture.
The dominant predictor is ego-vehicle speed (removing it drops AUC to 0.753). The
frozen AUC-selected baseline (the demo checkpoint) is AUC 0.932 ± 0.011, 95% bootstrap
CI [0.92, 0.95].

## Pipeline

The scripts live in [`pipeline/`](pipeline/) and run in numeric order. Run them
from the repository root so the relative paths resolve, for example
`python pipeline/04_train_bilstm.py`:

```
01_parse_annotations.py   PIE XML -> pie_annotations.pkl (one row per pedestrian per frame)
02_build_sequences.py     pkl -> sequences (N, 16, 5) windows + labels
03_bilstm_model.py        BiLSTMIntentPredictor (the baseline architecture)
04_train_bilstm.py        train the 5-D baseline
03b / 04b                 bounding-box-only (4-D) ablation
07_*                      attention variant
08_ablation_window.py     observation-window sweep {8, 16, 30}
09_ablation_tte.py        time-to-event sweep {30, 45, 60}
10_yolo_bytetrack_demo.py live demo: YOLO26 -> ByteTrack -> BiLSTM -> overlay
05_compare_runs.py        side-by-side results table
```

## Repository layout

| Path | Contents |
|---|---|
| `pipeline/` | the numbered pipeline scripts, the hand-maintained project docs (`THESIS_PLAN.md`, `PROGRESS_LOG.md`, `CODE_STATE.md`), the multi-seed result tables, and the live-demo outputs (`demo_out/`) |
| `journal_prep/` | the 12-issue journal-readiness program: leakage audit, clean protocol, bootstrap CIs, LOSO, ablations, latency, detector-in-the-loop, and the unified model-agnostic training engine (`issue12_unified_pipeline/`) |
| `transformer/` | the supervisor-requested Transformer-vs-BiLSTM extension (staged search; beats the BiLSTM on AUC, ties it on F1) |
| `f1_optimization/` | the F1-first optimization program (metric hierarchy F1 → acc → AUC) for both model families |
| `paper_and_artifacts/Journal_writing/` | the MDPI MTI manuscript workspace (LaTeX scaffold + bibliography) |
| `paper_and_artifacts/runs/` | trained checkpoints, per-feature normalization stats, and final metrics |
| `paper_and_artifacts/supervisor_review/` | presentation pack (figures and result tables) — ⚠ a dated 2026-06 snapshot on the pre-leakage-fix numbers; the current numbers are in `journal_prep/` + `f1_optimization/` |

The repository is being populated incrementally, so some folders above arrive over
the first days of commits.

## Inference contract

Anywhere the model is used, the input must match training exactly:

- Feature order `[x1, y1, x2, y2, vehicle_speed]` as raw PIE pixel coordinates
  (1920×1080), **not** normalized to image size.
- Standardize with `(x - mean) / std` using the `norm_mean.npy` / `norm_std.npy`
  saved in each run directory under `paper_and_artifacts/runs/`.
- Observation window is exactly 16 timesteps; decision threshold is 0.5 on
  `sigmoid(logit)`.
- Checkpoints load with `torch.load(..., weights_only=False)` (the `best.pt` files
  store numpy-scalar metrics next to the state dict).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch ultralytics numpy scipy
```

Training and metrics were run on Kaggle (T4) and an Apple M4 (MPS); the live demo
runs locally on MPS.

## Data

The PIE annotations and raw video clips are **not** included here (the clips are
~1.5 GB each and the dataset is distributed by its authors). Obtain PIE from the
[official source](https://data.nvision2.eecs.yorku.ca/PIE_dataset/) and place it
under `PIE/`, then run `01_parse_annotations.py` to build the annotation table.

## Status

Experimental work is complete. The current effort is writing the journal paper for
MDPI MTI (Multimodal Technologies and Interaction). See `THESIS_PLAN.md` and
`PROGRESS_LOG.md` for the full record.

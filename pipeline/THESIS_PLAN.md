# Thesis Plan — Pedestrian Crossing Intention Prediction (BiLSTM)

> **Repository layout (GitHub reorg):** this doc lives in `pipeline/` alongside the
> scripts. Run scripts from the repo root (e.g. `python pipeline/04_train_bilstm.py`).
> Trained runs are in `paper_and_artifacts/runs/`; demo outputs in `pipeline/demo_out/`;
> the presentation pack in `paper_and_artifacts/supervisor_review/`. The gitignored
> `sequences/`, `pie_annotations.pkl`, and `PIE/` stay at the repo root.

## Goal

Predict whether a pedestrian will cross in front of the ego vehicle, using
bidirectional LSTM on PIE dataset bbox sequences (+ ego-speed).

## Architecture (locked)

- 2-layer stacked BiLSTM
- Hidden size: 128
- Dropout: 0.3
- Input projection: 5D (x1, y1, x2, y2, ego_speed) -> 64D embed
- Output head: sigmoid (binary crossing)
- Loss: BCEWithLogits with class weight
- Optimizer: Adam, lr=1e-3, weight_decay=1e-5
- Variant: + temporal attention over BiLSTM outputs

## Dataset (PIE only)

- Train: set01, set02, set04
- Val: set05, set06
- Test: set03
- Observation: 16 frames (0.5s @ 30fps)
- Prediction horizon: TTE=45 frames (1.5s) [default]

## Day-by-Day

### Phase 1: Setup & Data (Days 1-3)

- Day 1: Local env, clone PIE repo, download annotations
- Day 2: Parse XML annotations -> pandas DataFrame -> pickle
- Day 3: Build sequence generator (16-frame windows + labels) -> .npy

### Phase 2: BiLSTM Main Model (Days 4-8)

- Day 4: DataLoader + 2-layer BiLSTM, one epoch sanity
- Day 5: Full training run, baseline BiLSTM numbers
- Day 6: Class weighting + bbox normalization
- Day 7: Add ego-vehicle speed (5D input)
- Day 8: Add temporal attention variant

### Phase 3: Ablations (Days 9-12)

- Day 9: Observation window ablation (8/16/30)
- Day 10: TTE ablation (30/45/60)
- Day 11: Hidden size ablation (64/128/256) — **deferred; run as `journal_prep/issue7_hidden_size/` (multi-seed) + depth companion `07b_num_layers_ablation.py`**
- Day 12: Re-runs + final numbers locked

### Phase 4: YOLO26 + ByteTrack Demo (Days 13-15)

- Day 13: YOLO26-M inference on PIE set03
- Day 14: ByteTrack via Ultralytics
- Day 15: Full demo: video -> YOLO -> ByteTrack -> BiLSTM -> overlay

### Phase 5: Writing (Days 16-20)

- Day 16: Outline + Intro + Related Work
- Day 17: Methodology
- Day 18: Experiments + Results
- Day 19: Discussion + Conclusion
- Day 20: Polish + references + buffer

## Metrics (always report all 4)

- Accuracy (don't trust alone — class imbalance)
- F1 score
- AUC-ROC
- Precision + Recall (with confusion matrix)

## File naming convention (ACTUAL — supersedes the original aspirational list)

The numbering below is what was really built. Quirk: this plan reserved `07_` for
the demo, but `07_` was taken by the attention model, so the demo became `10_`
(continuing the real 08/09 sequence). New scripts continue the real sequence.

- 01_parse_annotations.py
- 02_build_sequences.py
- 03_bilstm_model.py            (+ 03b_bilstm_model_flex.py — bbox-only variant)
- 04_train_bilstm.py            (+ 04b_train_bbox_only.py)
- 05_compare_runs.py
- 07_bilstm_attention.py        (+ 07_train_attention.py)
- 08_ablation_window.py
- 09_ablation_tte.py
- 10_yolo_bytetrack_demo.py     (Phase 4 demo — NOT 07_)
- journal_prep/                 reviewer-readiness work (issue1–issue10 folders); see journal_prep/README.md

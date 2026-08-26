# Pedestrian Crossing-Intention Prediction on PIE

Predicting whether a pedestrian is about to step into the road, from a short window of
their bounding-box motion plus the ego-vehicle's speed. Two input streams, no pose, no
optical flow, no segmentation. The work re-extracts the
[PIE dataset](https://data.nvision2.eecs.yorku.ca/PIE_dataset/) under a leakage-free
protocol, evaluates four temporal encoders on it under one engine, and ships a live
YOLO26 + ByteTrack demo.

This is a research pipeline for a master's thesis and a journal manuscript, not an
application. There is no build system and no test suite; the code runs as numbered
scripts executed in order.

## Start here

| If you want | Read |
|---|---|
| The paper | [`paper_and_artifacts/Journal_writing/submission/main.pdf`](paper_and_artifacts/Journal_writing/submission/main.pdf) |
| Every model in one table | [`journal_prep/Analysis/model_comparison.md`](journal_prep/Analysis/model_comparison.md) |
| How the project is organised and why | [`CLAUDE.md`](CLAUDE.md) |
| The chronological record of every run | [`pipeline/PROGRESS_LOG.md`](pipeline/PROGRESS_LOG.md) |
| What each script does and produces | [`pipeline/CODE_STATE.md`](pipeline/CODE_STATE.md) |

Every number below traces to a file in this repository. Where a claim rests on an
external source, the source is named in
[`paper_and_artifacts/Journal_writing/submission/evidence/`](paper_and_artifacts/Journal_writing/submission/evidence/).

## Headline result

Metric hierarchy is **F1 → accuracy → AUC** (supervisor directive). Test is PIE **set03**,
2,094 windows, 32.5% positive; five seeds; selection on validation only; the test set was
touched once. Each row is that family's F1-optimised model under an identical protocol.

| Family | Params | **F1** | Accuracy | ROC-AUC | Latency (CPU, ms/window) |
|---|---|---|---|---|---|
| BiLSTM | 2,237,313 | **0.844 ± 0.008** | 0.897 ± 0.006 | 0.940 ± 0.004 | 0.575 |
| Transformer | 794,241 | **0.847 ± 0.017** | 0.896 ± 0.011 | 0.947 ± 0.003 | 0.459 |
| GRU | 1,678,209 | **0.849 ± 0.011** | 0.901 ± 0.010 | 0.941 ± 0.007 | 0.721 |
| Vanilla RNN | 560,001 | **0.852 ± 0.012** | 0.902 ± 0.008 | 0.948 ± 0.002 | 0.316 |

Source: [`journal_prep/Analysis/model_comparison.md`](journal_prep/Analysis/model_comparison.md)
and [`latency_comparison.md`](journal_prep/Analysis/latency_comparison.md).

## What the study establishes

**1. The standard windowing protocol leaks, and the leak is large.** Anchoring the
observation window at the end of the annotated track puts the crossing inside the window
it is supposed to precede for **67.9%** of crossing sequences. Re-anchoring every window at
PIE's own `crossing_point`, with a minimum look-ahead of 30 frames, removes this by
construction (0 of 4,906 windows) and yields 3.5× as many windows.
→ [`journal_prep/issue1_leakage_audit/`](journal_prep/issue1_leakage_audit/),
[`journal_prep/issue2_clean_protocol/`](journal_prep/issue2_clean_protocol/)

**2. The input signal matters; the architecture does not.** Given the same search budget
and the same F1-first recipe on the same engine, an LSTM, a pre-LN Transformer, a GRU and
an un-gated Elman RNN are statistically indistinguishable on F1 under a pedestrian-cluster
bootstrap. Not even gating changes the answer over a 16-frame window, and the smallest,
fastest family is the un-gated RNN.
→ [`transformer/`](transformer/), [`gru/`](gru/), [`rnn/`](rnn/), [`f1_optimization/`](f1_optimization/)

**3. Ego-vehicle speed carries the signal.** Removing it and leaving bounding boxes alone
drops F1 from 0.828 to **0.551** on the same protocol. This is confirmatory rather than
novel, and it comes with a caveat the manuscript states plainly: the instrumented vehicle
was driven by a human who could see the pedestrian, so part of what the speed profile
encodes is that driver's own anticipation.

**4. The equivalence is horizon-bounded.** Extending the observation window to 32 and 64
frames lowers F1 for every family. The four still tie at 16 and 32 frames; at 64 the
un-gated RNN alone falls behind. Longer histories are where gating would earn its keep.
→ [`journal_prep/obs_window_extension/`](journal_prep/obs_window_extension/)

The manuscript's baseline table places these results against published PIE work without
claiming the top of any column. The highest F1 in that table is PIP-Net's 0.88, obtained
at a 0.5 s horizon with seven input streams.

## The manuscript

[`paper_and_artifacts/Journal_writing/submission/`](paper_and_artifacts/Journal_writing/submission/)
holds the MDPI MTI submission package: `main.tex`, `references.bib`, the figures, and an
`evidence/` folder recording the verification behind the text — baseline numbers checked
against the primary PDFs, method warrants including the two the literature contradicts, a
number-by-number audit, and the sources for the road-safety statistics. Figure generators
for the two Introduction figures are in `evidence/`, so both rebuild from the dataset and
the cited releases.

## Repository layout

| Path | Contents |
|---|---|
| [`pipeline/`](pipeline/) | the numbered scripts, the three hand-maintained project docs (`THESIS_PLAN.md`, `PROGRESS_LOG.md`, `CODE_STATE.md`), the multi-seed result tables, and the live-demo outputs |
| [`journal_prep/`](journal_prep/) | the 12-issue journal-readiness program: leakage audit, clean protocol, bootstrap CIs, LOSO, ablations, latency, detector-in-the-loop, and the unified model-agnostic training engine (`issue12_unified_pipeline/`) |
| [`journal_prep/Analysis/`](journal_prep/Analysis/) | every model from all four families in one table, plus latency and hyperparameter tables — **the fastest way to see all results at once** |
| [`journal_prep/obs_window_extension/`](journal_prep/obs_window_extension/) | the same four F1-optimised models re-run at 32- and 64-frame observation windows, plus the PSI cross-test |
| [`journal_prep/cross_dataset_validation/`](journal_prep/cross_dataset_validation/) | JAAD replication of the protocol and the architecture comparison (JAAD has no ego speed, so it cannot test the input claim) |
| [`idd_ped_crossdataset/`](idd_ped_crossdataset/) | IDD-PeD cross-dataset study in unstructured Indian traffic — the first out-of-domain test of the 5-D input contract, since IDD-PeD does carry ego speed |
| [`transformer/`](transformer/) | the Transformer-vs-BiLSTM extension (staged search; wins on AUC, ties on F1) |
| [`f1_optimization/`](f1_optimization/) | the F1-first optimization program for both original families |
| [`gru/`](gru/) | the GRU-vs-BiLSTM recurrent-cell study (the gated twin ties the LSTM) |
| [`rnn/`](rnn/) | the vanilla-RNN gating-isolation study (removing gating costs nothing at 16 frames) |
| [`paper_and_artifacts/Journal_writing/`](paper_and_artifacts/Journal_writing/) | the manuscript workspace: the MDPI template, the figure generators, the drawio diagram sources, and the `submission/` package |
| [`paper_and_artifacts/runs/`](paper_and_artifacts/runs/) | trained checkpoints, per-feature normalization stats, and final metrics |
| `paper_and_artifacts/supervisor_review/` | presentation pack — ⚠ a dated 2026-06 snapshot on the pre-leakage-fix numbers; current numbers live in `journal_prep/` and `Analysis/` |

Each of `transformer/`, `f1_optimization/`, `gru/`, `rnn/`, `obs_window_extension/`,
`cross_dataset_validation/` and `idd_ped_crossdataset/` carries its own `PLAN.md` and
`README.md` stating what it set out to test before it was run.

## Pipeline

The scripts live in [`pipeline/`](pipeline/) and run in numeric order. Run them from the
repository root so the relative paths resolve, for example `python pipeline/04_train_bilstm.py`:

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
11_demo_clean_ensemble.py five-seed ensemble through detector + tracker on held-out video
11a_select_demo_scenes.py rank demo scenes on tracker purity and ground-truth agreement
12_supervisor_demo.py     render the annotated presentation video
05_compare_runs.py        side-by-side results table
```

⚠ `04_train_bilstm.py` keeps legacy pre-leakage-fix defaults as a historical artifact.
**Clean-protocol training goes through the unified engine**, which is the entry point for
any new family or retraining:

```bash
python journal_prep/issue12_unified_pipeline/12_unified_engine.py \
    --family bilstm --seed 42 --device cpu --select f1 --out_dir <run_dir>
# families: bilstm | transformer | gru | birnn ; --select auc = the frozen legacy rule
```

## Inference contract

Anywhere the model is used, the input must match training exactly:

- Feature order `[x1, y1, x2, y2, vehicle_speed]` as raw PIE pixel coordinates
  (1920×1080), **not** normalized to image size.
- Standardize with `(x - mean) / std` using the `norm_mean.npy` / `norm_std.npy` saved in
  each run directory.
- Observation window is exactly 16 timesteps; decision threshold is 0.5 on `sigmoid(logit)`
  for the AUC-selected baseline, or the validation-tuned τ recorded with each F1 model.
- Checkpoints load with `torch.load(..., weights_only=False)` (the `best.pt` files store
  numpy-scalar metrics next to the state dict).
- Fixed splits by recording set, never a random split: train `set01/02/04`, validation
  `set05/06`, test `set03`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch ultralytics numpy scipy scikit-learn pandas matplotlib opencv-python
```

Training and the ablation sweeps were run on Kaggle (T4); analysis, the latency
measurements and the live demo run locally on an Apple M4. Recurrent training that needs
exact reproduction is run on CPU: `nn.LSTM` on Apple MPS is process-history-dependent,
which is measured in [`journal_prep/issue12_unified_pipeline/`](journal_prep/issue12_unified_pipeline/).

## Data

No dataset is redistributed here. Obtain each from its authors and place it where the
scripts expect it:

| Dataset | Expected path | Used by |
|---|---|---|
| PIE (annotations + clips) | `PIE/`, `PIE_clips/` | the main pipeline |
| JAAD | `journal_prep/cross_dataset_validation/JAAD/` | the JAAD replication |
| IDD-PeD | `idd_ped_crossdataset/data/` | the IDD-PeD study |

PIE annotations are the starting point: run `01_parse_annotations.py` to build the
annotation table, then `02_build_sequences.py` for the legacy windows or
`journal_prep/issue2_clean_protocol/02_build_sequences_clean.py` for the leakage-free ones.

## Status

Experimental work is complete across all four model families, the observation-window
extension and the cross-dataset studies. The current effort is the MDPI MTI manuscript in
`submission/`. `PROGRESS_LOG.md` carries the full chronological record.

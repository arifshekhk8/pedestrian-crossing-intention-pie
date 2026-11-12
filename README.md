# Pedestrian Crossing Prediction on PIE — research archive

This repository is the working archive of a thesis project on predicting whether a
pedestrian will cross in front of a vehicle, from a short window of bounding-box motion
plus the ego-vehicle's speed, on the
[PIE dataset](https://data.nvision2.eecs.yorku.ca/PIE_dataset/). It holds the original
numbered pipeline, four model-family studies, the cross-dataset work and a live
YOLO26 + ByteTrack demo.

The paper that came out of this work is now under peer review, and **its conclusions
differ from several that the studies in this tree first reached**. Read the next two
sections before quoting anything from the folders below.

This is a research pipeline, not an application. There is no build system and no test
suite; the code runs as numbered scripts executed in order.

## Current status (2026-11-12)

| | |
|---|---|
| Paper | *Temporal Validity and Sampling Bias in Pedestrian Crossing Prediction: A Multi-Dataset Audit with Controlled Evaluation on PIE* |
| Authors | Md. Arif Shekh, Nusrath Tabassum, Md Abdus Samad Kamal, Kou Yamada |
| Manuscript | MDPI 4601011. Submitted to *Multimodal Technologies and Interaction* (September 2026); the peer-review export is named `vehicles-4601011` (MDPI *Vehicles*) |
| Version under review | compiled 2026-10-21, 20 pages, 32 references |
| Stage | first-round reviewer comments received; the revision is worked in a separate folder, `vehicles-4601011-revision` |
| Thesis | BCSE final report and defence deck built from the manuscript (October 2026) |

The manuscript is written on Overleaf and is **not** in this repository. The code and the
machine-written results behind every number in it are in a separate repository,
`pedestrian-intention-temporal-validity`. The drafts under
`paper_and_artifacts/Journal_writing/` are earlier versions with a different argument.

## Start here

| If you want | Read |
|---|---|
| The code and numbers behind the paper | the `pedestrian-intention-temporal-validity` repository (`README.md`, `docs/CONTRIBUTIONS.md`, `docs/LIMITATIONS.md`) |
| What is current and what is superseded in this tree | [`CLAUDE.md`](CLAUDE.md), "Current status" |
| The chronological record of every run | [`pipeline/PROGRESS_LOG.md`](pipeline/PROGRESS_LOG.md) |
| What each script does and produces | [`pipeline/CODE_STATE.md`](pipeline/CODE_STATE.md) |
| The pre-audit results table for this tree | [`journal_prep/Analysis/model_comparison.md`](journal_prep/Analysis/model_comparison.md) |

## What the manuscript under review reports

All model training and evaluation is on PIE: train set01/02/04, validation set05/06, test
set03. Five seeds (42, 0, 1, 2, 3), pedestrian-clustered bootstrap with 10,000
replicates, Holm correction within each family of contrasts. JAAD and IDD-PeD are used
for the frame-level audit only.

**1. Track-end windows contain the action they are meant to predict.**

| Dataset | Temporal anchor | Samples | Windows containing a crossing frame |
|---|---|---|---|
| PIE | historical track-end | 570 positive sequences | 67.9% of positives |
| PIE | `crossing_point` | 4,906 windows | 0.0% |
| JAAD | naive anchor | 460 positive windows | 93.0% of positives |
| JAAD | event anchor | 972 windows | 0.0% |
| IDD-PeD | track-end | 8,651 windows | 81.3% of positives |
| IDD-PeD | `crossing_point` | 7,919 windows | 29.6% of positives |
| IDD-PeD | earlier of event and onset | 7,318 windows | 0.0% |

**2. Removing crossing frames does not make the two classes comparable in time.** On the
event-anchored PIE data, the distance from the observation window to the end of the track
is 32–62 frames for non-crossers and 88–6,606 for crossers. That variable alone separates
the classes with AUC 1.0000. A phase-matched control re-draws the non-crossing windows
earlier, using a distribution estimated from training positives only (4,520 windows). It
reduces that separation to 0.7919. It does not remove it.

**3. Model performance under the two protocols** (box + ego-speed input, five-seed means):

| Model | AUC | PR-AUC | F1 | Accuracy |
|---|---|---|---|---|
| BiLSTM | 0.9242 → 0.8923 | 0.8688 → 0.7897 | 0.8276 → 0.7698 | 0.8826 → 0.8184 |
| Transformer | 0.9447 → 0.8971 | 0.8964 → 0.7947 | 0.8250 → 0.7544 | 0.8849 → 0.7828 |
| GRU | 0.9375 → 0.8817 | 0.8890 → 0.7684 | 0.8419 → 0.7401 | 0.8945 → 0.7964 |
| Vanilla RNN | 0.9481 → 0.8872 | 0.8925 → 0.7749 | 0.8487 → 0.7553 | 0.8996 → 0.8104 |

Each cell is event-anchored → phase-matched. Under event anchoring 7 of 18 family
contrasts are significant after correction; after phase matching, 0 of 18.

**4. The measured benefit of ego-speed depends on the sampling protocol.** Under event
anchoring, adding speed to the boxes raises PR-AUC by +0.0565 to +0.2189 and 10 of 12
contrasts are significant. After phase matching the differences are small and mixed
(−0.0206 to +0.0049 in PR-AUC) and 0 of 12 are significant. The manuscript does not
conclude that speed is uninformative, only that its apparent benefit on PIE is sensitive
to how the windows are built.

## What changed from this tree's earlier conclusions

An independent audit in September 2026 re-derived every claim from code and
machine-written artifacts. These are the conclusions recorded in this tree's study
folders that the manuscript does not carry:

| Earlier conclusion in this tree | Status |
|---|---|
| The four families are statistically indistinguishable; the input, not the architecture, decides | Not supported. With matched seeds and Holm correction, 7 of 18 contrasts are significant under event anchoring. After phase matching none is, which is a failure to resolve differences, not a tie |
| Ego-speed carries the signal (box-only F1 0.551) | Holds under event anchoring only; not supported after the phase-matched control |
| Metric hierarchy F1 → accuracy → AUC | Not used as a framing; four metrics are reported side by side |
| BiLSTM test AUC 0.932 ± 0.011 (hidden 128) | A different configuration from the manuscript's BiLSTM row (0.9242 ± 0.0086, hidden 256) |
| F1 declines with observation-window length; the un-gated RNN alone falls behind at 64 frames | Not in the manuscript. The window sweep is confounded with prediction horizon |
| The four families tie on JAAD | Not in the manuscript. All 20 JAAD runs sit at chance |
| IDD-PeD zero-shot transfer, latency, LOSO | Not in the manuscript |

The reasons are tabulated at the bottom of `docs/CONTRIBUTIONS.md` in the
`pedestrian-intention-temporal-validity` repository.

## Repository layout

| Path | Contents |
|---|---|
| [`pipeline/`](pipeline/) | the numbered scripts, the three hand-maintained project docs (`THESIS_PLAN.md`, `PROGRESS_LOG.md`, `CODE_STATE.md`), the multi-seed result tables, and the live-demo outputs |
| [`journal_prep/`](journal_prep/) | the 12-issue journal-readiness program: leakage audit, clean protocol, bootstrap CIs, LOSO, ablations, latency, detector-in-the-loop, and the unified model-agnostic training engine (`issue12_unified_pipeline/`) |
| [`journal_prep/Analysis/`](journal_prep/Analysis/) | every model from all four families in one table, plus latency and hyperparameter tables — the pre-audit record for this tree, not the source of the manuscript's numbers |
| [`journal_prep/obs_window_extension/`](journal_prep/obs_window_extension/) | the same four F1-optimised models re-run at 32- and 64-frame observation windows, plus the PSI cross-test |
| [`journal_prep/cross_dataset_validation/`](journal_prep/cross_dataset_validation/) | JAAD replication of the protocol and the architecture comparison (JAAD has no ego speed, so it cannot test the input claim) |
| [`idd_ped_crossdataset/`](idd_ped_crossdataset/) | IDD-PeD cross-dataset study in unstructured Indian traffic — the first out-of-domain test of the 5-D input contract, since IDD-PeD does carry ego speed |
| [`transformer/`](transformer/) | the Transformer-vs-BiLSTM extension (staged validation-only search, 78 configurations) |
| [`f1_optimization/`](f1_optimization/) | the F1-first optimization program for both original families |
| [`gru/`](gru/) | the GRU-vs-BiLSTM recurrent-cell study (same search and optimization as the BiLSTM) |
| [`rnn/`](rnn/) | the vanilla-RNN gating-isolation study (`nn.LSTM` → `nn.RNN`, same search and optimization) |
| [`paper_and_artifacts/Journal_writing/`](paper_and_artifacts/Journal_writing/) | the **pre-audit** manuscript workspace: the MDPI template, the figure generators, the drawio diagram sources, and the `submission/` draft. Not the paper under review |
| [`paper_and_artifacts/runs/`](paper_and_artifacts/runs/) | trained checkpoints, per-feature normalization stats, and final metrics |
| `paper_and_artifacts/supervisor_review/` | presentation pack — ⚠ a dated 2026-06 snapshot on the pre-leakage-fix numbers; do not quote it |

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

Experimental work in this tree stopped in August 2026. The September audit, the rewritten
manuscript and the experiments behind it live in the `pedestrian-intention-temporal-validity`
repository. The manuscript is under peer review (see "Current status" at the top); the
first-round reviewer comments have been received and the revision is worked in a separate
folder, `vehicles-4601011-revision`, which also holds the paper's drafts, figure generators
and number-checking script that used to sit at this repository's root.
`pipeline/PROGRESS_LOG.md` carries the full chronological record.

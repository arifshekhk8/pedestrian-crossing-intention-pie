# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A master's-thesis project: predicting **pedestrian crossing intention** from the
PIE dataset with a bidirectional LSTM over short bbox + ego-speed sequences, plus
a YOLO26 + ByteTrack live-video demo. It is a linear, numbered research pipeline —
not an application. There is no build system, no test suite, no linter; "running
the code" means executing the numbered scripts in order.

A supervisor-requested extension in `transformer/` (own docs: `transformer/PLAN.md` /
`README.md` / `PROGRESS_LOG.md`) compares a staged-search Transformer encoder against
the frozen BiLSTM under the identical protocol — result: the searched Transformer
measurably beats the BiLSTM **on AUC** (test AUC 0.950 vs 0.932, 10k paired bootstrap
ΔAUC 95% CI excludes 0), while the same architecture run with the BiLSTM's un-searched
recipe ties it exactly, i.e. the win came from the search, not the architecture family.

A second supervisor directive (2026-07-12) set the metric hierarchy to **F1 →
accuracy → AUC**. The F1-first program in top-level `f1_optimization/` (own
PLAN/README/PROGRESS_LOG) optimized both families symmetrically: the LSTM improved
significantly (test F1 0.828 → 0.844 ± 0.008, config h256), the transformer did not,
and **on F1 the families TIE** — the AUC win is metric-specific. Everything
replicates under ONE model-agnostic engine
(`journal_prep/issue12_unified_pipeline/12_unified_engine.py`, families
bilstm/transformer + gru/birnn registered) — **use that engine for any new model
family or clean-protocol retraining**, not the legacy trainers.

A third supervisor directive (2026-07-13) asked for two more model families on the
same pipeline. The GRU study in top-level `gru/` (own PLAN/README/PROGRESS_LOG/
SUPERVISOR_SUMMARY, phase folders G1–G5) gave the GRU — the BiLSTM's gated recurrent
twin (only `nn.LSTM`→`nn.GRU`) — the identical Issue-8 search + F1-first optimization
on the same unified CPU engine. Result: the GRU **ties the BiLSTM** on F1 (vs BiLSTM-F1
ΔF1 +0.0071, CI includes 0) and on AUC at matched capacity/selection (vs frozen BiLSTM
ΔAUC −0.0008), both surviving the pedestrian-cluster bootstrap; it loses to the searched
transformer on AUC (ΔAUC −0.0070). So **the recurrent cell type doesn't matter — the
input signal does** (the transformer's AUC edge stays its *search*, not attention-vs-
recurrence). The parallel vanilla-RNN study in top-level `rnn/` (own
PLAN/README/PROGRESS_LOG/SUPERVISOR_SUMMARY, phase folders R1–R6) closed this out
(2026-07-14): the **un-gated** bidirectional tanh RNN (only `nn.LSTM`→`nn.RNN`, gating
removed), given the same search + F1-first optimization on the same engine, **ties
BiLSTM-F1 and GRU-F1 on F1** (ΔF1 +0.0033 / −0.0038, CIs include 0) and is
level-to-marginally-better than the frozen BiLSTM on AUC at matched h128 (ΔAUC +0.0059) —
**no cell-isolation endpoint is a loss** — and, unlike the GRU, its AUC-optimized version
**ties the searched transformer** on AUC (ΔAUC −0.0013, CI includes 0). So **not even the
LSTM's gating is what matters over a 16-step window — the input signal is** (LSTM ≈ GRU ≈
vanilla RNN); it is also the smallest and fastest of the four families (0.316 ms/window CPU).
All robust to the pedestrian-cluster bootstrap; caveat: horizon-specific (an un-gated RNN
would likely fall behind over long sequences).

That caveat was then measured rather than assumed. The **observation-window extension**
(`journal_prep/obs_window_extension/`, 2026-07-19) re-ran the F1-optimised model of each
family at 32- and 64-frame windows: F1 declines with window length for every family, the
four still tie at 16 and 32, and at OW64 the un-gated RNN alone falls behind (F1 −0.050,
about twice the gated cells) — so the equivalence is **horizon-bounded, and now shown to
be**. Two **cross-dataset** tracks followed. `journal_prep/cross_dataset_validation/`
replicates the protocol and the four-family comparison on JAAD, which can test the
leakage and architecture claims but **not** the input claim, because JAAD carries no
ego-vehicle speed. `idd_ped_crossdataset/` (an isolated folder that writes nothing
outside itself) uses IDD-PeD, which does carry per-frame OBD speed in PIE's units, for
the first out-of-domain test of the actual 5-D input contract — zero-shot transfer of the
frozen PIE checkpoints, plus an independent from-scratch replication.

Every model from all four families sits in one place:
`journal_prep/Analysis/model_comparison.md`, with companion latency and hyperparameter
tables. **Read that before re-deriving any number.** The manuscript is at
`paper_and_artifacts/Journal_writing/submission/` (`main.tex`, `references.bib`,
figures, and an `evidence/` folder recording the source behind every external claim).

Authoritative project state lives in three hand-maintained docs (in `pipeline/`) —
**read these first**, they are kept current with real numbers:
- `pipeline/THESIS_PLAN.md` — locked architecture, dataset splits, day-by-day plan.
- `pipeline/PROGRESS_LOG.md` — chronological results log (every run's numbers).
- `pipeline/CODE_STATE.md` — per-file status and what each script produces.

`paper_and_artifacts/supervisor_review/` is a self-contained presentation pack
(explainer + figures + demo videos); regenerate its figures/CSVs from the run
outputs, don't hand-edit.

## Repository layout (after the GitHub reorg)

Eight top-level folders. **Run scripts from the repo root** so relative paths resolve.
- `pipeline/` — all numbered scripts, the three project docs, the multi-seed
  result tables, and the live-demo outputs (`pipeline/demo_out/`). ⚠ the trainer
  `04_train_bilstm.py` keeps LEGACY leaky-era defaults (`sequences/`, pos_weight
  1.44) as a historical artifact — clean-protocol training goes through the unified
  engine (below).
- `journal_prep/` — the 12-issue journal-readiness program (one folder per issue);
  `issue12_unified_pipeline/` holds THE unified model-agnostic training engine.
  Three later folders sit alongside the issues: `Analysis/` (every model from all four
  families in one table — start here), `obs_window_extension/` (OW32/64 + the PSI
  cross-test), and `cross_dataset_validation/` (the JAAD replication).
- `paper_and_artifacts/` — `Journal_writing/` (manuscript workspace: MDPI template,
  figure generators, drawio sources, and `submission/` = the MDPI MTI submission package
  with its `evidence/` folder), `runs/` (trained checkpoints + norm stats), and
  `supervisor_review/` (presentation pack).
- `transformer/` — the Transformer-vs-BiLSTM extension, one subfolder per phase
  (`phase1_setup/` … `phase5_analysis/`); see `transformer/README.md`.
- `f1_optimization/` — the F1-first optimization program (metric hierarchy
  F1 → acc → AUC); see `f1_optimization/PLAN.md`.
- `gru/` — the GRU-vs-BiLSTM recurrent-cell study, one subfolder per phase
  (`phase1_setup/` … `phase5_analysis/`); GRU ties the BiLSTM on F1 and AUC; see
  `gru/README.md`.
- `rnn/` — the vanilla-RNN-vs-BiLSTM **gating**-isolation study (`birnn` = un-gated
  bidirectional tanh RNN), one subfolder per phase (`phase1_setup/` … `phase5_analysis/`);
  the un-gated RNN ties the LSTM/GRU on F1 and ties the searched transformer on AUC — gating
  buys nothing over 16 steps — and is the smallest/fastest family; see `rnn/README.md`.
- `idd_ped_crossdataset/` — the IDD-PeD cross-dataset study (unstructured Indian traffic).
  Self-contained: it `importlib`-loads project code read-only and monkey-patches in memory
  only, writing nothing outside its own folder. See `idd_ped_crossdataset/README.md`.

Gitignored data is NOT tracked: `PIE/`, `PIE_clips/`, `PIEPredict/`, `sequences/`,
`pie_annotations.pkl`, `yolo26m.pt`, `.venv/`, `venv/`, plus the two vendored datasets
(`journal_prep/cross_dataset_validation/JAAD/`, `idd_ped_crossdataset/data/`), the IDD
checkpoints, the Overleaf export zips, and the LaTeX build products of `submission/`
(`main.pdf` is tracked). Checkpoints elsewhere (`runs_*/`, 190+ `.pt`) ARE tracked.

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
- **`POS_WEIGHT`**: 1.44 (819 neg / 570 pos) in the LEGACY pipeline only; the clean
  protocol (everything journal-bound) uses **1.682** (1366/812). Held fixed unless
  it is the ablated factor.
- **Reproducibility (measured, `journal_prep/issue12_unified_pipeline/`):** CPU
  training is bit-reproducible and context-free; **nn.LSTM training on Apple MPS is
  process-history-dependent** (same cfg+seed, different result depending on what ran
  earlier in the process) — recurrent runs needing exact reproduction go on CPU;
  transformer training is context-free on MPS. Kaggle-GPU↔local-CPU inference drift
  is ~1e-6 (benign, parity-gated).
- **File-numbering quirk:** `THESIS_PLAN.md` reserved `07_` for the demo, but `07_`
  was taken by the attention model, so the demo is `10_`. New scripts continue the
  real sequence; don't reuse a taken number.

## Commands

Always activate the venv first (it holds torch/ultralytics/etc.):
```bash
source .venv/bin/activate
```

Train under the clean protocol (the journal-bound path — data already exists at
`journal_prep/issue2_clean_protocol/sequences_clean/`):
```bash
python journal_prep/issue12_unified_pipeline/12_unified_engine.py \
    --family bilstm --seed 42 --device cpu --select f1 --out_dir <run_dir>
# families: bilstm | transformer | gru | birnn; --select auc = the frozen legacy rule
```
Legacy pipeline (historical record — ⚠ reproduces the RETRACTED leaky-era number,
`sequences/` + pos_weight 1.44, not the published 0.932):
```bash
python pipeline/01_parse_annotations.py --pie-root PIE       # -> pie_annotations.pkl
python pipeline/02_build_sequences.py --obs-len 16 --tte 45  # -> sequences/ (leaky protocol)
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
  raw PIE clips (repo root) and `paper_and_artifacts/runs/` weights are present.
  `.venv` has torch 2.12, **scikit-learn 1.9 and scipy** (an older note claiming
  sklearn was missing is stale — sklearn is fine in local scripts).
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

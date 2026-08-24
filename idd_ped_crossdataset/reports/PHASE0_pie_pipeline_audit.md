# Phase 0 — Audit of the existing PIE pipeline (reference implementation for the IDD-PeD port)

**Date:** 2026-08-25 · **Auditor:** this session · **Repo:** `/Users/arif/Developer/pedestrian-thesis`
**Status:** read-only audit. **No existing file was modified, renamed, or deleted.**

This note answers Phase 0 of the IDD-PeD brief: what the current PIE pipeline does, what it
expects as input, what must be reproduced for IDD-PeD, and what cannot transfer directly.

---

## 1. Which experiment is canonical

The repo contains several generations of the pipeline. The **canonical / final** one — the
implementation the IDD-PeD work must mirror — is the **clean protocol running through the
unified engine**:

| Layer | Canonical file | Status |
|---|---|---|
| Sequence builder | `journal_prep/issue2_clean_protocol/02_build_sequences_clean.py` | **reference** |
| Sequence data | `journal_prep/issue2_clean_protocol/sequences_clean/` (X 4906×16×5, y, meta.pkl) | **reference** |
| Training engine | `journal_prep/issue12_unified_pipeline/12_unified_engine.py` | **reference** |
| Model classes | `pipeline/03_bilstm_model.py`, `transformer/phase1_setup/00_transformer_model.py`, plus `RecurrentIntentPredictor` (gru/birnn) defined inside the engine | **reference** |
| Eval / threshold / bootstrap helpers | `f1_optimization/00_common.py` | **reference** |
| Cluster bootstrap | `f1_optimization/07_cluster_bootstrap.py` | **reference** |
| Cross-dataset precedent | `journal_prep/cross_dataset_validation/` (JAAD, Track A) | **precedent to follow** |

**Explicitly NOT canonical** (kept as historical artifacts, must not be used or touched):
- `pipeline/02_build_sequences.py` — the leaky track-end anchor (Issue-1: 67.9 % of crossers leaked).
- `pipeline/04_train_bilstm.py` — legacy defaults (`sequences/`, `POS_WEIGHT=1.44`); reproduces the
  **retracted** leaky-era AUC, not the published 0.932.

CLAUDE.md states this directly: *"use that engine for any new model family or clean-protocol
retraining, not the legacy trainers."* The IDD-PeD work obeys that.

---

## 2. What the canonical pipeline does, layer by layer

### 2.1 Annotation parsing
`pipeline/01_parse_annotations.py` flattens PIE's XML into `pie_annotations.pkl` — one row per
(pedestrian, frame) with `set_id, video_id, ped_id, frame, x1, y1, x2, y2, vehicle_speed,
crossing_label`. `crossing_point` is *not* in that pkl; the clean builder re-parses it straight
from `PIE/annotations_attributes/set*/*_attributes.xml`.

### 2.2 Sequence construction (the leak-free core)
`02_build_sequences_clean.py`, per pedestrian:
1. Take the **contiguous** frame segment containing `crossing_point` (drop earlier disjoint
   segments — affects ~4 % of pedestrians).
2. **Truncate at `crossing_point` inclusive.** Let `L` = truncated length.
3. Exclude if `L < obs_len + TTE_MIN`.
4. Slide `obs_len`-frame windows at stride `(1-overlap)*obs_len`, constrained so the last observed
   frame sits between `TTE_MIN` and `TTE_MAX` frames before `crossing_point`. If `L` doesn't reach
   `TTE_MAX`, start at frame 0 instead (the official PIE fallback).

Settings: `obs_len=16`, `tte_min=30`, `tte_max=60`, `overlap=0.5`.
Output: `X.npy (N,16,5) float32`, `y.npy (N,) int8`, `meta.pkl` = list of
`{set_id, video_id, ped_id, anchor_frame, crossing_point, tte}`.

**Why this is leak-free:** `crossing_point` is validated against the per-frame `cross` ground truth
(Issue 1) — it equals the first `cross=="crossing"` frame in 516/519 crossers (99.4 %) and is never
earlier than true onset. Truncating there and requiring ≥30 frames of lead time makes post-onset
contamination impossible **by construction**, not by filtering.

### 2.3 Feature contract (must match exactly wherever the model is used)
- Order: `[x1, y1, x2, y2, vehicle_speed]` — **raw PIE pixel coordinates** on a 1920×1080 image,
  *not* normalized to image size.
- `vehicle_speed` is PIE's per-frame OBD speed.
- Window is exactly **16** timesteps.
- Standardize with **per-feature z-score** `(x - mean) / (std + 1e-6)`, statistics computed on the
  **flattened training split only** (`Xtr.reshape(-1, 5)`), saved per run as
  `norm_mean.npy` / `norm_std.npy`.
- Decision threshold **0.5** on `sigmoid(logit)` by default; τ\* variants are tuned **on validation
  probabilities only** (`best_threshold` in `00_common.py`: argmax F1 over achievable cutoffs,
  tie-break higher acc then smaller |τ−0.5|, bounded [0.05, 0.95]).

### 2.4 Splits
Fixed **by recording set**, never random (prevents pedestrian leakage across splits):
`train = {set01, set02, set04}` (2178 windows) · `val = {set05, set06}` (634) ·
`test = {set03}` (2094, 681 positive = 32.5 %). Asserted in both `load_splits()` implementations.

### 2.5 Training protocol (FROZEN — identical for every family)
From `12_unified_engine.py`: `BCEWithLogitsLoss(pos_weight=1.682)` (= 1366 neg / 812 pos on the
clean train split) · batch 32, shuffle, `num_workers=0` · ≤100 epochs, early stop patience 15 on
**val AUC** · `ReduceLROnPlateau(mode=max, factor 0.5, patience 5)` on val AUC (or
`warmup_cosine` if the cfg asks) · checkpoint selection `select="f1"` = best val F1, tie-broken by
acc then AUC (`select="auc"` = the older frozen rule) · metrics from sklearn at threshold 0.5,
full-batch eval · seeds `[42, 0, 1, 2, 3]`.

`train_run()` **has no test code path at all** — test evaluation is delegated to exactly one
designated script per study, which touches test once. This is a deliberate discipline and the
IDD-PeD port reproduces it.

### 2.6 Seeding / reproducibility
One `set_seed()` seeds `random`, `numpy`, `torch`, `torch.cuda`, and sets the cuDNN determinism
flags unconditionally. Measured finding (Issue 12): **CPU training is bit-reproducible and
context-free; `nn.LSTM` on Apple MPS is process-history-dependent.** Therefore **recurrent families
must train on CPU** for exact reproduction. Transformer training is context-free on MPS.

### 2.7 Model families and their headline configs
All four are registered in `MODEL_REGISTRY` and take `input_dim` as a constructor argument
(the *engine's wrapper functions* hardcode `input_dim=5`, the classes do not — this matters for
any dimension change, see §4).

| Family | Class | Headline ("-F1") cfg | Params | Checkpoints |
|---|---|---|---|---|
| `bilstm` | `BiLSTMIntentPredictor` | `lr 1e-3, dropout 0.3, hidden 256, num_layers 2` | 2,237,313 | `f1_optimization/runs_f1/lstm_lr1e-03_do0.3_h256_nl2/pw1.682/seed{42,0,1,2,3}/` |
| `transformer` | `TransformerIntentPredictor` | `d_model 128, nhead 4, num_layers 4, dim_ff 512, dropout 0.1, pool last, pos sin, lr 1e-3, plateau, wd 1e-5, adam` | 794,241 | `f1_optimization/runs_f1/transformer_searched/pw1.682/seed*/` |
| `gru` | `RecurrentIntentPredictor("gru")` | `lr 5e-4, dropout 0.3, hidden 256, num_layers 2` | 1,678,209 | `gru/phase4_final/runs_final/gru_f1_winner/seed*/` |
| `birnn` | `RecurrentIntentPredictor("rnn", tanh)` | `lr 1e-4, dropout 0.2, hidden 256, num_layers 2` | 560,001 | `rnn/phase4_final/runs_final/rnn_f1_winner/seed*/` |

Each run dir holds `best.pt` (needs `weights_only=False` — it stores numpy-scalar `val_metrics`
beside the state_dict), `norm_mean.npy`, `norm_std.npy`, `final.json`, `history.json`.
**All 20 headline checkpoints (4 families × 5 seeds) are present locally**, with their PIE-train
normalization statistics — so Experiment A (zero-shot) needs no retraining.

### 2.8 Statistical evaluation
- **Multi-seed**: 5 seeds, report mean ± std ("per-seed-mean" = the paper number). A 5-seed
  **probability ensemble** is reported separately as one deployable predictor.
- **Bootstrap**: `bootstrap_ci` (percentile, B=10 000, `default_rng(42)`) and `paired_bootstrap`
  (same resample indices on both sides).
- **Pedestrian-cluster bootstrap** (`07_cluster_bootstrap.py`): resample **pedestrians** with
  replacement, each contributing all its windows — because the 2094 test windows come from only
  ~541 pedestrians at 50 % overlap, so window-level CIs are too narrow. This is the CI that the
  paper's claims are required to survive.
- **LOSO**: leave-one-set-out over the 6 PIE recording sets (`journal_prep/issue5_loso_cv/`,
  `*/phase4_final/06_*_loso.py`).
- **Parity gate**: before any new number is emitted, the frozen BiLSTM's per-seed test AUC is
  regenerated from checkpoints and asserted equal to the stored `final.json` values (`|Δ| < 1e-4`).

### 2.9 The published PIE findings that IDD-PeD must test
1. **Ego-speed dominance** — 5-D (bbox+speed) F1 0.828 vs bbox-only 4-D F1 0.551 on the same
   protocol; ego-speed is worth ~+0.18 AUC. *The single most important claim to re-test.*
2. **Architecture/gating irrelevance** — BiLSTM ≈ Transformer ≈ GRU ≈ vanilla RNN on F1
   (0.844 / 0.847 / 0.849 / 0.852), all CIs overlapping.
3. **Temporal-validity methodology** — the leakage audit + event-anchored clean protocol
   (0 % verified leakage vs 67.9 % under the naive anchor).
4. **Horizon-bounded equivalence** — F1 declines with observation window for all families;
   at OW 64 the un-gated RNN alone falls behind.

---

## 3. The existing cross-dataset precedent (JAAD) — and why IDD-PeD matters more

`journal_prep/cross_dataset_validation/` already ran **Track A on JAAD**: the clean protocol was
ported, the leakage audit re-run (**0 % leakage on 972 sequences** — the anchor generalizes), and
all four families retrained on JAAD.

**But JAAD has no ego-vehicle speed** (only 5 coarse driver-motion states), so it ran **bbox-only
(4-D)**. Result (5 seeds, test):

| family | test F1 | test AUC | test Acc |
|---|---|---|---|
| bilstm | 0.720 ± 0.035 | 0.510 ± 0.020 | 0.605 |
| transformer | 0.740 ± 0.040 | 0.520 ± 0.023 | 0.619 |
| gru | 0.707 ± 0.018 | 0.502 ± 0.022 | 0.584 |
| birnn | 0.729 ± 0.029 | 0.493 ± 0.013 | 0.608 |

**AUC ≈ 0.50 — chance.** The F1 numbers are an artifact of JAAD's high positive base rate, not
discrimination. So the JAAD track supports finding (2) and (3) but is, on its own, a weak
generalization result: it could not test finding (1) at all.

**This is exactly the gap IDD-PeD closes.** IDD-PeD ships per-frame OBD speed, so it is the first
dataset on which the project's *actual* 5-D input contract can be tested out-of-domain.

> Note: `journal_prep/cross_dataset_validation/PLAN.md` (written 2026-07-21) listed IDD-PeD as
> "access-gated forms — not now". **That assessment is now outdated** — the annotations are
> served as direct, unauthenticated CC BY 4.0 downloads from CVIT. See
> `reports/IDD_PeD_schema_audit.md`. The old PLAN.md is left untouched.

---

## 4. What must be reproduced for IDD-PeD, and what cannot transfer

### Reproduce exactly (or the comparison is not fair)
- Feature semantics `[x1, y1, x2, y2, ego_speed]`, raw pixel coordinates, `obs_len=16`.
- Event-anchored clean protocol: truncate at `crossing_point`, TTE ∈ [30, 60], 50 % overlap.
- Train-only z-score; test touched once; F1-first selection; seeds `[42, 0, 1, 2, 3]`.
- The frozen training loop — call `train_run()` from the unified engine **unmodified**.
- Pedestrian-cluster bootstrap for all CIs; multi-seed mean ± std, never best-seed.
- Recurrent families on CPU.

### Cannot transfer directly (dataset differences — each needs an explicit, documented decision)
| Issue | PIE | IDD-PeD | Resolution |
|---|---|---|---|
| **Image resolution** | fixed 1920×1080 | per-video `width`/`height` from the XML `<meta>` block | must be read per video; for **zero-shot** the boxes must be rescaled to PIE's 1920×1080 coordinate frame, since PIE norm stats are in raw pixels |
| **Ego-speed rate** | per-frame OBD | OBD logged at **10 Hz**, video at 30 fps → speed is step-held across ~3 frames | audit the hold pattern; document as a sampling-rate difference, do not interpolate silently |
| **Speed units** | PIE OBD speed | IDD-PeD `OBD_speed` (paper reports a median of 30 km/h) | unit reconciliation must be verified empirically and disclosed; a unit mismatch would invalidate zero-shot transfer |
| **Splits** | 6 recording sets, train/val/test = {01,02,04}/{05,06}/{03} | official train {0001,0002,0004,0006,0007} / test {0003,0005,0008,0009}; **no official val split** | keep the official train/test; carve a val split out of *train* sets only, at set granularity (never a random window split) |
| **`pos_weight`** | 1.682 (clean train) | must be recomputed from the IDD-PeD train split | recompute; it is a dataset property, not a tuned hyperparameter |
| **Frame rate** | 30 fps | GoPro 30 fps; the paper also mentions a DDPAI 25 fps camera | **only `gopro` sets are released** — verify, and exclude any non-30 fps source rather than resampling |
| **Per-frame crossing ground truth** | `cross` attribute (recovered in Issue 1) | `CrossingBehavior` ∈ {CU, CFU, CD, CFD, CI, N/A} | maps to a frame-level crossing state for the independent leakage audit; the CI ("crossing, but not in the ego path") class has no PIE analogue and needs an explicit decision |
| **Label semantics** | `crossing` binary | `crossing` {no:0, yes:1} = "seen crossing **in front of the ego-vehicle**" | close but not identical — must be stated in the paper |
| **Detector-in-the-loop** | YOLO26+ByteTrack on PIE clips (Issue 10) | would need the ~100 h video tars (not downloaded) | see Phase 9 decision in the final report |

### One engine-level incompatibility (handled without editing anything)
`12_unified_engine.py` hardcodes `input_dim=5` **inside its four builder wrapper functions**, and
`load_splits()` asserts the PIE shape `(4906, 16, 5)` and the PIE split sizes. The underlying model
classes accept `input_dim` freely. The JAAD track solved this by `importlib`-loading the engine
read-only and monkey-patching `MODEL_REGISTRY` in memory, calling `train_run()` itself unmodified —
**zero bytes changed on disk outside the isolated folder**. The IDD-PeD port uses the same
technique. For the main 5-D experiment no dimension patch is even needed; only the data loader is
replaced.

---

## 5. Environment (Phase 13 inspection)

| | |
|---|---|
| Machine | MacBook Air, **Apple M4**, 10 cores (4 performance + 6 efficiency) |
| RAM | 16 GB |
| Free disk | 52 GB |
| Python | 3.13.5 (`.venv`) |
| PyTorch | 2.12.0 · MPS **available** · CUDA unavailable |
| numpy / sklearn / scipy | 2.4.6 / 1.9.0 / 1.17.1 |

**Runtime estimate.** A single PIE run on this protocol takes 19–23 s (from the stored
`final.json` `seconds` fields, MPS) and roughly 1–2 min on CPU. IDD-PeD's usable window count is
expected to be of the same order as PIE's. Experiment B is 4 families × 5 seeds = 20 runs ≈
**20–40 min on CPU**; Experiment A (zero-shot) is inference only, seconds. **Local execution on
the M4 is comfortably feasible — Kaggle is not required.** A Kaggle package is still prepared as
a fallback per the brief, but is not expected to be needed.

---

## 6. Files touched by this work

**Created:** everything under `idd_ped_crossdataset/` (new top-level folder).
**Read only, never modified:** every path named in §1 and §2.7.
No file outside `idd_ped_crossdataset/` is written by any script in this experiment.

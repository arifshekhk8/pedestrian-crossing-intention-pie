# ProjectDescription.md — the single reference for writing the journal paper

> **Purpose.** This file is the one place to (re)learn the *entire* project before or
> while drafting the MDPI MTI manuscript. It consolidates everything from the code, the
> three authoritative pipeline docs, the 12-issue journal-readiness program, and the
> four model-family studies into one narrative with **every real number in place**.
> When drafting, read this instead of hunting across 80 files — but every claim here
> also cites its source folder so you can verify. **Golden rule (unchanged): every
> number in the paper must trace to a run output in `journal_prep/`, `transformer/`,
> `f1_optimization/`, `gru/`, or `rnn/`. Never invent a result or a citation.**
>
> Last synchronized with the repo: 2026-07-21. Numbers below match
> `journal_prep/Analysis/model_comparison.md`, `f1_optimization/README.md`,
> `journal_prep/issue3_baseline_comparison/03_baseline_comparison.md`, and the per-issue
> DONE blocks in `journal_prep/PLAN.md`.

---

> **▶ 2026-07-21 SESSION UPDATE (read alongside this file).** Since this reference was
> written: **(1)** MTI is locked and **F1-first** is confirmed as the reporting hierarchy.
> **(2)** The paper's **novelty was reframed** — parsimony ("2 streams is enough") is
> *conceded* to prior work (PedCMT, IEEE T-ITS 2024; GTransPDM-w/o-pose; Achaji 2022), and
> our headline is now the **leakage audit + statistical rigor + the four-family "input, not
> architecture, decides" isolation**. The honest competitive analysis + all recent-related-
> work links are in **`relatedwork.md`** (esp. §7 defensibility). **(3)** Cross-dataset
> validation is **paused-then-repivoted**: PSI blocked, **PePScenes confirmed dead**, so the
> plan is **JAAD** (methodology-only; no ego-speed) primary + **nuScenes** (self-derived
> labels) stretch — see `../../journal_prep/cross_dataset_validation/PLAN.md`; it is written
> as *future work* in the paper, not claimed. **(4)** The **full first draft** is written in
> the MDPI template: **`MDPI_Article_Template/main.tex`** (+ `references.bib`, compiled
> `main.pdf`), title *"Two Streams, Four Architectures…"*. The numbers/§13–14 tables below
> are exactly what feed that draft.

## 0. TL;DR — the paper in six sentences

1. We predict **pedestrian crossing intention** on the **PIE dataset** from a 16-frame
   (0.5 s) window of two cheap streams — the pedestrian's **bounding box** and the
   **ego-vehicle speed** — and output the probability they are about to cross.
2. We first found and fixed a **temporal-leakage** flaw in the common PIE extraction
   protocol (67.9% of crossers were already mid-crossing inside the observation
   window); re-anchoring windows at PIE's own `crossing_point` gives a **0%-leakage**
   dataset (N grows 1,389 → 4,906).
3. On that leakage-free protocol, two-stream models reach **F1 0.844–0.852** — within
   0.02–0.03 of the multimodal F1 ceiling (PedFormer 0.87) — while holding the
   **highest AUC in the standard-protocol table (0.94–0.95)**, at **0.32–0.72
   ms/window**, using **2 input streams vs the 3–7 of every method above them**.
4. **Ego-vehicle speed is the dominant predictor**: remove it and AUC collapses
   0.932 → 0.753.
5. We settled "is a fancier temporal model better?" empirically across **four model
   families** given the identical search + optimization: a searched **Transformer**
   beats the BiLSTM *on AUC* but ties it on F1; the **GRU** and even the un-gated
   **vanilla RNN** tie the BiLSTM on both — so the temporal model, and even its gating,
   is secondary; **the input signal carries the task**.
6. Everything is shown with full rigor: bootstrap + pedestrian-cluster CIs, LOSO across
   all 6 sets, multi-seed ablations, a documented hyperparameter search, and
   detector-in-the-loop (YOLO26-M + ByteTrack) realism.

**Metric hierarchy (supervisor directive, 2026-07-12): F1 → accuracy → AUC.** Report
F1-first; AUC (and PR-AUC) are the threshold-free corroboration where our models lead.

**Target venue:** MDPI **MTI** (Multimodal Technologies and Interaction), open access,
Overleaf + official MDPI LaTeX template, numbered BibTeX references (`mdpi.bst`).

---

## 1. The paper's spine (one-sentence thesis)

> On a **leakage-free, canonical PIE protocol**, two-stream (bounding-box motion +
> ego-vehicle speed) models reach **F1 0.844–0.847** — within 0.02–0.03 of the
> multimodal F1 ceiling — while holding the **highest AUC in the standard-protocol
> table (0.94–0.95)** at sub-millisecond latency, i.e. competitive with 3–7-stream
> multimodal SOTA at a fraction of the cost, shown with full statistical rigor and
> detector-in-the-loop realism. A BiLSTM, a staged-search Transformer, a GRU, and an
> un-gated vanilla RNN over the identical input **tie on F1**, so the parsimony result
> is about the input signal, not the architecture.

### Three contributions to hammer throughout
1. **A temporal-leakage audit + fix** — most PIE papers never check this. (Issues 1–2)
2. **Parsimony** — 2 streams ≈ top AUC and near-ceiling F1, with measured real-time
   latency. (Issues 3, 9; the four-family studies)
3. **Rigor** — bootstrap + cluster CIs, LOSO, multi-seed ablations, a documented HP
   search, detector-in-the-loop. (Issues 4, 5, 6, 7, 8, 10)

---

## 2. Repository map (where every number lives)

Seven top-level folders. **Run scripts from the repo root** so relative paths resolve.

| Path | Contents |
|---|---|
| `pipeline/` | the numbered legacy scripts (`01_`…`10_`), the three authoritative docs (`THESIS_PLAN.md`, `PROGRESS_LOG.md`, `CODE_STATE.md`), the leaky-era multiseed tables, and the live-demo outputs (`demo_out/`). ⚠ `04_train_bilstm.py` keeps LEGACY leaky defaults as a historical artifact. |
| `journal_prep/` | the 12-issue journal-readiness program (one folder per issue). `issue12_unified_pipeline/12_unified_engine.py` is **THE training engine** for every family. `Analysis/` is the consolidated cross-model pack. `obs_window_extension/` + `psi_crosstest/` are the newest extensions. |
| `transformer/` | the Transformer-vs-BiLSTM extension (phases 1–5). Beats BiLSTM on AUC, ties on F1. |
| `f1_optimization/` | the F1-first optimization program for both families (metric hierarchy F1 → acc → AUC). |
| `gru/` | the GRU-vs-BiLSTM recurrent-cell study (phases G1–G5). GRU ties BiLSTM. |
| `rnn/` | the vanilla-RNN gating-isolation study (phases R1–R6). Un-gated RNN ties LSTM/GRU. |
| `paper_and_artifacts/` | `Journal_writing/` (this manuscript workspace), `runs/` (trained checkpoints + norm stats), `supervisor_review/` (⚠ dated pre-leakage-fix presentation pack — do not cite its numbers). |

**Gitignored data (not tracked, at repo root):** `PIE/`, `PIE_clips/`, `PIEPredict/`,
`sequences/`, `pie_annotations.pkl`, `yolo26m.pt`, `.venv/`.

**Authoritative hand-maintained docs (read for provenance):**
`pipeline/THESIS_PLAN.md` (locked architecture + plan), `pipeline/PROGRESS_LOG.md`
(chronological results log), `pipeline/CODE_STATE.md` (per-file status),
`journal_prep/PLAN.md` (the 12 issues with DONE blocks), `journal_prep/Analysis/`
(the consolidated four-family pack — the best single source for final tables).

---

## 3. The dataset — PIE

**PIE (Pedestrian Intention Estimation)**, Rasouli et al., ICCV 2019. On-board
car-camera video (1920×1080, 30 fps), six recording sets (set01–06), with per-frame
pedestrian bounding boxes, per-pedestrian crossing labels, per-pedestrian
`crossing_point` event frames, and per-frame vehicle OBD speed.

- **Parsed** (`pipeline/01_parse_annotations.py` → `pie_annotations.pkl`): 582,376
  frame-level rows, 1,374 unique pedestrians (after dropping `crossing_label == -1`).
  Columns: `set_id, video_id, ped_id, frame, x1, y1, x2, y2, vehicle_speed, action,
  crossing_label`. No missing `vehicle_speed`.
- **Critical parser discovery (Issue 1):** PIE boxes also carry a per-frame `cross`
  attribute (`not-crossing`/`crossing`/`crossing-irrelevant`). The original parser kept
  only `action` and **dropped `cross`** — the exact "is crossing now" signal. It was
  re-parsed for the leakage audit (`issue1_leakage_audit/cross_state_map.pkl`).

### Fixed split by recording set (no random split — prevents identity/scene leakage)
- **Train:** set01, set02, set04
- **Val:** set05, set06
- **Test:** set03
- Because recording sets partition the videos, no pedestrian/video/scene appears in more
  than one split. This is the standard PIE benchmark split (PCPA/GTransPDM/PedFormer).

### Two datasets in the repo (do not confuse them)
| | leaky `sequences/` (retracted) | **clean `sequences_clean/`** (journal-bound) |
|---|---|---|
| Anchor | last annotated frame − TTE | **PIE `crossing_point` event, TTE ∈ [30,60]** |
| Windows/ped | one per contiguous segment | **50%-overlap sliding (stride 8)** |
| N total | 1,389 | **4,906** |
| Split N | 616 / 186 / 587 | **train 2,178 / val 634 / test 2,094** |
| Positive rate | 41.0% | **33.6% overall (train 37.3% / val 24.4% / test 32.5%)** |
| Window leakage | **387/570 crossers (67.9%)** | **0 / 4,906 (0.000%)** |
| `pos_weight` | 1.44 (819/570) | **1.682 (1366/812, train-only)** |
| Path | repo-root `sequences/` | `journal_prep/issue2_clean_protocol/sequences_clean/` |

Everything journal-bound uses the **clean** dataset. The leaky one exists only as the
historical record that produced the retracted 0.931.

---

## 4. The distinctive contribution — temporal leakage (Issues 1 & 2)

This is the finding most PIE papers never check, and the paper's methodological
centerpiece.

### 4.1 The leak (Issue 1 — `journal_prep/issue1_leakage_audit/`)
The old `02_build_sequences.py` anchored each 16-frame window at
`last_annotated_frame − TTE`, with no guarantee the window ends **before** crossing
onset. Audited against the recovered per-frame `cross` ground truth:

- **387/570 crossers (67.9%)** have ≥1 frame *already crossing* inside the window → leak.
- **369/570 (64.7%)** have **all 16 frames** already crossing → pure detection, not
  prediction, for ~2/3 of positives.
- Only **183/570 (32.1%)** have a genuinely clean window.
- Non-crossers: 0% leakage (correct by definition).
- **Static geometry shortcut too:** the anchor-frame bbox alone separates the classes —
  rank-biserial bbox_area **+0.65**, height **+0.63**, bottom-y **+0.49** (all large).
  Crossers are close/large/low-in-frame; the model could partly cheat on screen
  position with no temporal reasoning.

This explains the suspiciously fast **epoch-3** convergence and inflated **AUC 0.931**.

### 4.2 The fix (Issue 2 — `journal_prep/issue2_clean_protocol/`)
PIE's own `*_attributes.xml` carries a `crossing_point` frame for **every** pedestrian
(crossers and non-crossers). Validated: `crossing_point` equals the first
`cross=="crossing"` frame in **516/519 crossers (99.4%)** and is never earlier — so
truncating each track at `crossing_point` and requiring ≥30 frames of look-ahead is
**leak-free by construction**. This is exactly PIE's benchmark anchor
(`extract_tracks_tte`), not a thesis invention.

**Algorithm** (`02_build_sequences_clean.py`, mirrors the official code): per pedestrian,
take the contiguous segment containing `crossing_point` (drop earlier disjoint segments
on the ~4% of tracks with a gap), truncate at `crossing_point` inclusive (length `L`),
exclude if `L < obs_len + TTE_min`, else slide `obs_len=16` windows at 50% overlap
(stride 8) with the last observed frame `TTE ∈ [30,60]` frames before the crossing point
(fall back to sampling from frame 0 if the track is shorter than `TTE_max`).
**107/1,374 pedestrians (7.8%) excluded** as too short.

**Verification:** re-running the Issue-1 audit on `sequences_clean/` confirms **0/4,906
windows leak**. The static shortcut also collapses (bbox_area rank-biserial +0.65 →
+0.25, height +0.63 → +0.21, bottom-y +0.49 → +0.09 n.s.; all |r| < 0.3) — proving part
of the old shortcut was itself a side-effect of observing in-progress crossings.

**The honest headline:** on a single seed, AUC drops only **0.931 → 0.913** once leakage
is fully removed and N grows 3.5×, and convergence moves from a suspicious epoch 3 to a
believable **epoch 17**. So the leak fix is a **methodological win** (genuine pre-onset
prediction), *not* a deflated number — the 5-seed clean baseline is **0.932 ± 0.011**
(seed 42's 0.913 is the low end of the five).

**Eval-parity stress test** (`03_eval_parity_report.md`) — because 2 streams *beating*
multimodal baselines invites the "easier eval" reflex:
| check | AUC | meaning |
|---|---|---|
| per-window (headline) | 0.9131 | the `final.json` number |
| per-**pedestrian** (541 peds, mean prob) | 0.9143 | overlap does **not** inflate it (+0.0012) |
| benchmark-filter subset (track ≥76) | 0.9194 | our laxer 46-frame floor doesn't help |
| short tracks only (46–75) | 0.8634 | the extra tracks we admit are *harder*, not easier |

---

## 5. Features, preprocessing, and the inference contract

**Feature vector (5-D):** `[x1, y1, x2, y2, vehicle_speed]` — pedestrian bounding box +
ego OBD speed.

**The inference contract (get any of these wrong and results silently break):**
- Coordinates are **raw PIE pixels (1920×1080), NOT normalized to image size.**
- Standardize with per-feature **train-only** z-score: `(x − mean) / std`, using the
  `norm_mean.npy` / `norm_std.npy` saved in each run dir. (Train-only = no val/test
  statistics leak into training.)
- Observation window is exactly **obs_len = 16** timesteps.
- Decision threshold **0.5** on `sigmoid(logit)` for all cross-paper-comparable numbers.
- Checkpoints load with **`torch.load(..., weights_only=False)`** (best.pt stores
  numpy-scalar `val_metrics` next to the state dict; default True crashes on torch ≥2.6).
- Module imports use **importlib** because filenames start with digits (e.g.
  `import_module("03_bilstm_model").BiLSTMIntentPredictor`).

**Bbox-only (4-D) ablation** drops `vehicle_speed` → input_dim 4. This is the ego-speed
ablation, not a normal model.

---

## 6. Model architectures (four families + variants)

**All crossing-intention models are custom PyTorch architectures trained from scratch on
PIE — none are pretrained, no transfer learning.** (The only pretrained component
anywhere is YOLO26 in the live demo's perception front-end, a separate detector.) Each
model is the same small wrapper — `Linear(→proj)+ReLU → temporal model → last-step/pooled
read-out → Linear(→1)` — over a standard PyTorch block; only the temporal model changes,
which is precisely what lets each comparison isolate one design choice.

### 6.1 BiLSTM (the locked baseline) — `pipeline/03_bilstm_model.py`
`Linear(5→64) + ReLU` → **2-layer bidirectional LSTM, hidden 128, dropout 0.3** →
`Linear(256→1)` on the **last timestep**. **594,561 params.** Raw logit out.
Cite: LSTM (Hochreiter & Schmidhuber 1997), BiRNN (Schuster & Paliwal 1997).

### 6.2 Transformer — `transformer/phase1_setup/00_transformer_model.py`
Small **pre-LN Transformer encoder**: `Linear(5→d_model)` → positional encoding →
`nn.TransformerEncoder` (GELU, `norm_first=True`) → pool → `Linear(d_model→1)`.
Config knobs: `d_model, nhead, num_layers, dim_ff, dropout, pool∈{cls,mean,last},
pos∈{learned,sin}`. Cite: Vaswani 2017 (attention), Xiong 2020 (pre-LN).
- **Searched winner:** d128 / ff512 / **4 layers** / **last-token pool** / **sinusoidal
  PE**, dropout 0.1. **794,241 params.**
- **Default (un-searched):** d128 / ff256 / 2 layers / CLS / learned-PE. 268,417 params.

### 6.3 GRU & vanilla RNN — `RecurrentIntentPredictor` in `12_unified_engine.py`
Exact twins of the BiLSTM wrapper with only the cell swapped:
- **GRU** (`nn.LSTM` → `nn.GRU`): the BiLSTM's *gated recurrent twin*. Isolates **gated
  cell type**. Cite Cho 2014, Chung 2014.
- **Vanilla RNN** (`nn.LSTM` → `nn.RNN`, tanh): the BiLSTM's twin with **gating
  removed**. Isolates **gating itself**. Cite Elman 1990, Rumelhart 1986.

### 6.4 BiLSTM + attention (variant) — `pipeline/07_bilstm_attention.py`
Same backbone, additive (Bahdanau) temporal attention replaces last-step pooling.
611,265 params. Result: no benefit on clean data.

---

## 7. The training protocol (frozen, identical for all families)

Everything routes through the one model-agnostic engine
`journal_prep/issue12_unified_pipeline/12_unified_engine.py` (Issue 12) — this is what
makes the cross-family comparison fair by construction. Legacy trainers
(`pipeline/04_*`) are historical artifacts.

**FROZEN (identical for every family):**
- Data: `sequences_clean/` (N=4,906); splits train 2,178 / val 634 / test 2,094.
- Loss: `BCEWithLogitsLoss(pos_weight=1.682)` (train-only class ratio 1366/812; applied
  in the training gradient only — eval uses plain threshold 0.5).
- Batch 32, ≤100 epochs, **early stop patience 15 on val AUC**, `ReduceLROnPlateau(max,
  factor 0.5, patience 5)` on val AUC.
- Optimizer Adam, lr per-config, weight_decay 1e-5 (a few transformer stages used AdamW).
- Train-only per-feature z-score normalization, saved per run.
- **5 seeds [42, 0, 1, 2, 3]**; **test set03 touched exactly once** by a designated final
  script — `train_run` has no test code path at all.
- **Checkpoint selection rule:** `select="auc"` = best val AUC (the original frozen
  rule); `select="f1"` = best val F1 (tie: acc, then AUC) = the F1-first rule. Both
  record `val_at_auc_best` for the G1 counterfactual audit.
- Metrics: sklearn f1/acc/auc/pr_auc/prec/rec at threshold 0.5.

**Overfitting control (5 mechanisms):** early stopping, best-val-checkpoint (never last
epoch), LR decay, dropout, weight decay. **Leakage control (3 layers):** set-level splits,
train-only normalization, event-anchored windows (0% verified). See
`journal_prep/Analysis/documentation.md` for the full reviewer Q&A.

**Command (the journal-bound training path):**
```bash
python journal_prep/issue12_unified_pipeline/12_unified_engine.py \
    --family bilstm --seed 42 --device cpu --select f1 --out_dir <run_dir>
# families: bilstm | transformer | gru | birnn ; --select auc = the frozen legacy rule
```

---

## 8. The 12-issue journal-readiness program (`journal_prep/`)

Each issue fixes a specific reviewer objection. All done. Numbers below are final.

| # | Issue | Headline result |
|---|---|---|
| **1** | Temporal-leakage audit | 🔴 **leakage found:** 67.9% of crossers observed mid-crossing → old AUC 0.931 inflated. |
| **2** | Clean protocol + retrain | clean AUC **0.932 ± 0.011** (5-seed); **ego-speed dominant (+0.18)** — bbox-only collapses 0.932 → **0.753**; attention no benefit (0.925). |
| **3** | Published-baseline comparison | ours = **top AUC (0.94–0.95), near-ceiling F1 (0.84–0.85), 2 streams, sub-ms latency**, mid Acc. Positioning matrix carries a measured number in every cell. (§13 here) |
| **4** | Bootstrap CIs | AUC 0.932, **95% CI ≈ [0.92, 0.95]** (10k window bootstrap), PR-AUC 0.876; pedestrian-cluster CI [0.92, 0.96]. Ego-speed gap statistically unambiguous. |
| **5** | Leave-one-set-out CV | **6-fold AUC 0.928 ± 0.041**; **set03 fold 0.931 ≈ fixed split** → set03 is representative, not an easy fold. (excl. tiny set05: 0.915 ± 0.029) |
| **6** | Window + TTE ablation (multi-seed) | **window insensitive** (obs 8/16/30 → 0.931/0.933/0.937, spread 0.006 < seed noise 0.007); **TTE declines significantly** (30/45/60 → 0.960/0.948/0.919, every pairwise p ≤ 0.008), **confirmed on a matched cohort** (0.961/0.946/0.919, sample effect ≤ 0.002). Overturns the old leaky "insensitive to TTE." |
| **7** | Hidden-size + depth | hidden 64/128/256 → 0.927/0.933/0.938 (**128 justified** — 256 n.s. p=0.34 at 3.8× params); depth 1/2/3 → 0.930/0.932/0.931 (insensitive). Model is small-data-limited, not capacity-limited. **Metric-conditional:** under F1-first, h256 *is* the selected config (see §10). |
| **8** | Hyperparameter grid search | full **36-config grid**, **val-only selection + test touched once**. Winner lr1e-4/do0.2/h256/2L: val 0.969 ± 0.006, test **0.930 ± 0.005**; baseline test 0.929 ± 0.012 → Δ **+0.0006, p=0.914 (n.s.)** → **the search confirms the hand-set baseline**. Hyperparameters now documented, not asserted. |
| **9** | Isolated latency | **BiLSTM 0.575 ms/window** (CPU, batch 1, ~58× inside a 30 fps budget); CPU beats MPS at batch 1 (GPU dispatch overhead). Pipeline **detection-bound:** YOLO26-M 33.7 ms (93%) + ByteTrack ~1 ms + BiLSTM 1.6 ms (4.5%) → **27.5 fps**. |
| **10** | GT-box vs YOLO-box | **prediction robust to detector box noise:** GT 0.962/0.958 vs YOLO 0.953/0.948 (drop **+0.009/+0.010**, 3% decision flips, 98 peds / 311 windows). Weak links are **perception**: detector recall **88%**, ByteTrack **track purity 39%** (severe identity fragmentation). |
| **11** | Doc cleanup | root docs reconciled with reality (file numbering, deferred-then-done hidden-size, softened demo "AUC 1.000 N=10"). |
| **12** | Unified engine + F1-first integration | **ONE model-agnostic engine** (bilstm/transformer/gru/birnn), equivalence gates ALL PASS, single-device CPU replication of the F1-first verdicts. Removes the "trained by different code on different hardware" confound. |

**Key invariants preserved across all issues:** test touched once per experiment;
train-only normalization; `pos_weight=1.682` fixed unless it's the ablated factor;
`set_seed(42)` canonical; `weights_only=False`; threshold 0.5 for comparable numbers.

---

## 9. Extension 1 — Transformer vs BiLSTM (`transformer/`)

Answers the reviewer question *"why not a Transformer?"* with a measurement, not a
small-data argument. **Supervisor-requested.**

**What was frozen identical:** data, splits, `pos_weight=1.682`, 5 seeds, threshold 0.5,
early-stop rule; the BiLSTM's checkpoints were never retouched. **Only the Transformer's
own architecture + recipe were searched.**

**The search (78 distinct configs, >2× the BiLSTM's Issue-8 budget), val-only, test
touched once:** Stage A (36 architectures: width × depth × pooling × PE) → Stage B (36
recipes) → transfer check (6) → Stage C (top-5 + a pre-registered default, ×5 seeds;
winner = best **5-seed mean**, not best single seed).

**Result (5-seed test set03):**
| model | params | AUC | Acc | F1 |
|---|---|---|---|---|
| BiLSTM (frozen baseline) | 594,561 | 0.932 ± 0.011 | 0.883 | 0.828 |
| Transformer, default (zero search) | 268,417 | 0.934 ± 0.006 | 0.878 | 0.816 |
| **Transformer, searched (winner)** | 794,241 | **0.950 ± 0.003** | 0.894 | 0.845 |

- **The win:** 10k **paired** bootstrap on the identical 2,094 windows → **ΔAUC =
  +0.0135, 95% CI [+0.0097, +0.0174]** (excludes 0); paired t-test over 5 seeds p=0.025
  (stated as low-power; the bootstrap is primary).
- **The nuance that makes it credible:** the *un-searched* Transformer (same family, the
  BiLSTM's own recipe, zero tuning) **ties** the BiLSTM (Δ=+0.0005, CI [−0.0034,
  +0.0043], p=0.83). **Attention does not trivially beat recurrence on this 16×5 signal
  — the deliberate search is what found the improvement** (deeper: 4 vs 2 layers;
  last-token pooling; sinusoidal PE).
- **LOSO:** 0.939 ± 0.044 vs BiLSTM 0.928 ± 0.041 (directional, 6 folds too few to test).
- **Latency:** the Transformer is *faster* per window (0.459 vs 0.575 ms CPU batch-1)
  despite 1.3× the params — parallel self-attention over 16 tokens beats sequential
  recurrence here.
- **Metric scope:** this WIN is **AUC-specific**; under F1-first the families TIE (§10).

Full verdict: `transformer/phase5_analysis/05_comparison_report.md`,
`transformer/SUPERVISOR_SUMMARY.md`, `transformer/PLAN.md`.

---

## 10. Extension 2 — F1-first optimization (`f1_optimization/`)

**Supervisor's metric hierarchy F1 → acc → AUC**, applied symmetrically to both families
under the same val-only / test-once discipline.

**Pre-registered levers:** (1) **val-tuned decision threshold τ\*** (argmax val F1, tie
acc then |τ−0.5|; never test); (2) **config re-selection** from each family's own search
by 5-seed mean val F1 (LSTM: h128 → **h256**; the transformer's searched winner was
already F1-optimal, rank 1/78 on both metrics); (3) **hybrid F1 checkpointing** (stop/
schedule on val AUC, checkpoint = best val F1); (4) **symmetric pos_weight sweep**
{1.0…2.5}, val-selected (LSTM kept 1.682; the transformer's marginal val pick of 2.5 did
not transfer to test — reported plainly).

**Result (test set03, per-seed mean; τ\* landed ≈0.5 so @0.5 is table-comparable):**
| model | test F1 | ens F1 | Acc | AUC |
|---|---|---|---|---|
| BiLSTM frozen @0.5 (the old 0.828) | 0.8275 ± 0.0123 | 0.8370 | 0.883 | 0.9423 |
| **BiLSTM-F1 (h256, F1-ckpt, τ\*)** | **0.8444 ± 0.0078** | **0.8557** | **0.897** | 0.9467 |
| Transformer frozen @0.5 | 0.8446 ± 0.0129 | 0.8490 | 0.894 | 0.9558 |
| **Transformer-F1 (pw2.5, F1-ckpt, τ\*)** | **0.8470 ± 0.0178** | **0.8565** | 0.896 | 0.9550 |

**Three verdicts (10k paired bootstrap, ensemble vectors):**
1. **BiLSTM IMPROVED on F1:** ΔF1 **+0.0187, CI [+0.0073, +0.0300]** (also survives the
   stricter pedestrian-cluster resampling: [+0.004, +0.035]).
2. **Transformer NO significant change** (ΔF1 +0.0075, CI [−0.0021, +0.0173]) — already
   near its F1 ceiling.
3. **Families TIE on F1:** ΔF1 **+0.0008, CI [−0.0124, +0.0142]**, p=0.762. **The
   Transformer's AUC win does not carry to the primary metric.**

Deployable 5-seed probability ensembles reach **F1 0.856 / 0.857** (a *different
statistic* from per-seed means — always labeled, never mixed into the comparison table).
All replicated end-to-end under the unified CPU engine (Issue 12), removing the
engine/device confound; pedestrian-cluster bootstrap confirms every CI
(`f1_optimization/07_cluster_bootstrap.md`).

---

## 11. Extensions 3 & 4 — the recurrent-cell isolation (`gru/`, `rnn/`)

**Third supervisor directive: two more model families on the same pipeline.** Each got
the identical Issue-8 search + F1-first optimization on the unified CPU engine, with the
frozen BiLSTM parity-gated (|Δ| = 0.00 across all 5 seeds).

### GRU (`gru/`) — isolates the gated cell type
GRU-F1 (h256, 1,678,209 params): AUC 0.941, **F1 0.849**.
- **Ties BiLSTM-F1 on F1:** ΔF1 +0.0071, CI [−0.0043, +0.0187].
- **Ties frozen BiLSTM on AUC at matched size:** ΔAUC −0.0008, CI [−0.0039, +0.0021]
  (the 446k h128 GRU vs the 595k BiLSTM).
- **Loses to the searched Transformer on AUC:** ΔAUC −0.0070, CI [−0.0101, −0.0038].
- Verdict: **the gated cell type doesn't matter.** LOSO 0.946.

### Vanilla RNN (`rnn/`) — isolates gating itself (the sharpest test)
Vanilla RNN-F1 (h256, **560,001 params** — smaller than the BiLSTM): AUC 0.948 ± 0.002,
**F1 0.852 ± 0.012** (the highest per-seed F1 of any model, though statistically level).
- **Ties BiLSTM-F1 on F1:** ΔF1 +0.0033, CI [−0.0083, +0.0150].
- **Ties GRU on F1:** ΔF1 −0.0038, CI [−0.0117, +0.0039]. → **LSTM ≈ GRU ≈ vanilla RNN.**
- **Edges frozen BiLSTM on AUC at matched h128:** ΔAUC +0.0059, CI [+0.0032, +0.0088]
  (framed as "level-to-marginally-better" — no evidence gating helps).
- **Ties the searched Transformer on AUC:** ΔAUC −0.0013, CI [−0.0041, +0.0015]. The GRU
  *lost* this; the AUC-optimized RNN *reaches* ~0.95 — **direct confirmation the
  Transformer's AUC edge was its search, not attention.**
- **Search notes:** independently landed on h256 (`lr1e-4/do0.2/h256/2L`) — the BiLSTM's
  own Issue-8 AUC winner; **0 diverged runs** (vanishing-gradient risk mild over 16
  steps). Smallest + **fastest** family (0.316 ms/window CPU, ~105× inside 30 fps).
  LOSO 0.937.
- Verdict: **not even the LSTM's gating is what matters over a 16-step window — the input
  signal is.** All verdicts survive the pedestrian-cluster bootstrap.
- **Caveat:** horizon-bounded (see §12).

**The through-line the four families establish:** attention beats the BiLSTM on AUC
*only via its search*; the gated GRU and un-gated vanilla RNN both tie it — so the
temporal model, and even its gating, is secondary. **The two-stream input (bbox +
ego-speed) carries the task.** The bbox-only collapse (0.932 → 0.753) is the other half.

---

## 12. Extension 5 — observation-window sweep across all families

`journal_prep/obs_window_extension/` (supervisor directive 2026-07-19). Extends the
BiLSTM-only Issue-6 window sweep to the F1-optimized model of **all four families** at
OW **16 / 32 / 64** frames (0.53 / 1.07 / 2.13 s). ⚠ Not a matched cohort: longer windows
need longer pre-crossing tracks, so **test N shrinks 2,094 → 1,009 → 458** (val 634 → 302
→ 138). Read within a window and as a per-family *trend*, not across windows as absolutes.

**Findings:**
1. **Longer windows do not help — F1 declines monotonically for every family** (≈0.85 @16
   → ≈0.836 @32 → ≈0.815 @64). A 0.5 s window already carries the predictive dynamics;
   more history dilutes the signal. **This justifies the OW-16 design choice.**
2. **At OW 16 and 32 the four families still tie** (F1 spread ≤0.004 at OW 32, inside
   per-seed noise). The "cell type / gating doesn't matter" conclusion holds here.
3. **At OW 64 the un-gated vanilla RNN alone falls behind** (F1 0.802; drop from OW-16
   −0.050, ~2× the gated cells' −0.026/−0.027/−0.028; lowest AUC 0.929, Acc 0.857).
   Directional (per-seed CIs still overlap), matching theory — **the family equivalence
   is horizon-bounded: gating is redundant over 0.5 s but begins to re-earn its keep by
   ~2 s.** This is the exact experiment the RNN study pre-registered.

Table: `journal_prep/Analysis/model_comparison.md` (the "Observation-window extension"
section).

**Extension 6 (in progress, BLOCKED on data) — PSI 2.0 cross-dataset test**
(`journal_prep/obs_window_extension/psi_crosstest/`): train on full PIE, zero-shot test
on PSI 2.0 (2023) at OW 16/32/64, starting with the vanilla RNN-F1. PIE side done (15
models saved). **Blocked** — PSI Google Drive links are dead; needs the user to submit
the PSI access-request form and drop annotations into `psi_crosstest/PSI2.0/`. Honest
caveat for the paper: ego-speed is our dominant feature and PSI's speed is 1 Hz in a
different city/units, so a transfer drop may partly reflect speed domain-shift.

---

## 13. Consolidated results — every model, one table

Test = PIE **set03**, 2,094 windows (32.5% positive), obs_len 16, TTE∈[30,60], 2-stream
input unless noted. **Per-seed-mean ± std over 5 seeds** = the paper numbers. ⭐ = the
family headline. Source: `journal_prep/Analysis/model_comparison.md`.

| family | model | params | selection | Acc | AUC | **F1** |
|---|---|---|---|---|---|---|
| BiLSTM | ⭐ **BiLSTM (baseline)** | 594,561 | val AUC | 0.883 ± .009 | 0.932 ± .011 | **0.828 ± .012** |
| BiLSTM | BiLSTM **bbox-only (4-D)** | 594,497 | val AUC | 0.744 ± .007 | 0.753 ± .020 | **0.551 ± .028** |
| BiLSTM | BiLSTM + attention | 611,265 | val AUC | 0.879 ± .010 | 0.925 ± .010 | **0.821 ± .009** |
| BiLSTM | **BiLSTM-F1 (h256)** | 2,237,313 | val F1 | 0.897 ± .006 | 0.940 ± .004 | **0.844 ± .008** |
| Transformer | ⭐ **Transformer (searched)** | 794,241 | val AUC | 0.894 ± .009 | **0.950 ± .003** | **0.845 ± .013** |
| Transformer | Transformer (default) | 268,417 | val F1 | 0.878 ± .006 | 0.942 ± .004 | **0.821 ± .006** |
| Transformer | **Transformer-F1** | 794,241 | val F1 | 0.896 ± .011 | 0.947 ± .003 | **0.847 ± .017** |
| GRU | ⭐ **GRU-F1 (h256)** | 1,678,209 | val F1 | 0.901 ± .010 | 0.941 ± .007 | **0.849 ± .011** |
| GRU | GRU (default h128, F1) | 446,081 | val F1 | 0.898 ± .010 | 0.939 ± .007 | **0.844 ± .020** |
| GRU | GRU (default h128, AUC) | 446,081 | val AUC | 0.898 ± .007 | 0.933 ± .010 | **0.840 ± .012** |
| RNN | ⭐ **Vanilla RNN-F1 (h256)** | 560,001 | val F1 | 0.902 ± .008 | 0.948 ± .002 | **0.852 ± .012** |
| RNN | Vanilla RNN (h256, AUC) | 560,001 | val AUC | 0.910 ± .006 | 0.948 ± .006 | **0.845 ± .022** |
| RNN | Vanilla RNN (default h128, F1) | 149,121 | val F1 | 0.897 ± .007 | 0.942 ± .007 | **0.844 ± .013** |
| RNN | Vanilla RNN (default h128, AUC) | 149,121 | val AUC | 0.889 ± .010 | 0.942 ± .008 | **0.836 ± .021** |

(Ensemble @ τ metrics + full confusion cells and per-model hyperparameters are in
`journal_prep/Analysis/model_comparison.md` and `hyperparameters.md`.)

**Latency (M4 CPU batch-1, ms/window):** RNN **0.316** (fastest) · Transformer 0.459 ·
BiLSTM 0.575 · GRU 0.721 — all ~50–105× inside a 30 fps budget. Latency is not a
deployment discriminator; the live pipeline is detection-bound.

---

## 14. Baseline comparison & positioning vs prior work (Issue 3)

**Standard PIE protocol** = train set01/02/04 · val set05/06 · test set03; obs 16 frames
(0.5 s); TTE 30–60 frames (1–2 s); metrics Acc/AUC/F1. ✅ = verified first-hand against
the source (mostly GTransPDM Table I / PedFormer Table I, read directly).

| Method | Venue/Year | Acc | AUC | F1 | Modalities (streams) |
|---|---|---|---|---|---|
| PCPA | WACV 2021 | 0.87 | 0.86 | 0.77 | bbox+pose+context+speed (4) |
| Pedestrian Graph+ | 2022 | 0.89 ‡ | 0.90 ‡ | 0.81 ‡ | pose graph + ego (2–3) |
| IntFormer | 2021 | 0.89 | 0.92 | 0.81 | multimodal |
| PIT | 2023 | 0.91 | 0.92 | 0.82 | multimodal |
| BiPed | 2023 | 0.91 ‡ | 0.90 ‡ | 0.85 ‡ | multimodal |
| **PedFormer** | 2023 | **0.93** | 0.90 | **0.87** | multimodal multitask — **F1/Acc ceiling** |
| GTransPDM | arXiv 2024 | 0.90 | 0.87 | 0.82 | bbox+pose+ego (3) |
| GTransPDM (w/o pose) | arXiv 2024 | 0.92 | 0.90 | 0.86 | bbox+ego (2) — **closest 2-stream cousin** |
| MFT | arXiv 2025 | 0.90 | **0.94** | 0.83 | 4 context streams — **ties our AUC** |
| **BiLSTM (ours)** | 2026 | 0.883 | 0.932 | 0.828 | **bbox+ego (2)** |
| **Transformer (ours, searched)** | 2026 | 0.894 | **0.950** | 0.845 | **bbox+ego (2)** |
| **BiLSTM-F1 (ours)** | 2026 | 0.897 | 0.940 | 0.844 | **bbox+ego (2)** |
| **Transformer-F1 (ours)** | 2026 | 0.896 | 0.947 | 0.847 | **bbox+ego (2)** |
| **GRU-F1 (ours)** | 2026 | 0.901 | 0.941 | 0.849 | **bbox+ego (2)** |
| **Vanilla RNN-F1 (ours)** | 2026 | 0.902 | 0.948 | 0.852 | **bbox+ego (2)** |

‡ GTransPDM flags Ped-Graph+ and BiPed as configured differently ("Except BiPed and
Pedestrian Graph+…") — verify vs originals in the BibTeX pass.

**Table corrections already made (2026-07-13) — do not regress them:**
- **PedFormer and BiPed are SEPARATE rows** (PedFormer 0.93/0.90/0.87, BiPed
  0.91/0.90/0.85); an earlier draft misattributed BiPed's numbers to both.
- **GTransPDM's abstract "92%" is the *w/o-pose* ablation** (0.92/0.90/0.86), given its
  own 2-stream row; the full model is 0.90/0.87/0.82.
- **PIP-Net was REMOVED** — its own paper uses a **custom random split** (~880/719/243),
  not the standard protocol; cite only as prose context.
- **Occlusion-Aware Diffusion is a *modality precedent*, not a comparison row** —
  occluded-only, ~1-frame-ahead TTE. It confirms bbox+ego-velocity is a legitimate
  minimal modality; do **not** put its 0.95 in the table.

**The honest framing (F1-first):** *On the standard PIE protocol, a 2-stream (bbox +
ego-speed) model reaches F1 within 0.02–0.03 of the multimodal SOTA (0.844–0.852 vs
PedFormer's 0.87), with the table's highest AUC (0.94–0.95), at a fraction of the
feature-extraction cost and latency; this holds for a BiLSTM, a searched Transformer, a
GRU, and an un-gated RNN, and the families tie on F1 — so the finding is about the input
signal, not the architecture.* We are **mid-band on Accuracy** (0.897–0.902 vs
PedFormer's 0.93) — stating that plainly is what keeps the AUC claim credible rather than
"suspiciously easy."

**Positioning matrix** (`04_positioning_vs_prior_work.md`) — every "their limitation → our
response" cell now carries a measured number: heavy modality/latency → Issue 9; no CIs →
Issue 4; single split → Issue 5; single-seed ablations → Issue 6; GT-box assumption →
Issue 10; unjustified hyperparameters → Issues 7–8 + the four searches; the distinctive
leakage fix → Issues 1–2.

---

## 15. The live perception pipeline (demo, Issues 9–10)

`pipeline/10_yolo_bytetrack_demo.py`: raw set03 video → **YOLO26-M** (person detection,
`yolo26m.pt`, required, no fallback) → **ByteTrack** (per-pedestrian track IDs,
`persist=True`, fed frame-by-frame from an OpenCV reader so absolute PIE frame numbers
stay aligned with the ego-speed lookup) → per-track rolling 16-frame buffer of
`[x1,y1,x2,y2,ego_speed]` → normalize (train stats) → BiLSTM → `P(cross)` overlay →
annotated mp4. Stages `--stage {detect,track,demo}`; device auto cuda→mps→cpu. Cite
YOLO (Ultralytics) + ByteTrack (Zhang 2022).

Two demo clips (set03, downloaded via 24-way parallel range download because the York
server throttles single connections): **video_0016** (crowd crossing at a stop,
qualitative) and **video_0012** (moving vehicle, mixed crossing/not-crossing, superseded
by Issue 10's proper 98-ped measurement). Demo outputs in `pipeline/demo_out/`.

**Deployment reality (from Issues 9–10):** the pipeline is **detection-bound** (YOLO26-M
is ~20× the BiLSTM's per-frame cost), the predictor is **robust to detector box noise**
(+0.009 AUC drop GT→YOLO), and the real engineering gaps are **detector recall (88%)**
and **ByteTrack identity fragmentation (track purity 39%)** — which need re-ID in
deployment, and are separate from the prediction model.

---

## 16. Honest limitations (state before a reviewer does)

1. **Ego-speed carries much of the signal and partly encodes the ego-driver's
   anticipation** — the instrumented car slows for expected crossers, so speed is a
   legitimate inference-time signal but not purely vision-based. Candidate mitigation:
   a speed-perturbation robustness check. **(The one limitation to always state.)**
2. **ByteTrack fragments identities** (track purity 39%) → needs re-ID in deployment.
3. **Detector-in-the-loop is indicative** (2 clips, 98 peds); the YOLO-box association is
   GT-guided best-IoU (isolates box-quality noise), so the +0.009 drop is a **lower
   bound** on fully end-to-end degradation.
4. **Single dataset (PIE)** — the PSI cross-dataset test (§12) is the planned answer, but
   is data-blocked.
5. **The family equivalence is horizon-bounded** — the un-gated RNN falls behind by
   OW 64 (~2 s); gating is redundant over 0.5 s, not universally.
6. **Statistical power:** 5 seeds is low-power, so the paired/cluster bootstraps over the
   2,094 test windows are the primary evidence throughout, not the n=5 seed t-tests.

---

## 17. Metric hierarchy & the "which model" answer

**Metric hierarchy: F1 → accuracy → AUC** (supervisor directive). Report F1 first, then
accuracy, then AUC/PR-AUC as threshold-free corroboration. Every PIE benchmark paper
reports Acc/AUC/F1 jointly, so this is consistent with the field (we do not claim the
field treats F1 as primary — it's the imbalance-appropriate operating-point metric and
our supervisor's priority). The operating point is deliberately **recall-favoring**
(`pos_weight=1.682` → R > P for the frozen baseline): in AV safety a missed crosser costs
more than a false alarm.

**The model-choice answer, empirically:** *Transformer for the best AUC; any of the four
families for F1; the smaller BiLSTM remains fully defensible as the headline model under
F1-first.* The model-choice question dissolves into a **metric-priority question** — which
is itself a publishable finding.

---

## 18. Reproducibility & execution environments

- **Determinism (measured, Issue 12):** CPU training is **bit-reproducible and
  context-free**; **`nn.LSTM` training on Apple MPS is process-history-dependent** (same
  cfg+seed gives different results depending on what ran earlier in the process) — so
  recurrent runs needing exact reproduction go on **CPU**. Transformer training is
  context-free on MPS. Kaggle-GPU ↔ local-CPU inference drift is ~1e-6 (benign,
  parity-gated).
- **Local:** MacBook Air M4. The live demo runs on MPS; `.venv` has torch 2.12,
  scikit-learn 1.9, scipy, ultralytics 8.4.68, opencv 4.13.
- **Kaggle (T4):** some early training + ablation sweeps ran there (paths hard-code
  `/kaggle/…` — adjust when running locally). The journal-bound work runs locally on the
  unified engine.

---

## 19. Manuscript status & the writing plan

**Where the writing stands:** experimental work is **complete**; the task is drafting the
MDPI MTI paper. The scaffold `paper_skeleton.tex` is already **F1-first** and has the
baseline table filled; `references.bib` is seeded (~40 entries, many `% VERIFY` flags);
`PLAN.md` maps issues → sections; `HANDOVER_PROMPT.md` is the prior briefing.

**MDPI MTI section order + our content (from `PLAN.md`):**
| Section | Our content | Source |
|---|---|---|
| Abstract (~200 w, unstructured) | problem → leakage finding → clean protocol → F1 0.844–0.852 + top AUC + latency → rigor | all |
| 1 Introduction | safety motivation; intention ≠ trajectory; the gap (leakage, heavy modality, weak rigor); 3 contributions; roadmap | 1, 2, 3 |
| 2 Related Work | PIE landscape (PCPA, Ped-Graph+, PIT, IntFormer, BiPed/PedFormer, GTransPDM, MFT); Occlusion-Diffusion as minimal-modality precedent; field-wide gaps | 3 |
| 3 Materials & Methods | PIE + splits; **leakage problem + clean crossing_point protocol**; features + train-only norm; the four architectures; frozen training protocol + documented HP search; YOLO26+ByteTrack pipeline | 1, 2, 7, 8, 12, (10) |
| 4 Results | main result + bootstrap/cluster CI; baseline table; **ego-speed dominance (+0.18)**; LOSO; window/TTE; capacity; **Transformer comparison**; **F1-first**; (GRU/RNN cell studies); latency; detector-in-the-loop | 2–10, transformer, f1_opt, gru, rnn |
| 5 Discussion | parsimony interpretation; the metric-hierarchy finding; high-AUC/mid-Acc honesty; positioning vs prior work; deployment realism; **Limitations** (§16) | 3, 9, 10 + limitations |
| 6 Conclusions | restate contributions + headline; future work (re-ID, speed-perturbation robustness, PSI cross-dataset, more clips) | all |
| Back matter | Author Contributions; Funding; Data Availability (PIE public + our code); Conflicts; Abbreviations; References | — |

**Drafting order (write concrete → framing):** Methods → Results → Related Work →
Introduction → Discussion → Conclusions → **Abstract last**. Target ~7,000 words + ~9
floats. Keep prose human (run the `/humanizer` pass — strip em-dash overuse, "moreover/
furthermore" stacks, rule-of-three, inflated phrasing).

**Figures to pull (don't regenerate; they exist as PNGs):** feature-ablation / ROC (ego-
speed gap), window+TTE (`issue6_*`), capacity (`issue7_*`), grid search (`issue8_*`),
latency breakdown (`issue9_*`), GT-vs-YOLO (`issue10_*`), transformer three-bar
(`transformer/phase5_analysis/05_comparison_figure.png`), F1-first
(`f1_optimization/06_figure.png`), and the consolidated `journal_prep/Analysis/figures/`
(confusion grids, metrics bar, ROC/PR overlays, efficiency frontier).

---

## 20. Citation / reference status (the debt to clear before submission)

`references.bib` is seeded but carries **`% VERIFY` flags** on many fields (DOIs, author
lists, venues, pages). **Do not submit a guessed DOI or split.** Specific external
verifications still owed (need paper access):
- **IntFormer** and **PIT** bib entries (currently no keys; appear via GTransPDM Table I).
- **PIP-Net** split (confirmed custom; context citation only) and **GTransPDM** 0.90-vs-
  0.92 (resolved: 0.92 = w/o-pose ablation).
- Ped-Graph+ / BiPed configuration flag (‡) vs their originals.
- Full BibTeX/venue/DOI pass. arXiv ids on hand: GTransPDM 2409.20223, PIP-Net
  2402.12810, Occlusion-Diffusion 2511.00858, IntFormer 2105.08647, PedFormer 2210.07886,
  MFT 2511.20011, ACIT 2511.20020.
- Architecture-source citations (for Methods) are listed in
  `journal_prep/Analysis/README.md` refs [1]–[16] (LSTM, BiRNN, Transformer, pre-LN, GRU,
  Chung, Elman, Bahdanau, Rumelhart, Adam, AdamW, PyTorch, PIE, PCPA, ByteTrack).

**What Claude must NOT do:** invent citations/DOIs, fabricate numbers, or claim results
we didn't run. Flag anything needing an external source or a new experiment.

---

## 21. Critical conventions & gotchas (quick reference)

- **Training entrypoint** = `journal_prep/issue12_unified_pipeline/12_unified_engine.py`
  (`--family bilstm|transformer|gru|birnn`, `--select f1|auc`). **Not** the legacy
  `pipeline/04_*` (leaky defaults `sequences/` + pos_weight 1.44).
- **Clean data** = `journal_prep/issue2_clean_protocol/sequences_clean/` (N=4,906).
- **pos_weight = 1.682** (clean) everywhere unless it's the ablated factor. 1.44 is
  retracted legacy.
- **Test set03 touched once** per experiment, on the best-val checkpoint. Never select on
  test.
- **`weights_only=False`** to load any `best.pt`.
- **Two statistics, never mixed in one sentence:** *per-seed mean* (the paper numbers,
  cross-paper-comparable) vs *5-seed probability ensemble* (deployable, slightly higher,
  source of the confusion matrices).
- **The headline is F1-first**, not "AUC 0.932 first." 0.932 is the frozen AUC-selected
  baseline (still the demo checkpoint, still valid) but not the headline framing.
- **`paper_and_artifacts/supervisor_review/`** is a **stale pre-leakage-fix** snapshot —
  regenerate figures from current run outputs; do not cite its numbers.
```

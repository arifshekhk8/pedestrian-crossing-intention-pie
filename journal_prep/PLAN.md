# Journal Preparation Plan
# Pedestrian Crossing Intention — BiLSTM on PIE

**Goal:** Fix every issue identified in the journal reviewer analysis before
submission. Issues are grouped by severity (🔴 reject-risk / 🟠 major concern /
🟡 minor). Work through them in the numbered order — earlier issues gate later ones.

**Working directory:** `/Users/arif/Developer/pedestrian-thesis/`  
**Outputs of this work live in:** `journal_prep/`  
**Training environment:** Kaggle T4 (sklearn present); local M4 for demo/latency.

---

## Status Key
- [ ] Not started  
- [~] In progress  
- [x] Done  

---

## Issue 1 — Temporal Leakage Audit 🔴
**Risk level:** If leakage is found, the headline AUC 0.931 is invalid.

**Problem:**  
`02_build_sequences.py` anchors each observation window at
`last_annotated_frame − TTE`, where "last annotated frame" is the last row for
that pedestrian in the PIE XML — **not** the crossing event boundary. There is no
check that the 16-frame observation window is entirely pre-crossing. If a crosser's
window overlaps frames where they are already stepping off the curb, the task
collapses to "detect someone mid-crossing," which is trivially easy and explains
the fast epoch-3 convergence.

**What to build:**  
`journal_prep/01_leakage_audit.py`  
- Load `pie_annotations.pkl` (has `action` and `crossing_label` columns).  
- For each sequence in `sequences/meta.pkl`, identify the observation window
  (frames `anchor_frame − obs_len + 1` … `anchor_frame`).  
- Look up the `action` column for that pedestrian + those frames. Flag any window
  where `action` contains a value associated with mid-crossing (e.g., action ≠
  "standing" / "walking" prior to crossing event — read PIE action label docs).  
- Also check whether crossers and non-crossers differ in anchor-frame bbox
  position (bottom-of-frame proximity, bbox height) in a way that would be a
  trivial visual shortcut.  
- Output: `journal_prep/01_leakage_report.md` — every flagged sequence, summary
  statistics, and a verdict: CLEAN or LEAKAGE FOUND.

**Success criterion:** Report generated, verdict clearly stated, numbers reported
in the paper's "Data" or "Experimental Setup" section.

### ✅ DONE — VERDICT: 🔴 LEAKAGE FOUND (2026-06-21)

Files in `journal_prep/issue1_leakage_audit/`:
`01_leakage_audit.py`, `01_leakage_report.md`, `leakage_per_sequence.csv`,
`cross_state_map.pkl`, `figures/leakage_gap_hist.png`,
`figures/anchor_bbox_shortcut.png`.

**Critical data discovery:** PIE boxes carry a per-frame `cross` attribute
(`not-crossing` / `crossing` / `crossing-irrelevant`). The original
`01_parse_annotations.py` kept only `action` (walking/standing) and **dropped
`cross`** — the exact per-frame "is crossing now" signal. We re-parsed it.

**Findings (570 crossers / 819 non-crossers):**
- **387/570 crossers (67.9%)** have ≥1 frame *already crossing* inside the
  16-frame observation window → leakage.
- **369/570 crossers (64.7%)** have **all 16 frames** already crossing → the task
  is *detection of an in-progress crossing*, not prediction, for ~2/3 of positives.
- Only **183/570 (32%)** crossers have a genuinely clean (0-crossing) window.
- Non-crossers: 0% leakage (correct, by definition).
- **Static shortcut also present:** anchor-frame bbox alone separates classes —
  area (rank-biserial +0.65), height (+0.63), bottom-y (+0.49), all large effects.
  Crossers are close/large/low-in-frame; non-crossers far/small. Model can partly
  cheat on geometry with no temporal info.

**This explains the suspicious AUC 0.931 @ epoch 3.** The headline number is
inflated by (a) observing crossings in progress and (b) a static size/position
shortcut. **Both must be fixed in Issue 2** (anchor the window strictly before
crossing onset = event-relative TTE; adopt the canonical overlapping protocol)
before any baseline comparison (Issue 3) is valid.

- [x] Write `01_leakage_audit.py`  
- [x] Run it locally (pure pandas/scipy/matplotlib, ~30 s on M4)  
- [x] Write `01_leakage_report.md`  
- [x] Verify findings by manual spot-check of flagged vs clean sequences  

---

## Issue 2 — Canonical PIE Extraction Protocol 🔴
**Risk level:** Bars comparison to any published number on PIE.

**Problem:**  
Current extraction (`02_build_sequences.py`):  
1. Extracts **one window per pedestrian per contiguous segment** — yields only
   1,389 sequences from 1,374 pedestrians.  
2. Anchors TTE at **track end**, not the **crossing event frame**.  

Standard PIE protocol (see `PIE/scenarioEval/README.md`):  
1. Sliding window with **50% overlap** → many more sequences.  
2. TTE defined from the **first frame where the pedestrian starts crossing** (or the
   last frame with TTE-to-event for non-crossers).  

**Options (discuss with supervisor first):**  
- **Option A (preferred for journal):** Re-implement extraction to match PIE
  benchmark protocol, retrain all models, report new numbers.  
- **Option B (acceptable for thesis):** Keep custom protocol but state it
  explicitly, only compare within your own study. Add a paragraph in
  "Experimental Setup" documenting the deviation and why.

**What to build (Option A path):**  
`journal_prep/02_build_sequences_pie_protocol.py`  
- 50% overlapping sliding window  
- TTE anchored to crossing-event frame from PIE annotation  
- Same 5-D feature, same train/val/test set splits  
- Output: `journal_prep/sequences_pie/` (new X.npy, y.npy, meta.pkl)

**Success criterion:** Either new canonical sequences produced and retrained, or
a written justification in the paper that is airtight.

- [x] Read PIE annotation structure carefully (action labels, event frames)
- [x] Decide Option A vs B with supervisor — **Option A taken**
- [x] Implement and retrain

### ✅ DONE — Option A implemented, leak-free, retrained (2026-06-21)

Files in `journal_prep/issue2_clean_protocol/`:
`02_build_sequences_clean.py`, `sequences_clean/{X.npy,y.npy,meta.pkl}`,
`02_leakage_report_clean.md` (+ `leakage_per_sequence.csv`, `figures/`),
`runs_clean/bilstm_baseline_clean/{best.pt,history.json,final.json,norm_mean.npy,norm_std.npy}`.

**Design decision — no need to invent an event-relative TTE definition.** PIE's own
`annotations_attributes/*.xml` already carries a `crossing_point` frame attribute
for *every* pedestrian, crossers and non-crossers alike — the exact anchor PIE's
own benchmark code uses (`PIE/utilities/data_gen_utils.py::extract_tracks_tte`,
the basis for PCPA/GTransPDM/PIP-Net-style protocols). Validated against the
per-frame `cross` ground truth cached in `issue1_leakage_audit/cross_state_map.pkl`:
`crossing_point` exactly equals the first `cross=="crossing"` frame in 516/519
crossers (99.4%), never earlier. For non-crossers it sits ~2 frames before the
track's last annotated frame ("when the vehicle reaches them") — so the same
mechanism anchors both classes, no separate non-crosser rule needed.

**Algorithm implemented** (mirrors `extract_tracks_tte` exactly): per pedestrian,
take the contiguous segment containing `crossing_point` (drop earlier disjoint
segments on the ~4% of tracks with a gap), truncate at `crossing_point`
inclusive, then slide `obs_len=16` windows with 50% overlap (stride 8)
constrained to `TTE ∈ [30,60]` frames before that point — falling back to
sampling from frame 0 if the track doesn't reach the full TTE=60 range (same
fallback as the official code). 107/1,374 pedestrians (7.8%) excluded for being
too short even for TTE=30.

**Results:**

| | sequences | crosser % | leakage | test N | test AUC | test F1 | test Acc | best epoch |
|---|---|---|---|---|---|---|---|---|
| Old (leaky, `sequences/`) | 1,389 | 41.0% | 387/570 crossers (67.9%) | 587 | 0.931 | 0.844 | 0.874 | 3 |
| **New (clean, `sequences_clean/`)** | **4,906** | **33.6%** | **0/4,906 (0.000%)** | **2,094** | **0.913** | **0.823** | **0.884** | **17** |

Re-running `01_leakage_audit.py --seq-dir sequences_clean/` (now generalized with
`--seq-dir`/`--out-dir` flags, backward-compatible) confirms **0/4,906 windows
leak** — verdict ✅ CLEAN. The static anchor-bbox shortcut also weakened sharply
(bbox_area rank-biserial +0.65 → +0.25, bbox_height +0.63 → +0.21,
bbox_bottom_y +0.49 → +0.09 n.s.), confirming part of the old shortcut was itself
a side-effect of observing pedestrians already mid-crossing.

**The honest headline:** AUC drops only 0.931 → 0.913 (−0.018) once leakage is
fully removed and N grows 3.5× — model convergence also moved from a suspicious
epoch 3 to a believable epoch 17. This AUC is the number to carry into Issue 3's
baseline comparison table, not 0.931.

**Eval-parity verification (2026-06-21) — `03_eval_parity_check.py` +
`03_eval_parity_report.md`.** Because 0.913 with only bbox+speed *beats* the
multimodal baselines in the Issue 3 table (PCPA 0.86, GTransPDM 0.87, PIP-Net
0.90), we stress-tested whether our evaluation is merely easier. It is not:

| check | result | meaning |
|---|---|---|
| per-window AUC (headline) | 0.9131 | the final.json number |
| per-**pedestrian** AUC (mean prob, 541 peds) | 0.9143 | gap +0.0012 → overlap does **not** inflate it |
| benchmark-filter subset (track ≥76 = `min_track_size` 75) | 0.9194 | our laxer 46-frame floor doesn't help us |
| short tracks only (46–75, the 37 we extra-admit) | 0.8634 | the extra tracks are *harder*, not easier |

So 0.913 reflects genuine signal in bbox motion + ego-speed, not an easier
protocol — consistent with the Occlusion-Diffusion paper's 0.93–0.95 on bbox+ego
only. Only documented deviation: 0.5 overlap (PIE's trajectory value; 0.3 is its
action default), shown immaterial since per-ped ≈ per-window.

**Class balance changed** (33.6% positive vs. old 41.0%), so `POS_WEIGHT=1.44`
no longer matches; recomputed from the new train split: `neg/pos = 1366/812 =
1.682`. Added a `--pos_weight` CLI override to `04_train_bilstm.py` (default
unchanged at 1.44, so the original baseline command is reproducible byte-for-byte)
rather than hardcoding a second constant into the shared script.

**All three variants now multi-seeded on clean data** (`05_variant_comparison.md`,
5 seeds [42,0,1,2,3]; baseline local, variants on Kaggle T4 — `kaggle_result/`):

| Model | Inputs | OLD AUC | **NEW (clean, 5-seed) AUC** | Δ |
|---|---|---|---|---|
| BiLSTM baseline | bbox + ego-speed (5-D) | 0.931 | 0.932 ± 0.011 | +0.001 |
| BiLSTM bbox-only | bbox (4-D) | 0.889 | **0.753 ± 0.020** | **−0.136** |
| BiLSTM + attention | bbox + ego-speed (5-D) | 0.933 | 0.925 ± 0.010 | −0.008 |

**Key finding (validated, multi-seed): ego-vehicle speed is the dominant
predictor — +0.179 AUC** (baseline 0.932 − bbox-only 0.753). bbox-only **collapses
−0.136** once the window ends strictly before crossing onset and the
static-geometry shortcut is gone — confirming the old evaluation measured
detection-of-in-progress-crossing. **Attention gives no measurable benefit** on
clean data (0.925 vs baseline 0.932, within seed noise; its leaky-data 0.945 edge
was an artifact). The leak fix itself is **methodological** (baseline 0.931→0.932
unchanged, N 3.5×, epoch 3→17), not a deflation. **Honest limitation:** ego-speed
partly encodes the ego-driver's own anticipation (instrumented car slows for
expected crossers) — a legitimate inference-time signal, not purely vision-based;
flag in Limitations + a speed-perturbation robustness check.

Validated two ways: Kaggle clean ≈ independent local CPU cross-check
(`06b_local_verify_seed42.py`: bbox 0.770±0.016, attn 0.933±0.004), and every
checkpoint reproduces its `final.json` on the local clean test (bbox seed42 0.7325,
attn seed42 0.9228 — exact). ⚠ The **first** Kaggle run was discarded: it silently
trained on the OLD LEAKY `sequences/` (test N=587 not 2,094; bbox 0.883, attn
0.945) because `find_seq_dir()` grabbed the leaky `X.npy` from the `pie-bilstm`
input — now hard-errors unless it loads the clean N=4,906 data. Cosmetic (old
notebook version): `kaggle_result/summary.csv` labels swapped + header says
pos_weight 1.44 — use `.md`/`results.csv`.

**Multi-seed baseline** (`04_multiseed_summary.md`, 5 seeds): **test AUC 0.932 ±
0.011**; seed 42 (0.913) is the *lowest* of the five so the single-seed headline
is conservative. best-epoch scatters 5–17 because val (set05/06; set05 = 13 peds,
24% pos vs 37% train) is small and skewed — noisy model selection that directly
motivates Issue 4 (bootstrap CI) and Issue 5 (LOSO). Seed 42 reproduced
bit-for-bit → determinism confirmed.

**Eval parity verified** (`03_eval_parity_report.md`): the 5-D 0.913 is robust to
per-window-vs-per-pedestrian (per-ped 0.914) and to min-track-length
(benchmark-filter subset 0.919) — not an easier-evaluation artifact.

**Variant multi-seeding DONE (Kaggle T4, clean):** bbox-only **0.753 ± 0.020**,
attention **0.925 ± 0.010** (`kaggle_result/`), confirmed by local cross-check and
checkpoint reproduction. Net result: ego-speed dominant (+0.18), attention no
benefit. (No backend issue — an earlier "fragility" theory was just a leaky-vs-clean
data mixup from a discarded first run.)

---

## Issue 3 — Published Baseline Comparison 🔴
**Risk level:** "No external comparison" is a standard major revision comment.

**Problem:**  
No published PIE baseline number appears anywhere in the project. Without a
comparison table, AUC 0.931 is uninterpretable to a reviewer — and currently it
sits at the *top* of the published band (0.86–0.92) despite using the *fewest*
inputs, which reads as "easier evaluation" rather than "better model." This is
only meaningful once Issues 1 + 2 (leakage + canonical protocol) are resolved.

**Chosen baselines (verified from the papers' own result tables, web search
2026-06):**  
All report on PIE crossing prediction with the standard protocol (observation
T=16 frames / 0.5 s; TTE 30–60 frames / 1–2 s; sliding windows with ~0.5 overlap)
— the same five metrics this thesis uses (Acc / AUC / F1 / P / R).

| Baseline | Venue / Year | Acc | AUC | F1 | Inputs | Role in our table |
|---|---|---|---|---|---|---|
| **PCPA** (Kotseruba et al.) | WACV 2021 | 0.87 ✅ | 0.86 ✅ | 0.77 ✅ | bbox + pose + local context + speed | **Benchmark anchor (mandatory)** — defines the standard protocol & metric format |
| **GTransPDM** (graph + transformer) | arXiv, Sept 2024 | 0.90 ✅ (abs. 0.92) | 0.87 ✅ | 0.82 ✅ | bbox + pose + ego motion | Recent SOTA; its **Table I** tabulates PCPA/PIT/IntFormer/Ped-Graph+/BiPed for us |
| **PIP-Net** (Azarmi et al.) | arXiv Feb 2024 → IEEE **T-ITS 2025** | 0.915 ✅ | 0.897 ✅ | 0.846 ✅ | 7 features (bbox, pose, speed, context, optical flow, semseg, depth) | Multimodal "full-feature upper bound" contrast |
| **Occlusion-Aware Diffusion** (Liu et al.) | **IEEE T-ITS, Nov 2025** | ✗ occluded-only | ✗ (~0.95 at EO5) | ✗ | **bbox + ego velocity ONLY** | ⚠ **NOT apples-to-apples** — occlusion-robustness task, ~1-frame-ahead TTE; cite as **modality precedent** only |
| **PIEPredict** (Rasouli et al.) | ICCV 2019 | — | — | — | bbox + ego + context | Foundational; *trajectory* model (no classifier row) — optional "run on our split" |

Secondary methods that recur in every comparison table (cite for landscape, no
need to reproduce): **Pedestrian Graph+** (2022, 0.89 / 0.90 / 0.81), **PIT**
(2023, 0.91 / 0.92 / 0.82), **IntFormer** (~2022, 0.89 / 0.92 / 0.81),
**BiPed/PedFormer** (2023, 0.91 / 0.90 / 0.85).

**Key framing (the honest, strong claim to make in the paper):**  
On the canonical protocol (Issue 2) and leakage-cleared (Issue 1), our bbox+speed
BiLSTM lands **AUC 0.932** — top of the published band on AUC, **mid-pack on Acc
(0.883)**, with only **2 input streams**. Claim *"competitive with recent
multimodal SOTA using only bbox + ego-velocity, at a fraction of the latency"* —
and lean on the high-AUC/mid-Acc split to pre-empt the "easier eval" reflex (we
don't dominate every metric). **Correction (web-verified 2026-06):** the
**Occlusion-Aware Diffusion** paper is *not* an apples-to-apples row as first
assumed — it reports only occluded scenarios at a ~1-frame-ahead TTE (no standard
fully-observed number), so cite it as a **precedent that bbox + ego-velocity is a
legitimate minimal modality**, with the protocol caveat stated, not as a head-to-head
AUC comparison. Full verified table + sources in
`issue3_baseline_comparison/03_baseline_comparison.md`.

**Positioning thread (runs through ALL remaining issues).** The comparison is not
just a table — the paper must say *how we address each recent work's limitation*.
Seeded in `issue3_baseline_comparison/04_positioning_vs_prior_work.md` as a "their
limitation → our response → evidence (issue)" matrix, and it is the *purpose* behind
several later issues: heavy-modality/latency (PCPA 3D-CNN, PIP-Net 7-feat, GTransPDM
pose-graph) → **Issue 9**; no CIs → **Issue 4**; single split → **Issue 5**;
single-seed ablations → **Issue 6**; GT-box assumption → **Issue 10**; unjustified
hyperparams → **Issue 7–8**; our distinctive leakage fix → **Issue 1–2**. Finalize
that matrix (with each issue's numbers) at the very end.

**What to do:**  
1. Pull each baseline's PIE numbers from its results table (done for the four
   above; lock exact figures + citations when drafting).  
2. Confirm split alignment: standard PIE benchmark uses train set01/02/04,
   val set05/06, test set03 — verify each baseline used this (footnote any that
   differ).  
3. (Optional, strong) Run `PIEPredict/` on our split for a directly comparable
   "original PIE model, our split" row.  
4. Build the final table with our number (post-Issue-2) inserted, modalities
   column included so the input-parsimony story is visible.

**Output:** `journal_prep/issue3_baseline_comparison/03_baseline_comparison.md`
(+ `README.md`) — draft comparison table + modality column + protocol footnotes,
ready for the paper's Related Work / Results.

### ✅ DONE — internal finalization (2026-06-28); external verify + BibTeX remain

`issue3_baseline_comparison/03_baseline_comparison.md` + `04_positioning_vs_prior_work.md`
are finalized now that Issues 1–10 are complete. Our row: **AUC 0.932 [0.92–0.95]
(top of the standard-protocol table), 2 streams, 0.575 ms/window**; honest mid-pack
Acc (0.883) framed via threshold-free ROC/PR-AUC + the recall-favoring operating
point. Occlusion-Diffusion reclassified as a *modality precedent* (occluded protocol,
~1-frame TTE), not a comparison row. The positioning matrix now carries a measured
number in every "our response" cell (CI, LOSO, multi-seed ablations, grid search,
latency, detector-in-the-loop). **Only external items remain** (need paper access),
intentionally left for the manuscript-drafting pass:

- [x] Lock Acc/AUC/F1 for PCPA/GTransPDM/PIP-Net + landscape — confirmed from sources  
- [x] Reclassify Occlusion-Diffusion (modality precedent, not a row)  
- [x] Fold in our complete evidence (Issues 4/5/6/7/8/9/10) — framing + matrix done  
- [ ] **External:** `[verify]` PIP-Net's PIE split + GTransPDM 0.90/0.92 variant vs source PDFs  
- [ ] **External:** full BibTeX + venue/DOI pass  
- [ ] (Optional, strong) Run vendored `PIEPredict/` on our split for a directly comparable row  

---

## Issue 4 — Bootstrap CIs on Test AUC 🔴
**Risk level:** Increasingly required by IEEE/Springer journals.

**Problem:**  
Multi-seed std (±0.013) captures *training noise*, not *test-set sampling noise*.
With 587 test sequences the sampling uncertainty on AUC is meaningful (~±0.02).
No CI is currently reported.

**What to build:**  
`journal_prep/04_bootstrap_ci.py`  
- Load the saved raw predictions from the best checkpoint (re-run test eval to
  get `probs` and `labels` — already computed in `evaluate()` in
  `04_train_bilstm.py`, just not saved).  
- Bootstrap resample `(probs, labels)` 10,000 times and compute AUC each time.  
- Report 95% CI (percentile method).  
- Repeat for all 3 model variants (baseline / bbox-only / attention).  
- Output: `journal_prep/04_bootstrap_ci_results.md`

**Success criterion:** Every reported test AUC in the paper has a 95% CI alongside it.

- [x] Write `04_bootstrap_ci.py` (in `issue4_bootstrap_ci/`; regenerates probs from
      saved checkpoints + reuses `06b` model classes)  
- [x] Save raw test probs from baseline, bbox-only, attention checkpoints  
- [x] Run and report CIs  

### ✅ DONE (2026-06-23) — `issue4_bootstrap_ci/`

10k percentile bootstrap on the clean test (set03, N=**2,094** — note the old
"587" above was the leaky size). All 3 variants × 5 seeds, ROC-AUC + PR-AUC.

**Headline: baseline ROC-AUC 0.932, 95% CI ≈ [0.92, 0.95]; PR-AUC 0.876.** Three
findings: (1) test-sampling CI half-width (±0.013) ≈ seed std (±0.010) — the
0.932 ± 0.011 already reflects the right uncertainty; (2) the ego-speed gap is
statistically unambiguous (bbox-only CIs ≤0.80 never touch baseline ≥0.90); (3)
attention ≈ baseline (CIs overlap → no significant gain). Full table +
interpretation in `04_bootstrap_ci_results.md`. Addresses the "point estimate, no
CI" gap (positioning matrix, Issue 3).

---

## Issue 5 — Leave-One-Set-Out Cross-Validation 🔴
**Risk level:** Single fold (set03) means the test number could be
set03-specific; reviewers may ask "what if set03 is easy?"

**Problem:**  
PIE has 6 sets (set01–06). You use one fixed split. Rotating the held-out set
across all 6 and averaging gives a much stronger generalization claim.

**What to build:**  
`journal_prep/05_loso_cv.py` (Kaggle notebook or script)  
- For each fold i in {01,02,03,04,05,06}:  
  - test = set_i, train+val = remaining 5 sets (hold out one randomly for val,
    or use all 5 for train with no early-stop, or use a fixed val set)  
  - Train baseline BiLSTM with identical hyperparams, evaluate on test fold  
- Report AUC mean ± std across 6 folds.  
- Note: set sizes differ, so do NOT average N-weighted — report per-fold too.  
- Output: `journal_prep/05_loso_results.md` + `05_loso_results.csv`

**Success criterion:** Table showing per-fold AUC and mean ± std over 6 folds.

- [x] Write `05_loso_cv.py` (in `issue5_loso_cv/`; 85/15 **pedestrian-grouped** val
      split, per-fold norm + pos_weight, baseline arch)  
- [x] Run — **locally on M4 GPU/MPS (~80 s)**, not Kaggle (6 baseline trainings are
      cheap; 5-D model is backend-stable)  
- [x] Report results  

### ✅ DONE (2026-06-23) — `issue5_loso_cv/`

| | 6-fold AUC | excl. tiny set05 (N=47) | set03 fold | softest large fold |
|---|---|---|---|---|
| LOSO | **0.928 ± 0.041** | 0.915 ± 0.029 | **0.931** (≈ fixed-split 0.932) | set04 0.892 (N=1610) |

**set03 is representative, not an easy cherry-picked fold** (0.931 ≈ the 0.932
multi-seed number) — directly answers "what if set03 is easy?". Model generalizes
across all 6 recording sets; per-fold class balance ranges 24–57% positive and AUC
holds. Honest weak spot: set04 (PR-AUC 0.791). set05 (N=47) excluded as
uninterpretable. Addresses the "single fixed split" gap (positioning matrix,
Issue 3). Full table + interpretation in `05_loso_results.md`.

---

## Issue 6 — Multi-Seed Ablations (Window + TTE) 🟠
**Risk level:** Current "insensitive to window/TTE" conclusions are weaker than
the seed-to-seed variance — undefendable as written.

**Problem:**  
Window ablation (obs_len 8/16/30) and TTE ablation (TTE 30/45/60) were each run
at **seed 42 only**. The maximum spread in the window ablation is 0.005 AUC, but
the seed-to-seed std for the baseline at the **same obs_len=16** is ±0.013.
Conclusion "insensitive" is noise-level — need multi-seed to confirm.

**What to build:**  
`journal_prep/06_multiseed_ablations.ipynb` (Kaggle notebook)  
- Re-run obs_len ∈ {8, 16, 30} × 5 seeds → 15 models  
- Re-run TTE ∈ {30, 45, 60} × 5 seeds → 15 models  
- Report mean ± std per condition  
- Run a simple significance test (Mann-Whitney U or paired t-test) between
  conditions to support "no significant difference" claim  
- Output: `journal_prep/06_window_multiseed.csv`, `06_tte_multiseed.csv`,
  `06_multiseed_ablation_summary.md`

**Success criterion:** "AUC insensitive to obs_len/TTE" supported by multi-seed
mean ± std and a p-value > 0.05 between conditions.

- [x] Write `06_multiseed_ablations.py` (in `issue6_window_tte_ablation/`; self-contained
      MPS harness, locked baseline arch/hyperparams, builds per-config clean sequences)
- [x] Run — **locally on M4 GPU/MPS (~11 s/training, 30 trainings)**, not Kaggle
- [x] Run significance tests (paired t-test + Mann-Whitney U + Kruskal–Wallis) and report

### ✅ DONE (2026-06-26) — `issue6_window_tte_ablation/`

Clean (Issue-2), leak-free, **5-seed** [42,0,1,2,3] re-run of both ablations,
everything locked to the baseline (pos_weight=1.682 fixed across cells; only the
ablated factor moves). **TTE-band mapping** (the builder takes `--tte-min/--tte-max`,
not a single `--tte`): window sweep = obs_len∈{8,16,30} at the canonical band
[30,60] (obs16/[30,60] reuses `sequences_clean/`); TTE sweep = single-point band
[T,T] for T∈{30,45,60} at obs_len 16 (faithful to the old single-point `09_`).

**Two different answers** — the multi-seed clean data splits the old joint claim:

| Axis | mean AUC (5 seeds) | spread vs seed std | significance | verdict |
|---|---|---|---|---|
| **obs_len 8/16/30** (band [30,60]) | 0.931 / 0.933 / 0.937 | 0.0058 < ±0.007 | all pairwise p>0.21; Kruskal 0.566 | **insensitive — confirmed** |
| **TTE 30/45/60** (single-point) | 0.960 / 0.948 / 0.919 | 0.0417 ≫ ±0.005 | every pairwise p≤0.008; Kruskal 0.002 | **significant decline — OVERTURNED** |

**Observation window: insensitive, old single-seed claim survives** (obs_len=16
safe). **Prediction horizon: significant, monotonic decline** — AUC drops 0.960
(1.0 s) → 0.948 (1.5 s) → 0.919 (2.0 s) as the model predicts further ahead. This
**corrects** the old *leaky* single-seed "insensitive to TTE": on leaky data the
model detected in-progress crossings regardless of nominal horizon (Issues 1–2);
on leak-free `crossing_point`-anchored data it degrades gracefully — the expected,
sensible behaviour (a point *for* the model's validity, not against it). The
obs16/[30,60] cell on MPS reproduces the existing CPU baseline (0.933 ± 0.007 vs
0.932 ± 0.011) — backend cross-check. ⚠ Single-point TTE cells use a smaller,
single-horizon test set (N≈500), so their absolute AUCs (0.92–0.96) are not
comparable to the band-based headline 0.932 — the result is the **relative trend**.
Full tables + figure in `06_multiseed_ablation_summary.md` / `06_ablation_figure.png`.

**Matched-cohort TTE control (`06b_matched_track_tte.py`) — makes the horizon
result publication-defensible.** The single-point cells use *nested* pedestrian
sets (TTE=30 needs L≥46, TTE=60 needs L≥76), so TTE=30 carries 48 extra short/harder
tracks — a reviewer can call the decline a sample artifact. The control restricts
all three horizons to the **common TTE=60-eligible cohort** (identical peds + labels
in train/test, only the window slides; matched-tte60 reuses the existing run). The
decline is **essentially unchanged** (0.961/0.946/0.919; sample effect ≤0.002 AUC;
every pairwise paired-t p≤0.004; Kruskal 0.002) → the horizon effect is genuine, not
track-length eligibility. The window null is argued from **effect size / equivalence**
(spread 0.006 < seed std 0.007), not the underpowered n=5 p-value. See
`06b_matched_tte_report.md` / `06b_matched_tte_figure.png`.

---

## Issue 7 — Hidden-Size Ablation (Planned Day 11, Never Run) 🟠
**Risk level:** THESIS_PLAN.md promised it; absence is a gap in methodology.

**Problem:**  
`THESIS_PLAN.md` Day 11 specifies a hidden-size ablation {64, 128, 256}.
It was never run. The central capacity choice (hidden=128) is asserted, not
justified. A reviewer may ask: "Did you select hidden=128 because it was best,
or because you decided it first?"

**What to build:**  
`journal_prep/07_hidden_size_ablation.py` (or Kaggle notebook)  
- Train baseline BiLSTM with hidden ∈ {64, 128, 256}, seed 42  
- All other hyperparams held fixed  
- Report AUC / F1 / params per config  
- Output: `journal_prep/07_hidden_ablation_results.md`

**Success criterion:** Table justifying hidden=128 (either best, or "sufficient
— bigger gives no improvement at higher cost").

- [x] Write `07_hidden_size_ablation.py` (in `issue7_hidden_size/`; self-contained MPS
      harness, locked baseline contract, reports param counts)
- [x] Run — **locally on M4 GPU/MPS (~13 s/training, 15 trainings)**, not Kaggle;
      **multi-seeded** (5 seeds) rather than the plan's single seed 42, per the Issue-6
      lesson that single-seed ablation conclusions sit below seed noise
- [x] Report results + significance

### ✅ DONE (2026-06-26) — `issue7_hidden_size/`

hidden_dim ∈ {64,128,256} on the clean baseline data (`sequences_clean/`), 5 seeds,
everything else locked. hidden=128 reproduces the baseline (0.933 ± 0.007 vs 0.932 ±
0.011).

| hidden | params | AUC | F1 |
|---|---|---|---|
| 64 | 166 k | 0.927 ± 0.009 | 0.809 |
| **128 (baseline)** | **595 k** | **0.933 ± 0.007** | 0.828 |
| 256 | 2.24 M | 0.938 ± 0.003 | 0.835 |

**hidden=128 is justified — no size significantly beats it.** hidden=256 is nominally
+0.0045 AUC but **not significant** (paired-t p=0.34, Kruskal 0.12) at **3.8× the
params**; hidden=64 is no better (p=0.35). Honest nuance: a *mild, non-significant*
upward trend with capacity (0.927→0.933→0.938; spread 0.010 slightly > seed noise
0.006), so we don't claim saturation — only that nothing beats 128 significantly, so
128 is kept as the **accuracy/cost compromise** (the standard justification). These 15
runs are the hidden-size rows **Issue 8 reuses**. Full table + figure in
`07_hidden_size_results.md` / `07_hidden_size_figure.png`.

---

## Issue 8 — Hyperparameter Search (Grid Search) 🟠
**Risk level:** "No search = unjustified hyperparameters" is a standard critique.

**Problem:**  
All hyperparameters (lr, dropout, hidden size, batch size) are hand-set with no
documented search. A reviewer will ask why lr=1e-3, dropout=0.3, etc.

**What to build:**  
`journal_prep/08_hyperparam_search.py` (Kaggle — sklearn present for scoring)  
- Define a **coarse grid** (do not need Bayesian; grid is more transparent):  
  - `lr` ∈ {1e-3, 5e-4, 1e-4}  
  - `dropout` ∈ {0.2, 0.3, 0.5}  
  - `hidden` ∈ {64, 128, 256} (reuses Issue 7 runs)  
  - `num_layers` ∈ {1, 2}  
- Selection criterion: **validation AUC** (set05/06), test set NEVER touched.  
- Report best config + full grid results table.  
- Output: `journal_prep/08_grid_search_results.csv`, `08_grid_search_summary.md`

**Note:** If Bayesian optimization is preferred (supervisor request), replace the
grid with Optuna (TPE sampler, 50 trials). The important invariant is the same:
selection on val AUC, test touched once at the end with the winning config.

**Success criterion:** A written, documented search procedure with the selected
config clearly justified by val-AUC ranking.

- [x] Write `08_grid_search.py` (in `issue8_grid_search/`; full grid, val-only
      selection, multi-seed candidate confirmation, test touched once)
- [x] Run — **locally on M4 GPU/MPS (~25 min, full grid)**, not Kaggle
- [x] Report results and justify final config

### ✅ DONE (2026-06-26) — `issue8_grid_search/`

**Supervisor-requested section.** Transparent grid over lr{1e-3,5e-4,1e-4} ×
dropout{0.2,0.3,0.5} × hidden{64,128,256} × num_layers{1,2}. Inter-layer LSTM
dropout is inert at num_layers=1, so those cells merge → **36 distinct configs**.
**Leakage-proof protocol:** Stage 1 ranks all 36 at seed 42 by **val AUC** (test
never evaluated); Stage 2 multi-seeds (5) the top-5 + baseline, picks winner by
**mean val AUC**; Stage 3 trains winner + baseline ×5 and touches **test once**.

| | config | val AUC | test AUC (once) | params |
|---|---|---|---|---|
| winner | lr1e-4 / do0.2 / h256 / 2L | 0.969 ± 0.006 | **0.930 ± 0.005** | 2.24 M |
| baseline | lr1e-3 / do0.3 / h128 / 2L | 0.964 ± 0.004 | 0.929 ± 0.012 | 595 k |

**Verdict: the search confirms the hand-set baseline.** Winner beats baseline on
test by **Δ +0.0006, paired-t p=0.914 (n.s.)** at 3.8× params + 10× lower lr — so the
baseline is **statistically as good**, retained for efficiency; hyperparameters now
documented, not asserted. **Selection-noise control mattered:** the single-seed val
leader (lr1e-4/do0.3/h128) was *not* the 5-seed-mean winner (lr1e-4/do0.2/h256, the
most stable) — a single-seed grid would have chosen differently. The highest val
AUCs cluster on lr=1e-4 but don't carry to test (small skewed val set), supporting
the baseline lr=1e-3. Full 36-row grid in `08_grid_full.csv`; figure + verdict in
`08_grid_search_summary.md`.

---

## Issue 9 — Isolated Inference Latency Benchmark 🟠
**Risk level:** "~18 fps" conflates YOLO+ByteTrack+BiLSTM; BiLSTM latency is
unreported. Deployment claims without latency numbers are weak.

**Problem:**  
The only timing number in the project is "900 frames in ~50 s on MPS" which is
the full pipeline dominated by YOLO26-M. The BiLSTM (0.6 M params, 16×5 input)
latency in isolation has never been measured.

**What to build:**  
`journal_prep/09_inference_latency.py` (run locally on M4)  
- Load `runs/bilstm_baseline/best.pt`  
- Warm up with 100 forward passes (batch=1, input shape (1,16,5))  
- Time 1000 forward passes, report mean ± std latency in **ms/window**  
- Repeat for: batch=1, batch=8, batch=32 (simulating parallel tracks)  
- Repeat on CPU and MPS  
- Also: for comparison, time the obs_len=8 vs obs_len=16 vs obs_len=30 models
  (from ablation runs) — verify the "shorter window = lower latency" claim  
- Report pipeline breakdown:
  - YOLO26-M ms/frame (already measured implicitly: 50s/900f = 55 ms/frame)
  - ByteTrack ms/frame (estimate from track stage)
  - BiLSTM ms/window (new measurement)  
- Output: `journal_prep/09_latency_report.md`

**Success criterion:** Table of isolated BiLSTM latency on CPU + MPS; pipeline
breakdown table; "shorter window = lower latency" claim either confirmed with
numbers or dropped.

- [x] Write `09_inference_latency.py` (in `issue9_latency/`; proper MPS sync, warmup,
      CPU+MPS × batch sweep, obs_len sweep, YOLO + pipeline breakdown)
- [x] Run locally on M4 (inference only, ~4 min)
- [x] Report latency table and pipeline breakdown

### ✅ DONE (2026-06-26) — `issue9_latency/`

Inference only (no training); MPS timings synchronise inside each timed call.

**Isolated BiLSTM latency (the previously-missing number): 0.575 ms/window** (CPU,
batch 1) = ~58× inside a 30 fps budget (33.3 ms). Counter-intuitive but honest:
**CPU beats MPS at batch 1** (0.575 vs 1.647 ms) — GPU dispatch overhead dominates a
0.6 M-param model; MPS only wins when batching many tracks (batch 32: MPS 0.083 vs
CPU 0.135 ms/window). "Shorter window = lower latency" **confirmed** (obs_len 8/16/30
→ 0.32/0.56/1.01 ms CPU). **Pipeline is detection-bound:** YOLO26-M 33.7 ms (92.7%) +
ByteTrack ~1 ms + BiLSTM 1.6 ms (4.5%) → 27.5 fps on MPS; YOLO is ~20× the BiLSTM, so
the headline fps is a property of the detector, not the prediction model. Tables +
figure in `09_latency_report.md` / `09_latency_figure.png`.

---

## Issue 10 — GT-Box vs YOLO-Box AUC Drop (Demo Experiment) 🟠
**Risk level:** Phase 4 demo is qualitative-only; N=10 "AUC 1.000" is
not a result. The pipeline contribution is unquantified.

**Problem:**  
The key science of Phase 4 is: "how much does noisy detector output hurt the
BiLSTM prediction vs. using ground-truth boxes?" This gap has never been measured.
The reported "ped-level AUC 1.000 on 10 pedestrians" is too small (N=10) and a
perfect score actually hides whether the tracker is adding noise or not.

**What to build:**  
`journal_prep/10_gt_vs_detector_auc.py`  
- For the demo clips (video_0012, video_0016) on set03:  
  1. **GT-box path:** feed PIE ground-truth boxes into the BiLSTM (already the
     offline test — AUC 0.931 from `final.json`)  
  2. **YOLO-box path:** feed YOLO26-M + ByteTrack boxes into the BiLSTM for the
     same pedestrians (matched by IoU); use the predictions.csv already saved  
  3. Report AUC on the matched subset for both paths  
  4. Report ByteTrack quality: # unique IDs, # ID switches, # fragmented tracks  
- Output: `journal_prep/10_gt_vs_detector_results.md`

**Note:** The matched subset will be small (maybe 30–50 peds across both clips),
so report it as an *indicative* experiment, not a full benchmark — but it directly
quantifies the perception→prediction degradation.

**Success criterion:** A table showing GT-box AUC vs YOLO-box AUC for the same
pedestrians, + tracker quality metrics.

- [x] Write `10_gt_vs_detector_auc.py` (in `issue10_gt_vs_detector/`; segment YOLO+ByteTrack,
      IoU match, per-frame-best YOLO window, clean-BiLSTM dual scoring, tracker quality)
- [x] Run on matched clips (local, inference only ~4 min; YOLO dets cached)
- [x] Report degradation table

### ✅ DONE (2026-06-28) — `issue10_gt_vs_detector/`

Quantifies the perception→prediction gap on demo clips video_0012 + video_0016
(replaces the old qualitative "N=10, AUC 1.000"). For each clean GT window, YOLO26-M
+ ByteTrack is run over the needed frames; the YOLO-box window is the best-IoU
detection per frame (isolates box noise; ego-speed unchanged); both paths scored by
the clean BiLSTM. Indicative subset: **98 peds / 311 windows**.

| path | per-window AUC | per-ped AUC |
|---|---|---|
| GT boxes | 0.962 | 0.958 |
| YOLO boxes | 0.953 | 0.948 |
| **drop** | **+0.009** | **+0.010** |

**Prediction is robust to detector box noise** (drop ≈0.01, decision flips 3%, mean
IoU 0.75). **The weak links are perception, not prediction:** detector recall 88% of
peds (~12% never predictable — safety gap); **tracker fragmentation severe** — a
single ByteTrack ID covers a mean of only **39%** of a ped's frames, 59% of windows
have a competing ID (needs re-ID in deployment). These are detector/tracker
engineering gaps, separate from the BiLSTM. Table + scatter in
`10_gt_vs_detector_results.md` / `10_gt_vs_detector_figure.png`.

---

## Issue 11 — Clean Up Plan/Docs to Match Reality 🟡
**Risk level:** Minor, but supervisors and co-authors read these.

**Problem:**  
- `THESIS_PLAN.md` still lists the old (wrong) file numbering and the unrun
  Day 11 hidden-size ablation as a future item, making it look like a commitment.  
- `CODE_STATE.md` doesn't mention `journal_prep/` at all.  
- The demo N=10 "AUC 1.000" language in `PROGRESS_LOG.md` should be softened.

**What to do:**  
- Update `THESIS_PLAN.md` file-numbering section to reflect actual script names.  
- Mark Day 11 as "moved to journal_prep/07" in the plan.  
- Add an entry to `CODE_STATE.md` for the `journal_prep/` folder.  
- Soften "AUC 1.000 on 10 peds" to "illustrative qualitative check, N=10."

- [x] Update `THESIS_PLAN.md` — Day-11 hidden-size marked deferred to `journal_prep/issue7`;
      file-naming convention corrected to actual script names (07-reserved-for-demo quirk noted)
- [x] Update `CODE_STATE.md` — `journal_prep/` header updated; stale Issue-2 variant numbers
      fixed (bbox 0.753, attn 0.925 multi-seed); per-issue DONE entries for Issues 3–11 added
- [x] Update `PROGRESS_LOG.md` language — demo "AUC 1.000 on 10 peds" softened to an
      illustrative check, marked superseded by Issue 10 (N=98); headline = clean 0.932

### ✅ DONE (2026-06-28) — root docs now match reality

`THESIS_PLAN.md`, `CODE_STATE.md`, `PROGRESS_LOG.md` reconciled with the real
file numbering, the deferred-then-done hidden-size ablation, the completed
`journal_prep/` Issues 1–10, and the corrected clean headline (0.932) — no more
"AUC 1.000 / 0.931" leftovers in the narrative docs.

---

## Execution Order

```
1  → 2  (leakage audit must pass before spending time on anything else)
2  → 3  (know the protocol before comparing to baselines)
4  → 5  (CIs first, then LOSO — both are held-out eval questions)
6  → 7  (multi-seed ablations + hidden-size can run in parallel on Kaggle)
8       (grid search uses hidden-size results from 7)
9       (latency — independent, run locally any time)
10      (demo experiment — independent, run locally after demo deps confirmed)
11      (doc cleanup — do last, after all numbers are final)
```

---

## Output Files Expected (all in `journal_prep/`)

| File | Produced by |
|---|---|
| `01_leakage_audit.py` + `01_leakage_report.md` | Issue 1 |
| `02_build_sequences_pie_protocol.py` (if Option A) | Issue 2 |
| `03_baseline_comparison.md` | Issue 3 |
| `04_bootstrap_ci.py` + `04_bootstrap_ci_results.md` | Issue 4 |
| `05_loso_cv.py` + `05_loso_results.csv` + `05_loso_results.md` | Issue 5 |
| `06_multiseed_ablations.ipynb` + CSVs + `06_multiseed_ablation_summary.md` | Issue 6 |
| `07_hidden_size_ablation.py` + `07_hidden_ablation_results.md` | Issue 7 |
| `08_hyperparam_search.py` + `08_grid_search_results.csv` + summary | Issue 8 |
| `09_inference_latency.py` + `09_latency_report.md` | Issue 9 |
| `10_gt_vs_detector_auc.py` + `10_gt_vs_detector_results.md` | Issue 10 |

---

## Key Invariants to Preserve Across All New Work

1. **Test set (set03) is touched exactly once per experiment**, at the end, on
   the best-val checkpoint. Never select hyperparameters on test.  
2. **Normalization is train-only**: `mean/std` computed on train split, applied to
   val and test. No recomputing per-fold without saving the fold's own stats.  
3. **`POS_WEIGHT = 1.44`** stays fixed everywhere unless explicitly varying it as
   the ablated factor.  
4. **`set_seed(42)` is the canonical single-seed run** for all issues that need a
   reference point consistent with Day 5.  
5. **`weights_only=False`** when loading `best.pt` (numpy-scalar in checkpoint).  
6. **`threshold = 0.5`** for all binary classification decisions.

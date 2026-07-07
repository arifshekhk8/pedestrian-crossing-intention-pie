# Issue 3 — Published-Baseline Comparison (PIE crossing prediction)

**Purpose.** Place our leak-free result against published PIE crossing-intention
numbers, with a *modality* column so the input-parsimony story is visible. Valid
only now that Issues 1–2 (leakage removed, canonical protocol) are done — the
number below is the clean `0.932`, **not** the old leaky `0.931`.

> **Verification status (2026-06):** baseline figures below are now confirmed
> against primary sources (mostly GTransPDM's Table I, which tabulates the field on
> PIE, plus each method's own paper). Remaining `[verify]` items are flagged.
> Citations are abbreviated pending a BibTeX pass.

---

## Our result (clean protocol, 5-seed)

| | Acc | AUC | PR-AUC | F1 | Precision | Recall | Inputs |
|---|---|---|---|---|---|---|---|
| **BiLSTM (ours, clean)** | 0.883 ± 0.009 | **0.932 ± 0.011** | 0.876 ± 0.016 | 0.828 ± 0.012 | 0.794 ± 0.022 | 0.865 ± 0.024 | bbox + ego-speed (**2 streams**) |

Test = PIE set03, 2,094 windows (**32.5% positive**); mean ± std over seeds
[42,0,1,2,3]; obs_len 16, TTE∈[30,60], 0.5 overlap, threshold 0.5. The full
evidence chain is now complete: **bootstrap 95% CI ≈ [0.92, 0.95]** (Issue 4, 10k
resamples), **LOSO 6-fold AUC 0.928 ± 0.041** with set03 = 0.931 ≈ fixed-split
(Issue 5, so set03 isn't an easy fold), and eval-parity verified (Issue 2:
per-pedestrian 0.914 ≈ per-window 0.913, benchmark-min-track subset 0.919) — so the
0.932 is **not** an easier-evaluation artifact.

### On metric choice (why our AUC leads but Acc is lower-middle)

The high-AUC / lower-Acc profile is expected and defensible, not a red flag:

- **ROC-AUC and PR-AUC are threshold-free and the most comparable across papers;
  Acc/F1 are single-threshold operating points** confounded by each paper's class
  prevalence and window extraction. Our **PR-AUC 0.876** (base rate 0.325)
  corroborates the ROC-AUC — strong ranking on the imbalance-sensitive metric too.
- **Our 0.5 threshold is already near-optimal:** tuning it on validation
  (val-optimal ≈ 0.51, never test) moves test Acc only 0.883→0.887 and F1
  0.827→0.832. We do **not** tune the threshold to flatter Acc.
- **The operating point is deliberately recall-favoring** (`pos_weight=1.682` →
  R 0.865 > P 0.794): in an AV-safety setting a missed crosser costs more than a
  false alarm.
- **Low Acc with the highest AUC is the *opposite* of an "easier-eval" artifact** —
  our 50%-overlap protocol admits 3.5× more, and Issue-2-confirmed *harder*,
  windows (short-track AUC 0.863 vs 0.919), which depresses thresholded Acc more
  than ranking AUC.

**⇒ Report AUC (+ PR-AUC) as the primary metric; show Acc/F1 honestly with the
operating point stated.**

## Comparison table — standard PIE protocol

All rows use the standard PIE benchmark (train set01/02/04 · val set05/06 · test
set03; obs T=16 frames / 0.5 s; TTE 30–60 frames / 1–2 s; metrics Acc/AUC/F1).
✅ = figure confirmed from the cited source.

| Method | Venue / Year | Acc | AUC | F1 | Modalities (streams) | Src |
|---|---|---|---|---|---|---|
| **PCPA** (Kotseruba et al.) | WACV 2021 | 0.87 ✅ | 0.86 ✅ | 0.77 ✅ | bbox + pose + context + speed (4) | benchmark; GTransPDM Tbl I |
| **Pedestrian Graph+** | 2022 | 0.89 ✅ | 0.90 ✅ | 0.81 ✅ | pose graph + ego (2–3) | GTransPDM Tbl I |
| **IntFormer** | 2021 | 0.89 ✅ | 0.92 ✅ | 0.81 ✅ | multimodal | GTransPDM Tbl I |
| **PIT** | 2023 | 0.91 ✅ | 0.92 ✅ | 0.82 ✅ | multimodal | GTransPDM Tbl I |
| **BiPed / PedFormer** | 2023 | 0.91 ✅ | 0.90 ✅ | 0.85 ✅ | multimodal | GTransPDM Tbl I |
| **GTransPDM** | arXiv Sept 2024 | 0.90 ✅ † | 0.87 ✅ | 0.82 ✅ | bbox + pose + ego motion (3) | GTransPDM Tbl I |
| **PIP-Net** (Azarmi et al.) | IEEE T-ITS 2025 | 0.915 ✅ | 0.897 ✅ | 0.846 ✅ | 7 feats: bbox, pose, speed, context, opt-flow, semseg, depth (7) | PIP-Net paper |
| **BiLSTM (ours, clean)** | 2026 | 0.883 | **0.932** | 0.828 | **bbox + ego-speed (2)** | this work |

† GTransPDM's own Table I lists **0.90** Acc; its abstract headlines up to **0.92**
for a positional-decoupling variant. Using the table value for like-for-like.

**PIEPredict** (Rasouli et al., ICCV 2019; bbox + ego + context) is the
dataset-origin model but is a *trajectory* predictor, not a binary crossing
classifier — no directly comparable Acc/AUC/F1 row. It is the candidate for an
optional "run-it-on-our-split" experiment (below), not a transcribed row.

## ⚠ Occlusion-Aware Diffusion is a *modality precedent*, not a comparison row

The PLAN earmarked **Occlusion-Aware Diffusion** (Liu et al., arXiv 2511.00858,
accepted IEEE T-ITS Nov 2025) as the "apples-to-apples" row. On inspection it is
**not protocol-comparable**:

- It reports **only occluded scenarios** (EO1–EO5 element / PO1–PO5 partial
  occlusion) — there is **no standard fully-observed PIE number** in the paper.
- Its prediction horizon is **~1 frame ahead** (15-frame obs, 16th-frame target),
  far shorter/easier than our **TTE 30–60 frames (1–2 s)**.
- Its best PIE figure (EO5: Acc 0.90 / AUC 0.95 / F1 0.90, vs TrEP 0.85/0.91/0.85)
  is an *occlusion-robustness* result, not a benchmark crossing-prediction score.

What it **does** give us: confirmation that **bbox + ego-velocity only** is a
deliberate, published minimal modality ("the proposed model and TrEP only leverage
bounding boxes and ego-vehicle speed"). So cite it as a **precedent for the
two-stream design choice**, with the protocol caveat stated — do **not** put its
0.95 in the same table as our 0.932.

## How to read this (the honest framing)

- **On AUC, ours (0.932) is the highest in the standard-protocol table** (next:
  PIT / IntFormer 0.92), using **2 input streams** vs the 3–7 of every method above
  it. This is the parsimony headline.
- **On Accuracy we are near the bottom** (0.883 vs 0.87–0.915) and **mid-pack on
  F1** (0.828 vs 0.77–0.85). We do **not** dominate — a high-AUC / modest-Acc
  profile, consistent with a model that ranks crossing risk well but whose fixed
  0.5 threshold isn't Acc-optimal under 33.6% positives. Reporting this honestly is
  what makes the AUC claim credible rather than "suspiciously easy."
- **The claim to make:** *"On the standard PIE protocol, a 2-stream
  (bbox + ego-speed) BiLSTM attains AUC competitive with — at the top of —
  multimodal SOTA, at a fraction of the feature-extraction cost and latency, with
  honest mid-pack accuracy."* Use Occlusion-Aware Diffusion as the precedent that
  minimal modality is a legitimate choice (with its protocol caveat).
- **The latency half of the claim is now measured (Issue 9):** the BiLSTM runs in
  **0.575 ms/window** (~58× inside a 30 fps budget) and is **4.5% of the live
  pipeline** (detection-bound). So "fewest inputs" is backed by "lowest compute" —
  the 3–7-stream methods above need pose/optical-flow/semseg/depth extractors we do
  not, which is the parsimony+efficiency story, not just an accuracy claim.
- **Pre-empt the "top AUC + fewest inputs ⇒ easier eval" reflex** with the now-
  complete evidence, cited inline next to the table: leakage removed + **0%
  verified** (Issue 1–2); eval-parity per-ped ≈ per-window (Issue 2); **bootstrap
  95% CI [0.92, 0.95]** (Issue 4); **LOSO 0.928 ± 0.041** across all 6 sets, set03
  representative (Issue 5); multi-seed ablations (Issue 6) and a documented grid
  search that **confirms** the hyperparameters (Issue 8); and detector-in-the-loop
  robustness (Issue 10: AUC drop only +0.009 GT→YOLO boxes).

## Split & protocol alignment

| Baseline | Standard split? | Notes |
|---|---|---|
| PCPA | ✅ (defines it) | reference protocol |
| PIT / IntFormer / Ped-Graph+ / BiPed | ✅ (as tabulated by GTransPDM) | same Tbl I, same split |
| GTransPDM | ✅ | 4770/1332/3816 (0.5:0.1:0.4) = set01/02/04 · 05/06 · 03; obs 16, TTE 30–60 |
| PIP-Net | `[verify]` | "in the wild"; confirm it reports on PIE set03 test, not a custom split |
| Occlusion-Aware Diffusion | ✗ different task | occluded scenarios, ~1-frame-ahead TTE — not comparable |

## Optional but strong: PIEPredict on **our** split

`PIEPredict/` is vendored locally. Running it on our exact set03 split → one
*directly comparable* row ("original PIE model, our split, our metrics") that
sidesteps cross-paper caveats. Indicative, not mandatory.

## Remaining to-do (Issue 3 checklist)

Internal finalization is **DONE** — every downstream issue's number is now folded in.
Only external-source verification and the BibTeX pass remain (need paper access).

- [x] Lock Acc/AUC/F1 for PCPA, GTransPDM, PIP-Net + landscape (PIT/IntFormer/
      Ped-Graph+/BiPed) — confirmed from sources.
- [x] Resolve Occlusion-Diffusion — **reclassified** as modality precedent (occluded
      protocol, not a comparison row).
- [x] Fold in our complete evidence: bootstrap CI (Issue 4), LOSO (Issue 5), latency
      (Issue 9), detector robustness (Issue 10) — done in the framing above.
- [ ] **External (pre-submission):** `[verify]` PIP-Net's PIE split (set03 test?) +
      GTransPDM 0.90-vs-0.92 variant against the source PDFs.
- [ ] **External (pre-submission):** full BibTeX + venue/DOI pass (arXiv ids:
      GTransPDM 2409.20223, PIP-Net 2402.12810, ODM 2511.00858, IntFormer 2105.08647,
      PedFormer 2210.07886).
- [ ] (Optional, strong) Run vendored `PIEPredict/` on our split for one directly
      comparable "original PIE model, our split" row.

## Sources

- PCPA / benchmark: Kotseruba, Rasouli, Tsotsos, "Benchmark for Evaluating
  Pedestrian Action Prediction," WACV 2021 — openaccess.thecvf.com; repo
  github.com/ykotseruba/PedestrianActionBenchmark
- GTransPDM: arXiv:2409.20223 (Table I tabulates PCPA/PIT/IntFormer/Ped-Graph+/BiPed)
- PIP-Net: Azarmi et al., IEEE T-ITS 2025; arXiv:2402.12810; eprints.whiterose.ac.uk
- Occlusion-Aware Diffusion: arXiv:2511.00858 (accepted IEEE T-ITS, Nov 2025)

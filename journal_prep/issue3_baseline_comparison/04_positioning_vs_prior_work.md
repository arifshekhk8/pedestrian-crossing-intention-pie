# Issue 3 (companion) — Positioning vs prior work: their limitations → our response

**Why this file.** A baseline *table* shows we're competitive; a journal reviewer
also wants the *narrative*: what does each recent work get wrong or leave open, and
how does our design address it. This is the "Related Work / Discussion" backbone.

> **Status: FINAL (2026-06-28).** Issues 1–10 are all complete, so every "our
> response" cell now carries a measured number. The only items left are external
> (verify two baselines' protocol details against their PDFs + BibTeX) and the
> writing pass (compress into Related-Work/Discussion prose). Per-paper limitations
> needing in-depth protocol reading are still marked `[verify]` — do not assert a
> flaw we haven't confirmed.

## Per-baseline: limitation → our response → evidence

| Prior work | Its limitation / open issue | How our work responds | Evidence (issue) |
|---|---|---|---|
| **PCPA** (WACV'21) | Heavy multimodal: 3D-CNN over RGB + pose + context + speed (4 streams); oldest anchor, AUC 0.86 | Match/beat AUC with **2 streams**, no RGB/pose encoder; real-time — **BiLSTM = 0.575 ms/window**, ~58× inside a 30 fps budget; pipeline is detection-bound (BiLSTM only 4.5% of per-frame cost) | Issue 3 table; **Issue 9 ✅ (latency)** |
| **GTransPDM** (2024) | Requires **skeleton/pose estimation + graph** — extra model, degrades at distance/occlusion; reports near-perfect acc at TTE=0 (at onset) `[verify favorable-TTE]` | No pose pipeline; we evaluate strictly **pre-onset** (TTE∈[30,60]) on leak-free windows | Issue 1–2 (pre-onset, leak-free); Issue 3 |
| **PIP-Net** (T-ITS'25) | **7 modalities** incl. optical flow, semantic segmentation, depth — prohibitive feature pipeline, many failure points, not real-time | 2 cheap streams (bbox + ego-speed); competitive AUC at a fraction of compute (BiLSTM 0.575 ms/window) | Issue 9 ✅ (latency); Issue 3 |
| **Occlusion-Aware Diffusion** (T-ITS'25) | Short horizon (**~1 frame ahead**), occlusion-focused, diffusion sampling is expensive; no standard full-horizon benchmark number | Genuine **1–2 s** prediction horizon; lightweight single forward pass | Issue 3 note; Issue 9 |

## Field-wide methodological gaps we close (the stronger story)

| Common gap in PIE-prediction papers | Our response | Evidence (issue) |
|---|---|---|
| **Temporal leakage not checked** — windows often anchored at track end, can overlap frames where the pedestrian is *already crossing* | Explicit leakage audit + re-anchor at PIE `crossing_point`; **0% leakage verified** | **Issue 1–2** (our most distinctive contribution) |
| Single **fixed split** (set03 test only) | Leave-one-set-out across all 6 sets | Issue 5 |
| **Point estimates**, no confidence intervals | 10k-bootstrap 95% CIs on test AUC | Issue 4 |
| **Single-seed** ablations reported as conclusions | Multi-seed (5-seed) mean ± std + significance tests, which *split* the old joint claim: window length insensitive (effect-size/equivalence: spread 0.006 < seed noise 0.007), but prediction-horizon AUC declines significantly (0.960→0.948→0.919 @1.0→2.0s) — a sensible degradation the leaky single-seed run had masked, and **confirmed on a matched cohort** (06b: same peds at all 3 horizons, decline unchanged, sample effect ≤0.002) so it is not a track-length artifact | **Issue 6** (done; and done for the model table) |
| **Ground-truth boxes assumed** at inference | Detector-in-the-loop (YOLO→ByteTrack) degradation measured (Issue 10): prediction **robust to box noise** (AUC drop only +0.009/+0.010, 3% decision flips on 98 peds); honestly flags the real pipeline weak links — detector recall 88% and severe ByteTrack identity fragmentation (track purity 39%) — as detector/tracker engineering gaps, not prediction-model flaws | **Issue 10 ✅** |
| Architecture / hyperparameters **unjustified** | Hidden-size ablation (Issue 7: 64/128/256, hidden=128 kept — 256 n.s. at 3.8× params) **+ a documented 36-config grid search** (Issue 8) with leakage-proof val-only selection (test touched once): the search **confirms the hand-set config** — its val-winner beats baseline on test by only Δ+0.0006 (p=0.91, n.s.). Hyperparameters justified by search, not asserted | Issue 7 ✅ / Issue 8 ✅ |

## Our distinctive contribution (one paragraph for the intro/discussion)

*A leakage-free, canonical-protocol re-evaluation showing that **bbox + ego-speed
alone** (2 streams) reaches **AUC 0.932 [0.92–0.95]** — competitive with, at the top
of, recent multimodal SOTA on PIE — at **0.575 ms/window** (4.5% of the live
pipeline; the 3–7-stream baselines need pose/flow/semseg/depth extractors we do not),
with full statistical rigor (bootstrap CIs, LOSO across all 6 sets, multi-seed
ablations, a documented grid search that confirms the hyperparameters) and deployment
realism (detector-in-the-loop: AUC drops only +0.009 from GT to YOLO boxes). We trade
raw accuracy for parsimony, recall-favoring safety behaviour, and reproducibility.*

## Our own limitation (state it before a reviewer does)

The AUC lead is substantially carried by **ego-speed**, which partly encodes the
ego-driver's anticipation (the instrumented car slows for expected crossers) — a
legitimate inference-time signal but not purely vision-based. Disclose in
Limitations; candidate **speed-perturbation robustness check**. (See
`../issue2_clean_protocol/05_variant_comparison.md`.)

## Remaining (external + writing only)

- [x] Fill the "our response" column with the actual Issue 4/5/6/7/8/9/10 numbers — **done**.
- [ ] `[verify]` each per-paper limitation against the source PDF (don't assert
      unverified flaws); confirm whether any baseline shares our Issue-1 leakage
      (a strong point if so). *Needs paper access.*
- [ ] Compress into 1–2 Related-Work/Discussion paragraphs + this matrix as a
      figure/table when drafting the manuscript.

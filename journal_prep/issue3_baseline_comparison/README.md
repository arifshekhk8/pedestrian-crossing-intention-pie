# Issue 3 — Published-Baseline Comparison

Places our clean result (AUC **0.932 ± 0.011**, bbox + ego-speed) in the context
of published PIE crossing-prediction numbers, with a modality column so the
two-stream parsimony story is visible. Only valid after Issues 1–2 (leakage
removed, canonical protocol). Full plan + framing: [`../PLAN.md`](../PLAN.md)
(Issue 3).

## Files

| File | What it is |
|---|---|
| `03_baseline_comparison.md` | **FINAL (internal)** comparison table (Acc/AUC/PR-AUC/F1 + modalities + venue) + metric-choice + honest framing, with the complete evidence chain (CI, LOSO, latency, detector robustness) folded in |
| `04_positioning_vs_prior_work.md` | **FINAL (internal)** matrix: each baseline's limitation → our response → evidence — every "our response" cell now carries a measured Issue-4–10 number |

## Status

**Finalized internally (2026-06-28)** now that Issues 1–10 are complete. Figures
verified against primary sources (mostly GTransPDM's Table I); Occlusion-Aware
Diffusion reclassified as a modality precedent (occluded-only, ~1-frame TTE), not a
comparison row. The "ours" framing now folds in bootstrap CI (Issue 4), LOSO (Issue
5), latency (Issue 9) and detector-in-the-loop robustness (Issue 10). **Only external
items remain** (need paper access, left for the manuscript pass): `[verify]` PIP-Net's
PIE split + GTransPDM 0.90/0.92 variant, the BibTeX/DOI pass, and the optional
`PIEPredict/`-on-our-split row. Checklist at the bottom of `03_baseline_comparison.md`.

## The one-line claim

On the standard PIE protocol, our 2-stream (**bbox + ego-speed**) BiLSTM attains
the **highest AUC in the comparison table (0.932)** with the fewest inputs —
honest mid-pack Acc (0.883). Minimal modality is a deliberate choice (precedent:
Occlusion-Aware Diffusion, bbox+ego only).

# Issue 3 — Published-Baseline Comparison

Places our clean result in the context of published PIE crossing-prediction numbers,
with a modality column so the two-stream parsimony story is visible. Only valid after
Issues 1–2 (leakage removed, canonical protocol). Full plan + framing:
[`../PLAN.md`](../PLAN.md) (Issue 3). **Metric order: F1 → Acc → AUC** (supervisor
directive; implemented 2026-07-13 throughout `03_baseline_comparison.md`).

**Update (2026-07-12):** the supervisor-requested `../../transformer/` extension found
a Transformer encoder (same 2-stream input, staged 78-config search) that measurably
beats the BiLSTM **on AUC** (0.950 vs 0.932, 10k paired bootstrap ΔAUC CI excludes 0).

**Update (2026-07-13, F1-first + source-verification audit):**
- The F1-first program (`../../f1_optimization/`) added two rows: **BiLSTM-F1
  0.897/0.940/0.844** and **Transformer-F1 0.896/0.947/0.847** (Acc/AUC/F1 @0.5) —
  and on F1 the two families are a statistical **TIE**, so the AUC win is
  metric-specific.
- Table corrected against primary sources: the old "BiPed / PedFormer 0.85" row
  misattributed BiPed's numbers — **PedFormer is 0.93 Acc / 0.90 AUC / 0.87 F1**
  (own Table I, default split), now a separate row and the F1/Acc ceiling of the
  table; **PIP-Net removed** (its own paper states a custom random ~50/40/10 split —
  context citation only); **GTransPDM w/o-pose (0.92/0.90/0.86)** added as the
  closest published 2-stream cousin. The standard-protocol F1 band is **0.77–0.87**.

## Files

| File | What it is |
|---|---|
| `03_baseline_comparison.md` | **FINAL (internal)** — the lean, **publication-bound** table (Acc/AUC/F1 + modalities + venue, F1-first framing, own-paper-verified rows only) + metric-choice section + honest framing, with the complete evidence chain (cluster CI, LOSO, latency, detector robustness) folded in |
| `05_master_comparison_table.md` | **Comprehensive INTERNAL reference** (for-understanding) — a superset with every column (Acc/AUC/F1/**Precision/Recall/Params/Latency/hyperparameters**) and **new 2024–2026 methods** (MFT, Faster-PCPNet, PedCMT, RAIDN, LSOP-Net, GTransPDM, …). Keeps PIP-Net + PIEPredict flagged **⚠ CAN REMOVE** (off-protocol). Every row carries a trust flag (✅ own-paper / ◻ secondary-transcription / ⚠ off-protocol); ◻/⚠ rows must be first-hand verified before entering the publication table |
| `04_positioning_vs_prior_work.md` | **FINAL (internal)** matrix: each baseline's limitation → our response → evidence — every "our response" cell carries a measured Issue-4–10 number |

## Status

**Finalized internally (2026-06-28; F1-first + source corrections 2026-07-13).**
Remaining external items (need paper access, left for the manuscript pass): verify
PIP-Net's *published* T-ITS version numbers (context citation), IntFormer vs its own
paper (single-source row), Ped-Graph+/BiPed configs (flagged by GTransPDM itself),
the BibTeX/DOI pass, and the optional `PIEPredict/`-on-our-split row. Checklist at
the bottom of `03_baseline_comparison.md`.

## The one-line claim (F1-first)

On the standard PIE protocol, our 2-stream (**bbox + ego-speed**) models reach
**F1 0.844–0.847** — within 0.02–0.03 of the multimodal ceiling (PedFormer 0.87,
multitask, many streams) — while holding the **highest AUC in the table**
(0.940–0.950) and honest mid-band Acc (0.896–0.897), at a fraction of the
feature-extraction cost. Once both of our architecture families receive identical
F1-first optimization they are statistically indistinguishable on F1 — the parsimony
finding is about the input signal, not an architecture artifact.

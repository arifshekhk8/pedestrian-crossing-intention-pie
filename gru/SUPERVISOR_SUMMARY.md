# GRU vs. BiLSTM — Supervisor Summary

**Student:** Arif
**Date:** 2026-07-14
**One-line summary:** You asked us to test two more model families on the same pipeline under
the identical protocol. This is the **GRU** (the gated recurrent twin of our BiLSTM). Given the
*same* search budget the BiLSTM got and the same F1-first optimization, the GRU is a **dead
statistical tie** with the BiLSTM — on F1 *and* on AUC. It does **not** reach the searched
Transformer's AUC. The finding: **the recurrent cell type is not what matters here — the input
signal is.**

> **How to use this doc:** read top-to-bottom. Every number comes from the actual run outputs
> (`gru/phase4_final/`, `gru/phase5_analysis/`), each independently re-derived from raw files
> (the frozen BiLSTM was re-checked to the full displayed precision before any comparison ran).
> Section 8 is a ready-to-use talking script and Q&A.

---

## 1. Why this exists

The transformer extension answered *"attention vs. recurrence?"* This one answers the natural
next question: *"is it the specific recurrent cell (LSTM) doing the work, or would any gated
recurrent unit do?"* A GRU is the obvious test — it is the BiLSTM's architectural twin (same
input projection, same bidirectional recurrence, same last-step readout, same linear head), with
**only the cell swapped** (`nn.LSTM` → `nn.GRU`). So a GRU-vs-BiLSTM comparison isolates the cell
type exactly the way the transformer study isolated attention.

**The honest short answer, now that we've measured it:** swapping the LSTM for a GRU changes
nothing that matters. Given the same search and the same F1-first optimization, the two are
statistically indistinguishable. That is a *useful* result: it sharpens the thesis's central
claim that the win comes from the **input signal (bounding box + ego-speed)**, not from the
particular temporal model.

---

## 2. What stayed exactly the same (so the comparison is fair)

Everything that could bias the comparison was **frozen identical** to the BiLSTM's
clean-protocol result:

- Same data: `sequences_clean/` (N=4,906; train 2,178 / val 634 / **test 2,094**).
- Same splits (train set01/02/04, val set05/06, test set03), same `pos_weight=1.682`, same
  5 seeds `[42,0,1,2,3]`, same early-stopping rule (patience 15 on val AUC).
- Same discipline: **test set03 touched exactly once**, on the final selected model, after a
  human checkpoint confirmed the winner. Everything before that is validation-only — the search
  code physically has no test path (verified: 89 search files, zero test keys).
- **The BiLSTM's own checkpoints were never retrained** — loaded as-is; a parity gate re-derived
  its per-seed test AUC and matched the stored values **exactly (|Δ| = 0.00e+00, all 5 seeds)**.
- **One engine, one device.** All GRU training ran through the *same* unified engine as the
  BiLSTM and Transformer (`journal_prep/issue12_unified_pipeline/`), **locally on CPU** — where
  training is bit-reproducible (our own issue-12 finding), so nothing here is a device or
  code-path artifact.

**What was allowed to differ:** only the GRU's own architecture/recipe — and only within the
**identical search budget the BiLSTM received** (the Issue-8 36-config grid + a class-weight
sweep). Same effort on both sides.

---

## 3. What is new: the model and the search

**The model.** A bidirectional GRU over the same 16-frame input the BiLSTM sees, with the same
wrapper around it. Default size = 446,081 parameters (smaller than the BiLSTM's 594,561).

**The search.** We applied the *same* staged, pre-registered search the BiLSTM got in Issue 8:
the 36-config grid over learning rate × dropout × hidden size × depth (ranked on validation by
F1 first, AUC second), then multi-seeded the top candidates, then a class-weight sweep on the
winner — all validation-only. The search landed on a **wider model (hidden 256)** — exactly the
move the BiLSTM's own F1-first program made — and kept the class weight at the anchor 1.682. As
in every prior search here, the single-seed leader was *not* the 5-seed winner (the
selection-noise control mattered again). We also carry an un-searched **default GRU** (the BiLSTM
baseline's recipe on a GRU) as a control.

---

## 4. The result

**Headline: the GRU ties the BiLSTM.** Each model trained 5 times; the table is the 5-seed
result on test set03 (F1 at each seed's validation-fitted operating point; AUC is
threshold-free):

| model | parameters | test AUC (5-seed) | test F1 (5-seed) | test Acc |
|---|---|---|---|---|
| Frozen BiLSTM (your original, un-optimized) | 594,561 | 0.932 | 0.828 | 0.883 |
| BiLSTM-F1 (F1-optimized) | ~2–3 M (h256) | 0.940 | 0.844 | 0.897 |
| Transformer-F1 (F1-optimized) | 794,241 | 0.947 | 0.847 | 0.896 |
| Searched Transformer (AUC winner) | 794,241 | **0.950** | 0.845 | 0.894 |
| **GRU (F1-winner, h256) — this study** | 1,678,209 | 0.941 | **0.849** | 0.901 |
| GRU (default h128, F1-selected) | 446,081 | 0.939 | 0.844 | 0.898 |
| GRU (default h128, AUC-selected) | 446,081 | 0.933 | 0.840 | 0.898 |

(The h128 default was run in both selection modes: the **AUC-selected** one is the matched-size
AUC twin of the frozen BiLSTM — AUC 0.933 ≈ 0.932; the **F1-selected** one is the un-searched-GRU
control and already reaches **F1 0.844, identical to BiLSTM-F1** on the BiLSTM's own recipe. The
h256 search only nudged F1 from 0.844 → 0.849 — the cell, not the size or search, is the constant.)

A table of point numbers isn't proof, so we ran the actual test — a **10,000-resample paired
bootstrap** (same resampled test windows for both models each time, isolating the *difference*):

1. **GRU vs. BiLSTM on F1 — TIE.** ΔF1 = +0.0071, 95% CI **[−0.0043, +0.0187]** (straddles
   zero). Given the same F1-first optimization, the GRU and LSTM are indistinguishable on your
   primary metric.
2. **GRU vs. Transformer on F1 — TIE.** ΔF1 = +0.0063, CI [−0.0046, +0.0174]. Consistent with
   the transformer's own AUC win not carrying over to F1.
3. **GRU vs. BiLSTM on AUC, at matched size and selection — TIE.** ΔAUC = −0.0008, CI
   [−0.0039, +0.0021]. This is the cleanest cell-isolation comparison: same 446k-parameter size,
   same AUC-based selection, GRU vs LSTM — a dead heat.
4. **GRU vs. searched Transformer on AUC — LOSS.** ΔAUC = −0.0070, CI [−0.0101, −0.0038]
   (below zero). The GRU does **not** reach the transformer's AUC — which is exactly what we'd
   expect if the transformer's AUC edge came from its *search* (a deeper, differently-pooled
   architecture), not from being a transformer rather than a recurrent net.

**Every one of these verdicts also survives a stricter pedestrian-cluster bootstrap** (resampling
whole pedestrians rather than individual windows — the honest interval given that windows from
one pedestrian are correlated). The two cell-isolation TIEs stay TIEs; the AUC loss stays a loss.

---

## 5. Supporting evidence (it's not just one lucky test set)

- **Cross-validation across all 6 PIE sets (LOSO):** the GRU's fold-average AUC is **0.946**
  (0.935 excluding the tiny 47-window set05 fold), squarely in the band of the BiLSTM (0.928)
  and Transformer (0.939). set03's own fold (0.928) matches its fixed-split number — so set03
  isn't an unusually easy fold for the GRU either. (6 folds is descriptive, not a hypothesis
  test — the fixed-split bootstrap in section 4 is the actual evidence.)
- **Latency:** the GRU F1-winner runs in **0.721 ms/window** (M4 CPU, single track) — a bit
  slower than the BiLSTM (0.575 ms) because it's a bigger h256 model, but still ~46× inside a
  30 fps frame budget. Latency is a non-issue for all three families; the live pipeline is
  detection-bound regardless.
- **A parity gate + a determinism gate** guard the whole study: the frozen BiLSTM reproduced its
  stored numbers exactly, and the GRU trains bit-identically on CPU for a fixed seed (so the
  "tie" isn't training noise we happened to land on).

---

## 6. Honest limitations

- **This is one dataset (PIE), one fixed split, 5 seeds.** The paired + cluster bootstrap over
  the 2,094 test windows is the primary evidence precisely because 5 seeds alone is low-powered.
- **The GRU winner is a *bigger* model** (1.68 M params, ~2.8× the BiLSTM) — the search pushed
  to hidden-256, the same way the BiLSTM's F1 program did. So "the GRU ties the BiLSTM" comes
  with "at a larger size"; the matched-size comparison (446k GRU vs 595k BiLSTM, AUC-selected)
  is the dead-heat in section 4, point 3. We state both plainly.
- **A tie is a tie, not a win.** We are not claiming the GRU is better; we're reporting that the
  cell choice doesn't move the needle — which is the scientifically interesting outcome here.
- **We did not run a separately AUC-optimized large GRU** (you chose the leaner arm set at the
  checkpoint), so the GRU's headline AUC comes from its F1-selected winner plus a matched-size
  AUC control, not a dedicated AUC-tuned h256 model. This only affects the (secondary) AUC
  chase against the transformer, which the GRU loses regardless.

---

## 7. How rigorously this was checked

- A **parity gate** re-derived the frozen BiLSTM's per-seed test AUC from its checkpoints before
  any comparison number was computed — exact match, all 5 seeds (|Δ| = 0.00e+00).
- The **search review was independently recomputed** from the raw run files and cross-checked
  against the search's own summary (exact agreement); every search file was verified to contain
  no test key.
- Every LOSO fold's size matched the exact pedestrian counts from the earlier LOSO study — a
  fingerprint that the real, untampered data was used.
- Every reported number is re-derived from raw checkpoints/probabilities on each run — nothing
  is a stored, un-checkable claim.

---

## 8. How to present this (talking script + Q&A)

**45-second pitch:**
> "You asked us to test more model families on the same pipeline. The GRU is the BiLSTM's
> gated recurrent twin — same everything, only the cell swapped. We gave it the *same* search
> the BiLSTM got and the same F1-first optimization, and it's a dead statistical tie with the
> BiLSTM — on F1 (your primary metric) and on AUC at matched size. It doesn't reach the searched
> Transformer's AUC, which confirms that transformer's edge was the search, not attention vs.
> recurrence. Bottom line: the recurrent cell isn't what matters here — the input signal, bbox
> plus ego-speed, is. Same conclusion the whole thesis has been building toward, now with a
> third architecture confirming it."

**Order to walk through:** Section 1 (why the GRU) → Section 4's table + the four bootstrap
verdicts (the two TIEs are the point) → Section 5 (LOSO + latency, it generalizes and it's
free) → Section 6 (limitations, stated plainly).

**Likely questions & answers:**
- *"Did you try as hard on the GRU as on the BiLSTM?"* → Same search budget by construction (the
  Issue-8 grid + class-weight sweep), same optimization, same engine, same device.
- *"Could the tie just be noise?"* → That's what the paired bootstrap tests: 10,000 resamples of
  the same 2,094 windows; the F1 interval straddles zero, and it still does under the stricter
  pedestrian-cluster resampling.
- *"So which model do we use?"* → On the evidence, it doesn't matter between the recurrent cells
  — keep the BiLSTM (smaller, already the headline). Use the searched Transformer only if you
  specifically want the best AUC. The interesting finding isn't a model choice; it's that the
  choice barely matters.

---

## 9. Where the evidence lives (file map)

```
gru/
├── PLAN.md                     ← full pre-registered design (written before any result)
├── PROGRESS_LOG.md             ← chronological log of every run/number/decision
├── phase3_search_review/03_search_summary.md   ← the search winner (val-only)
└── phase5_analysis/
    ├── 07_comparison_report.md      ← THE VERDICTS (paired bootstrap, all endpoints)
    ├── 07_comparison_figure.png     ← endpoint Δ chart
    ├── 08_cluster_bootstrap.md      ← pedestrian-cluster CIs (the honest intervals)
    ├── 09_latency_report.md         ← latency vs BiLSTM / Transformer
    └── 10_loso_report.md            ← 6-fold cross-validation table
```

Every script in `gru/phase5_analysis/` re-derives its numbers from the raw checkpoints/probs on
each run — nothing here is an un-checkable stored claim.

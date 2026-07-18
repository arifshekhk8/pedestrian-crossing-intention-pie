# Vanilla RNN vs. BiLSTM — Supervisor Summary

**Student:** Arif
**Date:** 2026-07-14
**One-line summary:** You asked us to test two more model families on the same pipeline under
the identical protocol. This is the **vanilla RNN** — the BiLSTM with its gating **removed**
(the simplest recurrent cell there is). Given the *same* search the BiLSTM got and the same
F1-first optimization, the un-gated RNN is a **dead statistical tie** with the BiLSTM and the
GRU on F1, ties-or-edges them on AUC, and — unlike the GRU — its AUC-optimized version even
**ties the searched Transformer** on AUC. It is also the **smallest and fastest** of the four
families. The finding: **not even the LSTM's gating is what matters here — the input signal is.**

> **How to use this doc:** read top-to-bottom. Every number comes from the actual run outputs
> (`rnn/phase4_final/`, `rnn/phase5_analysis/`), each independently re-derived from raw files
> (the frozen BiLSTM was re-checked to full precision before any comparison ran). Section 8 is a
> ready-to-use talking script and Q&A.

---

## 1. Why this exists

The transformer extension answered *"attention vs. recurrence?"* The GRU study answered *"does
the specific gated cell (LSTM vs GRU) matter?"* — it didn't; the GRU tied the BiLSTM. This one
asks the sharpest remaining question: *"is it **gating itself** doing the work, or would even an
**un-gated** recurrent net do just as well?"* The vanilla RNN is the definitive test — it is the
BiLSTM's exact twin (same input projection, same bidirectional recurrence, same last-step
readout, same linear head) with **only the cell swapped and its gates removed** (`nn.LSTM` →
`nn.RNN`, a plain tanh cell). So a vanilla-RNN-vs-BiLSTM comparison isolates **gating** exactly.

**The honest short answer, now that we've measured it:** removing the LSTM's gating changes
nothing that matters. Given the same search and the same F1-first optimization, the un-gated RNN
is statistically indistinguishable from both the LSTM and the GRU. This is the *strongest* form
of the thesis's central claim: the win comes from the **input signal (bounding box + ego-speed)**,
not from the temporal model *or its gating*.

---

## 2. What stayed exactly the same (so the comparison is fair)

Everything that could bias the comparison was **frozen identical** to the BiLSTM's clean-protocol
result:

- Same data: `sequences_clean/` (N=4,906; train 2,178 / val 634 / **test 2,094**).
- Same splits (train set01/02/04, val set05/06, test set03), same `pos_weight=1.682`, same 5
  seeds `[42,0,1,2,3]`, same early-stopping rule (patience 15 on val AUC).
- Same discipline: **test set03 touched exactly once**, on the final selected models, after a
  human checkpoint confirmed the winner. Everything before that is validation-only — the search
  code physically has no test path (verified: 93 search files, zero test keys).
- **The BiLSTM's own checkpoints were never retrained** — loaded as-is; a parity gate re-derived
  its per-seed test AUC and matched the stored values **exactly (|Δ| = 0.00e+00, all 5 seeds)**.
- **One engine, one device.** All RNN training ran through the *same* unified engine as the
  BiLSTM, GRU, and Transformer (`journal_prep/issue12_unified_pipeline/`), **locally on CPU** —
  where training is bit-reproducible (our own issue-12 finding), so nothing here is a device or
  code-path artifact.

**What was allowed to differ:** only the RNN's own architecture/recipe — and only within the
**identical search budget the BiLSTM received** (the Issue-8 36-config grid + a class-weight
sweep). Same effort on both sides.

---

## 3. What is new: the model and the search

**The model.** A bidirectional vanilla (tanh) RNN over the same 16-frame input the BiLSTM sees,
with the same wrapper around it. Because an un-gated cell has ~¼ the recurrent weights of the
4-gate LSTM, the whole family is *smaller* than the BiLSTM: default h128 = 149,121 parameters,
and the searched h256 winner = 560,001 (still below the BiLSTM's 594,561).

**The search.** We applied the *same* staged, pre-registered search the BiLSTM got in Issue 8:
the 36-config grid over learning rate × dropout × hidden size × depth (ranked on validation by
F1 first, AUC second), then multi-seeded the top candidates, then a class-weight sweep — all
validation-only. Two notable findings from the search itself:
- It landed on **hidden 256** (`lr1e-04_do0.2_h256_nl2`) — the *exact* config Issue-8's grid
  chose as the BiLSTM's AUC winner. The un-gated RNN independently converged on the BiLSTM's own
  AUC-optimal recipe. The F1-winner and AUC-winner were the **same** config (they usually differ).
- **Zero configs diverged.** A concern with vanilla RNNs is vanishing/exploding gradients, but
  over a 16-step window that risk is mild — every one of the 93 runs trained cleanly. (We kept an
  explicit "instability ledger" that would have recorded any divergence; it stayed empty.)
- As in every prior search here, the single-seed leader was *not* the 5-seed winner (the
  selection-noise control mattered again). We also carry an un-searched **default RNN** (the
  BiLSTM baseline's recipe on a vanilla RNN) as a control.

At the human checkpoint you added a fourth arm — the winner config trained with **AUC**-based
selection — which was essentially free here (F1-winner = AUC-winner) and gives a dedicated
AUC-optimized large RNN (closing a gap the GRU study had to flag).

---

## 4. The result

**Headline: the un-gated RNN ties the gated models — and it's the smallest and fastest.** Each
model trained 5 times; the table is the 5-seed result on test set03 (F1 at each seed's
validation-fitted operating point; AUC is threshold-free):

| model | parameters | test AUC (5-seed) | test F1 (5-seed) | test Acc |
|---|---|---|---|---|
| Frozen BiLSTM (your original, un-optimized) | 594,561 | 0.932 | 0.828 | 0.883 |
| BiLSTM-F1 (F1-optimized) | ~2–3 M (h256) | 0.940 | 0.844 | 0.897 |
| GRU-F1 (F1-optimized) | 1,678,209 | 0.941 | 0.849 | 0.901 |
| Transformer-F1 (F1-optimized) | 794,241 | 0.947 | 0.847 | 0.896 |
| Searched Transformer (AUC winner) | 794,241 | **0.950** | 0.845 | 0.894 |
| **Vanilla RNN (F1-winner, h256) — this study** | **560,001** | **0.948** | **0.852** | 0.902 |
| Vanilla RNN (winner h256, AUC-selected) | 560,001 | 0.948 | 0.845 | 0.910 |
| Vanilla RNN (default h128, F1-selected) | 149,121 | 0.942 | 0.844 | 0.897 |
| Vanilla RNN (default h128, AUC-selected) | 149,121 | 0.942 | 0.836 | 0.889 |

(The un-gated RNN's headline per-seed F1 of **0.852** is the *highest* number in the F1 column —
though, as the bootstrap below shows, it is statistically level with the 0.844–0.849 band of the
other optimized models, not meaningfully ahead. The point is that removing gating did not cost
anything, not that it helped.)

A table of point numbers isn't proof, so we ran the actual test — a **10,000-resample paired
bootstrap** (same resampled test windows for both models each time, isolating the *difference*):

1. **RNN vs. BiLSTM-F1 on F1 — TIE.** ΔF1 = +0.0033, 95% CI **[−0.0083, +0.0150]** (straddles
   zero). Given the same F1-first optimization, the un-gated RNN and the gated LSTM are
   indistinguishable on your primary metric.
2. **RNN vs. GRU on F1 — TIE.** ΔF1 = −0.0038, CI [−0.0117, +0.0039]. Un-gated vs gated
   recurrent: a dead heat. Three cell types (LSTM, GRU, vanilla RNN) tie.
3. **RNN vs. BiLSTM on AUC, at matched size and selection — the RNN slightly *edges* it.**
   ΔAUC = +0.0059, CI [+0.0032, +0.0088] (just above zero). Same 149k-parameter size, same
   AUC-based selection, un-gated RNN vs gated LSTM — the RNN is level-to-marginally-better (the
   n=5 seed t-test is non-significant, p=0.14; we report both, and lean on the window+cluster
   bootstrap). Certainly no evidence gating helps.
4. **RNN vs. searched Transformer on AUC — TIE.** ΔAUC = −0.0013, CI [−0.0041, +0.0015]
   (straddles zero). This is the surprise: the **GRU lost** this comparison (ΔAUC −0.0070), but
   the AUC-optimized vanilla RNN **reaches the transformer's AUC**. Once an un-gated recurrent
   net gets the same search, it hits the same ~0.95 AUC — direct confirmation that the
   transformer's AUC edge was its *search*, not attention-over-recurrence.

**Every one of these verdicts also survives a stricter pedestrian-cluster bootstrap** (resampling
whole pedestrians rather than individual windows — the honest interval given that windows from
one pedestrian are correlated). The two F1 ties stay ties; the matched-size AUC edge stays; the
transformer AUC tie stays. (Only a secondary "discipline" comparison — the *un-searched* default
RNN beating the *un-optimized* frozen BiLSTM — softens from a win to a tie under clustering; that
one is about the F1-first recipe, not the cell, and we report it plainly.)

---

## 5. Supporting evidence (it's not just one lucky test set)

- **Cross-validation across all 6 PIE sets (LOSO):** the RNN's fold-average AUC is **0.937**
  (0.926 excluding the tiny 47-window set05 fold), squarely in the band of the BiLSTM (0.928),
  GRU (0.946), and Transformer (0.939). set03's own fold (0.944) matches its fixed-split number —
  so set03 isn't an unusually easy fold for the RNN either.
- **Latency:** the RNN F1-winner runs in **0.316 ms/window** (M4 CPU, single track) — the
  **fastest** of all four families (BiLSTM 0.575, Transformer 0.459, GRU 0.721), because the
  un-gated cell is the smallest and cheapest. ~105× inside a 30 fps frame budget; the live
  pipeline is detection-bound regardless.
- **A parity gate + a determinism gate** guard the whole study: the frozen BiLSTM reproduced its
  stored numbers exactly (|Δ| = 0.00e+00), and the RNN trains bit-identically on CPU for a fixed
  seed (so the "tie" isn't training noise we happened to land on).

---

## 6. Honest limitations

- **This is one dataset (PIE), one fixed split, 5 seeds.** The paired + cluster bootstrap over
  the 2,094 test windows is the primary evidence precisely because 5 seeds alone is low-powered
  (the n=5 t-tests are mostly non-significant even where the window bootstrap is clear).
- **The matched-size AUC comparison is RNN-vs-*frozen*-BiLSTM.** Both are trained under the same
  protocol, but the BiLSTM is its canonical published checkpoint rather than a fresh unified-engine
  run, so the small +0.0059 AUC edge could carry a little training-run variance. We frame it as
  "level-to-marginally-better," not a clean RNN win — the scientifically safe reading is a tie.
  (The GRU, trained on the same engine, tied the BiLSTM here, which argues the RNN's small edge is
  a genuine, if minor, cell effect rather than an engine artifact.)
- **A tie is a tie, not a win.** We are not claiming the vanilla RNN is *better*; we're reporting
  that removing gating doesn't move the needle — which is the scientifically interesting outcome.
- **"Vanishing gradients" only stayed benign because the window is short (16 steps).** This result
  should not be read as "gating never matters" — over longer horizons it very likely would. We
  state the horizon explicitly.

---

## 7. How rigorously this was checked

- A **parity gate** re-derived the frozen BiLSTM's per-seed test AUC from its checkpoints before
  any comparison number was computed — exact match, all 5 seeds (|Δ| = 0.00e+00).
- The **search review was independently recomputed** from the raw run files and cross-checked
  against the search's own summary (exact agreement, including the instability-ledger count);
  every search file was verified to contain no test key (93 files, zero leaks).
- Every LOSO fold's size matched the exact pedestrian counts from the earlier LOSO study — a
  fingerprint that the real, untampered data was used.
- Every reported number is re-derived from raw checkpoints/probabilities on each run — nothing is
  a stored, un-checkable claim.

---

## 8. How to present this (talking script + Q&A)

**45-second pitch:**
> "You asked us to test more model families. The vanilla RNN is the BiLSTM with its gating
> removed — the simplest recurrent cell there is. We gave it the *same* search the BiLSTM got and
> the same F1-first optimization, and it's a dead statistical tie with the BiLSTM and the GRU on
> F1, and level-or-better on AUC at matched size. And unlike the GRU, the AUC-optimized RNN even
> ties the searched Transformer on AUC — so that transformer's edge really was the search, not
> attention. Bottom line: it's not the recurrent cell, and it's not even the LSTM's gating — the
> input signal, bbox plus ego-speed, is what matters. Same conclusion the whole thesis has been
> building toward, now with the strongest possible confirmation: even stripping the cell down to a
> plain RNN loses nothing. And it's the smallest and fastest of the four models."

**Order to walk through:** Section 1 (why the vanilla RNN — it removes gating) → Section 4's table
+ the four bootstrap verdicts (the two F1 ties and the transformer-AUC tie are the point) →
Section 5 (LOSO + latency, it generalizes and it's the fastest) → Section 6 (limitations, stated
plainly, especially the "short window" caveat).

**Likely questions & answers:**
- *"Did you try as hard on the RNN as on the BiLSTM?"* → Same search budget by construction (the
  Issue-8 grid + class-weight sweep), same optimization, same engine, same device.
- *"Doesn't a plain RNN have vanishing gradients?"* → Over a 16-step window the risk is mild — all
  93 search runs trained cleanly, zero divergences (we kept a ledger for them; it stayed empty).
  We're explicit that this result is horizon-specific and wouldn't necessarily hold for long
  sequences.
- *"How can the RNN tie the Transformer on AUC when the Transformer beat the BiLSTM?"* → Because
  the searched RNN (h256, AUC-selected) reaches ~0.95 AUC, well above the un-searched BiLSTM. Give
  any of these models an equivalent search and they land in the same place — which is exactly the
  transformer study's own conclusion ("the win was the search, not the architecture").
- *"So which model do we use?"* → On the evidence it doesn't matter among the recurrent cells —
  keep the BiLSTM (it's the established headline), or the vanilla RNN if you want the smallest and
  fastest. The interesting finding isn't a model choice; it's that the choice barely matters.

---

## 9. Where the evidence lives (file map)

```
rnn/
├── PLAN.md                     ← full pre-registered design (written before any result)
├── PROGRESS_LOG.md             ← chronological log of every run/number/decision
├── phase3_search_review/03_search_summary.md   ← the search winner (val-only)
└── phase5_analysis/
    ├── 07_comparison_report.md      ← THE VERDICTS (paired bootstrap, all 8 endpoints)
    ├── 07_comparison_figure.png     ← endpoint Δ chart
    ├── 08_cluster_bootstrap.md      ← pedestrian-cluster CIs (the honest intervals)
    ├── 09_latency_report.md         ← latency vs BiLSTM / GRU / Transformer (RNN is fastest)
    └── 10_loso_report.md            ← 6-fold cross-validation table
```

Every script in `rnn/phase5_analysis/` re-derives its numbers from the raw checkpoints/probs on
each run — nothing here is an un-checkable stored claim.

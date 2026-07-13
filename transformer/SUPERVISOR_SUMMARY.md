# Transformer vs. BiLSTM — Supervisor Summary

**Student:** Arif
**Date:** 2026-07-12
**One-line summary:** You asked us to build the best Transformer we could on the same
data and compare it to our BiLSTM, properly. We did — the Transformer measurably wins
**on AUC**, and the *reason* it wins is the real finding: a hyperparameter search
found a better architecture, whereas an un-searched Transformer running our own recipe
is a dead statistical tie with the BiLSTM. **Update (F1-first directive, 2026-07-13):**
under your newer instruction to prioritize F1 → accuracy → AUC, we ran the same
optimization discipline on both models (`f1_optimization/`) — the BiLSTM improved
significantly on F1 (0.828 → 0.844) and on the F1 metric the two families are now a
**statistical tie**; the AUC win stands, but it is metric-specific. See the new
section 4b below.

> **How to use this doc:** read top-to-bottom — it's written so you can explain the
> whole extension to your supervisor in order, exactly as the earlier
> `paper_and_artifacts/supervisor_review/` pack was written for the base thesis. All
> numbers here come from the actual run outputs (`transformer/phase4_kaggle_final/`,
> `transformer/phase5_analysis/`), independently re-verified from raw files at every
> stage — not estimates, not the notebook's own summary taken on faith. Section 8 is a
> ready-to-use talking script and Q&A cheat sheet.

---

## 1. Why this exists

The base thesis's own supervisor-review pack poses this question directly, in its own
FAQ: *"Why BiLSTM, not a Transformer?"* — and answers it on **small-data grounds**
("1,389 sequences is small; a 0.6M-param BiLSTM is the right capacity"). That's a
reasonable argument, but it's an argument, not a measurement. You asked for the
measurement. This extension builds the best Transformer we reasonably can on the exact
same data and protocol, and settles the question with numbers instead of intuition.

**The honest short answer, now that we've measured it:** the argument wasn't wrong, but
it wasn't the whole story either. A Transformer *can* beat the BiLSTM here — but only
if you spend a real search budget finding the right one. Handed the BiLSTM's own
untuned recipe, a Transformer doesn't beat it at all.

---

## 2. What stayed exactly the same (so the comparison is fair)

Everything that could bias the comparison in either model's favor was **frozen
identical** to the BiLSTM's own clean-protocol result (the `journal_prep/` work):

- Same data: `sequences_clean/` — the leakage-free windows anchored at PIE's
  `crossing_point`, N=4,906 (train 2,178 / val 634 / test 2,094).
- Same splits (train = set01/02/04, val = set05/06, **test = set03**), same
  `pos_weight=1.682`, same 5 training seeds `[42, 0, 1, 2, 3]`, same 0.5 decision
  threshold, same early-stopping rule (patience 15 on validation AUC).
- Same discipline: the **test set is touched exactly once**, on the final,
  already-selected model — never during the search. Everything before that point is
  validation-only: the search notebook never invokes test evaluation (every job passes
  `eval_test=False`), and its outputs were verified to contain zero test keys.
- The BiLSTM's own checkpoints were **never retouched or retrained** — they're loaded
  as-is from the original run. There's no way to accidentally improve the baseline to
  make the comparison look different than it really is.

**What was allowed to differ:** only the Transformer's own architecture (width, depth,
pooling, positional encoding) and training recipe (learning rate, schedule, dropout,
weight decay). That's the one axis this extension is actually about.

---

## 3. What is new: the model and the search

**The model.** A small Transformer encoder over the same 16-frame input the BiLSTM
sees (bounding box + ego-speed, standardized): a linear input projection, a positional
encoding, a few self-attention + feed-forward layers, then a pooling step and a linear
output — architecturally the standard "small Transformer classifier," nothing exotic.

**The search.** Rather than hand-pick a Transformer configuration and hope, we ran a
staged, pre-registered search — the same discipline already applied to the BiLSTM
itself (the grid search from `journal_prep` Issue 8), but with **more than double the
search budget** (78 distinct architecture/recipe combinations vs. the BiLSTM's 36):

1. **Stage A** — 36 architecture variants (width × depth × pooling × positional
   encoding), one seed each, ranked by validation AUC.
2. **Stage B** — 36 training-recipe variants (learning rate, schedule, dropout, weight
   decay) on the best Stage-A architecture.
3. **Transfer check** — 6 more runs, confirming the best recipe still works on the
   2nd/3rd-best architectures (guards against a recipe that only looks good by luck of
   pairing with one specific architecture).
4. **Stage C** — the top 5 candidates overall, *plus* a pre-registered "default"
   configuration (the Transformer's most ordinary settings, trained with the BiLSTM's
   own exact recipe — zero tuning), each re-run across all 5 seeds. **The winner is
   whichever config has the best *5-seed average*, not the best single lucky seed** —
   this matters: the single-seed leader across all 78 configs was *not* the actual
   5-seed winner, exactly the same phenomenon we'd already seen with the BiLSTM's own
   grid search.

All 102 of those runs are validation-only — test set03 is not evaluated anywhere in
this stage. Only after the winning configuration was reviewed and confirmed did we
build the final notebook that touches test set03, exactly once.

---

## 4. The result

**Headline: the searched Transformer wins on AUC.** Each model was trained 5 times
(different random seeds); the table below is the plain average of those 5 runs' own
test metrics — the same "0.932" you already know for the BiLSTM from the base thesis,
now shown with accuracy and F1 alongside (all at the fixed 0.5 threshold):

| model | parameters | test AUC (5-seed) | test Acc | test F1 |
|---|---|---|---|---|
| BiLSTM (your original baseline, frozen) | 594,561 | 0.932 ± 0.011 | 0.883 | 0.828 |
| Transformer, default recipe (zero search) | 268,417 | 0.934 ± 0.006 | 0.878 | 0.816 |
| **Transformer, searched (the winner)** | 794,241 | **0.950 ± 0.003** | 0.894 | 0.845 |

(Note the default transformer *ties* the BiLSTM on AUC but is *below* it on F1 —
metric order matters, which is exactly why section 4b exists.)

A table of point numbers isn't proof of anything on its own — two models could differ by
this much just from noise in which 2,094 test windows happened to be in the test set.
So we ran the actual test: a **10,000-resample paired bootstrap** — repeatedly
re-sampling the test windows (the same resampled windows for both models each time, so
we're isolating the *difference* between the models rather than each model's own noise)
and asking how often the Transformer's advantage could be a fluke.

**Result: it isn't.** The searched Transformer beats the BiLSTM by
**ΔAUC = +0.0135, with a 95% confidence interval of [+0.0097, +0.0174]** — an interval
that sits entirely above zero. (The bootstrap's own Δ is computed by first combining
each model's 5 seeds into one probability per window, which is why it reads +0.0135
rather than the table's simple +0.018 gap — a marginally more conservative version of
the same number, not a different finding.) A second, independent check (a paired
statistical test across the 5 training seeds) agrees: p = 0.025, significant at the
conventional 0.05 level.

**The finding that actually matters most — and the one to lead with in any
presentation:** run that *same* Transformer architecture with the BiLSTM's own
untouched, un-searched recipe, and it is a **dead statistical tie** with the BiLSTM
(Δ = +0.0005, confidence interval [−0.0034, +0.0043] — straddles zero completely, not
a real difference). **A Transformer does not automatically beat a BiLSTM here.** The
win came specifically from spending a real search budget and finding a genuinely
different, better-suited configuration (deeper — 4 layers instead of 2; reads out from
the *last* timestep instead of a learned summary token; uses a fixed sinusoidal
position encoding instead of a learned one). That is a much more defensible, much less
hand-wavy story than "we tried a Transformer and it won."

---

## 4b. What changed under your F1-first directive (2026-07-13)

You then asked us to prioritize **F1 first, then accuracy, then AUC**. Everything
above selects and reports AUC-first, so we ran a second pre-registered program
(`f1_optimization/`) that applied the *same* F1-first optimization to **both** models
symmetrically — a validation-tuned decision threshold, best-F1 checkpoint selection, a
class-weight sweep, and (for the BiLSTM) re-selecting its configuration from its own
Issue-8 grid by F1 instead of AUC. Test was still touched exactly once per model, and
every choice was made on validation only. Three verdicts (10,000-resample paired
bootstrap, same discipline as section 4):

1. **The BiLSTM improved significantly on F1**: 0.828 → **0.844 ± 0.008** (accuracy
   0.883 → 0.897), ΔF1 confidence interval [+0.007, +0.030] — entirely above zero.
   It also survives a stricter pedestrian-cluster resampling ([+0.004, +0.035]).
2. **The Transformer did not improve significantly** (0.845 → 0.847; interval
   straddles zero) — it was already near its F1 ceiling.
3. **On F1 the two models are now a dead statistical tie** (ΔF1 +0.001, interval
   [−0.012, +0.014]). **The Transformer's win is AUC-specific: it does not carry
   over to your primary metric once both models get the same F1-first treatment.**

Context that matters for the paper: the verified standard-protocol F1 range in the
PIE literature is 0.77–0.87 (ceiling: PedFormer, a multimodal multitask model, 0.87);
both our 2-stream models now sit at 0.844–0.847 — within ~0.02 of that ceiling using
a fraction of the inputs. All of this was additionally replicated end-to-end under
one unified training engine on one device
(`journal_prep/issue12_unified_pipeline/12_replication_report.md`) so the conclusion
cannot be an artifact of the two models having been trained by different code on
different hardware.

**Bottom line to present:** "the searched Transformer wins on AUC (0.950 vs 0.932);
under the F1-first hierarchy the F1-optimized BiLSTM catches up completely
(0.844 vs 0.847 — a tie), so the model choice is a wash on the primary metric and the
BiLSTM remains a fully defensible, smaller headline model."

---

## 5. Supporting evidence (it's not just one lucky test set)

- **Cross-validation across all 6 PIE recording sets (not just set03):** rotating
  which set is held out, the searched Transformer's fold-average is higher (**0.939
  AUC** vs the BiLSTM's **0.928**), though individual folds vary — the Transformer
  trails slightly on set01 and set04 — and 6 folds is descriptive, not a hypothesis
  test; the fixed-split bootstrap in section 4 is the actual evidence.
- **Latency — genuinely surprising, in the Transformer's favor:** despite having about
  1.3× the parameters, the Transformer is actually *faster* per prediction than the
  BiLSTM on your M4 (0.459 ms vs. 0.575 ms for one window). The likely reason: the
  BiLSTM's recurrence has to process its 16 timesteps one after another, while the
  Transformer's self-attention processes all 16 at once, in parallel — and that
  parallelism apparently wins out here, even with more total weights. Either way,
  **both models are so fast that this is a non-issue for the live demo** — both are
  roughly two orders of magnitude faster than a 30-fps video frame budget requires
  (BiLSTM ~58×, Transformer ~73×), and the actual bottleneck in the live pipeline is
  the object detector (YOLO26-M), not the intention model, by either architecture.
- **A determinism check, to make sure the headline number is real and reproducible:**
  we re-ran the winning configuration's canonical seed a second time, independently,
  on the same Kaggle GPU. It reproduced its own test AUC **exactly** — not
  approximately, to the full displayed precision. The reported number isn't a fluke
  of one lucky run.

---

## 6. Honest limitations

- **This is one dataset (PIE), one fixed train/val/test split, and 5 training seeds.**
  The paired bootstrap (10,000 resamples over the shared 2,094 test windows) is the
  primary evidence precisely because 5 seeds alone is a small, low-power sample — we
  say so explicitly in every report rather than leaning on the seed-level t-test as if
  it were the strong evidence.
- **The improvement, while statistically real, is not huge in absolute terms:** about
  1.3 AUC points. It's a genuine, reproducible win, not a transformative leap — and we
  report it exactly that plainly rather than oversell it.
- **The Transformer costs more to train and is a bigger model** (794k vs 595k
  parameters) — a legitimate trade-off to state alongside the win, even though it
  turned out not to cost anything at inference time.
- **The win is metric-specific.** This comparison's WIN verdict is about AUC. Under
  your F1-first directive, the follow-up program (section 4b) improved *both* models
  symmetrically and on F1 they tie — so "the Transformer is better" is only true with
  "on AUC" attached, and the paper says so explicitly.
- **Within THIS comparison the BiLSTM was deliberately not improved** — that would
  have been moving the goalposts mid-study. Its numbers here are the exact, frozen,
  already-published result from your journal-prep work. The later F1-first program
  then improved both models under its own pre-registration, symmetrically — which is
  the legitimate way to do it.

---

## 7. How rigorously this was checked

Every number in this document was independently re-derived from the raw output files
at least once — not taken on the notebook's own word for it:

- A **parity gate** re-loaded the BiLSTM's frozen checkpoints and re-computed its test
  AUC from scratch before any comparison number was allowed to be computed; it matched
  the originally recorded numbers **exactly, to the displayed precision, for all 5
  seeds** — confirming nothing about the baseline had drifted.
- All 6 leave-one-set-out fold sizes matched the exact pedestrian counts from your
  earlier LOSO study, which is essentially a fingerprint check that the real,
  untampered dataset was used throughout.
- Every per-epoch training log was checked to confirm the test set genuinely was never
  touched anywhere except the one designated final-evaluation step.
- A tiny (far below the 4th decimal place) numerical mismatch between the
  locally-recomputed and the Kaggle-GPU-computed Transformer probabilities was
  investigated rather than ignored — traced to an expected, harmless floating-point
  difference between CPU and GPU hardware, confirmed not to be a batching bug, and
  confirmed far too small to move any reported number.

---

## 8. How to present this (talking script + Q&A)

**60-second pitch:**
> "You asked whether a Transformer could beat our BiLSTM on the same crossing-intention
> task. We gave the Transformer a real chance — a search more than twice the size of
> the one we ran for the BiLSTM itself — and it measurably wins **on AUC**:
> 0.950 vs. 0.932, confirmed with a 10,000-sample bootstrap, not just a raw average.
> Two findings sharpen that. First, an un-searched Transformer run with our own
> BiLSTM's recipe ties the BiLSTM exactly — the win is 'a properly searched model
> beats a hand-set one,' not 'Transformers are better.' Second, under your F1-first
> priority we then optimized both models for F1 with the same discipline, and on F1
> they tie (0.847 vs 0.844) — the BiLSTM caught up completely. So: Transformer for
> the best AUC, either model for F1, and the BiLSTM stays a fully defensible,
> smaller headline model. Inference cost is a non-issue either way — the Transformer
> is actually slightly faster per window."

**Order to walk through this with your supervisor:** Section 1 (the question this
answers) → Section 4's table and the bootstrap result (the win) → the
searched-vs-default tie (**the key nuance — don't skip this**) → Section 5 (LOSO +
latency, it generalizes and it's free) → Section 6 (limitations, stated plainly).

**Likely questions & answers:**
- *"Is this a fair comparison, or did you just try harder on the Transformer?"* →
  Everything except the Transformer's own architecture/recipe was frozen identical to
  the BiLSTM's protocol; we also report the un-searched Transformer explicitly (a tie)
  precisely so this can't be read as "we only tuned one side."
- *"Could this just be a lucky test set?"* → That's exactly what the paired bootstrap
  rules out: 10,000 resamples of the same 2,094 windows, and the confidence interval
  for the Transformer's advantage never crosses zero.
- *"So do we switch to the Transformer?"* → Depends on the metric you lead with.
  On AUC it is measurably better (and no latency cost); on your primary metric (F1,
  after identical F1-first optimization of both models) it is a statistical tie with
  the smaller BiLSTM. The honest paper framing: "a searched Transformer beats the
  BiLSTM on AUC; an unsearched one doesn't; and under F1-first optimization the
  families are indistinguishable on F1" — the model-choice question dissolves into a
  metric-priority question, which is itself a publishable finding.
- *"Does this replace the demo?"* → Not yet done — the live YOLO+ByteTrack demo still
  runs the BiLSTM; swapping in the Transformer there is a small, optional follow-up if
  you want it, not something this extension required.

---

## 9. Where the evidence lives (file map)

```
transformer/
├── PLAN.md                     ← the full pre-registered design (read this for the
│                                   exact methodology, written BEFORE any result existed)
├── PROGRESS_LOG.md             ← chronological log of every run, every number, every
│                                   bug found and fixed, in the order it happened
├── phase3_search_review/03_search_summary.md   ← the search's own winner report
├── phase4_kaggle_final/README.md               ← the final-training run, verified
└── phase5_analysis/
    ├── 05_comparison_report.md      ← THE VERDICT — the bootstrap, the tie-breaking
    │                                   nuance, the full numbers (start here for detail)
    ├── 05_comparison_figure.png     ← the three-bar chart (BiLSTM / default / searched)
    ├── 04_final_summary.md          ← the 5-seed result tables
    ├── 06_latency_report.md         ← the latency finding
    └── 07_loso_report.md            ← the 6-fold cross-validation table
```

**To reproduce or dig deeper:** every script in `transformer/phase5_analysis/` can be
re-run directly (`python transformer/phase5_analysis/05_compare_vs_lstm.py`, etc.) —
they re-derive every number from the raw checkpoints and JSON files each time, so
nothing here is a stored, un-checkable claim.

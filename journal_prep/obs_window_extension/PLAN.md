# PLAN — Observation-window extension (OW 32 & 64 on all four F1-optimised families)

**Supervisor directive (2026-07-19):** try observation windows **32 and 64** on all four model
families, evaluating only the **F1-optimised** version of each family.

## 1. What we are actually running

The current headline comparison is at **OW = 16 frames** (0.5 s @ 30 fps). This study adds two
longer horizons — **OW = 32** (1.07 s) and **OW = 64** (2.13 s) — for the F1-optimised (headline)
model of each family, keeping everything else in the frozen protocol identical. That is:

| # | family | F1-optimised recipe (config · pos_weight · selection) | source |
|---|---|---|---|
| 1 | **BiLSTM-F1** | `lr1e-3 do0.3 h256 nl2` · pw 1.682 · select f1 · τ\* | `f1_optimization` A3 |
| 2 | **Transformer-F1** | `d128 nhead4 L4 ff512 do0.1 pool=last pos=sin, plateau, adam, wd1e-5` · pw 2.5 · select f1 · τ\* | `f1_optimization` B3 |
| 3 | **GRU-F1** | `lr5e-4 do0.3 h256 nl2` · pw 1.682 · select f1 · τ\* | `gru` gru_f1_winner |
| 4 | **Vanilla RNN-F1** | `lr1e-4 do0.2 h256 nl2` · pw 1.682 · select f1 · τ\* | `rnn` rnn_f1_winner |

**Grid:** 4 families × 2 windows (32, 64) × 5 seeds `[42,0,1,2,3]` = **40 training runs**, all on
**CPU** (bit-reproducible for the recurrent families; consistent for the transformer). OW-16 stays
as the already-published reference row — we do **not** retrain it.

## 2. Data — already built, real sample counts

Rebuilt the clean, leak-free sequences at each window with the *same* builder
(`journal_prep/issue2_clean_protocol/02_build_sequences_clean.py`, event-anchored at
`crossing_point`, TTE ∈ [30,60], overlap 0.5). Verified counts:

| window | total | train | val | test (set03) | train pos_weight | test pos% |
|---|---|---|---|---|---|---|
| OW 16 (ref) | 4906 | 2178 | 634 | 2094 | 1.682 | 32.5 % |
| **OW 32** | 2366 | 1055 | 302 | 1009 | **1.755** | 32.6 % |
| **OW 64** | 1097 | 501 | 138 | 458 | **1.831** | 32.5 % |

Two facts this table settles:
- **Class balance is stable** (~32.5 % positive at every window), so `pos_weight` barely drifts.
- **Sample size shrinks fast** — each doubling of the window roughly halves the data, because a
  longer window needs a longer pre-crossing track (`L ≥ obs_len + 30`). OW 64 keeps only 458 test /
  138 val windows. This is the main caveat (see §5).

**`pos_weight` decision (FINAL):** **hold each family's F1-recipe `pos_weight` fixed** at its
OW-16 value (BiLSTM/GRU/RNN = 1.682, Transformer = 2.5 — the latter a *searched* value, not the
class ratio). This makes OW the single changed variable and treats all four families consistently.
Because the class balance is essentially constant across windows (~32.5 % positive; the train-only
ratio would only drift 1.682 → 1.755 → 1.831) and F1 is insensitive to that drift, recomputing
would change nothing measurable — so we keep the recipe fixed for clean single-variable isolation.

## 3. Method (mirrors the existing F1 pipeline exactly)

For each family × window × seed, on **CPU**:
1. **Train** with `journal_prep/issue12_unified_pipeline/12_unified_engine.py::train_run` — the same
   model-agnostic engine every published run uses: train-only z-score normalization (recomputed for
   the new window), `BCEWithLogitsLoss(pos_weight)`, ≤100 epochs, early-stop patience 15, plateau LR,
   **select = F1** (F1→acc→AUC checkpoint). Validation-only; test never touched here.
2. **Tune τ\*** on the validation set (maximize val F1), exactly as `f1_optimization/05_final_test_eval.py`.
3. **Evaluate once on test set03** at τ\* (and also at 0.5 for reference), cache probabilities.
4. Aggregate **per-seed mean ± std** (the paper numbers) and the **5-seed probability ensemble**
   (the deployable predictor + confusion matrix), identical to how the OW-16 table is built.

**One required code tweak:** the transformer's positional-encoding buffer is sized to `seq_len`
(default 16). The driver must build `TransformerIntentPredictor(..., seq_len=OW)` so the sinusoidal
PE spans 32/64 tokens. Sinusoidal PE extrapolates to any length, so no other change is needed. The
recurrent families (LSTM/GRU/RNN) are length-agnostic already.

**Deliverables:** a small driver `01_run_ow_extension.py` (loads OW-specific sequences → `train_run`
per seed → τ\* + test eval), a results JSON/CSV, and the **updated comparison table in
`journal_prep/Analysis/model_comparison.md`** with a new "observation-window" section (OW-16 headline
row + OW-32 + OW-64 for each family, per-seed-mean F1/Acc/AUC and the trend per family).

## 4. Time estimate (local M4, CPU)

Measured per-seed CPU times at OW 16, h256: LSTM/GRU ≈ 30 s, vanilla RNN ≈ 21 s, transformer-L4
≈ 20 s. Per-epoch cost ≈ `N_train × seq_len`; at longer windows `N_train` shrinks ~as fast as
`seq_len` grows, so per-seed time stays **roughly constant** (transformer OW 64 is a bit slower —
attention is O(L²) — but on far fewer windows).

- 40 training runs × ~40 s (conservative avg) ≈ **~27 min**
- τ\* tuning + test eval + aggregation ≈ **~5 min**
- **Total ≈ 30–45 min of compute; budget ~1 hour with driver debugging.** Fully local, no Kaggle.

## 5. Caveats to state in the paper

- **Not a matched cohort across windows.** The OW-64 test set (458 windows) is a *subset-defining*
  selection — only pedestrians tracked ≥ 94 frames (~3.1 s) before crossing survive — so absolute
  F1 is not perfectly comparable *across* windows. The clean comparisons are **within each window,
  across families** and **the per-family trend**. (Issue 6 hit the same issue for the BiLSTM-only
  {8,16,30} sweep and addressed it with a matched-cohort check — we can optionally replicate that.)
- **Smaller val at OW 64** (138 windows) makes τ\* noisier → we also report F1@0.5.

## 6. Why this strengthens the journal

1. **Directly tests the standing caveat.** The RNN study concluded gating buys nothing *over a
   16-step window*, explicitly flagging that "an un-gated RNN would likely fall behind over long
   sequences." OW 32/64 is the experiment that confirms or refutes that — either the vanilla RNN
   degrades at 64 while gated cells hold (a clean, publishable "gating matters once the horizon is
   long enough" nuance), or all four stay tied (a *stronger* "input signal dominates, architecture
   secondary — across horizons" claim). Either outcome is a result, not a null.
2. **Closes an obvious reviewer gap.** Right now only the BiLSTM had its window varied ({8,16,30},
   Issue 6). Extending the window sweep to all four families at longer horizons pre-empts the
   "you only varied this on one model" objection.
3. **Adds a temporal-context robustness axis** to the headline four-family comparison, at ~1 hour of
   local compute and zero new data collection.

## 7. Decisions (resolved 2026-07-19)

- **Run all 8 configs at once** (done) — no staged OW-32-first check.
- **No matched-cohort analysis** — kept to the per-family per-window comparison the supervisor asked
  for (the cohort caveat is stated in §5 and in the results table).

## 8. Results (2026-07-19) — DONE

40 runs, CPU, 1478 s total (~25 min). Full numbers in `01_ow_results.json` / `.csv`; the comparison
table is in `journal_prep/Analysis/model_comparison.md` (§ Observation-window extension).
Per-seed-mean test-F1:

| family | OW 16 (ref) | OW 32 | OW 64 | Δ(64−16) |
|---|---|---|---|---|
| BiLSTM-F1 | 0.844 | 0.837 | 0.818 | −0.026 |
| Transformer-F1 | 0.847 | 0.838 | 0.819 | −0.028 |
| GRU-F1 | 0.849 | 0.834 | 0.822 | −0.027 |
| Vanilla RNN-F1 | 0.852 | 0.834 | **0.802** | **−0.050** |

**Findings.** (1) F1 declines monotonically with window for all four families — longer context does
not help; this justifies the OW-16 design choice. (2) Families still tie at OW 16 and OW 32. (3) At
**OW 64 the un-gated vanilla RNN alone falls behind** — lowest F1 (0.802), AUC (0.929), Acc (0.857),
and ensemble F1 (0.784), with ~2× the F1 drop of the gated cells. Consistent across all metrics and
matches theory, though per-seed CIs overlap (directional, not bootstrap-confirmed). This confirms
the pre-registered caveat *in direction*: the four-family equivalence is **horizon-bounded** —
gating is redundant over 0.5 s but begins to re-earn its keep by ~2 s.

**Reproduce:** `python journal_prep/obs_window_extension/01_run_ow_extension.py` (rebuild sequences
first with `02_build_sequences_clean.py --obs-len {32,64}` if `seq_ow*/` are absent).

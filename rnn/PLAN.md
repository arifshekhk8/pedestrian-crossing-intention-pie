# Vanilla-RNN Extension — Master Plan

**Created:** 2026-07-14 · **Status:** ALL PHASES (R1–R6) DONE — verdict: **TIE with the gated
recurrent models on F1 (gating buys nothing over a 16-step window); on AUC the searched RNN even
ties the transformer the GRU could not reach** ·
**Requested by:** supervisor (test two more model families — GRU in a parallel session,
**vanilla RNN here** — on the existing clean pipeline, identical protocol).
**Companion docs:** `README.md` (index + how to run) · `PROGRESS_LOG.md` (chronological
numbers log) · `SUPERVISOR_SUMMARY.md` (plain-English write-up, R6).

This is the **un-gated recurrent-cell analogue of the `gru/` and `transformer/` studies**:
same pre-registration discipline, same frozen protocol, same comparison machinery. Where the
transformer study isolated *attention vs recurrence* and the GRU study isolated *the gated
cell type*, this one isolates **gating itself** — a bidirectional vanilla RNN (`birnn`, tanh
cell) is the exact twin of our BiLSTM with **only `nn.LSTM` → `nn.RNN`** (identical input
projection, bidirectional recurrence, last-step readout, linear head). This plan is written
**before any RNN result exists**; the search space, selection rules, and verdict templates
are pre-registered so no decision is made after seeing test numbers. Each phase is closed by
appending a `### ✅ DONE` block with the real numbers (journal_prep convention).

**The question this answers:** *under the identical clean protocol and the F1-first hierarchy,
does the LSTM's gating buy anything over a 16-step window, or would an un-gated recurrent net
do just as well?* This is the sharpest test of the thesis's central claim: the GRU (still
gated) tied the BiLSTM, so if the **un-gated** RNN also ties, "the input signal, not the
recurrent cell, is what matters" is strongly supported; if it clearly loses, gating matters.
**Hypothesis (data decides):** either outcome is a clean, publishable result — the vanilla
RNN is the family most likely to break the tie, which is exactly why it is worth running.

---

## 1. The models we compare against (never retrained)

| model (5 seeds [42,0,1,2,3]) | params | test AUC | F1 | Acc | source |
|---|---|---|---|---|---|
| **Frozen BiLSTM (locked, Issue 2)** | 594,561 | **0.9324 ± 0.0114** | 0.828 | 0.883 | `journal_prep/issue2_clean_protocol/runs_clean/multiseed/` |
| Transformer (searched) | 794,241 | ~0.950 (0.9497 ± 0.0025) | 0.845 | 0.894 | `transformer/phase4_kaggle_final/` |
| BiLSTM-F1 (F1-first program) | ≈2–3M (h256) | 0.940 | 0.844 ± 0.008 | 0.897 | `f1_optimization/runs_f1/lstm_..._h256/` |
| Transformer-F1 (F1-first program) | 794,241 | 0.947 | 0.847 ± 0.017 | 0.896 | `f1_optimization/runs_f1/transformer_searched/` |
| GRU-F1 (recurrent-cell twin) | 1,678,209 | 0.941 | 0.849 | 0.901 | `gru/phase4_final/` |
| **Vanilla RNN (default h128) — this study** | **149,121** | TBD | TBD | TBD | this folder |

Frozen-BiLSTM per-seed test AUC (parity-gate reference): seed42 0.913114, seed0 0.933424,
seed1 0.943189, seed2 0.936295, seed3 0.935822.

**Cached comparison probs** already exist (val+test vectors + `y_val`/`y_test`):
`f1_optimization/probs_cache/` → `lstm_frozen`, `lstm_a3` (BiLSTM-F1), `tf_frozen` (Transformer
searched), `tf_b3` (Transformer-F1); `gru/phase4_final/probs_cache/` → `gru_f1_winner`,
`gru_default_f1`, `gru_default_auc`. The RNN study reuses these for the paired bootstrap rather
than regenerating them.

---

## 2. Fairness contract (byte-for-byte identical to the BiLSTM/GRU/Transformer protocol)

**FROZEN — not searchable:**

| item | value |
|---|---|
| engine | `journal_prep/issue12_unified_pipeline/12_unified_engine.py --family birnn` — **no new trainer** |
| device | **local CPU only** (`--device cpu`) — recurrent CPU training is bit-reproducible & context-free (issue12); MPS is process-history-dependent |
| data | `journal_prep/issue2_clean_protocol/sequences_clean/` reused verbatim (engine's canonical path) |
| splits | train {set01,02,04} N=2178 · val {set05,06} N=634 · test {set03} N=2094 (681 pos, 32.5%) — asserted in `load_splits()` |
| normalization | per-feature z-score from **train only**, saved per run |
| loss | `BCEWithLogitsLoss(pos_weight=1.682)` (= 1366/812), except the pos_weight sweep and LOSO (per-fold) |
| threshold | 0.5 (+ val-tuned τ\* reported for F1) |
| batch / epochs | 32 / max 100 |
| early stopping | patience 15 on **val AUC** |
| seeds | [42, 0, 1, 2, 3]; seed 42 canonical |
| selection | **validation only**; test set03 evaluated **exactly once**, in `phase4_final/05_rnn_test_eval.py`, **after the user confirms the winner (R3 gate)** |
| checkpoints | `torch.load(..., weights_only=False)` |
| reporting | always label **per-seed-mean vs 5-seed probability-ensemble** |

**SEARCHED — the RNN's own knobs (identical budget to the BiLSTM's Issue-8 search):**
`lr × dropout × hidden × num_layers` (the 36-config Issue-8 grid) + a pos_weight sweep — no
more, no less than the BiLSTM/GRU got. This isolates gating.

**Metric hierarchy: F1 → accuracy → AUC.** Primary verdict on F1; AUC reported as secondary
corroboration (incl. vs frozen BiLSTM 0.9324 and searched transformer ~0.9497).

---

## 3. The model

`RecurrentIntentPredictor("rnn")` — defined in `12_unified_engine.py`, the un-gated twin of
`pipeline/03_bilstm_model.py::BiLSTMIntentPredictor`:
```
input (B,16,5) raw [x1,y1,x2,y2,vehicle_speed] pixels, z-scored
 → Linear(5, 64) + ReLU                       # input projection (proj_dim=64)
 → nn.RNN(64, hidden, num_layers, bidirectional=True, dropout, nonlinearity="tanh", batch_first=True)
 → last-step readout  h[:, -1, :]  (dim = hidden*2)
 → Linear(hidden*2, 1)                         # logits — same contract as the BiLSTM
```
Default (h128/nl2/do0.3) = **149,121 params** (input_proj 384 + RNN 148,480 + head 257),
confirmed analytically and by the R1 gate. Only the cell (`nn.RNN` vs `nn.LSTM`) differs from
the BiLSTM. The vanilla RNN has ~1/4 the recurrent weights of the 4-gate LSTM, so the whole
search ladder (h64/h128/h256 × {1,2}L, 17k → 560k params) sits **below** the BiLSTM's 594,561.

---

## 4. Pre-registered staged search (selection on val only, Issue-8 protocol)

All search runs are **val-only by construction** (the engine has no test code path). Every run
cached to `phase2_search/runs_search/<cfg_id>/seed<k>.json` (= the engine result dict;
**no `test` key ever**), so interrupted sessions resume free.

**Key efficiency:** a single `--select f1` run records both `val` (F1-best epoch) and
`val_at_auc_best` (AUC-best epoch); early stopping is on val AUC and independent of `--select`,
so the trajectory — and hence the AUC-best metrics — are identical across modes. ⇒ **one run
per config yields both the val-F1 and val-AUC rankings.**

**Stage 1 — grid (36 configs, seed 42, `--select f1`):**
`lr ∈ {1e-3, 5e-4, 1e-4}` × `dropout ∈ {0.2, 0.3, 0.5}` × `hidden ∈ {64, 128, 256}` ×
`num_layers ∈ {1, 2}` — dropout inert at nl=1, cells merged → 36 distinct configs
(reuses `issue8/08_grid_search.py::build_grid()`/`cfg_id()`). Rank by **val F1 (primary)** and
**val AUC (secondary)**.

**Stage 2 — multiseed (5 seeds, `--select f1`):** union of {top-5 by val-F1} ∪ {top-5 by
val-AUC} ∪ `birnn_default` (h128/nl2/do0.3, the BiLSTM-baseline recipe twin). **F1-winner =
highest mean val F1** (tie: acc, then AUC); **AUC-winner = highest mean val AUC** (noted
separately). Selection-noise control (Issue-8 precedent: the seed-42 leader was not the 5-seed
winner).

**Stage 3 — pos_weight sweep (5 seeds, `--select f1`):** F1-winner × pos_weight
∈ {1.0, 1.3, 1.682, 2.1, 2.5}, val-only, ranked by mean val F1. Keep 1.682 unless a value
beats it (mirrors `f1_optimization` G2 / GRU Stage 3). The AUC track keeps pos_weight 1.682
fixed (Issue 8 never swept it).

**Instability ledger (RNN-specific — vanilla tanh RNNs can vanish/explode).** The search
records **every** config's outcome even if it diverges; any run with val AUC < 0.70 or a
non-finite loss is **kept and flagged** ("recorded, not dropped") in `_stage_summary.json`
and `03_search_summary.md`, mirroring `transformer/PLAN.md §4`'s divergence contingency. The
watch list is higher-lr and/or 2-layer cells. Over a 16-step window vanishing gradients are
mild (R1's default cfg trained cleanly to val AUC 0.966), so most configs should train — but
nothing is silently discarded.

---

## 5. Final training + test-once (Phase R4, after R3 sign-off)

- `04_rnn_final.py`: arms **(confirmed at the R3 checkpoint, 2026-07-14 — the 4-arm set:
  "add the AUC-selected winner", free here because the search's F1-winner and AUC-winner are the
  same config `lr1e-04_do0.2_h256_nl2`)** —
  `rnn_f1_winner` (`--select f1`, headline) · `rnn_winner_auc` (same winner cfg, `--select auc`,
  the dedicated AUC-optimized h256 — closes the "no AUC-tuned large model" gap the GRU study had
  to flag) · `rnn_default_f1` (h128, `--select f1`, un-searched-RNN control, analogue of
  `transformer_default`/`gru_default_f1`) · `rnn_default_auc` (same h128 cfg, `--select auc`,
  matched-capacity + matched-selection AUC twin of the frozen BiLSTM). Each × 5 seeds → full run
  dirs `runs_final/<arm>/seed<k>/{best.pt, final.json, history.json, norm_*.npy}`. Val-only
  (engine has no test path); checkpoints saved for the paired comparison.
- `05_rnn_test_eval.py` — **THE single designated test-touch** (mirrors
  `f1_optimization/05_final_test_eval.py` / `gru/.../05_gru_test_eval.py`): load each arm's
  checkpoints, regen val+test probs on CPU; **parity gate** (frozen-BiLSTM per-seed test AUC
  must match stored `final.json` before any number is emitted); per-seed τ\* on that seed's own
  val probs, ensemble τ\* on averaged val probs; report F1/acc/AUC per-seed-mean and ensemble,
  at 0.5 and at τ\*.
- `06_rnn_loso.py` — 6-fold LOSO (Issue-5 protocol): per-fold grouped 85/15 val split by
  `set_id/ped_id`, per-fold pos_weight = n_neg/n_pos, seed 42; train via `engine.train_run`,
  score the held-out set from the saved checkpoint. Assert fold `test_n` =
  258/310/2094/1610/47/587 (Issue-5 fingerprint of genuine data).

---

## 6. Statistical comparison plan (Phase R5, local)

- `07_compare.py` — **10k paired percentile bootstrap** of ΔF1 (primary) and ΔAUC (secondary),
  same resample indices for both models (`np.random.default_rng(42)`), τ fixed from val,
  reusing `f1_optimization/00_common.py::paired_bootstrap`. Compare `rnn_f1_winner` /
  `rnn_default_auc` vs each target (frozen BiLSTM, BiLSTM-F1, Transformer-F1, searched
  Transformer, **and GRU-F1** — the un-gated-vs-gated recurrent landscape) from `probs_cache/`.
- `08_cluster_bootstrap.py` — **pedestrian-cluster** CI (the honest interval) reusing
  `f1_optimization/07_cluster_bootstrap.py::cluster_paired_delta`/`cluster_ci`; reported
  alongside every primary ΔF1.
- `09_latency.py` — Issue-9 protocol (50 warmup + 1000 timed, CPU & MPS × batch{1,8,32}).
- `10_loso_report.py` — fold table vs BiLSTM 0.928 / GRU 0.946 / Transformer 0.939.

### Pre-registered outcome templates (verbatim in the report)
Primary — **F1**, paired bootstrap on the 2094 test windows; report the cluster bootstrap
alongside. Δ = RNN − comparison. **WIN** if 95% CI excludes 0 positive / **TIE** if CI
includes 0 / **LOSS** if excludes 0 negative.

1. **RNN-F1 vs frozen BiLSTM** (F1 0.828) — expect WIN from the F1-first *discipline*, not a
   cell effect (mirrors GRU #1).
2. **RNN-F1 vs BiLSTM-F1** (0.844) — the key gating-isolation F1 comparison.
3. **RNN-F1 vs Transformer-F1** (0.847).
4. **RNN-F1 vs GRU-F1** (0.849) — un-gated vs gated recurrent, the sharpest cell-landscape test.
5. **RNN-default-F1 vs frozen BiLSTM** — un-searched-RNN control (F1-first discipline check).

Secondary — Δaccuracy, ΔAUC:
6. **RNN-default-AUC (h128, matched selection) vs frozen BiLSTM** AUC (0.9324) — matched-
   capacity AUC cell-isolation (analogue of GRU #5).
7. **RNN-winner-AUC (h256, AUC-selected) vs frozen BiLSTM** AUC — the AUC-optimized large RNN.
8. **RNN-winner-AUC vs searched Transformer** AUC (~0.9497).

A TIE with the recurrent models strengthens the "input signal, not the cell" story; a clean
LOSS would show gating matters over this window. Either is reported as plainly as a win.
**No hedging, no best-seed cherry-picking.**

---

## 7. Phases

### Phase R1 — Setup & sanity gates — ✅ DONE (2026-07-14)
`phase1_setup/01_sanity_checks.py` → `01_sanity_report.md`. **ALL 5 GATES PASS.** birnn default
= **149,121** params exact; CPU determinism |Δ|=0 (bit-identical same-seed, val F1 0.825083 /
val AUC 0.966233, best epoch 13, ~8s/run); engine reproduces the published BiLSTM (594,561,
val AUC 0.951687 twice). Param ladder h64/h128/h256 × {1,2}L spans **0.03×–0.94×** the BiLSTM
— the whole vanilla-RNN family is smaller than the BiLSTM. Orientation (not a result): the
un-gated default already reaches val F1 0.825 / val AUC 0.966 at seed 42, in the BiLSTM
ballpark and with no sign of divergence — the real comparison awaits the searched winner on
the untouched test set.

### Phase R2 — Search (val-only, local CPU) — ✅ DONE (2026-07-14)
`phase2_search/02_rnn_search.py` — grid + multiseed + pos_weight sweep on CPU (93 run files,
**0 test keys, 0 diverged runs**). **F1-winner = AUC-winner (they agree): `lr1e-04_do0.2_h256_nl2`**
(h256/nl2/do0.2/lr1e-4, **560,001 params**), mean val F1 0.8554 ± 0.0141 / val AUC 0.9721 ± 0.0051
— the exact config Issue-8's grid picked as the BiLSTM AUC-winner (the un-gated RNN independently
converged on the BiLSTM's own AUC-optimal recipe). pos_weight anchor 1.682 retained. Search
converged on h256 (as BiLSTM-F1 and the GRU did); selection-noise control mattered again (seed-42
F1-leader ≠ 5-seed winner).

### Phase R3 — Search review (human checkpoint) — ✅ DONE (2026-07-14)
`phase3_search_review/03_search_report.py` re-derived every ranking from the raw JSONs and
cross-checked `_stage_summary.json` — **ALL MATCH** (incl. the instability-ledger count); 93 files
scanned, zero test keys → `03_search_summary.md`. Instability ledger: **0 diverged runs** (the
vanilla tanh RNN is stable at every searched setting). **⏸ Awaiting user confirmation of the
winner + pos_weight + R4 arm set before R4.**

### Phase R4 — Final + test-once (local CPU) — ✅ DONE (2026-07-14)
`04_rnn_final.py` (20 checkpoints, val-only; reproduces the search's 0.8554 val F1 exactly) ·
`05_rnn_test_eval.py` (**PARITY GATE PASS, |Δ|=0.00e+00 all 5 seeds**; test read once; RNN-F1
per-seed F1 **0.852**, AUC **0.948**) · `06_rnn_loso.py` (6-fold LOSO AUC **0.937 ± 0.040**; fold
sizes match Issue-5 fingerprint; set03 fold 0.944).

### Phase R5 — Analysis — ✅ DONE (2026-07-14) — VERDICT
`07_compare.py`: **RNN-F1 TIES BiLSTM-F1 (ΔF1 +0.0033), Transformer-F1 (+0.0025), and GRU-F1
(−0.0038) on F1; RNN ties-or-edges the frozen BiLSTM on AUC at matched h128 (ΔAUC +0.0059, WIN);
the AUC-selected RNN TIES the searched transformer (ΔAUC −0.0013) — which the GRU could NOT reach.
No cell-isolation endpoint is a loss.** `08_cluster_bootstrap.py`: **all cell-isolation verdicts
survive** clustering (only the un-searched-control discipline WIN softens to TIE). `09_latency.py`:
RNN 0.316 ms/window CPU b1 — the **fastest** family. `10_loso_report.py`: LOSO 0.937, same band.

### Phase R6 — Docs & integration — ✅ DONE (2026-07-14)
`SUPERVISOR_SUMMARY.md` written; RNN rows added to
`journal_prep/issue3_baseline_comparison/{03_baseline_comparison,05_master_comparison_table}.md`;
root docs (`CLAUDE.md`, `pipeline/CODE_STATE.md`) + memory updated. Manuscript integration +
RNN–GRU–LSTM landscape cross-link flagged (not forced).

---

## 8. Key invariants (carry-over from journal_prep / transformer / gru)
1. Test set03 touched **exactly once**, in `05_rnn_test_eval.py`, on best-val checkpoints —
   search/final-training code physically contains no test-eval path.
2. Normalization train-only, saved per run.
3. pos_weight 1.682 everywhere except the sweep and LOSO (per-fold n_neg/n_pos).
4. seed 42 = canonical single-seed reference; 5-seed mean±std (ddof=1) is the reported number.
5. `torch.load(..., weights_only=False)` for every `best.pt`.
6. Threshold 0.5; τ\* reported for F1 but never chosen on test.
7. The BiLSTM/GRU/Transformer baselines are never retrained; their stored numbers/probs are the
   comparison rows.
8. `num_workers=0` in every DataLoader (engine default).
9. Every results table states the device that produced it (all M4-CPU here).

## 9. Risks & pitfalls
- **Vanishing/exploding gradients** — the vanilla RNN's specific risk; handled by the
  instability ledger (record, don't drop). R1 shows the default trains cleanly.
- **Val N=634 selection noise** — candidate gaps < ~0.01 are noise; hence Stage-2 5-seed mean
  selection (Issue-8 precedent).
- **Seed σ vs effect size** — the window-paired + pedestrian-cluster bootstrap is the primary
  evidence, not the n=5 t-test.
- **F1 vs AUC winners may differ** — carry both; primary verdict is F1-first.
- **Engine confound honesty** — the RNN is trained on the unified CPU engine; the frozen
  BiLSTM is loaded as its canonical published checkpoints (same discipline the GRU/transformer
  studies used). Issue-12 proved the engine is bit-equivalent to the published pipeline.

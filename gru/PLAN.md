# GRU Extension — Master Plan

**Created:** 2026-07-13 · **Status:** ALL PHASES (G1–G6) DONE — verdict: **TIE with the
BiLSTM on F1 and on AUC** (the recurrent cell doesn't matter; the input signal does) ·
**Requested by:** supervisor (test two more model families — GRU here, vanilla RNN in a
parallel session — on the existing clean pipeline, identical protocol).
**Companion docs:** `README.md` (index + how to run) · `PROGRESS_LOG.md` (chronological
numbers log) · `SUPERVISOR_SUMMARY.md` (plain-English write-up, G6).

This is the **recurrent-cell analogue of the `transformer/` study**: same pre-registration
discipline, same frozen protocol, same comparison machinery. Where the transformer study
isolated *attention vs recurrence*, this one isolates *the recurrent cell type* — a GRU is
the gated recurrent twin of our BiLSTM (identical input projection, bidirectional recurrence,
last-step readout, linear head; only `nn.LSTM` → `nn.GRU`). This plan is written **before
any GRU result exists**; the search space, selection rules, and verdict templates are
pre-registered so no decision is made after seeing test numbers. Each phase is closed by
appending a `### ✅ DONE` block with the real numbers (journal_prep convention).

**The question this answers:** *under the identical clean protocol and the F1-first hierarchy,
how does a GRU compare to the frozen BiLSTM and the searched Transformer?*
**Hypothesis (data decides):** a GRU typically ties the LSTM — a TIE strengthens the thesis
story that *the input signal, not the recurrent cell, is what matters*. Any clean difference
is still a valid, plainly-reported result.

---

## 1. The models we compare against (never retrained)

| model (5 seeds [42,0,1,2,3]) | params | test AUC | F1 | Acc | source |
|---|---|---|---|---|---|
| **Frozen BiLSTM (locked, Issue 2)** | 594,561 | **0.9324 ± 0.0114** | 0.828 | 0.883 | `journal_prep/issue2_clean_protocol/runs_clean/multiseed/` |
| Transformer (searched) | 794,241 | ~0.950 (0.9497 ± 0.0025) | 0.845 | 0.894 | `transformer/phase4_kaggle_final/` |
| BiLSTM-F1 (F1-first program) | ≈2–3M (h256) | 0.940 | 0.844 ± 0.008 | 0.897 | `f1_optimization/runs_f1/lstm_..._h256/` |
| Transformer-F1 (F1-first program) | 794,241 | 0.947 | 0.847 ± 0.017 | 0.896 | `f1_optimization/runs_f1/transformer_searched/` |
| **GRU (default h128) — this study** | **446,081** | TBD | TBD | TBD | this folder |

Frozen-BiLSTM per-seed test AUC (parity-gate reference): seed42 0.913114, seed0 0.933424,
seed1 0.943189, seed2 0.936295, seed3 0.935822.

**Cached comparison probs** already exist in `f1_optimization/probs_cache/` (val+test vectors
+ `y_val`/`y_test`): `lstm_frozen`, `lstm_a3` (BiLSTM-F1), `tf_frozen` (Transformer searched),
`tf_b3` (Transformer-F1). The GRU study reuses these for the paired bootstrap rather than
regenerating them.

---

## 2. Fairness contract (byte-for-byte identical to the BiLSTM/Transformer protocol)

**FROZEN — not searchable:**

| item | value |
|---|---|
| engine | `journal_prep/issue12_unified_pipeline/12_unified_engine.py --family gru` — **no new trainer** |
| device | **local CPU only** (`--device cpu`) — recurrent CPU training is bit-reproducible & context-free (issue12); MPS is process-history-dependent |
| data | `journal_prep/issue2_clean_protocol/sequences_clean/` reused verbatim (engine's canonical path) |
| splits | train {set01,02,04} N=2178 · val {set05,06} N=634 · test {set03} N=2094 (681 pos, 32.5%) — asserted in `load_splits()` |
| normalization | per-feature z-score from **train only**, saved per run |
| loss | `BCEWithLogitsLoss(pos_weight=1.682)` (= 1366/812), except the pos_weight sweep and LOSO (per-fold) |
| threshold | 0.5 (+ val-tuned τ\* reported for F1) |
| batch / epochs | 32 / max 100 |
| early stopping | patience 15 on **val AUC** |
| seeds | [42, 0, 1, 2, 3]; seed 42 canonical |
| selection | **validation only**; test set03 evaluated **exactly once**, in `phase4_final/05_gru_test_eval.py`, **after the user confirms the winner (G3 gate)** |
| checkpoints | `torch.load(..., weights_only=False)` |
| reporting | always label **per-seed-mean vs 5-seed probability-ensemble** |

**SEARCHED — the GRU's own knobs (identical budget to the BiLSTM's Issue-8 search):**
`lr × dropout × hidden × num_layers` (the 36-config Issue-8 grid) + a pos_weight sweep — no
more, no less than the BiLSTM got. This isolates the cell type.

**Metric hierarchy: F1 → accuracy → AUC.** Primary verdict on F1; AUC reported as secondary
corroboration (incl. vs frozen BiLSTM 0.9324 and searched transformer ~0.9497).

---

## 3. The model

`RecurrentIntentPredictor("gru")` — defined in `12_unified_engine.py`, the twin of
`pipeline/03_bilstm_model.py::BiLSTMIntentPredictor`:
```
input (B,16,5) raw [x1,y1,x2,y2,vehicle_speed] pixels, z-scored
 → Linear(5, 64) + ReLU                       # input projection (proj_dim=64)
 → nn.GRU(64, hidden, num_layers, bidirectional=True, dropout, batch_first=True)
 → last-step readout  h[:, -1, :]  (dim = hidden*2)
 → Linear(hidden*2, 1)                         # logits — same contract as the BiLSTM
```
Default (h128/nl2/do0.3) = **446,081 params** (input_proj 384 + GRU 445,440 + head 257),
confirmed analytically and by the G1 gate. Only the cell (`nn.GRU` vs `nn.LSTM`) differs from
the BiLSTM, so the comparison isolates the cell type exactly as the transformer study isolated
attention.

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
val-AUC} ∪ `gru_default` (h128/nl2/do0.3, the BiLSTM-baseline recipe twin). **F1-winner =
highest mean val F1** (tie: acc, then AUC); **AUC-winner = highest mean val AUC** (noted
separately). Selection-noise control (Issue-8 precedent: the seed-42 leader was not the 5-seed
winner).

**Stage 3 — pos_weight sweep (5 seeds, `--select f1`):** F1-winner × pos_weight
∈ {1.0, 1.3, 1.682, 2.1, 2.5}, val-only, ranked by mean val F1. Keep 1.682 unless a value
beats it (mirrors `f1_optimization` G2). The AUC track keeps pos_weight 1.682 fixed (Issue 8
never swept it).

---

## 5. Final training + test-once (Phase G4, after G3 sign-off)

- `04_gru_final.py`: arms = `gru_f1_winner` (--select f1) · `gru_auc_winner` (--select auc,
  only if ≠ F1-winner) · `gru_default` (both --select f1 and auc). Each × 5 seeds → full run
  dirs `runs_final/<arm>/seed<k>/{best.pt, final.json, history.json, norm_*.npy}`. Val-only
  (engine has no test path); checkpoints saved for the paired comparison.
- `05_gru_test_eval.py` — **THE single designated test-touch** (mirrors
  `f1_optimization/05_final_test_eval.py`): load each arm's checkpoints, regen val+test probs
  on CPU; **parity gate** (frozen-BiLSTM per-seed test AUC must match stored `final.json`
  before any number is emitted); per-seed τ\* on that seed's own val probs, ensemble τ\* on
  averaged val probs; report F1/acc/AUC per-seed-mean and ensemble, at 0.5 and at τ\*.
- `06_gru_loso.py` — 6-fold LOSO (Issue-5 protocol): per-fold grouped 85/15 val split by
  `set_id/ped_id`, per-fold pos_weight = n_neg/n_pos, seed 42; train via `engine.train_run`,
  score the held-out set from the saved checkpoint. Assert fold `test_n` =
  258/310/2094/1610/47/587 (Issue-5 fingerprint of genuine data).

---

## 6. Statistical comparison plan (Phase G5, local)

- `07_compare.py` — **10k paired percentile bootstrap** of ΔF1 (primary) and ΔAUC (secondary),
  same resample indices for both models (`np.random.default_rng(42)`), τ fixed from val,
  reusing `f1_optimization/00_common.py::paired_bootstrap`. Compare `gru_f1_winner` /
  `gru_auc_winner` vs each target (frozen BiLSTM, BiLSTM-F1, Transformer-F1, searched
  Transformer) from `probs_cache/`.
- `08_cluster_bootstrap.py` — **pedestrian-cluster** CI (the honest interval) reusing
  `f1_optimization/07_cluster_bootstrap.py::cluster_paired_delta`/`cluster_ci`; reported
  alongside every primary ΔF1.
- `09_latency.py` — Issue-9 protocol (50 warmup + 1000 timed, CPU & MPS × batch{1,8,32}).
- `10_loso_report.py` — fold table vs BiLSTM 0.928 ± 0.041 / Transformer 0.939.

### Pre-registered outcome templates (verbatim in the report)
Primary — **F1**, paired bootstrap on the 2094 test windows; report the cluster bootstrap
alongside:
- **GRU vs frozen BiLSTM** — ΔF1: **WIN** if 95% CI excludes 0 positive / **TIE** if CI
  includes 0 / **LOSS** if excludes 0 negative.
- **GRU vs BiLSTM-F1 (0.844)** and **GRU vs Transformer-F1 (0.847)** — same templates.
Secondary — Δaccuracy, ΔAUC (incl. vs frozen BiLSTM 0.9324, searched transformer ~0.9497).
A TIE with the LSTM strengthens the "input signal, not the cell" story; any clean difference
is reported as plainly as a win. **No hedging, no best-seed cherry-picking.**

---

## 7. Phases

### Phase G1 — Setup & sanity gates — ✅ DONE (2026-07-13)
`phase1_setup/01_sanity_checks.py` → `01_sanity_report.md`. **ALL 5 GATES PASS.** GRU default
= **446,081** params exact; CPU determinism |Δ|=0 (bit-identical same-seed); engine reproduces
the published BiLSTM (594,561). Param ladder h64/h128/h256 × {1,2}L spans ~0.08×–2.8× the BiLSTM.

### Phase G2 — Search (val-only, local CPU) — ✅ DONE (2026-07-14)
`phase2_search/02_gru_search.py` — 89 run files, 0 test keys. F1-winner
`lr5e-04_do0.3_h256_nl2` (val F1 0.8683 ± 0.0241); AUC-winner `lr1e-03_do0.2_h256_nl2`
(val AUC 0.9760 ± 0.0017, differs); pos_weight sweep kept anchor 1.682. Search converged on
h256 (as the BiLSTM's own F1 program did); selection-noise control mattered again.

### Phase G3 — Search review (human checkpoint) — ✅ DONE (2026-07-14)
`phase3_search_review/03_search_report.py` re-derived every ranking from raw JSONs and
cross-checked `_stage_summary.json` — **ALL MATCH**; 89 files, zero test keys →
`03_search_summary.md`. **User confirmed the leaner arm set ("F1-winner + default only")** —
green light for G4.

### Phase G4 — Final + test-once (local CPU) — ✅ DONE (2026-07-14)
`04_gru_final.py` (15 checkpoints, val-only) · `05_gru_test_eval.py` (**PARITY GATE PASS,
|Δ|=0.00e+00 all 5 seeds**; test read once; GRU-F1 per-seed F1 0.849, AUC 0.941) ·
`06_gru_loso.py` (6-fold LOSO AUC 0.946; fold sizes match Issue-5 fingerprint).

### Phase G5 — Analysis — ✅ DONE (2026-07-14) — VERDICT
`07_compare.py`: **GRU-F1 TIES BiLSTM-F1 (ΔF1 +0.0071) and Transformer-F1 (ΔF1 +0.0063);
GRU-AUC TIES frozen BiLSTM at matched h128/selection (ΔAUC −0.0008); GRU LOSES to the searched
transformer on AUC (ΔAUC −0.0070).** `08_cluster_bootstrap.py`: **all verdicts survive**
pedestrian clustering. `09_latency.py`: GRU 0.721 ms/window CPU b1. `10_loso_report.py`:
LOSO 0.946, same band as BiLSTM/Transformer.

### Phase G6 — Docs & integration — ✅ DONE (2026-07-14)
`SUPERVISOR_SUMMARY.md` written; GRU rows added to
`journal_prep/issue3_baseline_comparison/{03_baseline_comparison,05_master_comparison_table}.md`;
root docs + memory updated. Manuscript integration + vanilla-RNN cross-link flagged (not forced).

---

## 8. Key invariants (carry-over from journal_prep / transformer)
1. Test set03 touched **exactly once**, in `05_gru_test_eval.py`, on best-val checkpoints —
   search/final-training code physically contains no test-eval path.
2. Normalization train-only, saved per run.
3. pos_weight 1.682 everywhere except the sweep and LOSO (per-fold n_neg/n_pos).
4. seed 42 = canonical single-seed reference; 5-seed mean±std (ddof=1) is the reported number.
5. `torch.load(..., weights_only=False)` for every `best.pt`.
6. Threshold 0.5; τ\* reported for F1 but never chosen on test.
7. The BiLSTM/Transformer baselines are never retrained; their stored numbers/probs are the
   comparison rows.
8. `num_workers=0` in every DataLoader (engine default).
9. Every results table states the device that produced it (all M4-CPU here).

## 9. Risks & pitfalls
- **Val N=634 selection noise** — candidate gaps < ~0.01 are noise; hence Stage-2 5-seed mean
  selection (Issue-8 precedent).
- **Seed σ vs effect size** — the window-paired + pedestrian-cluster bootstrap is the primary
  evidence, not the n=5 t-test.
- **F1 vs AUC winners may differ** — carry both; primary verdict is F1-first.
- **Engine confound honesty** — the GRU is trained on the unified CPU engine; the frozen
  BiLSTM is loaded as its canonical published checkpoints (same discipline the transformer
  study used). Issue-12 proved the engine is bit-equivalent to the published pipeline.

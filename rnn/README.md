# rnn/ — Vanilla-RNN extension (bidirectional RNN vs BiLSTM vs GRU vs Transformer)

Supervisor-requested follow-up: test additional model families on the existing clean pipeline
under the identical protocol. This folder is the **un-gated recurrent-cell analogue of `gru/`
and `transformer/`** — it isolates **gating** (a bidirectional vanilla tanh RNN vs the
BiLSTM's gated LSTM cell) exactly as the GRU study isolated the gated cell type and the
transformer study isolated attention. Master plan with all pre-registered decisions:
**[PLAN.md](PLAN.md)**. Chronological numbers: **[PROGRESS_LOG.md](PROGRESS_LOG.md)**.

**One-paragraph story.** The vanilla RNN (`birnn`) is the un-gated twin of our BiLSTM
(identical input projection, bidirectional recurrence, last-step readout — only `nn.LSTM` →
`nn.RNN`, tanh cell). It gets the identical frozen protocol (same windows, splits, pos_weight
1.682, seeds, val-only selection, test-set03-touched-once) and the **identical search budget
the BiLSTM got in Issue 8** (the 36-config `lr × dropout × hidden × num_layers` grid + a
pos_weight sweep), all **local on CPU** through the one unified engine. Under the supervisor's
**F1 → accuracy → AUC** hierarchy, we then compare it — with a 10k window-paired and
pedestrian-cluster bootstrap — against the frozen BiLSTM, the searched Transformer, both
F1-first models, and the GRU. Because the GRU (still gated) already tied the BiLSTM, the
vanilla RNN is the sharpest remaining test of "the input signal, not the recurrent cell, is
what matters". **Verdict: the un-gated RNN TIES the gated models on F1** (vs BiLSTM-F1 ΔF1
+0.0033, vs GRU-F1 −0.0038, both CIs include 0) and ties-or-edges the frozen BiLSTM on AUC at
matched capacity — **no cell-isolation endpoint is a loss, so removing the LSTM's gating costs
nothing measurable over a 16-step window.** And — unlike the GRU, which lost to the searched
transformer on AUC — the **AUC-optimized vanilla RNN ties the searched transformer** (ΔAUC
−0.0013, CI includes 0), confirming that transformer's edge was its *search*, not attention over
recurrence. The vanilla RNN is also the **smallest and fastest** of the four families. All
findings survive the pedestrian-cluster bootstrap.

## Status at a glance — **ALL PHASES DONE ✅**

| phase | what | where | status |
|---|---|---|---|
| R1 | Setup + sanity gates 0–4 (149,121 params, determinism) | local CPU | ✅ all pass |
| R2 | Search (grid + multiseed + pos_weight sweep, val-only) | local CPU | ✅ winner h256, 0 diverged |
| R3 | Search review (human checkpoint, test untouched) | local | ✅ user confirmed (4-arm set) |
| R4 | Final + **test once** (parity |Δ|=0) + LOSO | local CPU | ✅ done |
| R5 | Paired + cluster bootstrap, latency, LOSO report | local | ✅ **verdict: TIE (gating unnecessary)** |
| R6 | Docs, baseline-table integration | local | ✅ done |

See **[SUPERVISOR_SUMMARY.md](SUPERVISOR_SUMMARY.md)** for a plain-English write-up.

## Headline result (5-seed, test set03)

| model | params | test AUC | test F1 (per-seed) | Acc | vs BiLSTM |
|---|---|---|---|---|---|
| **Vanilla RNN (F1-winner, h256) — this study** | 560,001 | 0.948 (ens 0.955) | **0.852** (ens 0.859) | 0.902 | **TIE** |
| Vanilla RNN (winner h256, AUC-selected) | 560,001 | 0.948 (ens 0.955) | 0.845 (ens 0.863) | 0.910 | **ties searched TF on AUC** |
| Vanilla RNN (default h128, F1) | 149,121 | 0.942 (ens 0.947) | 0.844 (ens 0.851) | 0.897 | TIE (= BiLSTM-F1) |
| Vanilla RNN (default h128, AUC) | 149,121 | 0.942 (ens 0.948) | 0.836 (ens 0.852) | 0.889 | WIN on AUC (matched size) |
| Frozen BiLSTM | 594,561 | 0.9324 ± 0.0114 | 0.828 | 0.883 | — |
| BiLSTM-F1 | ≈2–3M (h256) | 0.940 | 0.844 ± 0.008 | 0.897 | — |
| GRU-F1 | 1,678,209 | 0.941 | 0.849 | 0.901 | — |
| Transformer (searched) | 794,241 | ~0.950 | 0.845 | 0.894 | RNN **ties** on AUC |
| Transformer-F1 | 794,241 | 0.947 | 0.847 ± 0.017 | 0.896 | — |

Formal verdicts (10k paired bootstrap, F1-first): RNN-F1 vs BiLSTM-F1 **TIE** (ΔF1 +0.0033); vs
Transformer-F1 **TIE** (+0.0025); vs GRU-F1 **TIE** (−0.0038); RNN-default-AUC vs frozen BiLSTM
(matched h128) **WIN** (ΔAUC +0.0059); RNN-winner-AUC vs searched Transformer **TIE** (ΔAUC
−0.0013). All cell-isolation verdicts survive the pedestrian-cluster bootstrap. Latency: RNN CPU
batch-1 **0.316 ms/window** (fastest family). LOSO 6-fold AUC 0.937 (same band).

Data: `journal_prep/issue2_clean_protocol/sequences_clean/` (N=4906; train 2178 / val 634 /
test 2094, 32.5% pos) — read from the canonical path via the unified engine (no local copy).
Frozen BiLSTM checkpoints stay in
`journal_prep/issue2_clean_protocol/runs_clean/multiseed/seed*/` (never retrained here).

## How to run (phase by phase)

```bash
source .venv/bin/activate            # torch 2.12, sklearn 1.9, scipy 1.17 (all present)

# R1 — local sanity gates
python rnn/phase1_setup/01_sanity_checks.py            # gates 0–4 → 01_sanity_report.md

# R2 — search (val-only, CPU, resumable/cached)
python rnn/phase2_search/02_rnn_search.py              # grid + multiseed + pos_weight sweep

# R3 — search review (BEFORE any test evaluation exists)
python rnn/phase3_search_review/03_search_report.py    # → 03_search_summary.md; user confirms

# R4 — final + test-once (only after R3 sign-off)
python rnn/phase4_final/04_rnn_final.py                # winner + default × 5 seeds, val-only
python rnn/phase4_final/05_rnn_test_eval.py            # THE single test-touch (parity-gated)
python rnn/phase4_final/06_rnn_loso.py                 # 6-fold LOSO

# R5 — analysis
python rnn/phase5_analysis/07_compare.py               # paired bootstrap ΔF1/ΔAUC + verdicts
python rnn/phase5_analysis/08_cluster_bootstrap.py     # pedestrian-cluster CIs
python rnn/phase5_analysis/09_latency.py               # Issue-9 latency protocol
python rnn/phase5_analysis/10_loso_report.py           # LOSO fold table
```

## Files

```
rnn/
├── PLAN.md, README.md, PROGRESS_LOG.md   ← tracking docs (span all phases)
├── SUPERVISOR_SUMMARY.md                 ← R6: plain-English write-up
├── phase1_setup/                         ← R1: sanity gates
├── phase2_search/                        ← R2: staged search (val-only)
├── phase3_search_review/                 ← R3: local winner review
├── phase4_final/                         ← R4: final training + test-once + LOSO
└── phase5_analysis/                      ← R5: comparison, cluster CI, latency, LOSO report
```

Conventions inherited from `journal_prep/` / `transformer/` / `gru/`: val-only selection,
test-once, train-only norm, pos_weight 1.682, seeds [42,0,1,2,3], `weights_only=False`,
threshold 0.5, run scripts from the repo root. All training via
`journal_prep/issue12_unified_pipeline/12_unified_engine.py --family birnn` — **no new trainer.**

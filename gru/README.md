# gru/ — GRU extension (GRU vs BiLSTM vs Transformer, clean PIE protocol)

Supervisor-requested follow-up: test additional model families on the existing clean pipeline
under the identical protocol. This folder is the **recurrent-cell analogue of `transformer/`**
— it isolates the *recurrent cell type* (GRU vs the BiLSTM's LSTM) exactly as the transformer
study isolated *attention vs recurrence*. Master plan with all pre-registered decisions:
**[PLAN.md](PLAN.md)**. Chronological numbers: **[PROGRESS_LOG.md](PROGRESS_LOG.md)**.

**One-paragraph story.** The GRU is the gated recurrent twin of our BiLSTM (identical input
projection, bidirectional recurrence, last-step readout — only `nn.LSTM` → `nn.GRU`). It gets
the identical frozen protocol (same windows, splits, pos_weight 1.682, seeds, val-only
selection, test-set03-touched-once) and the **identical search budget the BiLSTM got in
Issue 8** (the 36-config `lr × dropout × hidden × num_layers` grid + a pos_weight sweep), all
**local on CPU** through the one unified engine. Under the supervisor's **F1 → accuracy → AUC**
hierarchy, we then compare it — with a 10k window-paired and pedestrian-cluster bootstrap —
against the frozen BiLSTM, the searched Transformer, and both F1-first models.
**Verdict: the GRU TIES the BiLSTM** — statistically indistinguishable on F1 (vs BiLSTM-F1
ΔF1 +0.0071, CI includes 0) and on AUC at matched capacity/selection (vs frozen BiLSTM
ΔAUC −0.0008). The searched transformer keeps its AUC edge (GRU loses on AUC), confirming that
edge is the *search*, not attention-vs-recurrence. **The recurrent cell doesn't matter — the
input signal does.** All findings survive the pedestrian-cluster bootstrap.

## Status at a glance — **ALL PHASES DONE ✅**

| phase | what | where | status |
|---|---|---|---|
| G1 | Setup + sanity gates 0–4 (446,081 params, determinism) | local CPU | ✅ all pass |
| G2 | Search (grid + multiseed + pos_weight sweep, val-only) | local CPU | ✅ F1-winner h256 |
| G3 | Search review (human checkpoint, test untouched) | local | ✅ user confirmed |
| G4 | Final + **test once** (parity |Δ|=0) + LOSO | local CPU | ✅ done |
| G5 | Paired + cluster bootstrap, latency, LOSO report | local | ✅ **verdict: TIE** |
| G6 | Docs, baseline-table integration | local | ✅ done |

See **[SUPERVISOR_SUMMARY.md](SUPERVISOR_SUMMARY.md)** for a plain-English write-up.

## Headline result (5-seed, test set03)

| model | params | test AUC | test F1 (per-seed) | Acc | vs BiLSTM |
|---|---|---|---|---|---|
| **GRU (F1-winner, h256) — this study** | 1,678,209 | 0.941 (ens 0.949) | **0.849** (ens 0.863) | 0.901 | **TIE** |
| GRU (default h128, F1-selected) | 446,081 | 0.939 (ens 0.946) | 0.844 (ens 0.852) | 0.898 | TIE (= BiLSTM-F1) |
| GRU (default h128, AUC-selected) | 446,081 | 0.933 (ens 0.942) | 0.840 (ens 0.847) | 0.898 | TIE on AUC |
| Frozen BiLSTM | 594,561 | 0.9324 ± 0.0114 | 0.828 | 0.883 | — |
| BiLSTM-F1 | ≈2–3M (h256) | 0.940 | 0.844 ± 0.008 | 0.897 | — |
| Transformer (searched) | 794,241 | ~0.950 | 0.845 | 0.894 | GRU loses on AUC |
| Transformer-F1 | 794,241 | 0.947 | 0.847 ± 0.017 | 0.896 | — |

Formal verdicts (10k paired bootstrap, F1-first): GRU-F1 vs BiLSTM-F1 **TIE** (ΔF1 +0.0071);
vs Transformer-F1 **TIE** (ΔF1 +0.0063); GRU-AUC vs frozen BiLSTM (matched h128) **TIE**
(ΔAUC −0.0008); GRU vs searched Transformer **LOSS on AUC** (ΔAUC −0.0070). All survive the
pedestrian-cluster bootstrap.

Data: `journal_prep/issue2_clean_protocol/sequences_clean/` (N=4906; train 2178 / val 634 /
test 2094, 32.5% pos) — read from the canonical path via the unified engine (no local copy).
Frozen BiLSTM checkpoints stay in
`journal_prep/issue2_clean_protocol/runs_clean/multiseed/seed*/` (never retrained here).

## How to run (phase by phase)

```bash
source .venv/bin/activate            # torch 2.12, sklearn 1.9, scipy 1.17 (all present)

# G1 — local sanity gates
python gru/phase1_setup/01_sanity_checks.py            # gates 0–4 → 01_sanity_report.md

# G2 — search (val-only, CPU, resumable/cached)
python gru/phase2_search/02_gru_search.py              # grid + multiseed + pos_weight sweep

# G3 — search review (BEFORE any test evaluation exists)
python gru/phase3_search_review/03_search_report.py    # → 03_search_summary.md; user confirms

# G4 — final + test-once (only after G3 sign-off)
python gru/phase4_final/04_gru_final.py                # winner + default × 5 seeds, val-only
python gru/phase4_final/05_gru_test_eval.py            # THE single test-touch (parity-gated)
python gru/phase4_final/06_gru_loso.py                 # 6-fold LOSO

# G5 — analysis
python gru/phase5_analysis/07_compare.py               # paired bootstrap ΔF1/ΔAUC + verdicts
python gru/phase5_analysis/08_cluster_bootstrap.py     # pedestrian-cluster CIs
python gru/phase5_analysis/09_latency.py               # Issue-9 latency protocol
python gru/phase5_analysis/10_loso_report.py           # LOSO fold table
```

## Files

```
gru/
├── PLAN.md, README.md, PROGRESS_LOG.md   ← tracking docs (span all phases)
├── SUPERVISOR_SUMMARY.md                 ← G6: plain-English write-up
├── phase1_setup/                         ← G1: sanity gates
├── phase2_search/                        ← G2: staged search (val-only)
├── phase3_search_review/                 ← G3: local winner review
├── phase4_final/                         ← G4: final training + test-once + LOSO
└── phase5_analysis/                      ← G5: comparison, cluster CI, latency, LOSO report
```

Conventions inherited from `journal_prep/` / `transformer/`: val-only selection, test-once,
train-only norm, pos_weight 1.682, seeds [42,0,1,2,3], `weights_only=False`, threshold 0.5,
run scripts from the repo root. All training via
`journal_prep/issue12_unified_pipeline/12_unified_engine.py --family gru` — **no new trainer.**

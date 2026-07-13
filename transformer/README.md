# transformer/ — Transformer extension (BiLSTM vs Transformer, clean PIE protocol)

Supervisor-requested extension: build the **best Transformer** intention predictor on the
existing pipeline and dataset, then compare it against the locked BiLSTM under the exact
clean protocol. Master plan with all pre-registered decisions: **[PLAN.md](PLAN.md)**.
Chronological numbers: **[PROGRESS_LOG.md](PROGRESS_LOG.md)**.

**Folder layout: one subfolder per phase.** Each phase folder holds only what that phase
needs; shared inputs (`sequences_clean/`, this file, `PLAN.md`, `PROGRESS_LOG.md`) sit at
the `transformer/` root. See "Files" below for the full map.

**One-paragraph story.** The journal-prep program closed with a leakage-free BiLSTM at
**test AUC 0.932 ± 0.011** (5 seeds) using only bbox + ego-speed, and an explicit open
question ("Why BiLSTM, not a Transformer?"). This folder answers it empirically: a small
pre-LN Transformer encoder gets the identical frozen protocol (same windows, splits,
pos_weight 1.682, seeds, val-only selection, test-set03-touched-once) plus its own
staged 78-config search — more than 2× the search budget the LSTM got in Issue 8 — and is
then compared with a 10k window-paired bootstrap of ΔAUC against the frozen LSTM
checkpoints. **Verdict: WIN on AUC** — the searched transformer measurably beats the BiLSTM on AUC
(ΔAUC = +0.0135, 95% CI [+0.0097, +0.0174], excludes 0; paired t-test p=0.025), while the
un-searched default recipe ties it exactly — the search, not the architecture family
alone, is what won.

## Status at a glance

| phase | what | where | status |
|---|---|---|---|
| T1 | Model + sanity gates 0–3 | local (M4) | ✅ done — all gates pass |
| T2 | Staged search A/B/transfer/C (val-only, 102 runs) | **Kaggle T4×2** | ✅ done — winner found |
| T3 | Winner review (human checkpoint, test still untouched) | local | ✅ done — winner confirmed by user |
| T4 | Final: winner + default × 5 seeds (**test once**) + LOSO | **Kaggle T4×2** | ✅ done — test AUC 0.9497 ± 0.0025 (winner) |
| T5 | Paired comparison, latency, LOSO report | local | ✅ done — **verdict: WIN** |
| T6 | Docs, baseline-table + manuscript integration | local | ✅ done |

**All six phases complete.** See **[SUPERVISOR_SUMMARY.md](SUPERVISOR_SUMMARY.md)** for
a plain-English write-up of the whole extension and the result, ready to walk a
supervisor through.

## Headline result (Phase T5 — the formal verdict)

| model | params | test AUC (seed-avg, 95% CI) | test latency (CPU, batch-1) |
|---|---|---|---|
| BiLSTM baseline (frozen) | 594,561 | 0.9423 [0.9306, 0.9533] | 0.575 ms/window |
| transformer_default | 268,417 | 0.9428 [0.9312, 0.9538] | — |
| **transformer_searched (winner)** | 794,241 | **0.9558 [0.9453, 0.9656]** | 0.459 ms/window |

**Why the BiLSTM shows 0.9423 here, not the 0.9324 ± 0.0114 everywhere else in the
repo:** this table's number combines the 5 seeds' predicted probabilities into one
vector before scoring (needed so the paired bootstrap below has a single probability
vector per model) — a different, both-correct statistic from the plain average of the
5 seeds' own AUCs. **0.9324 ± 0.0114 remains the canonical, citable BiLSTM number**;
0.9423 is specific to this comparison. Full explanation:
`phase5_analysis/05_comparison_report.md`.

**WIN.** Paired bootstrap (10k resamples, same 2094 test windows, same resample indices
across models) of ΔAUC (`transformer_searched` − BiLSTM) = **+0.0135, 95% CI
[+0.0097, +0.0174]** (excludes 0); paired t-test over the 5 seeds t=3.50, p=0.025
(significant). `transformer_default` vs BiLSTM: Δ=+0.0005, CI [-0.0034, +0.0043]
(includes 0) — a clean tie, confirming the win came from the search, not just from
being a transformer. LOSO 6-fold: 0.939 ± 0.044 vs BiLSTM's 0.928 ± 0.045 (both
recomputed with `ddof=1`), consistent with the fixed-split result. Latency: the
searched transformer is actually *faster* than the BiLSTM per window on this M4
(0.459 vs 0.575 ms CPU batch-1) despite 1.3× the parameters — both are ~2 orders of
magnitude inside a 30 fps budget either way. Full reports:
`phase5_analysis/05_comparison_report.md` (the verdict), `04_final_summary.md`,
`06_latency_report.md`, `07_loso_report.md`.

**Phase T2/T3 result:** the search picked `d128_ff512_L4_last_spe__adam_lr1e-03_plateau_
do0.1_wd1e-05` (794,241 params) over `transformer_default`: **val AUC 0.9789 ± 0.0038**
vs 0.9629 ± 0.0056 — a deeper (4-layer), last-token-pooled, sinusoidal-PE architecture
that meaningfully beat the un-searched default (ranked 66th/78). Full report:
`phase3_search_review/03_search_summary.md` / `03_search_figure.png`. **This is
validation-set only** — the real LSTM-vs-transformer comparison happens in Phase T5 on
the untouched test set.

**Phase T1 result:** all 4 gates pass (`phase1_setup/01_sanity_report.md`).
`transformer_default` (d128/L2/ff256/cls/learned) = 268,417 params vs the BiLSTM's
594,561. Model verified forward+backward on both CPU and MPS. See PLAN.md's Phase T1
block for the full report, including a construction-order seeding bug found and fixed in
the sanity script (the real training engine, `02_train_transformer.py`, was already
correct).

**Phase T2 status:** ran on Kaggle T4×2, output downloaded and verified (102 run files,
78 distinct configs, zero test-set contamination). **Phase T3 status:**
`03_search_report.py` independently recomputed every number from the raw files and
confirmed exact agreement with the notebook's own summary — winner confirmed by the user
2026-07-11.

**Phase T4 status: DONE.** Ran on Kaggle T4×2, all 17 trainings (Stage D's 10 + the
determinism rerun + 6 LOSO folds) completed in ~8 minutes. One bug hit the notebook's
own final cell (`KeyError: 'n_params'`, a field deliberately not persisted in
`final.json`) — fixed and re-run in the same live Kaggle session at zero retraining
cost (full writeup in `PROGRESS_LOG.md`). Output independently re-verified locally
(own script, not committed): every 5-seed/6-fold mean±std recomputed from the raw
JSONs matches `_final_summary.json` exactly; all 6 LOSO fold sizes match Issue 5's
real per-set counts exactly (confirms genuine data, no corruption); determinism rerun
is an exact bit-for-bit match (`|delta| = 0.0`); zero test-key leakage in any
per-epoch history log.

**Raw 5-seed means (mean-vs-mean, superseded by Phase T5's formal comparison below):**

| config | params | test AUC (5-seed) |
|---|---|---|
| `transformer_searched` (winner) | 794,241 | 0.9497 ± 0.0025 |
| `transformer_default` | 268,417 | 0.9337 ± 0.0058 |
| BiLSTM baseline (frozen) | 594,561 | 0.9324 ± 0.0114 |

**Phase T5 status: DONE — verdict WIN.** All four scripts ran clean (`04_final_report.py`,
`05_compare_vs_lstm.py`, `06_latency_transformer.py`, `07_loso_report.py`). The LSTM
parity gate passed exactly (0.00e+00 delta, all 5 seeds); a non-mandated sanity check
on the transformer's own checkpoints found a ~1e-6 CPU/GPU device-drift artifact
(PLAN.md §10's pre-registered risk), confirmed far below the bug threshold. See the
Headline result above and `phase5_analysis/README.md` for the full writeup.

## The number to beat

| model (5 seeds) | params | test AUC | PR-AUC | F1 | Acc |
|---|---|---|---|---|---|
| BiLSTM baseline (locked, Issue 2) | 594,561 | **0.9324 ± 0.0114** | 0.876 | 0.828 | 0.883 |
| BiLSTM + attention (reference) | ~611k | 0.925 ± 0.010 | 0.865 | — | — |

Data: `transformer/sequences_clean/` (N=4906; train 2178 / val 634 / test 2094, 32.5% pos)
— a byte-identical copy of `journal_prep/issue2_clean_protocol/sequences_clean/`, kept
local to this folder for discoverability. LSTM checkpoints stay in
`journal_prep/issue2_clean_protocol/runs_clean/multiseed/seed*/` (frozen — never
retrained here, so they aren't duplicated).

## How to run (phase by phase)

```bash
source .venv/bin/activate            # local prerequisites: torch 2.12, sklearn, scipy (all present)

# T1 — local gates (seconds-scale probes only; no experiment training happens locally)
python transformer/phase1_setup/00_transformer_model.py   # param table + shape self-test
python transformer/phase1_setup/01_sanity_checks.py       # gates 0–3 → 01_sanity_report.md

# T2 — Kaggle search (training happens on Kaggle, user directive)
#   1. one-time: upload sequences_clean/{X.npy,y.npy,meta.pkl} as private dataset "pie-sequences-clean"
#   2. upload phase2_kaggle_search/03_search_kaggle.ipynb, attach dataset,
#      accelerator = GPU T4 x2, Run All
#   3. download output zip → unzip into transformer/phase2_kaggle_search/ (runs_search/ lands there)

# T3 — local winner review (BEFORE any test evaluation exists)
python transformer/phase3_search_review/03_search_report.py   # → 03_search_summary.md; user confirms winner

# T4 — DONE. Ran on Kaggle T4x2, 2026-07-11: winner test AUC 0.9497 +/- 0.0025
#   (see phase4_kaggle_final/runs_final/, runs_loso/, _final_summary.json)

# T5 — DONE. local analysis, verdict: WIN (deltaAUC +0.0135, 95% CI [+0.0097,+0.0174])
python transformer/phase5_analysis/04_final_report.py
python transformer/phase5_analysis/05_compare_vs_lstm.py    # paired bootstrap + t-test vs frozen LSTM
python transformer/phase5_analysis/06_latency_transformer.py  # M4 CPU+MPS vs BiLSTM 0.575 ms/window
python transformer/phase5_analysis/07_loso_report.py

# T6 — DONE. docs + manuscript integration (see PLAN.md's Phase T6 DONE block for the
# full list of files touched inside/outside transformer/)
```

## Files

```
transformer/
├── PLAN.md, README.md, PROGRESS_LOG.md   ← tracking docs (span all phases)
├── SUPERVISOR_SUMMARY.md                 ← T6: plain-English write-up for a supervisor
├── sequences_clean/                      ← shared data, used by every phase
├── phase1_setup/                         ← T1: model + engine + sanity gates
├── phase2_kaggle_search/                 ← T2: Kaggle staged search (val-only)
├── phase3_search_review/                 ← T3: local winner review
├── phase4_kaggle_final/                  ← T4: DONE (final training + LOSO)
└── phase5_analysis/                      ← T5: DONE (comparison, latency, LOSO report — verdict WIN)
```

| file | role |
|---|---|
| `PLAN.md` | pre-registered design: fairness contract, grids, default config, verdict templates, DONE blocks, full file map (§8) |
| `PROGRESS_LOG.md` | chronological log of every run's numbers |
| `SUPERVISOR_SUMMARY.md` | plain-English explainer (problem → protocol → search → result → limitations → talking script/Q&A) for presenting this extension to a supervisor |
| `phase1_setup/00_transformer_model.py` | `TransformerIntentPredictor` — single source of truth |
| `phase1_setup/01_sanity_checks.py` / `01_sanity_report.md` | gates 0–3 |
| `phase1_setup/02_train_transformer.py` | `train_run()` engine + CLI (embedded verbatim in the notebooks) |
| `phase2_kaggle_search/03_search_kaggle.ipynb` | Stages A+B+transfer+C, val-only, cached JSONs, WORKERS=2 — ran, verified |
| `phase2_kaggle_search/gen_search_nb.py` | regenerates the notebook above; re-run after editing `phase1_setup/00_`/`02_` |
| `phase2_kaggle_search/runs_search/` | downloaded Kaggle output (102 val-only JSONs + `_stage_summary.json`) |
| `phase2_kaggle_search/kaggle_outputs/` | raw Kaggle download archive (zip + `worker.py`), for provenance |
| `phase3_search_review/03_search_report.py` → `03_*.csv/md/png` | local re-derivation of the search ranking |
| `phase4_kaggle_final/gen_final_nb.py` | regenerates the notebook below; re-run after editing `phase1_setup/00_`/`02_` |
| `phase4_kaggle_final/04_final_loso_kaggle.ipynb` | Stage D (test touched once) + determinism rerun + 6-fold LOSO — ran, verified |
| `phase4_kaggle_final/runs_final/`, `runs_loso/`, `_final_summary.json` | downloaded Kaggle output (10 full run dirs + determinism-check dir + 6 LOSO JSONs) |
| `phase4_kaggle_final/kaggle_outputs/` | raw Kaggle download archive (zip + `worker_final.py`), for provenance |
| `phase5_analysis/04_final_report.py` → `04_final_*.csv/md` | final 5-seed tables — ran |
| `phase5_analysis/05_compare_vs_lstm.py` → `05_comparison_*` | 10k paired bootstrap ΔAUC + paired t-test + Verdict — ran, **WIN** |
| `phase5_analysis/06_latency_transformer.py` → `06_latency_*` | Issue-9 latency protocol on M4 — ran, transformer 1.25× faster |
| `phase5_analysis/07_loso_report.py` → `07_loso_*` | LOSO fold table vs LSTM's LOSO — ran |

Conventions inherited from `journal_prep/` (see `PLAN.md` §9): val-only selection,
test-once, train-only norm, pos_weight 1.682, seeds [42,0,1,2,3], `weights_only=False`,
threshold 0.5, run scripts from the repo root. Cross-phase file references resolve via
`Path(__file__).resolve().parent.parent / "<other_phase>"` (same pattern journal_prep
uses for `pipeline/03_bilstm_model.py`).

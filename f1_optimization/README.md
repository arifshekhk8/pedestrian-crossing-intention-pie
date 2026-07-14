# f1_optimization/ — F1-first optimization of both models (supervisor directive)

The supervisor's rule: **F1 first, then accuracy, then AUC.** This folder re-optimizes
both the BiLSTM and the Transformer for that hierarchy — threshold tuning, F1-based
config re-selection, F1-aware checkpoint retraining, and a pos_weight sweep — under the
same val-only / test-once discipline as the rest of the repo. Pre-registration:
`PLAN.md` (read it first). Chronology + real numbers: `PROGRESS_LOG.md`.

## Status — ALL DONE (2026-07-12)

| phase | what | status |
|---|---|---|
| F0 | Pre-registration + shared modules (`00_*.py`) | ✅ done |
| F1 | `01_threshold_audit.py` — frozen models, val τ\* audit | ✅ done |
| F2 | `02_rerank_searches.py` — cached-search F1 re-rank | ✅ done (transformer cfg unchanged) |
| F3 | `03_lstm_shortlist.py` — shortlist fresh 5-seed measurement | ✅ done (determinism-gate amendment) |
| F4 | `04_train_f1_protocol.py` — F1-protocol training + pos_weight sweep | ✅ done (LSTM→h256; TF→pw2.5) |
| F5 | `05_final_test_eval.py` — the single test pass (all arms) | ✅ done (parity gates PASS) |
| F6 | `06_f1_first_comparison.py` — endpoints, verdicts, figure | ✅ done |

## How to run (from the repo root, venv active)

```bash
source .venv/bin/activate
python f1_optimization/01_threshold_audit.py        # val-only, no training, ~2 min
python f1_optimization/02_rerank_searches.py        # cached JSONs only, seconds
python f1_optimization/03_lstm_shortlist.py         # MPS training, ~3 min, cached
python f1_optimization/04_train_f1_protocol.py --part lstm         # ~10 min, cached
python f1_optimization/04_train_f1_protocol.py --part transformer  # ~1.5-3 h, cached
python f1_optimization/05_final_test_eval.py        # CPU, single test pass
python f1_optimization/06_f1_first_comparison.py    # bootstraps + verdicts + figure
```

All training runs are cached by JSON (re-invoking skips finished cells). Training uses
MPS; every probability regeneration/evaluation is CPU for exactness.

## Folder map

- `00_common.py` — data/split asserts, metrics, τ\* selection, bootstrap utils,
  checkpoint→probs loaders (models imported from `pipeline/03_bilstm_model.py` and
  `transformer/phase1_setup/00_transformer_model.py`).
- `00_train_engines.py` — forked train loops (LSTM from issue8's grid `train()`,
  transformer from phase1's `train_run()`) with the hybrid F1-checkpoint rule
  (stop/schedule on val AUC, checkpoint on best val F1) and dual `val` /
  `val_at_auc_best` recording. The frozen engines are deliberately NOT edited.
- `01`–`06` numbered scripts (see PLAN.md §9) + their `NN_*` outputs.
- `runs_shortlist/` — AUC-protocol completion runs (val-only JSONs).
- `runs_f1/<name>/pw<w>/seed<k>/` — F1-protocol run dirs (best.pt, final.json,
  history.json, norm stats).
- `probs_cache/` — per-arm val/test probability vectors written by 05, read by 06.

## Headline result

Test = set03 (touched once per arm, in 05 only). Per-seed = mean ± std over 5 seeds at
each seed's own val-fitted τ\*; ens = the 5 seeds' averaged probabilities at τ\*_ens
(a deployable mini-ensemble — a different statistic from the per-seed mean; always say
which one you are citing).

| model | test F1 (5-seed) | ens F1 | acc | AUC (ens) |
|---|---|---|---|---|
| BiLSTM frozen @0.5 (A0, the old 0.828) | 0.8275 ± 0.0123 | 0.8370 | 0.8827 | 0.9423 |
| **BiLSTM F1-optimized (A3: h256, F1-ckpt, τ\*)** | **0.8444 ± 0.0078** | **0.8557** | **0.8990** | 0.9467 |
| Transformer frozen @0.5 (B0, the old 0.845) | 0.8446 ± 0.0129 | 0.8490 | 0.8942 | 0.9558 |
| **Transformer F1-optimized (B3: pw2.5, F1-ckpt, τ\*)** | **0.8470 ± 0.0178** | **0.8565** | 0.8962 | 0.9550 |

Pre-registered endpoint verdicts (10k paired bootstrap, ensemble vectors, fixed
val-fitted thresholds):

1. **LSTM IMPROVED** — ΔF1 +0.0187, 95% CI [+0.0073, +0.0300] (excludes 0).
2. **Transformer NO SIGNIFICANT CHANGE** — ΔF1 +0.0075, CI [−0.0021, +0.0173].
3. **Family verdict under F1-first: TIE** — ΔF1 +0.0008, CI [−0.0124, +0.0142],
   paired-t p=0.762. **The transformer's AUC-first WIN does not carry to F1**: with
   both families given identical F1-first optimization, they are statistically
   indistinguishable on the supervisor's primary metric (the transformer keeps its
   AUC edge, tertiary).

Literature F1 ceiling on this protocol ≈0.85 (BiPed/PedFormer 0.85, PIP-Net 0.846):
the deployable ensembles clear it (0.8557 / 0.8565; B2's ens hits 0.8617), the
per-seed means sit just under. Full detail: `06_comparison_report.md`, `PLAN.md` §12.

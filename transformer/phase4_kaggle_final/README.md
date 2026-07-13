# phase4_kaggle_final/ — DONE, ran on Kaggle 2026-07-11

Phase T4 (see `../PLAN.md` §7): trains the Phase-T3 winner + `transformer_default` × 5
seeds each, on Kaggle T4×2, and touches the test set **exactly once**. Also runs a
seed-42 determinism-of-record rerun and the 6-fold leave-one-set-out CV (Issue-5
protocol) on the winner config.

Contains:
- `gen_final_nb.py` — regenerates `04_final_loso_kaggle.ipynb` from `../phase1_setup/
  00_`/`02_` (same "KEEP IN SYNC" convention as `../phase2_kaggle_search/
  gen_search_nb.py`; re-run after editing either of those files).
- `04_final_loso_kaggle.ipynb` — Stage D (winner + default × 5 seeds, `eval_test=True`)
  + a determinism rerun + 6-fold LOSO. The winner config is hardcoded in the CONFIG
  cell (copied from its raw `seed42.json`, not hand-parsed from the cfg_id string) — no
  manual paste step. Uses the same subprocess-per-GPU pattern as Phase T2's search
  notebook, via an embedded `worker_final.py` that dispatches on a job "kind" (`"final"`
  = fixed split + checkpoint dir; `"loso"` = per-fold pedestrian-grouped split, no
  checkpoint, per-fold `pos_weight`).
- `runs_final/transformer_{searched,default}/seed<k>/` — downloaded full run dirs
  (`best.pt`, `final.json`, `history.json`, `norm_mean.npy`, `norm_std.npy`).
- `runs_final/_determinism_check/transformer_searched_seed42_rerun/` — the independent
  rerun; reproduces the primary seed-42 run's val/test AUC exactly (`|delta| = 0.0`).
- `runs_loso/<fold>.json` — the 6 LOSO fold results.
- `_final_summary.json` — Kaggle's own aggregation (independently cross-checked, see below).
- `kaggle_outputs/` — raw download archive (`transformer_final_output.zip` +
  `worker_final.py`), kept for provenance. Duplicate `00_/02_.py` copies and the Kaggle
  editor's internal `.virtual_documents/` cache that came inside the download were
  removed as exact duplicates of files already canonical elsewhere (same convention as
  Phase T2's cleanup).

**Result (5 seeds [42,0,1,2,3], T4×2, ~8 min for all 17 trainings):**

| config | params | val AUC | test AUC |
|---|---|---|---|
| `transformer_searched` (winner) | 794,241 | 0.9789 ± 0.0038 | **0.9497 ± 0.0025** |
| `transformer_default` | 268,417 | 0.9629 ± 0.0056 | 0.9337 ± 0.0058 |
| BiLSTM baseline (frozen, for reference) | 594,561 | 0.9644 ± 0.0043 | 0.9324 ± 0.0114 |

LOSO (winner, 6-fold): 0.9392 ± 0.0436 (excl. tiny set05 N=47: 0.9270 ± 0.0357) vs the
BiLSTM's 0.928 ± 0.041 (Issue 5). **These are raw means for orientation only** — the
formal verdict is Phase T5's paired-bootstrap comparison, not yet run.

**Verified independently, not just trusted** (own script, mirrors
`../phase3_search_review/03_search_report.py`'s discipline): every 5-seed/6-fold
mean±std recomputed from the raw `final.json`/`runs_loso/*.json` files matches
`_final_summary.json` exactly (to 1e-9); `n_params` matches the known
architecture-derived values on every row; `winner_cfg`/`default_cfg`/`seeds` match the
pre-registered values exactly; all 6 LOSO fold `test_n` values match Issue 5's real
per-set sizes exactly (258/310/2094/1610/47/587 — confirms genuine data); every
`history.json` contains only per-epoch `val_*` keys, zero test-key leakage; determinism
rerun is an exact bit-for-bit match.

**One bug found and fixed on this run** (see `../PROGRESS_LOG.md` for the full
writeup): the notebook's own final cell raised `KeyError: 'n_params'` — that field is
deliberately not persisted in `final.json` (matches the pipeline's minimal schema) but
the summary cell wrongly assumed it would be there. Fixed by recomputing `n_params`
from the architecture config directly; the corrected cell was re-run in the same live
Kaggle session at zero retraining cost, since Stage D/determinism/LOSO had already
completed and cached.

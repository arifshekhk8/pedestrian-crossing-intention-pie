# Transformer Extension — Progress Log

Chronological log of every run's real numbers (pipeline/PROGRESS_LOG.md convention).
Newest entries at the bottom. Every table states the device that produced it.

---

## 2026-07-11 — Plan created (no code, no runs yet)

Supervisor requested a Transformer extension: build the best transformer on our pipeline
and dataset, compare against the LSTM, with the `journal_prep/` clean-protocol results as
the LSTM base. Full pre-registered design written to `PLAN.md` before any transformer code
exists. Decisions locked today:

- **Scope:** full staged search — Stage A architecture (36 configs) + Stage B recipe (36)
  + transfer check (6) + Stage C top-5+default × 5 seeds, all **val-only**; Stage D
  winner+default × 5 seeds touches test **once**. Budget 78 seed-42 configs ≥ 2× the
  LSTM's Issue-8 search (36).
- **Extras chosen by user:** latency benchmark (Issue-9 protocol, local M4) + LOSO CV
  (Issue-5 protocol). Declined: TTE-trend ablation, motion-feature variant.
- **Compute split (user directive):** all experiment-grade training on **Kaggle T4×2**
  (two self-contained notebooks, cached JSON/run-dir outputs downloaded back); everything
  else local on the M4 (`.venv`).
- **Pre-registered default transformer:** d_model=128, nhead=4, L=2, ff=256, dropout=0.1,
  learned PE, CLS pool (~268k params), trained with the LSTM's exact recipe (Adam 1e-3,
  wd 1e-5, plateau scheduler). Reported alongside the searched winner.
- **Comparison target (frozen, never retrained):** BiLSTM clean 5-seed test AUC
  **0.9324 ± 0.0114** (per-seed 0.9131 / 0.9334 / 0.9432 / 0.9363 / 0.9358), 594,561
  params, checkpoints in `journal_prep/issue2_clean_protocol/runs_clean/multiseed/`.
- **Primary statistic:** 10k window-paired percentile bootstrap of ΔAUC on the same 2094
  set03 windows (fixed RNG 42), secondary paired t-test over 5 seeds. Win/tie/loss verdict
  templates pre-registered in `PLAN.md` §6.
- **Environment facts verified:** local `.venv` = torch 2.12.0 + sklearn 1.9.0 +
  scipy 1.17.1 (CLAUDE.md "no sklearn locally" is outdated); torch 2.12 needs
  `enable_nested_tensor=False`; Issue-8 calibration = 10.3 s/run mean on M4 MPS →
  T4×2 estimates: search notebook ~30–90 min, final notebook ~10–30 min.

Next step: Phase T1 — write `00_transformer_model.py`, `01_sanity_checks.py`,
`02_train_transformer.py`; run gates 0–3 locally.

---

## 2026-07-11 — Phase T1 done: model + training engine + all sanity gates pass

Wrote `00_transformer_model.py` (`TransformerIntentPredictor`: pre-LN encoder, Linear(5→d)
input proj, optional CLS token, learned/sinusoidal PE, cls/mean/last pooling, GELU FFN,
`enable_nested_tensor=False` for torch 2.12), `02_train_transformer.py` (`train_run()` —
the shared engine for dev runs, the search notebook, and the final notebook; `DEFAULT_CFG`
= the pre-registered `transformer_default` from PLAN.md §3), and `01_sanity_checks.py`
(gates 0–3).

**All 4 gates pass** (full numbers in PLAN.md's Phase T1 DONE block / `01_sanity_report.md`):
- Gate 0 (protocol asserts): X (4906,16,5), splits 2178/634/2094, test positives 681,
  pos_weight 1.682, norm shape (5,) — all exact.
- Gate 1 (linear-probe floor): L=0 wrapper 0.899, sklearn LogisticRegression 0.939,
  L=2 transformer 0.931 (30 epochs, seed 42 — a wiring sanity check, not a benchmark).
- Gate 2 (overfit-64): train acc 1.0000 after 200 epochs, dropout 0.
- Gate 3 (determinism): seed 42 twice on CPU → val AUC 0.934379 vs 0.934379, exact match.
- Param ladder confirmed: (64,128)L2/L4 = 68,673/135,617; (128,256)L2/L4 = 268,417/533,377;
  (128,512)L2/L4 = 400,001/796,545. `transformer_default` = **268,417 params** (BiLSTM
  baseline: 594,561 — the transformer default sits at ~0.45×).
- Extra: forward+backward smoke-tested on **MPS** (user has an M4 GPU) — clean, no NaNs.
  Compute directive unchanged: sanity/smoke work may use local CPU or MPS, but all
  experiment-grade training (the search + final notebooks) runs on Kaggle T4×2.

**Bug found and fixed (sanity script only, not the training engine):** the first Gate-3
draft called `quick_train(build_default(), ...)` — Python builds the model argument
*before* `quick_train`'s internal `set_seed()` runs, so two "same seed" runs started from
different random initial weights (val AUC 0.9438 vs 0.9359, |Δ|≈0.008 over 5 epochs).
Initial hypothesis was CPU multi-threaded matmul non-associativity in attention (a real
phenomenon, worth remembering) — pinning `torch.set_num_threads(1)` did **not** fix the
gap, which ruled that out and pointed at construction order. Fixed by changing
`quick_train` to take a model *factory* and seed before calling it. Verified line-by-line
that `02_train_transformer.py::train_run()` already seeds before `build_model(cfg)` —
the real engine was never affected.

**Next step: Phase T2** — build `kaggle/03_search_kaggle.ipynb` (Stage A architecture
grid, Stage B recipe grid, transfer check, Stage C candidate multiseed — all val-only,
WORKERS=2 one-config-per-GPU), upload `sequences_clean/` as a private Kaggle dataset, run
on T4×2.

---

## 2026-07-11 — Phase T2 notebook built and locally validated (not yet run on Kaggle)

Built `kaggle/03_search_kaggle.ipynb` (13 cells: title, env check, dataset locate+assert,
write `00_transformer_model.py`/`02_train_transformer.py`/`worker.py` verbatim to
`/kaggle/working/`, orchestration helpers, Stage A, Stage B, transfer check, Stage C,
save-summary+zip, next-steps) via a generator script, `kaggle/gen_search_nb.py`, so the
notebook's embedded model/engine copies can never drift from the canonical files — it
reads them fresh and re-embeds on every regeneration.

**2-GPU design:** each stage shards its job list round-robin across the 2 T4s and
launches one `subprocess.Popen` per GPU (`worker.py`, with `CUDA_VISIBLE_DEVICES` set in
the subprocess's `env`), rather than threads or `multiprocessing`. Reasoning recorded in
PLAN.md's Phase T2 block: PyTorch's global RNG is process-global, so two threads training
concurrently would corrupt each other's "seeded" streams and break the reproducibility
the whole search depends on; plain `multiprocessing` can't reliably re-import a
notebook's `__main__` in a spawned worker. Separate OS subprocesses running a real
`worker.py` file sidestep both problems.

**Bug caught during design, before any code ran:** the planned `cfg_id()` only encoded
`lr/schedule/dropout/weight_decay`, not `optimizer`. Stage A trains with Adam (frozen
default recipe) and Stage B always uses AdamW — so a Stage-B cell can numerically match
Stage A's default-recipe cell on every other field while differing only in optimizer.
Without `optimizer` in the id, both would hash to the same cache path and one run's
result would be silently discarded. Fixed before writing the notebook (added
`optimizer` as the first token of the recipe half of `cfg_id`).

**Local validation (three layers, real code + real data, no mocks):**
1. Every one of the 11 code cells `compile()`s without a SyntaxError.
2. Extracted cells were actually run: environment check, dataset locate+assert (against
   the real local `sequences_clean/`), file-writing cells, and helper definitions all
   executed cleanly; a 2-config/1-seed job was run through the real `subprocess.Popen` +
   `worker.py` pipeline (CPU fallback locally) using the real data — produced correct
   JSON results, and a repeat run correctly hit the "already exists, skip" cache path
   with byte-identical results.
3. The full cross-stage chain was run for real on a shrunk grid (4 architecture configs,
   4 recipe configs, 6 transfer configs, 10 Stage-C jobs, instead of 36/36/6/~25): Stage
   A ranked correctly, `arch_top3` extraction worked, Stage B built its grid from
   `arch_top3[0]` correctly, the transfer check combined top-3 recipes × archs #2/#3
   correctly, Stage C's pooling+dedup correctly identified and tagged
   `transformer_default` among the candidates, multiseed aggregation and winner-by-mean
   selection worked, and the summary+zip cell completed. **No result at any stage ever
   carried a `test` key** (explicitly asserted).

**Status: notebook is ready to run on Kaggle. Not yet run — that is the user's next
action** (upload `pie-sequences-clean` dataset, attach GPU T4×2, Run All), followed by
Phase T3 (local winner review via `03_search_report.py`, not yet written).

---

## 2026-07-11 — data relocated into transformer/ for discoverability

User asked why `sequences_clean/` lived under `journal_prep/issue2_clean_protocol/`
instead of inside this workspace — fair point, since the user couldn't find it there.
The original reasoning (single source of truth, so the LSTM and transformer definitely
train on byte-identical data) doesn't actually require the data to live *outside*
`transformer/` — it only requires not maintaining a second copy that could drift.

Copied `X.npy`/`y.npy`/`meta.pkl` (1.8 MB total, already git-tracked in journal_prep) into
`transformer/sequences_clean/`, verified byte-identical via `diff`. Updated `SEQ_DIR` in
both `01_sanity_checks.py` and `02_train_transformer.py` to `HERE / "sequences_clean"`
(local sibling, was `HERE.parent / "journal_prep" / "issue2_clean_protocol" /
"sequences_clean"`), updated the data-location text in `PLAN.md`, `README.md`, and
`kaggle/gen_search_nb.py`'s markdown cell, then regenerated
`kaggle/03_search_kaggle.ipynb`.

Re-validated after the change: Phase T1's gates 0–3 all still pass against the new local
path; `02_train_transformer.py`'s CLI (`--preset default --seed 42`) ran a real training
to completion using the new default path (val AUC 0.9628, 268,417 params, matching prior
runs). No functional cells changed in the Kaggle notebook — only the markdown setup
instructions (the notebook always reads from `/kaggle/input/` regardless of the local
path, so this was purely a local-dev/discoverability fix). journal_prep's copy remains
the canonical original (where Issue 2 built it); this one is a static copy, not
independently regenerated, so the two stay identical by construction.

---

## 2026-07-11 — Phase T2 ran on Kaggle T4×2; Phase T3 report generated and verified

User ran `kaggle/03_search_kaggle.ipynb` on Kaggle (2×T4) and downloaded the output into
`transformer/kaggle_outputs/` (raw archive, kept for provenance) → copied `runs_search/`
into the canonical `transformer/runs_search/` (folder later renamed/cleaned, see the
2026-07-11 folder-cleanup entry below).

**Sanity-checked before trusting anything:** 78 distinct config directories, 102 total
`seed*.json` files (78 + 24 additional Stage-C seeds — exact match to the expected
count, confirming caching worked and nothing silently re-ran); `grep` for `"test"` or
`"test_confusion_matrix"` anywhere in `runs_search/` returned **zero hits**, confirming
the search notebook never touched the test set, as designed.

Wrote `03_search_report.py` (Phase T3): reloads every raw `seed*.json`, buckets rows
into Stage A / Stage B+transfer / Stage C purely from cfg fields (using
`_stage_summary.json` only to identify which architectures Stage B/transfer ran on, not
to borrow its conclusions), and **independently recomputes every ranking and every
mean/std from scratch** — then asserts the recomputation matches the notebook's own
summary. All assertions passed: Stage-A top-3 recomputed identically; all 6 Stage-C
candidates' 5-seed means matched the summary to 1e-9.

**Result:**

| | config | val AUC (5-seed) | params |
|---|---|---|---|
| **winner** | `d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` | **0.9789 ± 0.0038** | 794,241 |
| transformer_default | `d128_ff256_L2_cls_lpe__adam_lr1e-03_plateau_do0.1_wd1e-05` | 0.9629 ± 0.0056 | 268,417 |

Winner architecture: num_layers=4 (default: 2), pool="last" (default: "cls"), pos="sin"
(default: "learned") — recipe is unchanged from the default (Adam, lr=1e-3, plateau,
dropout=0.1, wd=1e-5); Stage B's recipe search didn't end up mattering for the overall
winner. `transformer_default` ranked 66th/78 in the full seed-42 pool — the search
meaningfully beat the un-searched default.

**Selection-noise control caught something real, again:** the single-seed (seed 42)
leader among all 78 configs (`d128_ff256_L4_last_spe__adam_...`, seed-42 val AUC 0.9839)
is *not* the 5-seed winner — its own mean drops to 0.9773, behind the actual winner's
0.9789. Exactly the phenomenon Issue 8 found for the LSTM grid search, now reproduced
independently for the transformer.

**Notable pattern worth keeping for the writeup:** nearly all of Stage A's top
architectures converged on `pool="last"` — i.e., the search rediscovered "read out from
the final timestep," which is exactly the BiLSTM's own readout strategy, without being
told to. Reassuring sign the search found a real signal, not an artifact.

**Caveat stated plainly (also in the generated report):** this is a **validation-set**
result only. It is not the LSTM-vs-transformer comparison — validation AUC from two
differently-searched models isn't directly comparable (a longer/luckier validation
search can inflate a val number without a matching test gain). The actual comparison is
a paired bootstrap on the test set in Phase T5, using checkpoints that don't exist yet.
For reference only: LSTM val AUC (Issue 8) = 0.9644 ± 0.0043; **LSTM test AUC — the
number that actually matters — is 0.9324 ± 0.0114.**

Outputs: `transformer/runs_search/` (102 files), `03_arch_grid.csv`, `03_recipe_grid.csv`,
`03_candidates_multiseed.csv`, `03_search_summary.md`, `03_search_figure.png`.

**Status: Phase T3's mechanical work is done. Waiting on the user to review and confirm
the winner** before Stage D (`kaggle/04_final_loso_kaggle.ipynb`, not yet written) gets
built — that notebook is the one that touches test set03, exactly once.

---

## 2026-07-11 — folder cleanup (duplicate files removed)

User flagged the folder as messy with duplicates. Audited before deleting anything:
verified with `diff -rq` that `kaggle_outputs/Transformer staged search/runs_search/`
was byte-identical to the canonical `transformer/runs_search/`, and with `diff` that its
`00_transformer_model.py`/`02_train_transformer.py` were byte-identical to the canonical
root copies — both are exactly what the notebook wrote to `/kaggle/working/` during the
run, so they're expected to match, not independent artifacts. `transformer/` is entirely
untracked in git so far (`git status` confirmed), so nothing here touched any committed
history.

Removed: `__pycache__/` and `.DS_Store` (both already `.gitignore`d repo-wide, just
messy on disk locally), the duplicate `00_/02_.py` files, the duplicate `runs_search/`,
and Kaggle's own `.virtual_documents/` autosave junk. Kept: `transformer_search_output.zip`
(the raw single-file download, a genuine provenance record) and `worker.py` (the one
generated file not otherwise saved anywhere in the repo — reconstructable from
`kaggle/gen_search_nb.py`'s `WORKER_CODE`, but cheap to keep as "exactly what ran").
Renamed the awkward space-containing `kaggle_outputs/Transformer staged search/` to
`kaggle_outputs/03_search/`, matching the repo's numbered-artifact convention.
`kaggle_outputs/` shrank from 572 KB to 72 KB. Fixed the two stale path references this
left in `PLAN.md` and above in this file.

No change to `transformer/runs_search/`, `transformer/sequences_clean/`, or any of the
`00_`–`03_` scripts/outputs at the folder root — those already matched the file map in
`PLAN.md` §8 and journal_prep's numbered-script convention (a script's `.csv`/`.md`/`.png`
outputs sit next to it, not in a subfolder), so nothing there needed reorganizing.

---

## 2026-07-11 — real restructure: one subfolder per phase

User clarified the previous cleanup wasn't what they wanted: they explicitly asked for
one subfolder per phase, each containing only what that phase needs — a genuine
restructure, not just dedup. Moved everything into:

- `phase1_setup/` — `00_transformer_model.py`, `02_train_transformer.py`,
  `01_sanity_checks.py`, `01_sanity_report.md`.
- `phase2_kaggle_search/` — `gen_search_nb.py`, `03_search_kaggle.ipynb`, `runs_search/`,
  and `kaggle_outputs/` (flattened one level — the redundant `03_search/` subfolder from
  the previous cleanup pass was dropped since the phase folder name already disambiguates).
- `phase3_search_review/` — `03_search_report.py` + its `.csv`/`.md`/`.png` outputs.
- `phase4_kaggle_final/`, `phase5_analysis/` — created empty, ready for Phases T4/T5.
- `sequences_clean/` stayed at `transformer/` root — it's a genuine shared input read by
  every phase (T1 local checks, T2/T4 Kaggle training), not a single phase's deliverable.

**Path-resolution fixes this required** (each script computes its data paths relative to
its own file location, so moving a script one level deeper breaks anything that was
`HERE / "sibling"` where the sibling moved to a different folder):
- `phase1_setup/01_sanity_checks.py` and `02_train_transformer.py`: `SEQ_DIR` changed
  from `HERE / "sequences_clean"` to `HERE.parent / "sequences_clean"` (data is now a
  parent-level sibling, not a same-folder one). The `00_transformer_model.py` import
  needed no change (still same-folder).
- `phase3_search_review/03_search_report.py`: `RUNS_DIR` changed from `HERE /
  "runs_search"` to `HERE.parent / "phase2_kaggle_search" / "runs_search"`.
- `phase2_kaggle_search/gen_search_nb.py`: reads `phase1_setup/00_/02_.py` via
  `HERE.parent.parent / "transformer" / "phase1_setup"` instead of `REPO / "transformer"`
  directly, and writes the notebook to its own folder (`HERE / "03_search_kaggle.ipynb"`)
  instead of a `kaggle/` subfolder that no longer exists. Also updated every path
  mentioned in the notebook's own markdown/print text (setup instructions, "next steps",
  the final download-instructions print) to the new phase-folder locations.

**Re-verified everything still works after the moves** (not just assumed it):
regenerated the notebook (11 cells still compile), re-ran `01_sanity_checks.py` from its
new location (all 4 gates still pass, identical numbers), re-ran `02_train_transformer.py`'s
CLI directly (matches its pre-move val AUC exactly, 0.9628), and re-ran
`03_search_report.py` from its new location (still finds the same winner, 0.9789 ±
0.0038, all internal cross-checks against `_stage_summary.json` still pass).

Rewrote `PLAN.md` §8 (file map) and `README.md`'s "Files"/"How to run" sections in full to
match the new structure; fixed every inline path reference elsewhere in both documents
that pointed at an old location.

---

## 2026-07-11 — Phase T3 confirmed; Phase T4 notebook built (not yet run)

User reviewed `phase3_search_review/03_search_summary.md` and confirmed the winner
("okay i have checked the Phase T3 summary. lets start building phase T4") — Phase T3
closes. Winner going into Phase T4:
`d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` (d_model=128,
num_layers=4, dim_ff=512, pool="last", pos="sin", Adam lr=1e-3/plateau/dropout=0.1/
wd=1e-5), copied from its raw `seed42.json` in `runs_search/` rather than hand-parsed
from the cfg_id string, to rule out a transcription mistake.

**Engine change required first:** LOSO needs a **per-fold** `pos_weight`
(`n_neg/n_pos` of that fold's train pool, Issue-5 protocol) but `train_run()` hardcoded
the frozen `POS_WEIGHT=1.682` module constant. Added an optional `pos_weight=None`
parameter to `phase1_setup/02_train_transformer.py::train_run()` (falls back to the
frozen constant when omitted) instead of writing a second training loop for LOSO —
`train_run()` is already the single, heavily-validated engine shared by Phase T1 dev
runs and Phase T2's entire 102-run search, so extending it keeps LOSO on the exact same
tested code path. Also added `pos_weight` to the returned result dict (visible in
every run's output now, useful for the per-fold LOSO report).

**Proven inert before relying on it:** re-ran the Phase-T1 CLI dev command
(`--preset default --seed 42 --device cpu`) — val AUC 0.9627719038319079,
byte-identical to the pre-change number. Re-ran all 4 Phase-T1 sanity gates (still ALL
PASS) and `03_search_report.py` (still finds the identical winner, 0.9789 ± 0.0038, all
cross-checks against `_stage_summary.json` still pass, still zero `test` keys anywhere)
after the change. Regenerated `phase2_kaggle_search/03_search_kaggle.ipynb` so its
embedded copy of the engine stays in sync (its own results are unaffected — the search
never passes `pos_weight`).

**Built `phase4_kaggle_final/gen_final_nb.py` + `04_final_loso_kaggle.ipynb`**
(13 cells, 11 code), reusing Phase T2's subprocess-per-GPU pattern
(`CUDA_VISIBLE_DEVICES` per worker, not threads/multiprocessing — same reasoning as
Phase T2). New per-Phase-T4 design: the embedded `worker_final.py` dispatches on a
`job["kind"]` field instead of assuming one job shape —
- `"final"`: the fixed train/{set05,06}-val/set03-test split (`engine.split_data`),
  `eval_test=True`, a real checkpoint dir. Used for Stage D (2 configs × 5 seeds = 10
  jobs) and a seed-42 determinism-of-record rerun (1 job, separate output dir, compared
  numerically against the primary run — Issue-2's "seed 42 reproduces 0.9131"
  precedent applied here).
- `"loso"`: loads `meta.pkl` for `ped_id`, builds the pedestrian-grouped 85/15 fold
  split itself (mirrors Issue 5's `train_fold()` exactly, composite `set_id/ped_id` key
  so pedestrian numbers can't collide across sets), computes that fold's `pos_weight`,
  calls `train_run(..., out_dir=None, pos_weight=pw)`, and writes the augmented result
  straight to `runs_loso/<fold>.json` (6 jobs, one per PIE set).

Total 17 Kaggle trainings (10 + 1 + 6), matching the estimate in `PLAN.md` §5.
Unlike Phase T2's worker, this one needs no `--out_root` CLI argument — every job
already carries its own absolute `out_dir`/`out_path`, since Stage D's two configs and
the 6 LOSO folds don't share one cache-key scheme the way the search's `cfg_id`s did.

**Validated locally, same three tiers as Phase T2:**
1. Syntax — all 11 code cells `compile()` without error.
2. Mechanics — both job kinds run for real through `worker_final.py` via subprocess
   (no GPU locally, falls back to CPU exactly as the code already handles), each
   including a cache-hit re-run confirmed via unchanged output-file mtimes. Ran against
   a **fabricated synthetic dataset** (6 fake sets × 5 peds × 4 rows, alternating
   labels) — never the real `sequences_clean/` — so test set03 remains untouched
   anywhere in the project even by this validation. `load_raw()`'s shape assert (which
   exists specifically to catch a wrong data directory in production) was relaxed only
   in a private in-memory copy of the engine used solely by the test harness; the real
   `phase1_setup/02_train_transformer.py` was never touched for this. Confirmed:
   `"final"` produces the exact expected run-dir (`best.pt`, `final.json`,
   `history.json`, `norm_mean.npy`, `norm_std.npy`, with `final.json` containing
   exactly `{best_epoch, val, test, test_confusion_matrix}`); `"loso"` produces the
   expected JSON schema with `train_n + val_n` correctly summing to the 5-set pool size
   per fold.
3. (No third real-data-shrunk-grid tier this time, unlike Phase T2 — Phase T4's
   `eval_test=True` code path always touches "test" in some sense, even on a rotating
   LOSO fold, so exercising it against real data locally would blur the "test set03
   touched exactly once, on Kaggle" invariant; synthetic data was judged the safer
   substitute and covers the same code paths.)

**Not yet done:** the notebook has not been uploaded to or run on Kaggle. No
`runs_final/` or `runs_loso/` data exists yet anywhere; test set03 has not been
touched anywhere in this project. Updated `PLAN.md` (Phase T3 DONE block, Phase T4
section, §8 file map, header status line) and `README.md` (status table, Phase T4
status paragraph, how-to-run, Files table) to match.

---

## 2026-07-11 — Bug found on the real Kaggle run: `KeyError: 'n_params'` in the save-zip cell

User ran the notebook on Kaggle (2×T4). All 17 trainings (Stage D's 10, the
determinism rerun, all 6 LOSO folds) completed successfully in **8 minutes**, well
inside the ~10–30 min estimate — but the last cell crashed:
`KeyError: 'n_params'` while building `final_summary["stage_d"]`.

**Root cause:** `CELL_SAVE_ZIP` read `r["n_params"]` from `stage_d_results`, but that
list is `{**job, **final}` where `job` only carries `kind/cfg/seed/tag/out_dir` and
`final` is `final.json`'s content — which is deliberately minimal
(`{best_epoch, val, test, test_confusion_matrix}`, matching `pipeline/
05_compare_runs.py`'s expected schema, confirmed by reading a real baseline
`final.json`). `n_params` was never persisted anywhere on disk for a `"final"`-kind
job (not in `final.json`, not in `best.pt`'s saved dict) — it only ever existed inside
`train_run()`'s in-memory return value, which the worker subprocess discards after
writing the reduced files. My mistake: assuming a field would survive a round-trip it
never actually crossed.

**Why this got past the earlier validation:** the previous mechanics test exercised
only `worker_final.py` (the subprocess side) — the exact bug lived in a main-process
notebook cell (`CELL_SAVE_ZIP`) that no test had ever executed. That's a real gap: a
worker-only test cannot catch a bug in the cells that never call the worker.

**Fix:** `CELL_SAVE_ZIP` now recomputes `n_params` from the config itself —
`engine.count_params(engine.build_model(r["cfg"]))` — since parameter count is a pure
function of architecture, not of the training run or the seed. Also tightened
`CELL_LOSO`'s "excluding small folds" guard from `if len(big) < len(df_loso)` to
`if 0 < len(big) < len(df_loso)`, so an all-folds-small edge case (only possible with
degenerate data, never with the real 6 PIE sets) can't compute mean/std of an empty
array.

**Deeper verification this time, to close the actual gap:** wrote a second local test
that execs **every one of the 11 code cells** from the generated notebook, in order, in
one shared namespace — not just the worker script — exactly reproducing what a live
Kaggle kernel does cell-by-cell. Only 2 of the 11 cells needed patching, and only
because they reference the real Kaggle filesystem, not because their logic was in
question: the locate-data cell (hardcoded `/kaggle/input` search → pointed at a local
synthetic `sequences_clean/`) and the write-engine cell (the real 4906-row shape assert
→ relaxed only in a private in-memory copy used solely by the test, never in the
repository file). The other 9 cells — env-check, write-model, write-worker, helpers,
config, Stage D, determinism, LOSO, and the fixed save-zip — ran completely
unmodified. Result: all 11 cells completed with no errors; `_final_summary.json` came
out with all 10 `stage_d` entries carrying the correct `n_params` (794,241 for
`transformer_searched`, matching the real, previously-confirmed number, since param
count doesn't depend on the data used to compute it), all 6 `loso` entries present,
determinism check `PASS` (delta 0.0, expected on CPU). Also separately verified the zip
step itself (`shutil.make_archive`) by redirecting only the zip destination to a local
path (the real cell targets `/kaggle/working/`, read-only on this machine) — confirmed
the archive's top-level entries are exactly `runs_final/`, `runs_loso/`,
`_final_summary.json`, so a plain `unzip -d transformer/phase4_kaggle_final/` lands
everything correctly with no extra manual nesting step.

**User's live Kaggle session (still running, 9m draft session) did not need to be
restarted** — Stage D, the determinism rerun, and all 6 LOSO folds were already cached
to disk and still held in the kernel's memory (`stage_d_results`, `loso_rows`,
`primary_final`, etc.), so only the corrected last cell needed to be pasted in and
re-run in place, at zero retraining cost.

Regenerated `04_final_loso_kaggle.ipynb` locally with both fixes; still 11 code cells,
still compiles.

---

## 2026-07-11 — Phase T4 complete: output verified, reorganized, and closed out

User applied the fix in their live Kaggle session and confirmed completion, attaching
the downloaded output.

**Independently re-verified everything from the raw files** (own script, not
committed to the repo; same discipline as `03_search_report.py`) rather than trusting
`_final_summary.json`:
- Structural completeness: all 10 Stage D run dirs (`best.pt`/`final.json`/
  `history.json`/`norm_mean.npy`/`norm_std.npy` each), the determinism-check dir, and
  all 6 LOSO fold JSONs present.
- Recomputed 5-seed val/test AUC mean±std for both configs from the raw `final.json`
  files, and 6-fold LOSO mean±std from the raw `runs_loso/*.json` files — both match
  `_final_summary.json` exactly (to 1e-9).
- `n_params` on every Stage D row matches the known architecture-derived values
  (794,241 for `transformer_searched`, 268,417 for `transformer_default`) — confirms
  yesterday's fix works correctly on the real run, not just the synthetic test.
- `winner_cfg`, `default_cfg`, and `seeds` in `_final_summary.json` match the
  pre-registered values exactly (byte-for-byte dict equality).
- **All 6 LOSO fold `test_n` values match Issue 5's real per-set sizes exactly**
  (set01=258, set02=310, set03=2094, set04=1610, set05=47, set06=587) — this is a
  strong, independent confirmation that the genuine, untampered `sequences_clean/` was
  used (these counts are a fingerprint of the real PIE recording sets, not
  reproducible by accident).
- Every `history.json` (per-epoch training log) contains only `val_*` keys — zero
  per-epoch test-metric leakage across all 17 runs.
- Determinism-of-record rerun: **exact bit-for-bit match** — `transformer_searched`
  seed 42 reproduces val AUC 0.9824634655532359 and test AUC 0.9488627211346704,
  `|delta| = 0.0` on both (not merely "close"), on both this run's own printed
  DETERMINISM CHECK and independent recomputation from the two raw `final.json` files.

**Results (5 seeds, T4×2):**

| config | params | val AUC | test AUC |
|---|---|---|---|
| `transformer_searched` (winner) | 794,241 | 0.9789 ± 0.0038 | **0.9497 ± 0.0025** |
| `transformer_default` | 268,417 | 0.9629 ± 0.0056 | 0.9337 ± 0.0058 |
| BiLSTM baseline (frozen, Issue 4, for reference) | 594,561 | 0.9644 ± 0.0043 | 0.9324 ± 0.0114 |

LOSO (winner, seed 42, 6-fold): mean 0.9392 ± 0.0436 (excl. set05 N=47: 0.9270 ±
0.0357) vs the BiLSTM's 0.928 ± 0.041 (Issue 5). Per-fold test AUC: set01 0.9050,
set02 0.9212, set03 0.9496, set04 0.8847, set05 1.0000, set06 0.9746 — notably the
transformer's 5-seed test-AUC std (0.0025) is much tighter than the BiLSTM's (0.0114),
i.e. more consistent across seeds on this run.

**These are plain mean-vs-mean numbers for orientation only — explicitly not the
formal verdict.** The pre-registered decision rule (`PLAN.md` §6: 10k paired bootstrap
of ΔAUC on the identical 2094 test windows + a paired t-test over seeds) has not been
run. That's Phase T5, next.

**Reorganized the download to match the established convention** (Phase T2's
`kaggle_outputs/` pattern): diffed `00_transformer_model.py`/`02_train_transformer.py`/
`worker_final.py` from the download against canonical `phase1_setup/`/generated
versions — all byte-identical, so the two `.py` copies were deleted (exact duplicates)
and `worker_final.py` moved to a new `kaggle_outputs/` alongside the raw zip (kept for
provenance). The Kaggle editor's `.virtual_documents/__notebook_source__.ipynb` cache
file was dropped — inspected first, but it's not valid notebook JSON (just the
editor's internal source-tracking blob) and held no execution output not already
covered by `_final_summary.json`. `runs_final/`, `runs_loso/`, and `_final_summary.json`
moved up out of the intermediate `phase4_out/` wrapper directly into
`phase4_kaggle_final/`, matching the notebook's own unzip instructions. `__pycache__`
and `.DS_Store` deleted. Re-ran the independent verification script against the new
location afterward — every check still passes, confirming the reorganization didn't
disturb anything.

Updated `PLAN.md` (Phase T4 DONE block with full results + verification summary,
header status line, §8 file map), `README.md` (status table, Phase T4 status
paragraph + headline table, how-to-run, Files table), and
`phase4_kaggle_final/README.md` (full rewrite: results table, verification summary,
bug writeup) to close out Phase T4.

**Next: Phase T5** (`phase5_analysis/`, not yet built) — `05_compare_vs_lstm.py`'s
paired bootstrap is what actually decides win/tie/loss per the pre-registered §6
templates, not the raw means above.

---

## 2026-07-11 — Phase T5 complete: the formal verdict is WIN

User said "lets start" — built all four Phase-T5 scripts and ran them.

**`04_final_report.py`** — pure aggregation of Phase T4's `runs_final/` into 5-seed
mean±std tables + the determinism-check cross-reference. Reproduces Phase T4's
verified numbers exactly (as it should — same source files, no new computation).

**`07_loso_report.py`** — aggregates `runs_loso/` against Issue 5's BiLSTM LOSO.
**Bug caught before running anything wrong**: an early draft hardcoded the LSTM's
per-fold reference numbers by hand from a terminal dump of the CSV, and the f1/acc
columns got transposed with prec (e.g. set01's f1 was written as 0.8295 when the raw
CSV has f1=0.8307692307692308, acc=0.8294573643410853 — the acc value ended up in the
f1 slot). Caught by the same discipline used throughout this project: printed the raw
CSV columns directly and diffed against the hardcoded dict before trusting it. Fixed
by loading Issue 5's CSV programmatically (`csv.DictReader`) instead of
hand-transcribing any of its numbers — eliminates this entire class of error going
forward. Also discovered the reason recomputed LSTM LOSO mean/std (0.928±0.045) don't
byte-match Issue 5's own published text (0.928±0.041): Issue 5's script used
`.std()` (`ddof=0`, population std) while every other 5-seed/6-fold aggregation in
this project (`03_search_report.py`, `04_final_report.py`, and this script) uses
`ddof=1` (sample std) — same underlying per-fold numbers, different formula. Kept
`ddof=1` for internal consistency (so the LSTM and transformer LOSO numbers in this
report are computed identically and are genuinely comparable) and added an explicit
note in the report's own text so a reader comparing against `PLAN.md`'s older
`0.928 ± 0.041` text doesn't mistake the difference for a data problem.

**`06_latency_transformer.py`** — Issue-9 protocol applied to `transformer_searched`
(loads the real `seed42/best.pt`, latency is weight-independent but this matches
Issue 9's own convention). **The result flipped an assumption written into the first
draft's report text**: I had written "the transformer costs more than the BiLSTM"
before actually running the benchmark. The real M4 numbers: transformer CPU batch-1 =
0.459 ms/window vs the BiLSTM's 0.575 ms — the transformer is **faster**, not slower,
despite ~1.3× the parameters (794,241 vs 594,561). Plausible mechanism: the BiLSTM's
recurrence is inherently sequential (must step through 16 timesteps one at a time,
can't parallelize across time), while the transformer's self-attention forward pass
processes all 16 tokens in one parallel matmul — apparently outrunning the recurrence
on this hardware/batch size, even with more total parameters. Rewrote the report-text
generation to be fully data-driven (computes the actual ratio and picks "faster" vs
"slower" from its sign) rather than asserting a fixed direction, so this class of
mistake can't recur if the numbers ever look different on a re-run.

**`05_compare_vs_lstm.py`** — the critical script. Mirrors Issue 4's `get_probs()` /
tie-corrected-Mann-Whitney-AUC / fixed-RNG-bootstrap pattern exactly, extended to a
*paired* bootstrap (same resampled window indices applied to both models' probability
vectors each of the 10,000 iterations, isolating sampling noise on the shared 2094
test windows from any genuine model difference) plus the LSTM parity gate PLAN.md §6
mandates before any Delta is trusted.

**Parity gate: PASS, exactly** — every recomputed LSTM per-seed test AUC matched its
stored `final.json` value to `0.00e+00` (seed42=0.913114, seed0=0.933424,
seed1=0.943189, seed2=0.936295, seed3=0.935822).

**Investigated a discrepancy rather than shipping past it**: the transformer's own
recomputed probabilities showed a small, consistent drift (~1e-6 to 8e-6 across all
10 Stage-D checkpoints) from their Kaggle-GPU-stored `final.json` test AUCs — nothing
in PLAN.md required checking this (the mandated parity gate only covers the LSTM), but
given the LSTM's gate matched *exactly*, the asymmetry was worth explaining rather than
ignoring. Ran a direct test: recomputed the same checkpoint's test AUC via a single
big forward pass vs via `DataLoader(batch_size=32)` (engine.py's own convention) —
both gave the identical result (0.9488637603623995), ruling out batch-size as the
cause. The remaining explanation is CPU/GPU floating-point device drift (Phase T4
trained on Kaggle T4 GPU; this recomputation runs on local CPU) — exactly the risk
`PLAN.md` §10 pre-registered ("exact bitwise reproducibility is per-device"). Added a
permanent, non-aborting sanity check to the script itself (`transformer_sanity_check()`,
tolerance 1e-4 — well above the observed ~1e-6-to-8e-6 drift but far below anything
that would indicate a real bug like a wrong config or wrong checkpoint) so this is
now self-documenting rather than something that has to be rediscovered by hand.

**Primary comparison result:**

| model | params | test AUC (seed-avg) | 95% CI |
|---|---|---|---|
| BiLSTM baseline (frozen) | 594,561 | 0.9423 | [0.9306, 0.9533] |
| transformer_default | 268,417 | 0.9428 | [0.9312, 0.9538] |
| **transformer_searched (winner)** | 794,241 | **0.9558** | [0.9453, 0.9656] |

Δ (searched − LSTM) = **+0.0135, 95% CI [+0.0097, +0.0174]** (excludes 0). Paired
t-test over the 5 seeds: t=3.498, p=0.0249 (significant at 0.05). **Both pre-registered
WIN conditions met.**

Secondary: Δ (default − LSTM) = +0.0005, 95% CI [-0.0034, +0.0043] (includes 0),
p=0.827 — a clean **TIE**. This is the most informative single number in the whole
report: the un-searched transformer, run with the LSTM's own exact recipe, does not
measurably beat it. The win is attributable to the 78-config staged search finding a
genuinely better architecture, not to the architecture family (attention vs
recurrence) on its own. Mirrors Issue 8's finding that the LSTM's own grid search
*confirmed* (rather than trivially rubber-stamped) its hand-set recipe — here the
search *did* find something better than the hand-set default, and the paired
bootstrap confirms that improvement survives on the untouched test set, not just on
the validation set that selected it.

**Verdict: WIN.** Full writeup: `phase5_analysis/05_comparison_report.md`.

**Re-ran all four scripts fresh in sequence** after all fixes (not just trusting the
first successful run) — identical numbers, no errors, confirming reproducibility.
Cleaned up `__pycache__`/`.DS_Store`; confirmed via `git status` that nothing leaked
outside `transformer/`.

Updated `PLAN.md` (Phase T5 DONE block with full results, header status line, §8 file
map), `README.md` (one-paragraph story, status table, new "Headline result" section
replacing the "lands here after T5" placeholder, Phase T4 status paragraph trimmed,
how-to-run, Files table), and `phase5_analysis/README.md` (full rewrite from
placeholder to results) to close out Phase T5.

**Next: Phase T6** — close remaining `PLAN.md` DONE-block bookkeeping, slot these
numbers into `journal_prep/issue3_baseline_comparison/` and the manuscript
(`tab:baselines` + Model Capacity section), one-line updates to root `CLAUDE.md` /
`pipeline/CODE_STATE.md`.

---

## 2026-07-12 — Phase T6 complete: documentation & integration, all six phases done

User said "lets start" on Phase T6, plus asked for a proper supervisor-facing
documentation file.

**`journal_prep/issue3_baseline_comparison/`:** added a Transformer row to both
tables in `03_baseline_comparison.md` (the "our result" table and the
standard-protocol comparison table), an "Update" note explaining the
searched-vs-default tie is the real finding (not just "transformer wins"), and revised
the "how to read this" framing so it covers both architectures rather than singling
out the BiLSTM. Extended `04_positioning_vs_prior_work.md`'s "architecture unjustified"
gap-closing row to mention the transformer's own 78-config search, and added a new
"Model-choice question, answered empirically" section — this directly answers the
question the base thesis's own supervisor-review pack poses in its FAQ ("Why BiLSTM,
not a Transformer?") but had only argued on small-data grounds. Updated
`README.md`'s one-line claim and added a pointer note at the top.

**`paper_and_artifacts/Journal_writing/paper_skeleton.tex`:** built the `tab:baselines`
LaTeX table in full — it was previously an empty stub (just a caption and a
"ask Claude to build this" note) — using `booktabs`, sourced from
`03_baseline_comparison.md`, with both the BiLSTM and new Transformer rows.
IntFormer/PIT rows flagged inline with their bib-entry-pending status rather than
inventing citation keys that don't exist in `references.bib` yet (checked directly —
confirmed absent, matches the project's already-tracked BibTeX debt). Extended the
"Model Capacity" NOTE to draw out a real contrast: the BiLSTM's own hidden-size
ablation found more capacity doesn't help (Issue 7), but the transformer's search
found the opposite (4 layers beat 2) — capacity insensitivity was a property of that
architecture, not a universal one. Added a new subsection with
`\label{sec:transformer-comparison}` covering the full result, still in NOTE form to
match the rest of the file's current state (this skeleton is a staging area for the
real Overleaf manuscript — every other Results subsection is likewise still notes, not
drafted prose, so filling in full paragraphs here alone would have been inconsistent
with how the file is actually being used).

**Root docs:** `CLAUDE.md` — "Three top-level folders" → "Four," added the
`transformer/` bullet, and one paragraph in "What this is" summarizing the extension
and the result. `pipeline/CODE_STATE.md` — one-line pointer added to its existing
repo-layout callout block, noting `transformer/` is untracked by this file and has its
own docs.

**New: `transformer/SUPERVISOR_SUMMARY.md`** (the user's explicit ask, beyond Phase
T6's original scope) — modeled directly on `paper_and_artifacts/supervisor_review/
00_README_START_HERE.md`'s structure (numbered sections, honest limitations, a
talking-script + Q&A section), since that pack's own format had already proven itself
for exactly this purpose on the base thesis. Content: why the extension exists (that
pack's own "why not a Transformer?" FAQ answer, now with a measured answer instead of
an argument), what was frozen identical for fairness, the model and the staged search,
the result, supporting evidence (LOSO, latency, determinism), honest limitations, the
verification discipline, and the talking script.

**Caught and fixed a real cross-document consistency issue while drafting this**,
before it shipped: an early draft's headline table used the AUC of each model's
5-seed-*averaged probability vector* (0.942 / 0.943 / 0.956 — the same statistic the
bootstrap comparison itself uses as "headline"), which is a real, correctly-computed
number, but is **different** from the plain average of 5 independent AUCs (0.932 /
0.934 / 0.950) that is the well-established "0.932" figure quoted everywhere else in
the repo for the BiLSTM (every journal_prep doc, the manuscript, this project's own
other tables). Shipping 0.942 as "the BiLSTM's number" in a document whose entire
purpose is clarity for a supervisor who already has "0.932" memorized would have read
as an unexplained discrepancy or an error. Fixed by using the plain 5-seed-mean
framing consistently as the headline (matching every other document), and explaining
the small gap to the bootstrap's own Δ (+0.018 simple vs +0.0135 bootstrap) in one
honest sentence instead of silently picking whichever number was locally convenient.
Also caught and tightened an overstated latency claim ("roughly 100× faster than
needed") against the actual measured factors (BiLSTM ~58×, Transformer ~73×) before
it shipped.

**All six phases (T1–T6) of the transformer extension are now complete.** Verdict:
**WIN** — a staged-search Transformer measurably beats the frozen BiLSTM
(ΔAUC = +0.0135, 95% CI [+0.0097, +0.0174], paired t-test p=0.025), while the same
architecture run with the BiLSTM's own un-searched recipe ties it exactly, meaning the
result is attributable to the search, not to switching architecture families alone.

**Not done, and explicitly out of scope:** swapping the transformer into the live
YOLO+ByteTrack demo (`pipeline/10_yolo_bytetrack_demo.py`) — `load_bilstm()` already
accepts any model meeting the forward contract, so this remains a small, optional
follow-up if requested later, not something this extension required.

Updated `PLAN.md` (Phase T6 DONE block, header status line) and `README.md` (status
table, Files section, `SUPERVISOR_SUMMARY.md` pointer) to close out the extension.

---

## 2026-07-12 — Fixed: two BiLSTM AUC numbers left unreconciled (0.932 vs 0.942)

User noticed `README.md`'s "Headline result" table showed the BiLSTM at 0.9423, while
every other document in the repo (`journal_prep`, the manuscript, this project's own
§1) cites 0.9324 ± 0.0114, and asked which one is accurate — correctly pointing at
`journal_prep/issue2_clean_protocol` as the authoritative, zero-leakage source.

**Re-verified directly from that source** (`runs_clean/multiseed/seed{42,0,1,2,3}/
final.json`): per-seed test AUCs 0.913114 / 0.933424 / 0.943189 / 0.936295 / 0.935822;
plain mean = 0.9324 (matches the canonical figure exactly), std (ddof=1) = 0.0114.
**Both numbers come from the identical zero-leakage checkpoints — there is no data or
correctness discrepancy.** The difference is purely which of two statistics is being
reported:
- **0.9324 ± 0.0114** — the plain average of the 5 seeds' own independent test AUCs.
  This is, and remains, the canonical BiLSTM number to cite everywhere.
- **0.9423** — the AUC obtained by averaging the 5 seeds' *predicted probabilities*
  together first (an implicit small ensemble across the 5 checkpoints), then scoring
  that one combined probability vector. This number only exists because
  `05_compare_vs_lstm.py`'s paired bootstrap needs a single probability vector per
  model to run the comparison (PLAN.md §6's "headline = seed-averaged probability
  vectors") — it was never meant to replace 0.9324 as a citable figure, but it was
  left sitting in "headline" tables without an explicit note saying so, which is
  exactly what caused the confusion.

**Fixed by adding an explicit reconciliation note everywhere the 0.9423 figure
appears as a table headline** — `phase5_analysis/05_compare_vs_lstm.py`'s
`write_report()` (so the generated `05_comparison_report.md` explains it every time
the script re-runs, not just this once), `phase5_analysis/README.md`, `README.md`,
and `PLAN.md`'s Phase T5 DONE block. Each note states both numbers, which one is
canonical, why the other exists, and points to the full explanation.
`SUPERVISOR_SUMMARY.md` already used only 0.932 throughout (fixed proactively before
it shipped, per the previous entry) and needed no change. Re-ran
`05_compare_vs_lstm.py` after the fix — identical verdict and numbers, only the
report text changed.

---

## 2026-07-13 — Post-F1-pivot metric qualification (audit T1-T5)

The F1-first program (`f1_optimization/`) showed the WIN does not carry to F1
(families TIE after identical F1-first optimization). All WIN loci in this folder now
carry the "on AUC" qualifier (README, PLAN header + §6 adoption line, phase5 README,
05_compare_vs_lstm.py WIN template — report regenerated, numbers unchanged);
SUPERVISOR_SUMMARY got an F1/acc column, a new §4b (F1-first verdicts), corrected
p-value/LOSO/"no code path" wording (T4), and a metric-scope note in
04_final_summary.md. Latency narrative synced to the artifact of record
(0.459 ms / 1.25×, was 0.457/1.26 — T3); parity-gate sentence now states the exact
0.00e+00 match (T5).

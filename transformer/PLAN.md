# Transformer Extension — Master Plan

**Created:** 2026-07-11 · **Status:** ALL PHASES (T1–T6) DONE — verdict: **WIN on AUC** (F1-first follow-up: families TIE on F1, see f1_optimization/) · **Requested by:** supervisor
**Companion docs:** `README.md` (index + how to run) · `PROGRESS_LOG.md` (chronological numbers log)

Supervisor request: *build the best Transformer model on our pipeline, on our dataset, and
compare it with the LSTM model.* The LSTM base = the `journal_prep/` clean-protocol results.

This plan is written **before any transformer code exists**. Grids, the default config,
selection rules, and the outcome verdict templates are pre-registered here so that no
decision is made after seeing test numbers. Each phase is closed by appending a
`### ✅ DONE — VERDICT` block with the real numbers (journal_prep convention).

The repo already poses the question this work answers:
`paper_and_artifacts/supervisor_review/00_README_START_HERE.md` argues *"Why BiLSTM, not a
Transformer?"* on small-data grounds. This extension replaces that argument with measured
numbers — **any outcome is publishable** (win → new headline model; tie/loss → the
model-choice argument is now empirical, and it slots into the manuscript's Model Capacity
section next to Issues 7/8).

---

## 1. The base model (what we compare against — never retrained)

Canonical LSTM = clean-protocol BiLSTM from `journal_prep/issue2_clean_protocol/`
(leakage-free windows anchored at PIE `crossing_point`, TTE band [30,60], obs_len 16,
overlap 0.5). **Its checkpoints and numbers are frozen; the comparison loads them, never
retrains them** — a retrained baseline would fork the canonical 0.932.

| model (5 seeds [42,0,1,2,3]) | params | test AUC | PR-AUC | F1 | Acc |
|---|---|---|---|---|---|
| **BiLSTM baseline (locked)** | 594,561 | **0.9324 ± 0.0114** | 0.876 ± 0.016 | 0.8275 ± 0.0123 | 0.8827 ± 0.0091 |
| BiLSTM + attention (Issue 2) | ~611k | 0.925 ± 0.010 | 0.865 | — | — |
| BiLSTM bbox-only 4-D (Issue 2) | — | 0.753 ± 0.020 | 0.610 | — | — |

Per-seed baseline test AUC: seed42 = 0.9131 (lowest), seed0 = 0.9334, seed1 = 0.9432,
seed2 = 0.9363, seed3 = 0.9358. Bootstrap 95% CI ≈ [0.92, 0.95] (Issue 4).
LOSO 6-fold: 0.928 ± 0.041 (excl. tiny set05: 0.915 ± 0.029) (Issue 5).
Latency: 0.575 ms/window CPU batch-1 on M4 (Issue 9).
Grid-search precedent (Issue 8): 36-config search **confirmed** the hand-set recipe
(winner Δ +0.0006, paired-t p = 0.914, n.s.).

Artifacts consumed by this extension:
- Data: `transformer/sequences_clean/` — `X.npy (4906,16,5) float32`, `y.npy (4906,) int8`,
  `meta.pkl` (set_id / video_id / ped_id / anchor_frame / crossing_point / tte). 1648 pos /
  3258 neg (33.6%). **A byte-identical copy of `journal_prep/issue2_clean_protocol/
  sequences_clean/`** (verified via `diff` at copy time, 2026-07-11) — kept local to this
  folder for discoverability, since this workspace is meant to be self-contained. The
  journal_prep copy remains the canonical source (it's where Issue 2 built it); this one
  is not regenerated independently, so the two stay identical by construction.
- LSTM checkpoints: `journal_prep/issue2_clean_protocol/runs_clean/multiseed/seed{42,0,1,2,3}/`
  (`best.pt` — load with `weights_only=False`; `final.json`; `norm_mean.npy` / `norm_std.npy`).
- Comparison machinery: `journal_prep/issue4_bootstrap_ci/04_bootstrap_ci.py` (`get_probs`,
  tie-corrected Mann-Whitney AUC, fixed-RNG paired bootstrap).

---

## 2. Fairness contract

**FROZEN — byte-for-byte identical to the LSTM protocol. Not searchable, not negotiable:**

| item | value |
|---|---|
| data | `sequences_clean/` reused verbatim (no rebuild) |
| splits (by recording set) | train {set01,02,04} N=2178 (812 pos / 1366 neg) · val {set05,06} N=634 · test {set03} N=2094 (681 pos, 32.5%) |
| normalization | per-feature z-score from **train only**; saved as `norm_mean.npy`/`norm_std.npy` (5,) per run |
| loss | `BCEWithLogitsLoss(pos_weight=1.682)` (= 1366/812) |
| threshold | 0.5 on `sigmoid(logit)` |
| batch / epochs | 32 / max 100 |
| early stopping | patience 15 on **val AUC**, best-on-val checkpoint `{"model": state_dict, "epoch", "val_metrics"}` |
| seeds | [42, 0, 1, 2, 3]; seed 42 canonical; `set_seed()` as in Issue 2/8 |
| selection | **validation only**; test set03 evaluated **exactly once** (Phase T4) |
| metrics | sklearn `roc_auc_score` on probs + acc/F1/prec/rec at 0.5; PR-AUC = average precision |

**SEARCHED — the transformer's own knobs (mirrors Issue 8 having searched the LSTM's recipe):**
architecture (width, depth, pooling, positional encoding) and training recipe
(lr, schedule/warmup, dropout, weight decay/optimizer). Seed-42 search budget = 78 configs
≥ 2× the LSTM's 36.

**Reported: both** `transformer_default` (pre-registered §3, LSTM's exact Adam recipe, zero
search) **and** `transformer_searched` (staged-search winner). This preempts both reviewer
attacks: "you tuned the transformer but not the LSTM" (no — Issue 8 tuned the LSTM) and
"you never tuned the transformer" (no — 78-config staged search).

---

## 3. The model

`TransformerIntentPredictor` — small **pre-LN** Transformer encoder, defined once in
`phase1_setup/00_transformer_model.py` (single source of truth; the Kaggle notebooks
embed it verbatim under a sync banner). All searchable dims are constructor args.

```
input (B,16,5)  raw [x1,y1,x2,y2,vehicle_speed] pixels, z-scored
 → Linear(5, d_model)                        # input projection
 → [prepend learned CLS token]               # only if pool="cls" (sequence length 17)
 → + positional encoding                     # "learned": nn.Parameter(1, T(+1), d_model), init N(0, 0.02)
                                             # "sin":     fixed sinusoidal buffer (same length)
 → Dropout(p)
 → nn.TransformerEncoder(
      nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=ff, dropout=p,
                                 activation="gelu", batch_first=True, norm_first=True),
      num_layers=L, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
 → pool: "cls" = h[:,0] | "mean" = mean over the 16 sequence tokens (CLS excluded) | "last" = h[:,-1]
 → Dropout(p) → Linear(d_model, 1)           # logits (B,1) — same contract as the BiLSTM
```

Design decisions (defensible, documented):
- **Pre-LN + terminal LayerNorm** — trains stably at the LSTM-parity lr 1e-3 without
  mandatory warmup; post-LN would force warmup and break the "default = LSTM recipe" cell.
- **nhead = 4 fixed** — head_dim 16/32 for d_model 64/128; with only 16 tokens more heads
  over-fragment; keeping it out of the grid controls search size. `d_model % nhead == 0`
  asserted in `__init__`.
- **One dropout knob** feeding embedding, encoder layers, and head; no extra dropout stacked
  on the CLS readout.
- **Scoped out** (documented, not silently omitted): stochastic depth, data augmentation,
  label smoothing, threshold tuning — the last three would breach the frozen protocol.

Parameter ladder (per encoder layer ≈ 4·d² + 2·d·ff) — brackets the 594,561-param BiLSTM:

| (d_model, ff) | L=2 | L=4 |
|---|---|---|
| (64, 128) | ~69k | ~136k |
| (128, 256) | ~268k | ~533k |
| (128, 512) | ~400k | ~797k |

(d=128, ff=512, L=3 ≈ 598k is the near-exact capacity anchor — cite in the report.
Issue 8 already showed 3.8× LSTM params bought nothing; N_train=2178 is a severe
small-data regime, so the grid deliberately spans ~0.12×–1.3× the BiLSTM budget.)

**Pre-registered `transformer_default`** (fixed here, before any search runs):
`d_model=128, nhead=4, L=2, ff=256, dropout=0.1, learned PE, CLS pool` (~268k params),
trained with the LSTM's exact recipe: **Adam lr 1e-3, wd 1e-5,
ReduceLROnPlateau(max, 0.5, 5) on val AUC**, batch 32, patience 15.

---

## 4. Pre-registered staged search (selection on val only, Issue-8 protocol)

All search runs are **val-only by construction** — the search notebook contains no
test-evaluation code path. Every run is cached as `runs_search/<cfg_id>/seed<k>.json`
(schema: `{**cfg, "seed", "n_params", "best_epoch", "val": {auc, pr_auc, f1, acc, prec,
rec}, "seconds"}` — **no `test` key ever**), so interrupted sessions resume free.
`cfg_id` = arch block `__` recipe block, e.g. `d128_ff256_L2_cls_lpe__lr1e-03_plateau_do0.1_wd1e-05`.

**Stage A — architecture (36 configs, seed 42)** @ default recipe (Adam 1e-3, wd 1e-5,
plateau, dropout 0.1):
- (d_model, ff) ∈ {(64,128), (128,256), (128,512)}
- num_layers ∈ {2, 4}
- pool ∈ {cls, mean, last}
- pos ∈ {learned, sin}

Rank by seed-42 val AUC; **top-3 architectures advance**.
*Pre-registered contingency:* if > 1/3 of cells diverge (val AUC < 0.70) at lr 1e-3, rerun
Stage A once at lr 3e-4 and document the amendment here (legal — selection never sees test).

**Stage B — training recipe (36 configs, seed 42)** on the Stage-A #1 architecture:
- lr ∈ {1e-4, 3e-4, 1e-3}
- schedule ∈ {plateau (LSTM parity), warmup5+cosine (linear 0→lr over 5 epochs, cosine to
  0.01·lr at epoch 100)}
- dropout ∈ {0.1, 0.3, 0.5}
- weight decay ∈ {1e-5, 1e-2} — AdamW throughout Stage B with no-decay parameter groups
  (bias / LayerNorm / PE / CLS); at wd=1e-5 AdamW ≈ Adam.

**+ transfer check (6 runs):** top-3 recipes × Stage-A architectures #2 and #3 — guards
against "the arch ranking only holds under the default recipe".
Total seed-42 pool: 36 + 36 + 6 = **78 configs**.

**Stage C — candidates (≤ 25 new runs):** top-5 (arch × recipe) combos from the pooled 78
**+ `transformer_default`** (always carried, like Issue 8 carried the baseline) × seeds
[42,0,1,2,3] (seed-42 cached). **Winner = highest MEAN val AUC** — the selection-noise
control that mattered concretely in Issue 8, where the seed-42 leader was not the 5-seed
winner. Val N=634 means differences < ~0.01 are noise; the 5-seed mean is the tiebreaker.

**Stage D — final (Phase T4, 10 full runs + 1 rerun):** `transformer_searched` (Stage-C
winner) + `transformer_default` × 5 seeds with **full run dirs** (checkpoints needed for the
paired comparison). **Test set03 evaluated here and nowhere else.** Includes a seed-42
determinism-of-record rerun (must reproduce its own final.json on the recorded device —
the Issue-2 "seed 42 reproduces 0.9131" precedent applied to the transformer).

---

## 5. Compute split & Kaggle workflow

**User directive: ALL experiment-grade training runs on Kaggle (T4×2). Everything else is
local (M4, `.venv`)** — model code, seconds-scale smoke probes, winner-selection review,
paired-bootstrap comparison, latency, reports, docs.
(Verified: `.venv` has torch 2.12.0 + sklearn 1.9.0 + scipy 1.17.1 locally — the CLAUDE.md
"no sklearn locally" note is outdated, Issue 8 already imported both. torch 2.12 requires
`enable_nested_tensor=False` on `nn.TransformerEncoder`.)

One-time setup: upload the three `sequences_clean/` files (~2 MB total) as a **private
Kaggle dataset** `pie-sequences-clean`.

The upload → run → download loop (both notebooks, precedent:
`issue2_clean_protocol/06_multiseed_variants_kaggle.ipynb`):
1. Upload the notebook, attach `pie-sequences-clean`, set accelerator **GPU T4 ×2**, Run All.
2. Cell 1 asserts the environment (CUDA count, `X.shape == (4906,16,5)` — refuses leaky data).
3. Training cells parallelize **2 worker processes, one config per GPU** (`WORKERS = 2`,
   fallback 1); every run cached to its JSON/run-dir, so a re-run resumes instead of repeating.
4. Final cell zips outputs; download and unzip into that phase's own folder locally — the
   local report scripts consume the identical layout unchanged.

| notebook | runs | est. wall-clock (2×T4) |
|---|---|---|
| `phase2_kaggle_search/03_search_kaggle.ipynb` — Stages A+B+transfer+C, val-only | ~103 trainings | ~30–90 min |
| `phase4_kaggle_final/04_final_loso_kaggle.ipynb` — Stage D + LOSO, test once | ~17 trainings | ~10–30 min |

Both embed the model class and `train_run()` **verbatim** from `phase1_setup/00_`/`02_`
under a "KEEP IN SYNC" banner (journal_prep precedent for self-contained notebooks).

---

## 6. Statistical comparison plan (`phase5_analysis/05_compare_vs_lstm.py`, local)

1. **Regenerate LSTM probs** on the identical 2094 set03 windows (meta order) from the 5
   frozen checkpoints — Issue-4 `get_probs()` pattern (`weights_only=False`, per-run norm).
2. **Parity gate:** recomputed per-seed LSTM test AUCs must match the stored `final.json`
   values (seed42 = 0.9131) before any Δ is computed; abort otherwise.
3. **Transformer probs** from `runs_final/transformer_{searched,default}/seed*/`.
4. **Primary evidence — 10k paired percentile bootstrap of ΔAUC:** `np.random.default_rng(42)`
   reset per comparison so resample indices are identical across models;
   Δ_b = AUC_T(idx) − AUC_L(idx). Headline = seed-averaged probability vectors → one Δ + 95%
   CI; plus per-seed-pair Δs. Absolute ROC-/PR-AUC CIs in Issue-4 format so the table slots
   next to the existing one.
5. **Secondary — paired t-test over seeds** (`scipy.stats.ttest_rel`, n=5, low power at
   σ_seed ≈ 0.011 — stated in the report; the window-paired bootstrap is primary).

**Pre-registered outcome templates (verbatim in the report):**
- **WIN** — paired-bootstrap 95% CI of ΔAUC excludes 0 **and** paired-t p < 0.05:
  adopt the transformer as headline model, with param/latency caveats; BiLSTM retained as
  the efficient baseline.
- **TIE** — CI includes 0: *"At matched protocol and a ≥2× search budget, the transformer
  does not measurably beat the BiLSTM (Δ = …, 95% CI […, …]) — consistent with Issue 2's
  finding that attention adds nothing on this 16×5 signal. BiLSTM retained."*
- **LOSS** — transformer clearly below: reported exactly as plainly; BiLSTM confirmed.
The thesis narrative ("ego-speed dominant; temporal-model choice secondary") survives any
outcome. **No hedging, no cherry-picking the best seed.**

---

## 7. Phases

### Phase T1 — Local scaffolding & sanity gates ✅ DONE — 2026-07-11
Folder: `phase1_setup/` — `00_transformer_model.py`, `01_sanity_checks.py`,
`02_train_transformer.py`.
Gates (all must pass before anything is uploaded to Kaggle; tiny/seconds-scale probes only):
- **Gate 0 — protocol asserts:** X (4906,16,5); splits 2178/634/2094; 681 test positives;
  recomputed pos_weight = 1366/812 = 1.682; norm shape (5,).
- **Gate 1 — linear-probe floor:** L=0 wrapper ≈ logistic regression on mean-pooled projected
  features (+ sklearn LogisticRegression on the flat 80-D input as reference). Any L≥1
  transformer must clearly beat this floor or the wiring is buggy.
- **Gate 2 — overfit-a-tiny-batch:** 64 train windows, dropout 0, 200 epochs → train acc 1.0.
- **Gate 3 — determinism probe:** default config, seed 42 twice on CPU (short run) →
  identical val AUC; param-count table for all Stage-A sizes + shape self-test.
Produces: `phase1_setup/01_sanity_report.md`.

**Result: ALL GATES PASS** (`01_sanity_report.md`).
- Gate 0: X (4906,16,5); train/val/test = 2178/634/2094; test positives 681; pos_weight
  1366/812 = 1.682; norm shapes (5,). All exact.
- Gate 1: L=0 wrapper (mean-pool, no encoder) val AUC 0.899; sklearn LogisticRegression
  (flat 80-D, balanced) val AUC 0.939; L=2 transformer (d64/ff128/cls/learned, 30 epochs)
  val AUC 0.931 — clearly in the same regime as both floors (wiring sound). Not a
  benchmark number (30 epochs, seed 42 only, tiny model) — just a sanity check.
- Gate 2: 64-window overfit, 200 epochs, dropout 0 → train acc 1.0000.
- Gate 3: seed 42 twice on CPU → val AUC 0.934379 vs 0.934379, |Δ|=0.
- Param ladder confirmed matching the design estimate: (64,128)×{2,4}L = 68,673 /
  135,617; (128,256)×{2,4}L = 268,417 / 533,377; (128,512)×{2,4}L = 400,001 / 796,545.
  `transformer_default` = **268,417 params** (BiLSTM: 594,561).
- Extra local check (not a formal gate, prompted by user note "I have a GPU too, M4
  Air"): forward+backward smoke test on **MPS** — output shape (32,1), no NaNs,
  `enable_nested_tensor=False` works cleanly on torch 2.12. MPS is available for local
  dev/smoke work; the compute directive (all experiment-grade training on Kaggle T4×2)
  is unchanged.

**Bug found and fixed during Gate 3 (documented for the record — not a training-engine
bug):** the first sanity-check draft called `quick_train(build_default(), ...)` — Python
evaluates `build_default()` *before* entering `quick_train`, so the model was constructed
(random init) before `quick_train`'s internal `set_seed()` ran. Two "same seed" runs
therefore started from different random weights and diverged (|Δ AUC| ≈ 0.008 over 5
epochs) even though training itself was seeded correctly. Initial diagnosis suspected
CPU multi-threaded matmul non-associativity (a real phenomenon for attention layers, and
still worth knowing about) but pinning `torch.set_num_threads(1)` did **not** fix it —
that ruled out threading and pointed at construction order instead. Fix: `quick_train`
now takes a zero-arg model **factory**, and seeds before calling it. **Verified this bug
does not exist in `02_train_transformer.py::train_run()`** — there, `set_seed(seed)` is
called before `build_model(cfg)` (line order checked directly). Only the throwaway
sanity-check helper was affected.

### Phase T2 — Kaggle staged search (val-only) ✅ DONE — ran on Kaggle 2026-07-11
Folder: `phase2_kaggle_search/` — `03_search_kaggle.ipynb` (Stages A, B, transfer, C from
§4), generated by `gen_search_nb.py` (re-run that script after any edit to
`phase1_setup/00_transformer_model.py` or `phase1_setup/02_train_transformer.py` — it
embeds both verbatim under "KEEP IN SYNC" banners).
Produces (downloaded): `runs_search/<cfg_id>/seed<k>.json` for all 78 + candidate runs,
plus `runs_search/_stage_summary.json`, and `kaggle_outputs/` (raw download archive:
the zip + `worker.py`).
**Acceptance:** ≥ 78 seed-42 JSONs + 6×5-seed candidate JSONs present; no `test` key in any;
`transformer_default` seed-42 val AUC > 0.90 (plausibility floor; LSTM seed-42 val ≈ 0.964)
— if far below, halt and debug rather than search over a broken model.

**Design notes / bug caught before running:**
- `cfg_id()` must encode `optimizer`, not just `lr/schedule/dropout/weight_decay`. Stage A
  trains with Adam (the frozen default recipe); Stage B always uses AdamW. A Stage-B cell
  can numerically match Stage A's default recipe on every other field (lr=1e-3, plateau,
  dropout=0.1, wd=1e-5) while differing only in optimizer — omitting it from the id would
  collide the two into the same cache path and silently discard one run. Caught during
  design, fixed before the notebook was generated.
- **2-GPU parallelism uses separate OS subprocesses (`subprocess.Popen`, one per GPU via
  `CUDA_VISIBLE_DEVICES`), not threads or `multiprocessing`.** Threads were rejected: PyTorch's
  global RNG (used by dropout, shuffling, and layer init) is process-global, not
  thread-local, so two threads training concurrently would corrupt each other's "seeded"
  random streams — silently breaking the seed-42 reproducibility the whole search protocol
  depends on. Plain `multiprocessing` was rejected too: Jupyter/Kaggle notebooks execute
  as `__main__`, which spawned worker processes can't reliably re-import. Subprocesses
  launching a real script file (`worker.py`, written to `/kaggle/working/` at runtime)
  sidestep both problems — each worker is a fully independent interpreter and RNG.
- Every job dict carries a pre-computed `cfg_id`, so `worker.py` never needs its own copy
  of the `cfg_id()` function — one less place for the two to drift out of sync.
- Validated locally before shipping: (1) syntax — every code cell `compile()`s without
  error; (2) mechanics — a 2-config/1-seed job run through the real subprocess/worker
  pipeline on CPU with the real `sequences_clean/` data, including a cache-hit re-run
  (confirmed identical results, no retraining); (3) full chain — Stage A → top-3 archs →
  Stage B → top-3 recipes → transfer → pooled dedup → Stage C candidates (correctly
  identified and tagged `transformer_default`) → 5-seed multiseed → winner selection →
  summary + zip, all run for real on a shrunk grid (4/4/6/10 jobs instead of
  36/36/6/~25). No result at any stage carried a `test` key.

### ✅ DONE — 2026-07-11 (ran on Kaggle T4×2)

Ran on Kaggle in one session. Working data lives at
`transformer/phase2_kaggle_search/runs_search/` (102 files: 78 distinct configs × seed
42, plus 4 additional seeds each for the 6 Stage-C candidates = 78 + 24 = 102 — exact
match, confirming the caching never re-ran a config that was already on disk). The raw
Kaggle download (zip + the notebook's own `worker.py` copy) is archived at
`phase2_kaggle_search/kaggle_outputs/` for provenance; the duplicate `00_/02_.py` files
and duplicate `runs_search/` that came inside that download were removed as exact copies
of files already canonical elsewhere (repo cleanup, see PROGRESS_LOG.md).

**Verified independently** (not just trusted): `03_search_report.py` reloads every raw
`seed*.json`, recomputes Stage A's top-3 and all 6 Stage-C candidates' mean/std from
scratch, and asserts the result matches `_stage_summary.json` exactly (to 1e-9) — all
assertions passed, plus a hard assert that zero files anywhere contain a `test` key.

**Winner:** `d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05` — d_model=128,
**num_layers=4** (vs default's 2), dim_ff=512, **pool="last"** (vs default's "cls"),
**pos="sin"** (vs default's "learned"), Adam/plateau/lr=1e-3/dropout=0.1/wd=1e-5 (i.e.
the LSTM's exact recipe — Stage B's recipe search did not end up mattering for the
overall winner). Val AUC (5-seed) **0.9789 ± 0.0038**, param count 794,241.
`transformer_default` scored 0.9629 ± 0.0056 and ranked **66th of 78** in the full
seed-42 architecture pool — the search meaningfully beat the un-searched default.

**Selection-noise control mattered again, exactly as in Issue 8:** the single-seed
(seed-42) leader across all 78 configs (`d128_ff256_L4_last_spe__adam_...`, seed-42 val
AUC 0.9839) is *not* the 5-seed-mean winner — its own mean drops to 0.9773, behind the
actual winner's 0.9789. A single-seed search would have picked the wrong config, which
is precisely why Stage C re-seeds the top candidates before deciding.

**Notable, reportable pattern:** every one of Stage A's top ~20 architectures uses
`pool="last"` or the sinusoidal `pos="sin"` — the search independently converged on
`pool="last"`, which mirrors the BiLSTM's own readout (its last timestep). That the
search rediscovered "read out from the final timestep" as the winning strategy, without
being told to, is a reassuring sign it found a real signal rather than an arbitrary
config.

**This is a validation-set result only — not the comparison.** Val AUC differences
between differently-searched models aren't directly comparable (a validation-selection
process can inflate a val number without a matching test gain). The actual comparison
happens in Phase T5, via a paired bootstrap on the untouched test set. For context only:
the LSTM's own 5-seed val AUC (Issue 8) is 0.9644 ± 0.0043; its test AUC — the number the
transformer must actually beat — is 0.9324 ± 0.0114.

Outputs (`phase2_kaggle_search/runs_search/`): 102 run files + `_stage_summary.json`.
Re-derived report (`phase3_search_review/`): `03_arch_grid.csv`, `03_recipe_grid.csv`,
`03_candidates_multiseed.csv`, `03_search_summary.md`, `03_search_figure.png`.

### Phase T3 — Winner review (local, human checkpoint) ✅ DONE — confirmed by user 2026-07-11
Folder: `phase3_search_review/` — `03_search_report.py` re-derives ranking/tables/figure
from the downloaded JSONs in `../phase2_kaggle_search/runs_search/`: `03_arch_grid.csv`,
`03_recipe_grid.csv`, `03_candidates_multiseed.csv`, `03_search_summary.md`,
`03_search_figure.png`.
**Acceptance: user reviews and confirms the winner config BEFORE notebook 2 exists-with-test.**
Test set still untouched at this point.

**Mechanical part done 2026-07-11:** script written and run; every number
independently recomputed from raw files and cross-checked against the notebook's own
summary (exact match, all assertions pass); zero `test` keys found anywhere. Winner:
`d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05`, val AUC 0.9789 ± 0.0038
vs default's 0.9629 ± 0.0056 (full numbers in Phase T2's DONE block above).

**User confirmed the winner 2026-07-11** ("okay i have checked the Phase T3 summary. lets
start building phase T4") — green light to build the notebook that touches test set03.

### Phase T4 — Kaggle final training + LOSO (test touched once) ✅ DONE — ran on Kaggle 2026-07-11
Folder: `phase4_kaggle_final/` — `04_final_loso_kaggle.ipynb`, generated by
`gen_final_nb.py` (re-run that script after any edit to `phase1_setup/00_
transformer_model.py` or `phase1_setup/02_train_transformer.py`; same "KEEP IN SYNC"
convention as `phase2_kaggle_search/gen_search_nb.py`). Winner config is hardcoded into
the notebook's CONFIG cell, copied from the raw `seed42.json` (not hand-parsed from the
cfg_id string) — no manual paste step needed.
- Stage D: winner + default × 5 seeds → full run dirs
  `runs_final/transformer_{searched,default}/seed<k>/{best.pt, final.json, history.json,
  norm_mean.npy, norm_std.npy}` — `final.json` in the pipeline schema (`best_epoch`,
  `val`/`test` {loss,acc,f1,auc,prec,rec}, `test_confusion_matrix`) so
  `pipeline/05_compare_runs.py` reads it directly. + seed-42 determinism-of-record rerun
  (separate `runs_final/_determinism_check/transformer_searched_seed42_rerun/`, compared
  numerically against the primary run; notebook prints PASS/WARN).
- LOSO (winner config, Issue-5 protocol): 6 leave-one-set-out folds, per-fold **grouped**
  85/15 val split (by `set_id/ped_id`), per-fold pos_weight + train-only norm, seed 42 →
  `runs_loso/<fold>.json`.
**Acceptance:** 10 complete run dirs + 6 LOSO JSONs downloaded; determinism rerun reproduces.

**Engine change needed for LOSO (2026-07-11):** `train_run()` in `phase1_setup/
02_train_transformer.py` gained an optional `pos_weight=None` parameter (defaults to the
frozen `POS_WEIGHT=1.682` when not given) — LOSO needs a **per-fold** pos_weight
(`n_neg/n_pos` of that fold's train pool, Issue-5 precedent), which the module-level
constant can't express. This reuses the exact same, already-validated training loop for
LOSO instead of duplicating it (Issue 5's own `05_loso_cv.py` duplicated the BiLSTM loop
instead; here the engine is shared across Phase T1 dev runs, Phase T2's search, and
Phase T4, so extending it was preferred over a third copy). The result dict also now
records the `pos_weight` actually used, for the per-fold report table.
**Proven inert for every existing use:** re-ran the Phase-T1 CLI dev command
(`--preset default --seed 42 --device cpu`, no `pos_weight` passed) — val AUC
0.9627719038319079, byte-identical to before the change. Re-ran all four Phase-T1 sanity
gates (still ALL PASS) and `03_search_report.py` (still finds the identical winner,
0.9789 ± 0.0038, all internal cross-checks still pass) after the change, and
regenerated `phase2_kaggle_search/03_search_kaggle.ipynb` so its embedded copy stays in
sync (its own behavior is unaffected — the search never passes `pos_weight`).

**Worker design:** `worker_final.py` (embedded in the notebook the same way
`worker.py` is in Phase T2) dispatches on a `job["kind"]` field — `"final"` uses the
fixed train/{set05,06}-val/set03-test split (`engine.split_data`) with `eval_test=True`
and a real checkpoint dir; `"loso"` loads `meta.pkl` for `ped_id`, builds the
grouped-by-pedestrian fold split itself (mirroring Issue 5's `train_fold()` exactly),
computes that fold's `pos_weight`, and calls `train_run(..., out_dir=None,
pos_weight=pw)`, writing the augmented result dict straight to `runs_loso/<fold>.json`
(no checkpoint needed there, per the file map). Both kinds cache-skip on their output
path already existing, same convention as Phase T2. Unlike Phase T2's worker, this one
needs no `--out_root` CLI arg — every job already carries its own absolute `out_dir` /
`out_path`, since Stage D's two configs and the 6 LOSO folds don't share one naming
scheme the way the search's `cfg_id`-keyed cache did.

**Validated locally before shipping (2026-07-11), same three-tier discipline as Phase
T2:** (1) syntax — all 11 code cells `compile()` without error; (2) mechanics — both job
kinds (`"final"` with `eval_test=True` + full checkpoint dir, `"loso"` with the
pedestrian-grouped fold split) run for real through `worker_final.py` via subprocess,
including a cache-hit re-run for each (confirmed identical output, no retraining) —
**against a fabricated synthetic dataset (6 fake sets, 5 peds each, alternating
labels), never the real `sequences_clean/`**, so test set03 stays untouched by this
validation exactly as the protocol requires; the one shape assert in `load_raw()` that
exists to catch a wrong data directory was relaxed only in a private in-memory copy of
the engine used solely by the test harness, never in the repository file. Confirmed:
run-dir schema correct (`best.pt`/`final.json`/`history.json`/`norm_mean.npy`/
`norm_std.npy`, `final.json` keys exactly `{best_epoch, val, test,
test_confusion_matrix}`); LOSO JSON schema correct (`test_set`/`train_n`/`val_n`/
`test_n`/`test_pos`/`pos_weight`/`val`/`test`/`test_confusion_matrix`/`best_epoch`,
`train_n + val_n` equal to the 5-set pool size); both kinds' cache-skip confirmed via
unchanged file mtimes on a second run.

### ✅ DONE — 2026-07-11 (ran on Kaggle T4×2, ~8 min for all 17 trainings)

Ran on Kaggle in one session (well inside the ~10–30 min estimate). One bug surfaced in
the notebook's own final cell (`KeyError: 'n_params'` — that field is deliberately not
persisted in `final.json`, matching the pipeline schema, but the summary cell wrongly
assumed it would be there) — fixed and the corrected cell re-run in the same live
session, at zero retraining cost since Stage D/determinism/LOSO had already completed
and cached. Full incident writeup, root cause, and the fix in `PROGRESS_LOG.md`.

Working data lives directly in `phase4_kaggle_final/`: `runs_final/transformer_
{searched,default}/seed<k>/`, `runs_final/_determinism_check/`, `runs_loso/<fold>.json`,
`_final_summary.json`. Raw provenance (the zip + `worker_final.py`) archived at
`kaggle_outputs/`; duplicate `00_/02_.py` copies and the Kaggle editor's internal
`.virtual_documents/` cache that came inside the download were removed as exact
duplicates of files already canonical elsewhere (same convention as Phase T2's cleanup).

**Verified independently, not just trusted** (own script, not committed — mirrors
`03_search_report.py`'s discipline): reloaded every raw `final.json`/`runs_loso/*.json`
and recomputed every 5-seed and 6-fold mean±std from scratch; matches
`_final_summary.json` exactly (to 1e-9). `n_params` in the summary matches the known
architecture-derived values (794,241 / 268,417) for every row. `winner_cfg` /
`default_cfg` / `seeds` in the summary match the pre-registered values exactly. All 6
LOSO fold `test_n` values (258/310/2094/1610/47/587) match Issue 5's real per-set sizes
**exactly**, confirming genuine, untampered `sequences_clean/` data was used. Every
`history.json` (per-epoch log) contains only `val_*` keys — no per-epoch test leakage
anywhere. Confusion-matrix shapes correct on all 17 runs.

**Determinism-of-record: exact bit-for-bit match.** `transformer_searched` seed 42
rerun independently reproduces val AUC 0.9824634655532359 and test AUC
0.9488627211346704 — `|delta| = 0.0` on both, not just "close."

**Stage D results (5 seeds [42,0,1,2,3], T4×2):**

| config | params | val AUC | test AUC |
|---|---|---|---|
| `transformer_searched` (winner) | 794,241 | 0.9789 ± 0.0038 | **0.9497 ± 0.0025** |
| `transformer_default` | 268,417 | 0.9629 ± 0.0056 | 0.9337 ± 0.0058 |
| BiLSTM baseline (frozen, for reference) | 594,561 | 0.9644 ± 0.0043 | 0.9324 ± 0.0114 |

**LOSO (winner config, seed 42, 6-fold):** mean 0.9392 ± 0.0436 (excl. tiny set05 N=47:
0.9270 ± 0.0357), vs the BiLSTM's 0.928 ± 0.041 (Issue 5). Per-fold: set01 0.9050,
set02 0.9212, set03 0.9496, set04 0.8847, set05 1.0000, set06 0.9746.

**These were raw numbers for orientation only, not the formal verdict** — the
pre-registered decision rule (§6: 10k paired bootstrap of ΔAUC on the same windows +
paired t-test) is Phase T5, below, which returned **WIN**.

### Phase T5 — Local analysis ✅ DONE — 2026-07-11 — VERDICT: WIN
Folder: `phase5_analysis/` (reads checkpoints from `../phase4_kaggle_final/runs_final/`
and `../phase4_kaggle_final/runs_loso/`, and the frozen LSTM checkpoints from
`journal_prep/issue2_clean_protocol/runs_clean/multiseed/`):
- `04_final_report.py` → `04_final_results.csv`, `04_final_summary.md` (5-seed mean±std,
  winner + default vs the frozen LSTM row).
- `05_compare_vs_lstm.py` → `05_comparison_results.csv`, `05_comparison_report.md` (bold
  **Verdict** from the §6 templates), `05_comparison_figure.png`.
- `06_latency_transformer.py` (M4 = deployment hardware): Issue-9 core — 50 warmup + 1000
  timed forwards, `torch.mps.synchronize()`, CPU+MPS × batch {1,8,32} → `06_latency_results.json`,
  `06_latency_report.md` (vs BiLSTM 0.575 ms/window CPU; expect the same "CPU beats MPS at
  batch 1" dispatch effect; T=16 makes O(T²) attention irrelevant).
- `07_loso_report.py` → `07_loso_results.csv`, `07_loso_report.md` (fold table vs LSTM's
  0.928 ± 0.041; set05 N=47 outlier caveat carried over).
**Acceptance:** parity gate passed; every report carries its Verdict paragraph.

**Result: WIN.** All four scripts built and run (mirroring Issue 4's `get_probs()`/
bootstrap pattern and Issue 9's latency protocol, reusing rather than reinventing).

**Parity gate: PASS, exactly.** Every recomputed LSTM per-seed test AUC matched its
stored `final.json` value to `0.00e+00` (seed42=0.913114, seed0=0.933424,
seed1=0.943189, seed2=0.936295, seed3=0.935822 — all exact).

**Primary comparison — 10k paired bootstrap of ΔAUC** (`transformer_searched` −
BiLSTM, seed-averaged probability vectors, same 2094 test-set03 windows, same resample
indices applied to both models each iteration):

| model | params | test AUC (seed-avg) | 95% CI (ROC) |
|---|---|---|---|
| BiLSTM baseline (frozen) | 594,561 | 0.9423 | [0.9306, 0.9533] |
| transformer_default | 268,417 | 0.9428 | [0.9312, 0.9538] |
| **transformer_searched (winner)** | 794,241 | **0.9558** | [0.9453, 0.9656] |

**The BiLSTM's 0.9423 here is NOT the canonical `0.9324 ± 0.0114` cited everywhere
else in the repo** (journal_prep, the manuscript, this doc's own §1) — both come from
the identical zero-leakage checkpoints, but this table's number is the AUC of the 5
seeds' *averaged probabilities* (needed so the paired bootstrap below has one
probability vector per model), not the plain mean of the 5 seeds' own AUCs. Treat
`0.9324 ± 0.0114` as the number to cite; `0.9423` is local to this comparison only.
Full explanation: `phase5_analysis/05_comparison_report.md`.

**Δ (searched − LSTM) = +0.0135, 95% CI [+0.0097, +0.0174]** — excludes 0. Paired
t-test over the 5 seeds: t=3.498, p=0.0249 — significant. **Both pre-registered WIN
conditions met → adopt `transformer_searched` as the headline model **for AUC-first reporting**, BiLSTM retained
as the efficient, lower-latency baseline.

**The secondary comparison is the tell:** `transformer_default` vs BiLSTM = **TIE**
(Δ=+0.0005, CI [-0.0034, +0.0043] includes 0, p=0.827). An un-searched transformer
running the LSTM's exact recipe does not measurably beat it — the win is attributable
to the 78-config staged search finding a genuinely better architecture (num_layers=4,
pool="last", pos="sin"), not to switching architecture families per se. This mirrors
Issue 8's own finding for the LSTM (search confirmed, didn't just rubber-stamp, the
hand-set config) and directly answers the "you only tuned the transformer" /
"you never tuned the transformer" preempts from §2 — both configs were reported, and
the un-tuned one is statistically indistinguishable from the baseline it's compared
against.

**Non-mandated transformer sanity check (device drift, not a bug):** recomputing the
transformer's own probabilities locally showed a consistent ~1e-6 to 8e-6 drift from
each run's Kaggle-GPU-stored `final.json` test AUC (all 10 Stage-D checkpoints
checked) — confirmed via a direct test to be device drift (CPU here vs T4 GPU during
training), not a batch-size artifact (single-batch and `DataLoader(batch_size=32)`
gave identical local results), and 2-3 orders of magnitude too small to move any
reported number. This is exactly the risk PLAN.md §10 pre-registered ("exact bitwise
reproducibility is per-device"). The LSTM parity gate's exact 0.0 match (vs the
transformer's tiny drift) is explained by the same logic in reverse — consistent, not
contradictory.

**LOSO (winner, 6-fold):** 0.939 ± 0.044 vs BiLSTM's 0.928 ± 0.045 (both recomputed
with `ddof=1`, reading Issue 5's raw CSV directly rather than hand-transcribing it —
a transcription error in an early draft of `07_loso_report.py`'s hardcoded LSTM
reference dict was caught by this exact discipline and fixed before running). Excl.
set05 (N=47): 0.927 ± 0.036 vs 0.915 ± 0.033. Directionally consistent with the
fixed-split win; 6 folds is too few for a hypothesis test, reported as a
generalization sanity check only.

**Latency (M4):** the searched transformer is **faster** than the BiLSTM per window
at CPU batch-1 — 0.459 ms vs 0.575 ms (1.25×) — despite ~1.3× the parameters. Fully
parallel self-attention over T=16 tokens apparently outruns the BiLSTM's sequential
recurrence on this hardware. Both are ~2 orders of magnitude inside a 30 fps budget;
latency is not a deployment concern for either model.

Outputs: `phase5_analysis/04_final_{results.csv,summary.md}`,
`05_comparison_{results.csv,results.json,report.md,figure.png}`,
`06_latency_{results.json,report.md}`, `07_loso_{results.csv,report.md}`.

### Phase T6 — Documentation & integration ✅ DONE — 2026-07-12
- Close all DONE blocks here; final PROGRESS_LOG entries; README headline table.
- Slot the transformer rows into `journal_prep/issue3_baseline_comparison/` and the
  manuscript (`paper_skeleton.tex` → Results "Comparison with Published Baselines"
  `tab:baselines` + "Model Capacity" section).
- One-line updates to root docs (`CLAUDE.md` repo-layout note, `pipeline/CODE_STATE.md`)
  recording the new top-level folder.
- Optional (out of scope unless asked): swap the transformer into the live demo
  (`pipeline/10_yolo_bytetrack_demo.py` `load_bilstm()` accepts any model meeting the
  forward contract). **Not done — remains optional/out of scope, not requested.**

**Done:**
- `journal_prep/issue3_baseline_comparison/03_baseline_comparison.md` — added a
  Transformer row to both the "our result" table and the standard-protocol comparison
  table, an "Update" note explaining the searched-vs-default tie nuance, and revised
  the "how to read this" framing to cover both architectures. `04_positioning_vs_prior_
  work.md` — extended the "architecture unjustified" gap-closing row to mention the
  transformer's own 78-config search, and added a new "Model-choice question, answered
  empirically" section. `README.md` — updated the one-line claim and added an "Update"
  pointer at the top.
- `paper_and_artifacts/Journal_writing/paper_skeleton.tex` — built the `tab:baselines`
  LaTeX table in full (it was previously an empty stub) from `03_baseline_comparison.md`,
  with both the BiLSTM and Transformer rows; extended the "Model Capacity" NOTE to
  contrast the BiLSTM's capacity-insensitivity finding (Issue 7) against the
  transformer's search finding that more capacity *did* help; added a new labeled
  subsection `\label{sec:transformer-comparison}` with the full result in NOTE form
  (matching the file's existing not-yet-drafted-into-prose convention throughout —
  this file is a scaffold for the real Overleaf manuscript, not the manuscript itself).
  IntFormer/PIT rows flagged inline as still needing bib keys (pre-existing debt, Issue
  3's own checklist), not fabricated.
- `CLAUDE.md` — "Three top-level folders" → "Four," added the `transformer/` bullet,
  and a paragraph in "What this is" summarizing the extension and its result.
- `pipeline/CODE_STATE.md` — one-line pointer in the existing repo-layout callout
  noting `transformer/` is a separate top-level folder with its own docs, not tracked
  in this file.
- **New: `transformer/SUPERVISOR_SUMMARY.md`** (user-requested, not in the original
  Phase T6 scope) — a supervisor-facing explainer mirroring `paper_and_artifacts/
  supervisor_review/00_README_START_HERE.md`'s proven structure (numbered sections,
  talking script + Q&A), specifically answering that pack's own FAQ entry ("Why
  BiLSTM, not a Transformer?") with the measured result instead of the small-data
  argument alone. Leads with the searched-vs-default tie as the key nuance, not just
  the win. Cross-checked every number against the source reports before writing (the
  headline table deliberately uses the plain 5-seed-mean AUC — the same "0.932" the
  supervisor already knows from the base thesis — rather than the bootstrap's own
  seed-averaged-probability statistic, which is technically a different, correct
  number (0.942 for the BiLSTM) but would have silently introduced a second "official"
  BiLSTM number into the record; the small gap between the two is explained inline
  rather than picked around).

**Not done (explicitly out of scope, per PLAN.md's own note above):** swapping the
transformer into the live YOLO+ByteTrack demo. `pipeline/10_yolo_bytetrack_demo.py`'s
`load_bilstm()` is already designed to accept any model meeting the forward contract,
so this remains a small, optional follow-up if requested later.

**All six phases (T1–T6) of the transformer extension are now complete.**

---

## 8. File map

**One subfolder per phase** — each holds everything (and only what's) needed for that
phase; shared inputs (`sequences_clean/`, the tracking docs) sit at `transformer/` root.
Cross-phase references (e.g. Phase 3 reading Phase 2's `runs_search/`, or the Kaggle
notebooks embedding Phase 1's model/engine files) resolve via `Path(__file__).resolve()
.parent.parent / "<other_phase>" / ...` — the same cross-folder importlib pattern
journal_prep already uses for `pipeline/03_bilstm_model.py`.

```
transformer/
├── PLAN.md                          ← this file (decisions + DONE blocks)
├── README.md                        ← index, status table, how to run
├── PROGRESS_LOG.md                  ← chronological numbers log
├── sequences_clean/                 ← shared data (X.npy, y.npy, meta.pkl) — used by every phase
│
├── phase1_setup/                    ← Phase T1: model + training engine + sanity gates
│   ├── 00_transformer_model.py      ← TransformerIntentPredictor (single source of truth)
│   ├── 02_train_transformer.py      ← train_run(cfg, seed, device, data, eval_test, out_dir) + CLI
│   ├── 01_sanity_checks.py          ← gates 0–3
│   └── 01_sanity_report.md
│
├── phase2_kaggle_search/            ← Phase T2: Kaggle staged search (val-only)
│   ├── gen_search_nb.py             ← regenerates the notebook below from phase1_setup/00_/02_
│   ├── 03_search_kaggle.ipynb       ← Stages A+B+transfer+C (val-only, T4×2, WORKERS=2)
│   ├── runs_search/<cfg_id>/seed<k>.json   (downloaded; val-only JSONs, no `test` key ever)
│   └── kaggle_outputs/              ← raw Kaggle download archive (zip + worker.py), for provenance
│
├── phase3_search_review/            ← Phase T3: local winner review (human checkpoint)
│   ├── 03_search_report.py          ← re-derives ranking from ../phase2_kaggle_search/runs_search/
│   ├── 03_arch_grid.csv
│   ├── 03_recipe_grid.csv
│   ├── 03_candidates_multiseed.csv
│   ├── 03_search_summary.md
│   └── 03_search_figure.png
│
├── phase4_kaggle_final/             ← Phase T4: DONE — ran on Kaggle 2026-07-11
│   ├── gen_final_nb.py              ← regenerates the notebook below from phase1_setup/00_/02_
│   ├── 04_final_loso_kaggle.ipynb   ← Stage D (winner+default ×5 seeds, test once) + determinism
│   │                                   rerun + 6-fold LOSO — winner config hardcoded, no manual paste
│   ├── runs_final/transformer_{searched,default}/seed<k>/   (downloaded; full run dirs)
│   ├── runs_final/_determinism_check/transformer_searched_seed42_rerun/
│   ├── runs_loso/<fold>.json                                (downloaded)
│   ├── _final_summary.json
│   └── kaggle_outputs/              ← raw Kaggle download archive (zip + worker_final.py), for provenance
│
└── phase5_analysis/                 ← Phase T5: DONE — verdict WIN
    ├── 04_final_report.py           ← aggregates ../phase4_kaggle_final/runs_final/
    ├── 05_compare_vs_lstm.py        ← paired bootstrap + t-test vs the frozen LSTM — WIN
    ├── 06_latency_transformer.py    ← Issue-9 latency protocol — transformer 1.25x faster
    └── 07_loso_report.py            ← aggregates ../phase4_kaggle_final/runs_loso/
```

Phase T6 (documentation & integration) has no dedicated subfolder — its output is edits
to the root tracking docs plus `journal_prep/`/manuscript files elsewhere in the repo.

Scripts run from the repo root with `.venv` activated, e.g.
`python transformer/phase1_setup/01_sanity_checks.py`. Digit-prefixed filenames load via
`importlib.util.spec_from_file_location` (repo convention — do not rename).

---

## 9. Key invariants (carry-over from journal_prep)

1. Test set03 is touched **exactly once per experiment**, on the best-val checkpoint —
   the search notebook physically contains no test-eval path.
2. Normalization is train-only and saved per run.
3. pos_weight fixed at 1.682 everywhere except LOSO (per-fold n_neg/n_pos, Issue-5 precedent).
4. seed 42 = canonical single-seed reference; 5-seed mean±std (ddof=1) is the reported number.
5. `torch.load(..., weights_only=False)` for every `best.pt` (numpy scalars inside).
6. Threshold 0.5; never tuned.
7. The LSTM baseline is never retrained; its stored numbers are the comparison row.
8. `num_workers=0` in every DataLoader (repo precedent; shuffle order governed by
   `torch.manual_seed`).
9. Every results table states the device that produced it (T4 / M4-CPU / M4-MPS).

---

## 10. Risks & pitfalls (acknowledged up front)

- **Val N=634 selection noise** — candidate gaps < ~0.01 val AUC are noise; hence Stage-C
  5-seed **mean** selection (Issue-8 precedent: the seed-42 leader lost).
- **Seed σ ≈ 0.011 vs realistic effect size** — a plausible transformer edge is smaller than
  seed noise; the window-paired bootstrap on the same 2094 windows is the primary evidence,
  not the n=5 t-test.
- **CLS/PE off-by-one** — learned & sinusoidal PE must be length T+1 when pool="cls";
  mean-pool must exclude the CLS token.
- **Pre-LN needs the terminal LayerNorm** (`norm=` arg) — without it the residual stream is
  never normalized. Do not "simplify" to `norm_first=False` (diverges at lr 1e-3 sans warmup).
- **Device drift** — search+final on T4 (CUDA, cudnn deterministic flags in `set_seed`),
  analysis local; exact bitwise reproducibility is per-device, which is why the
  determinism-of-record rerun happens on Kaggle in the same notebook.
- **Optimizer honesty** — `transformer_default` = Adam, LSTM's exact recipe; the searched
  recipe may adopt AdamW/warmup — framed as "recipe searched, mirroring Issue 8", with the
  no-decay groups stated.
- **Honesty requirement** — the §6 verdict templates are pre-registered; a negative result
  is reported as plainly as a win.

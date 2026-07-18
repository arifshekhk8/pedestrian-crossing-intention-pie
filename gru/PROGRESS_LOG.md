# GRU Extension — Progress Log (chronological)

Chronological record of every run's numbers, every gate, every decision — in the order it
happened. Authoritative results doc alongside `PLAN.md` (design) and `README.md` (index).

---

## 2026-07-13 — Kickoff & scaffolding

- Created top-level `gru/` (approved plan: recurrent-cell analogue of `transformer/`).
  Subfolders `phase1_setup/ … phase5_analysis/`; tracking docs `PLAN.md`, `README.md`,
  `PROGRESS_LOG.md`.
- Environment confirmed: `.venv` torch 2.12.0, sklearn 1.9.0, scipy 1.17.1 (local M4, CPU).
- Feasibility verified against the codebase before writing any code:
  - Unified engine (`journal_prep/issue12_unified_pipeline/12_unified_engine.py`) builds
    `gru` via `MODEL_REGISTRY["gru"]`; frozen protocol identical to BiLSTM; `--select {f1,auc}`,
    `--pos_weight`, cfg keys `{lr, dropout, hidden, num_layers}`; no test code path.
  - GRU default (h128/nl2) param count = **446,081** (analytic: input_proj 384 + GRU 445,440
    + head 257) — matches the issue12 smoke test.
  - Comparison machinery reused from `f1_optimization/00_common.py` +
    `07_cluster_bootstrap.py`; cached target probs in `f1_optimization/probs_cache/`
    (`lstm_frozen`, `lstm_a3`, `tf_frozen`, `tf_b3` + `y_val`/`y_test`).
  - Frozen BiLSTM checkpoints at `journal_prep/issue2_clean_protocol/runs_clean/multiseed/`;
    parity-gate reference per-seed test AUC 0.9131/0.9334/0.9432/0.9363/0.9358.

## 2026-07-13 — G1 sanity gates: ALL PASS ✅

`gru/phase1_setup/01_sanity_checks.py` → `01_sanity_report.md`. All 5 gates PASS on CPU.

- **G0 protocol:** X (16,5); splits 2178/634/2094; test pos 681; pos_weight 812pos/1366neg =
  1.6823; norm (5,). Exact.
- **G1 param count:** GRU default (h128/nl2/do0.3) = **446,081** exact. Ladder:
  h64 {1L 50,433 / 2L 124,929}, h128 {1L 149,633 / 2L 446,081}, h256 {1L 495,489 / 2L
  1,678,209}. So the search spans ~0.08×–2.8× the frozen BiLSTM's 594,561 params.
- **G2 fwd/bwd:** output (32,1), finite, one BCE+Adam step loss 0.6706, grads finite.
- **G3 determinism (load-bearing):** gru default seed 42 trained twice on CPU → **bit-identical**
  (val F1 0.837920, acc 0.916404, AUC 0.959149, val_at_auc_best AUC 0.966368; best epoch 6,
  ~15s/run; all |Δ|=0.0e+00). CPU is context-free as issue12 promised.
- **G4 engine parity:** engine builds the published BiLSTM cell — bilstm baseline n_params =
  **594,561** (fingerprint match), val AUC 0.951687 twice (|Δ|=0). Note the CPU val AUC
  (0.9517) is lower than issue8's cached-MPS 0.9644 — expected device difference, not a
  regression; the frozen *test* checkpoints are untouched and remain the comparison rows.

**Orientation note (not a result):** gru default already reaches val F1 0.838 / val AUC 0.959
at seed 42 — in the BiLSTM's ballpark, as expected for the recurrent twin. Real comparison
awaits the searched winner on the untouched test set.

## 2026-07-14 — G2 search complete + G3 independent review (val-only, test UNTOUCHED)

`02_gru_search.py` ran all three stages on CPU (89 run files, 0 test keys). `03_search_report.py`
independently re-derived every ranking from the raw JSONs and **cross-checked against
`_stage_summary.json` — ALL MATCH**; 89 files scanned, **zero test keys**.

**Stage 1 (36-config grid, seed 42):** the whole top of the val ranking is `hidden=256` — the
search pushed to the largest width, exactly as the BiLSTM's own F1-first program did (it also
selected h256). Full grid in `phase3_search_review/03_arch_grid.csv`.

**Stage 2 (7 candidates × 5 seeds, val):**

| config | val F1 | val acc | val AUC | note |
|---|---|---|---|---|
| `lr5e-04_do0.3_h256_nl2` | **0.8683 ± 0.0241** | 0.9331 | 0.9747 | **F1-winner** |
| `lr5e-04_do0.2_h256_nl2` | 0.8622 ± 0.0305 | 0.9309 | 0.9755 | |
| `lr1e-03_do0.3_h256_nl2` | 0.8616 ± 0.0158 | 0.9281 | 0.9737 | |
| `lr5e-04_do0.5_h256_nl2` | 0.8613 ± 0.0240 | 0.9297 | 0.9743 | |
| `lr1e-03_do0.2_h256_nl2` | 0.8608 ± 0.0095 | 0.9300 | **0.9760 ± 0.0017** | **AUC-winner** |
| `lr1e-03_doNA_h256_nl1` | 0.8573 ± 0.0192 | 0.9281 | 0.9705 | |
| `lr1e-03_do0.3_h128_nl2` | 0.8558 ± 0.0138 | 0.9268 | 0.9709 | gru_default (h128) |

- **F1-winner: `lr5e-04_do0.3_h256_nl2`** (h256/nl2/do0.3/lr5e-4, **1,678,209 params**), mean
  val F1 0.8683.
- **AUC-winner (different): `lr1e-03_do0.2_h256_nl2`** (h256/nl2/do0.2/lr1e-3, 1,678,209
  params), mean val AUC 0.9760.
- **Selection-noise control mattered again** (Issue-8 / transformer precedent): the seed-42
  val-F1 leader was `lr5e-04_do0.5_h256_nl2`, but on the 5-seed *mean* the winner is
  `lr5e-04_do0.3_h256_nl2` — a single-seed grid would have picked a different config.
- `gru_default` (h128) is the lowest-F1 candidate (0.8558), consistent with the search finding
  that width helps on val (same as the BiLSTM-F1 move to h256).

**Stage 3 (pos_weight sweep on F1-winner, 5-seed val F1):** pw 1.0 → 0.8547, 1.3 → 0.8612,
**1.682 → 0.8683**, 2.1 → 0.8667, 2.5 → 0.8621. Anchor **1.682 retained** (nothing beats it).

**⏸ Awaiting user confirmation of the winners + pos_weight before G4 (which touches test once).**

## 2026-07-14 — G3 sign-off: user chose "F1-winner + default only"

User confirmed the human checkpoint with the **leaner arm set** (dropped the separate
AUC-winner arm). G4 arms (each × 5 seeds [42,0,1,2,3], pos_weight 1.682, CPU):
- **`gru_f1_winner`** = `lr5e-04_do0.3_h256_nl2` (h256/nl2/do0.3/lr5e-4, 1,678,209 params),
  `--select f1` — the headline GRU (primary F1 comparison).
- **`gru_default_f1`** = `lr1e-03_do0.3_h128_nl2` (h128, 446,081), `--select f1` — the
  un-searched-GRU control (analogue of `transformer_default`).
- **`gru_default_auc`** = same h128 cfg, `--select auc` — the AUC-selected h128 twin of the
  frozen BiLSTM baseline (the matched-capacity + matched-selection "isolate the cell" AUC row).

Consequence (stated plainly): no dedicated AUC-optimized h256 arm, so the GRU's headline AUC
is reported from the F1-selected winner + the h128 AUC control, not a separately AUC-tuned
h256 model. The AUC-vs-searched-transformer comparison is secondary anyway (F1-first).

## 2026-07-14 — G4 final training + THE single test-touch (set03 read exactly once)

**04_gru_final.py** — 15 checkpoints (3 arms × 5 seeds), val-only, CPU. Reproduces the search
exactly (gru_f1_winner 5-seed mean val F1 0.8683; gru_default 0.8558 — bit-identical to G2).

**05_gru_test_eval.py** — **PARITY GATE PASS, |Δ|=0.00e+00 for all 5 seeds** (frozen BiLSTM
test AUC recomputed from checkpoints == stored final.json, seed42 0.913114 … seed3 0.935822).
Test set03 (2094 windows, 681 pos) read once; τ\* fit on val only; probs cached.

| arm | stat | F1@0.5 | F1@τ\* | Acc@τ\* | AUC |
|---|---|---|---|---|---|
| `gru_f1_winner` (h256) | per-seed | 0.8499 ± 0.0077 | **0.8488 ± 0.0111** | 0.9013 | **0.9408 ± 0.0066** |
| `gru_f1_winner` (h256) | ensemble | 0.8565 | **0.8628** | 0.9107 | 0.9489 |
| `gru_default_f1` (h128) | per-seed | 0.8443 ± 0.0185 | 0.8443 ± 0.0198 | 0.8983 | 0.9386 ± 0.0066 |
| `gru_default_f1` (h128) | ensemble | 0.8520 | 0.8520 | 0.9021 | 0.9460 |
| `gru_default_auc` (h128) | per-seed | 0.8352 ± 0.0119 | 0.8403 ± 0.0104 | 0.8980 | **0.9327 ± 0.0099** |
| `gru_default_auc` (h128) | ensemble | 0.8474 | 0.8466 | 0.8988 | 0.9415 |

**Orientation (formal deltas are Phase G5):**
- **F1 (primary):** headline GRU per-seed F1 **0.849** sits right in the tie band with BiLSTM-F1
  (0.844) and Transformer-F1 (0.847) — the "cell doesn't matter on F1" story looks intact.
- **AUC (secondary, the clean cell-isolation row):** `gru_default_auc` (h128, AUC-selected)
  AUC **0.9327 ± 0.0099** ≈ frozen BiLSTM **0.9324 ± 0.0114** — matched capacity + matched
  selection, essentially identical → the recurrent cell (GRU vs LSTM) is not what moves AUC.
- The GRU does **not** reach the searched transformer's AUC (0.9408/0.9489 vs 0.9497/0.9558) —
  consistent with "the transformer's AUC win came from its search, not attention-vs-recurrence."

## 2026-07-14 — G5 analysis: paired + cluster bootstrap, latency, LOSO — VERDICTS

**07_compare.py** — 10k paired percentile bootstrap (ens vectors, fixed val-τ\*), same
resample indices both sides. Endpoints (Δ = GRU − comparison):

| # | comparison | metric | Δ | 95% window CI | verdict |
|---|---|---|---|---|---|
| 1 | GRU-F1 vs frozen BiLSTM (0.828) | F1 | +0.0258 | [+0.0162, +0.0358] | **WIN** |
| 2 | **GRU-F1 vs BiLSTM-F1 (0.844)** | F1 | +0.0071 | [−0.0043, +0.0187] | **TIE** |
| 3 | GRU-F1 vs Transformer-F1 (0.847) | F1 | +0.0063 | [−0.0046, +0.0174] | **TIE** |
| 4 | GRU-default vs frozen BiLSTM | F1 | +0.0150 | [+0.0065, +0.0238] | WIN |
| 5 | **GRU-default-AUC vs frozen BiLSTM** (matched h128+sel) | AUC | −0.0008 | [−0.0039, +0.0021] | **TIE** |
| 6 | GRU-F1 vs searched Transformer | AUC | −0.0070 | [−0.0101, −0.0038] | **LOSS** |

(1)/(4) WIN = the F1-first *discipline* (F1-checkpoint + τ\*) lifting the GRU above the old
AUC-checkpointed 0.828, not a cell effect. The scientific verdicts are (2)(3)(5)(6).

**08_cluster_bootstrap.py** — pedestrian-cluster CI (541 clusters, all-windows-per-drawn-ped).
**All 6 verdicts survive clustering.** Cell-isolation TIEs hold: (2) cluster CI [−0.0089,
+0.0223], (5) cluster CI [−0.0067, +0.0045]; LOSS (6) still excludes 0 ([−0.0129, −0.0018]).
GRU-F1 absolute cluster CIs: ens F1 [0.826, 0.896], AUC [0.927, 0.968].

**09_latency.py** — GRU F1-winner (h256, 1.68M) CPU batch-1 = **0.721 ms/window** (~46× inside
30 fps) vs BiLSTM 0.575 / Transformer 0.459. Bigger model → slightly higher, still a non-issue;
pipeline stays detection-bound. Same CPU-beats-MPS-at-batch-1 pattern.

**10_loso_report.py** — GRU 6-fold LOSO AUC **0.946 ± 0.036** (excl set05: 0.935), F1 0.857 —
same band as BiLSTM 0.928 / Transformer 0.939. set03 fold AUC 0.928 ≈ fixed-split (not an easy
fold). All fold sizes matched the Issue-5 fingerprint.

### VERDICT (F1-first)
Under the identical clean protocol and F1 → acc → AUC hierarchy, **a GRU is statistically
indistinguishable from the BiLSTM** — it TIES BiLSTM-F1 and Transformer-F1 on F1, and TIES the
frozen BiLSTM on AUC at matched capacity/selection. The searched transformer keeps its AUC edge
(GRU loses on AUC), confirming that edge is the *search*, not attention-vs-recurrence.
**Strengthens the thesis story: the input signal (bbox + ego-speed), not the recurrent cell,
is what matters.** All findings robust to the pedestrian-cluster bootstrap.






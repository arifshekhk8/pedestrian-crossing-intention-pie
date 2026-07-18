# Vanilla-RNN Extension — Progress Log (chronological)

Chronological record of every run's numbers, every gate, every decision — in the order it
happened. Authoritative results doc alongside `PLAN.md` (design) and `README.md` (index).

---

## 2026-07-14 — Kickoff & scaffolding

- Created top-level `rnn/` (approved plan: un-gated recurrent-cell analogue of `gru/` /
  `transformer/`). Subfolders `phase1_setup/ … phase5_analysis/`; tracking docs `PLAN.md`,
  `README.md`, `PROGRESS_LOG.md`.
- Environment confirmed: `.venv` torch 2.12.0, sklearn 1.9.0, scipy 1.17.1 (local M4, CPU).
- Feasibility verified against the codebase before writing any code:
  - Unified engine (`journal_prep/issue12_unified_pipeline/12_unified_engine.py`) builds
    `birnn` via `MODEL_REGISTRY["birnn"]` = `RecurrentIntentPredictor("rnn")` (bidirectional
    tanh RNN); frozen protocol identical to BiLSTM; `--select {f1,auc}`, `--pos_weight`, cfg
    keys `{lr, dropout, hidden, num_layers}`; no test code path. Preset = GRU_DEFAULT_CFG
    (lr1e-3/do0.3/h128/nl2), the BiLSTM-baseline recipe twin.
  - birnn default (h128/nl2) param count = **149,121** (analytic: input_proj 384 + RNN 148,480
    + head 257) — matches the issue12 smoke test. The un-gated cell has ~1/4 the recurrent
    weights of the 4-gate LSTM, so the whole family is smaller than the BiLSTM's 594,561.
  - Comparison machinery reused from `f1_optimization/00_common.py` +
    `07_cluster_bootstrap.py`; cached target probs in `f1_optimization/probs_cache/`
    (`lstm_frozen`, `lstm_a3`, `tf_frozen`, `tf_b3` + `y_val`/`y_test`) and
    `gru/phase4_final/probs_cache/` (GRU arms, for the un-gated-vs-gated recurrent endpoint).
  - Frozen BiLSTM checkpoints at `journal_prep/issue2_clean_protocol/runs_clean/multiseed/`;
    parity-gate reference per-seed test AUC 0.9131/0.9334/0.9432/0.9363/0.9358.

## 2026-07-14 — R1 sanity gates: ALL PASS ✅

`rnn/phase1_setup/01_sanity_checks.py` → `01_sanity_report.md`. All 5 gates PASS on CPU.

- **G0 protocol:** X (16,5); splits 2178/634/2094; test pos 681; pos_weight 812pos/1366neg =
  1.6823; norm (5,). Exact.
- **G1 param count:** birnn default (h128/nl2/do0.3) = **149,121** exact. Ladder:
  h64 {1L 17,153 / 2L 41,985}, h128 {1L 50,305 / 2L 149,121}, h256 {1L 165,761 / 2L
  560,001}. So the search spans **0.03×–0.94×** the frozen BiLSTM's 594,561 params — the whole
  vanilla-RNN family sits below the BiLSTM (un-gated cell = ~1/4 the recurrent weights).
- **G2 fwd/bwd:** output (32,1), finite, one BCE+Adam step loss 0.8881, grads finite.
- **G3 determinism (load-bearing):** birnn default seed 42 trained twice on CPU →
  **bit-identical** (val F1 0.825083, acc 0.916404, AUC 0.966233, val_at_auc_best AUC 0.966233;
  best epoch 13, ~8s/run; all |Δ|=0.0e+00). CPU is context-free as issue12 promised.
- **G4 engine parity:** engine builds the published BiLSTM cell — bilstm baseline n_params =
  **594,561** (fingerprint match), val AUC 0.951687 twice (|Δ|=0). (CPU val AUC 0.9517 vs
  issue8's cached-MPS 0.9644 is the expected device difference, not a regression; the frozen
  *test* checkpoints are untouched and remain the comparison rows.)

**Orientation note (not a result):** birnn default already reaches val F1 0.825 / val AUC 0.966
at seed 42 — in the BiLSTM's ballpark, and with **no sign of divergence** at the default recipe
(tanh RNN trains cleanly over the 16-step window). Real comparison awaits the searched winner
on the untouched test set.

## 2026-07-14 — R2 search complete + R3 independent review (val-only, test UNTOUCHED)

`02_rnn_search.py` ran all three stages on CPU (93 run files, 0 test keys, **0 diverged runs**).
`03_search_report.py` independently re-derived every ranking from the raw JSONs and
**cross-checked against `_stage_summary.json` — ALL MATCH** (incl. the instability-ledger count);
93 files scanned, **zero test keys**.

**Stage 1 (36-config grid, seed 42):** width helps on val — the top of the ranking clusters on
`hidden=256` (and h256/nl1), exactly as the BiLSTM-F1 program and the GRU search both did. Full
grid in `phase3_search_review/03_arch_grid.csv`.

**Stage 2 (8 candidates × 5 seeds, val):**

| config | val F1 | val acc | val AUC | note |
|---|---|---|---|---|
| `lr1e-04_do0.2_h256_nl2` | **0.8554 ± 0.0141** | 0.9265 | **0.9721 ± 0.0051** | **F1-winner = AUC-winner** |
| `lr1e-03_doNA_h256_nl1` | 0.8509 ± 0.0130 | 0.9249 | 0.9714 | |
| `lr1e-03_do0.5_h256_nl2` | 0.8498 ± 0.0221 | 0.9246 | 0.9677 | |
| `lr5e-04_do0.2_h128_nl2` | 0.8491 ± 0.0194 | 0.9233 | 0.9670 | |
| `lr5e-04_doNA_h256_nl1` | 0.8473 ± 0.0065 | 0.9233 | 0.9689 | |
| `lr1e-03_do0.3_h128_nl2` | 0.8429 ± 0.0141 | 0.9202 | 0.9671 | rnn_default (h128) |
| `lr1e-03_do0.5_h128_nl2` | 0.8428 ± 0.0070 | 0.9196 | 0.9686 | |
| `lr1e-03_do0.3_h64_nl2` | 0.8417 ± 0.0120 | 0.9186 | 0.9670 | |

- **F1-winner = AUC-winner (they agree): `lr1e-04_do0.2_h256_nl2`** (h256/nl2/do0.2/lr1e-4,
  **560,001 params**), mean val F1 0.8554, mean val AUC 0.9721. Notably this is the **exact same
  config Issue-8's grid selected as the BiLSTM AUC-winner** (`lr1e-04_do0.2_h256_nl2`) — the
  un-gated RNN search independently converged on the BiLSTM's own AUC-optimal recipe.
- **Selection-noise control mattered again** (Issue-8 / GRU / transformer precedent): the seed-42
  val-F1 leader was `lr5e-04_doNA_h256_nl1`, but on the 5-seed *mean* the winner is
  `lr1e-04_do0.2_h256_nl2` — a single-seed grid would have picked a different config.
- `rnn_default` (h128) is mid-pack (0.8429), consistent with the search finding that width helps
  on val (same as BiLSTM-F1 / GRU moving to h256).

**Stage 3 (pos_weight sweep on F1-winner, 5-seed val F1):** pw 1.0 → 0.8511, 1.3 → 0.8498,
**1.682 → 0.8554**, 2.1 → 0.8510, 2.5 → 0.8551. Anchor **1.682 retained** (nothing beats it).

**Instability ledger: 0 diverged runs.** Every config across the grid, multiseed, and pw sweep
trained cleanly (all val AUC ≥ 0.70) — the vanilla tanh RNN is stable over the 16-step window at
every searched setting (short sequence ⇒ vanishing gradients are mild, as expected).

## 2026-07-14 — R3 sign-off: user chose the 4-arm set ("add the AUC-selected winner")

User confirmed the human checkpoint. Because the search's F1-winner and AUC-winner are the **same
config**, a dedicated AUC-selected large-RNN arm is essentially free, so the user added it (closes
the "no AUC-tuned large model" gap the GRU study had to flag). R4 arms (each × 5 seeds
[42,0,1,2,3], CPU):
- **`rnn_f1_winner`** = `lr1e-04_do0.2_h256_nl2` (h256/nl2/do0.2/lr1e-4, **560,001 params**),
  `--select f1` — the headline RNN (primary F1 comparison).
- **`rnn_winner_auc`** = same cfg, `--select auc` — the dedicated AUC-optimized h256.
- **`rnn_default_f1`** = `lr1e-03_do0.3_h128_nl2` (h128, 149,121), `--select f1` — the
  un-searched-RNN control (analogue of `transformer_default`/`gru_default_f1`).
- **`rnn_default_auc`** = same h128 cfg, `--select auc` — the AUC-selected h128 twin of the
  frozen BiLSTM baseline (matched-capacity + matched-selection "isolate the cell" AUC row).

## 2026-07-14 — R4 final training + THE single test-touch (set03 read exactly once)

**04_rnn_final.py** — 20 checkpoints (4 arms × 5 seeds), val-only, CPU. Reproduces the search
exactly (`rnn_f1_winner` 5-seed mean val F1 0.8554 — bit-identical to R2).

**05_rnn_test_eval.py** — **PARITY GATE PASS, |Δ|=0.00e+00 for all 5 seeds** (frozen BiLSTM test
AUC recomputed from checkpoints == stored final.json, seed42 0.913114 … seed3 0.935822). Test
set03 (2094 windows, 681 pos) read once; τ\* fit on val only; probs cached.

| arm | stat | F1@0.5 | F1@τ\* | Acc@τ\* | AUC |
|---|---|---|---|---|---|
| `rnn_f1_winner` (h256, F1) | per-seed | 0.8543 ± 0.0119 | **0.8518 ± 0.0120** | 0.9018 | **0.9480 ± 0.0015** |
| `rnn_f1_winner` (h256, F1) | ensemble | 0.8602 | **0.8590** | 0.9078 | 0.9546 |
| `rnn_winner_auc` (h256, AUC) | per-seed | 0.8490 ± 0.0152 | 0.8450 ± 0.0222 | 0.8937 | **0.9481 ± 0.0058** |
| `rnn_winner_auc` (h256, AUC) | ensemble | 0.8640 | 0.8634 | 0.9102 | 0.9545 |
| `rnn_default_f1` (h128, F1) | per-seed | 0.8430 ± 0.0133 | 0.8441 ± 0.0125 | 0.8968 | 0.9415 ± 0.0072 |
| `rnn_default_f1` (h128, F1) | ensemble | 0.8543 | 0.8510 | 0.9045 | 0.9470 |
| `rnn_default_auc` (h128, AUC) | per-seed | 0.8413 ± 0.0176 | 0.8360 ± 0.0208 | 0.8893 | **0.9421 ± 0.0085** |
| `rnn_default_auc` (h128, AUC) | ensemble | 0.8492 | 0.8519 | 0.9045 | 0.9483 |

**Orientation (formal deltas are Phase R5):** the un-gated RNN's headline per-seed F1 **0.852**
sits at the *top* of the tie band with BiLSTM-F1 (0.844), Transformer-F1 (0.847) and GRU-F1
(0.849), and its AUC (0.948 / ens 0.955) reaches the searched transformer's (0.9497 / 0.9558).

**06_rnn_loso.py** — 6-fold LOSO AUC **0.937 ± 0.040** (excl set05 0.926), F1 0.839; set03 fold
0.944 ≈ fixed-split (not an easy fold). All fold sizes matched the Issue-5 fingerprint
(258/310/2094/1610/47/587). Same band as BiLSTM 0.928 / GRU 0.946 / Transformer 0.939.

## 2026-07-14 — R5 analysis: paired + cluster bootstrap, latency, LOSO — VERDICTS

**07_compare.py** — 10k paired percentile bootstrap (ens vectors, fixed val-τ\*), same resample
indices both sides. Endpoints (Δ = RNN − comparison):

| # | comparison | metric | Δ | 95% window CI | verdict |
|---|---|---|---|---|---|
| 1 | RNN-F1 vs frozen BiLSTM (0.828) | F1 | +0.0220 | [+0.0111, +0.0327] | **WIN** |
| 2 | **RNN-F1 vs BiLSTM-F1 (0.844)** | F1 | +0.0033 | [−0.0083, +0.0150] | **TIE** |
| 3 | RNN-F1 vs Transformer-F1 (0.847) | F1 | +0.0025 | [−0.0079, +0.0131] | **TIE** |
| 4 | **RNN-F1 vs GRU-F1 (0.849)** | F1 | −0.0038 | [−0.0117, +0.0039] | **TIE** |
| 5 | RNN-default-F1 vs frozen BiLSTM | F1 | +0.0140 | [+0.0023, +0.0255] | WIN (→ cluster TIE) |
| 6 | **RNN-default-AUC vs frozen BiLSTM** (matched h128+sel) | AUC | +0.0059 | [+0.0032, +0.0088] | **WIN** |
| 7 | RNN-winner-AUC vs frozen BiLSTM | AUC | +0.0121 | [+0.0087, +0.0157] | WIN |
| 8 | **RNN-winner-AUC vs searched Transformer** | AUC | −0.0013 | [−0.0041, +0.0015] | **TIE** |

(1)/(5)/(7) WIN = the F1-first *discipline* + a searched h256 lifting the RNN above the old
AUC-checkpointed baseline, not a cell effect. The scientific cell-isolation verdicts are (2)(3)(4)(6)(8).

**08_cluster_bootstrap.py** — pedestrian-cluster CI (541 clusters, all-windows-per-drawn-ped).
**All cell-isolation verdicts survive clustering.** (2) cluster [−0.0130, +0.0187], (4) cluster
[−0.0128, +0.0049], (6) cluster [+0.0012, +0.0110], (8) cluster [−0.0061, +0.0033]. Only the
un-searched-control WIN (5) softens to TIE under clustering (a discipline artifact, reported
honestly). RNN-F1 absolute cluster CIs: ens F1 [0.822, 0.893], AUC [0.934, 0.972].

**09_latency.py** — RNN F1-winner (h256, 560,001) CPU batch-1 = **0.316 ms/window** (~105× inside
30 fps) — the **fastest** of all four families (vs BiLSTM 0.575 / Transformer 0.459 / GRU 0.721).
The un-gated cell is the smallest and cheapest. Pipeline stays detection-bound.

**10_loso_report.py** — RNN 6-fold LOSO AUC **0.937 ± 0.040** (excl set05 0.926), F1 0.839 — same
band as the other three families; set03 fold 0.944 (not an easy fold).

### VERDICT (F1-first)
Under the identical clean protocol and F1 → acc → AUC hierarchy, **the un-gated vanilla RNN is
statistically indistinguishable from the gated recurrent models** — it TIES BiLSTM-F1 and GRU-F1
on F1, and ties-or-edges the frozen BiLSTM on AUC at matched capacity/selection (no cell-isolation
endpoint is a loss). And — **unlike the GRU, which lost to the searched transformer on AUC — the
AUC-optimized vanilla RNN TIES it** (ΔAUC −0.0013, CI includes 0): once an un-gated recurrent net
gets the same search, it reaches the same AUC, confirming the transformer's edge was its *search*,
not attention-over-recurrence. **Removing the LSTM's gating costs nothing measurable over this
16-step window** — the strongest form of the thesis story: **the input signal (bbox + ego-speed),
not the recurrent cell or its gating, is what matters.** It is also the smallest and fastest of
the four families. All findings robust to the pedestrian-cluster bootstrap.

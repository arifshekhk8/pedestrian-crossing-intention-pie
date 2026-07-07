# Issue 2 — Canonical Leak-Free PIE Protocol

Rebuilds the dataset so the observation window ends **strictly before** crossing
onset, then retrains. Fixes the leakage proven in
[`../issue1_leakage_audit/01_leakage_report.md`](../issue1_leakage_audit/01_leakage_report.md).
Full write-up + DONE block: [`../PLAN.md`](../PLAN.md) (Issue 2).

## Files, in run order

| # | File | What it is |
|---|---|---|
| 02 | `02_build_sequences_clean.py` | builds `sequences_clean/` — anchors each window at PIE `crossing_point`, TTE∈[30,60], 50% overlap. **Run locally.** |
| 02 | `02_leakage_report_clean.md` | re-audit of the new data → **0/4,906 windows leak (✅ CLEAN)** |
| 03 | `03_eval_parity_check.py` | per-window vs per-pedestrian + min-track-size parity check |
| 03 | `03_eval_parity_report.md` | verdict: 0.913 is **not** an easier-evaluation artifact |
| 04 | `04_multiseed_baseline.py` | 5-seed multi-seed of the 5-D baseline (shells out to root `04_train_bilstm.py`) |
| 04 | `04_multiseed_summary.md` / `.csv` | **baseline AUC 0.932 ± 0.011** |
| 05 | `05_variant_comparison.md` | all 3 variants, clean vs leaky, multi-seed — **ego-speed dominant +0.18; bbox-only collapses; attention no benefit** |
| 06 | `06_multiseed_variants_kaggle.ipynb` | **Kaggle** notebook: multi-seed bbox-only + attention. Hardened to refuse leaky data (asserts N=4,906) |
| 06b | `06b_local_verify_seed42.py` | local CPU/MPS cross-check (5 seeds) — confirmed Kaggle clean numbers + exposed the leaky first run |
| — | `kaggle_result/` | downloaded notebook-06 outputs (clean re-run; ⚠ `summary.csv` labels swapped + header says pos_weight 1.44 — use `summary.md`/`results.csv`) |

## Outputs (not scripts)

- `sequences_clean/` — `X.npy (4906,16,5)`, `y.npy`, `meta.pkl`. **These three are
  what you upload to Kaggle** for notebook 06.
- `runs_clean/` — trained checkpoints + `final.json` per model:
  `bilstm_baseline_clean/`, `bbox_only_clean/`, `attention_clean/`, and
  `multiseed/seed{42,0,1,2,3}/`.
- `figures/`, `leakage_per_sequence.csv` — from the re-audit (step 02).

## Reproduce locally (M4, sklearn in `.venv`)

```bash
source .venv/bin/activate
# 1. build clean sequences
python journal_prep/issue2_clean_protocol/02_build_sequences_clean.py
# 2. prove 0% leakage
python journal_prep/issue1_leakage_audit/01_leakage_audit.py \
  --seq-dir journal_prep/issue2_clean_protocol/sequences_clean \
  --out-dir journal_prep/issue2_clean_protocol \
  --report-name 02_leakage_report_clean.md
# 3. eval-parity check     # 4. multi-seed baseline
python journal_prep/issue2_clean_protocol/03_eval_parity_check.py
python journal_prep/issue2_clean_protocol/04_multiseed_baseline.py --seeds 42 0 1 2 3 --skip-existing
```

Variant retrains (single-seed, what 05 reports) use the root scripts with the
clean `--pos_weight`:

```bash
python 04b_train_bbox_only.py  --seq_dir journal_prep/issue2_clean_protocol/sequences_clean \
  --out_dir journal_prep/issue2_clean_protocol/runs_clean/bbox_only_clean --pos_weight 1.682
python 07_train_attention.py   --seq_dir journal_prep/issue2_clean_protocol/sequences_clean \
  --out_dir journal_prep/issue2_clean_protocol/runs_clean/attention_clean --pos_weight 1.682
```

## Key numbers

| Model | Inputs | clean AUC (5-seed) | note |
|---|---|---|---|
| BiLSTM baseline (5-D) | bbox + ego-speed | **0.932 ± 0.011** | seed 42 = 0.913 (low end) |
| BiLSTM bbox-only (4-D) | bbox | **0.753 ± 0.020** | collapses from leaky 0.889 → ego-speed worth +0.18 |
| BiLSTM + attention (5-D) | bbox + ego-speed | **0.925 ± 0.010** | ≈ baseline; no measurable benefit on clean data |

`pos_weight` for clean data = **1.682** (train split 1366 neg / 812 pos), not the
leaky-era 1.44. The training scripts gained a `--pos_weight` flag (default 1.44,
so old commands are unchanged).

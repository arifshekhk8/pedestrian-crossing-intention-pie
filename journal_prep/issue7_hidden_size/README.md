# Issue 7 — Hidden-size ablation ✅

`THESIS_PLAN.md` Day 11 promised a hidden-size sweep `{64,128,256}` but it was
never run, so the central capacity choice (**hidden=128**) was *asserted*, not
justified. A reviewer asks: *"did you pick 128 because it was best, or because you
decided it first?"* This sweep answers that.

## How to run

```bash
source .venv/bin/activate
python journal_prep/issue7_hidden_size/07_hidden_size_ablation.py   # MPS, ~13 s/training
# per-run JSONs are cached, so re-running only regenerates the summary/figure
```

15 trainings (3 sizes × 5 seeds) on the M4 GPU (MPS). **Multi-seeded on purpose**
— the plan said seed 42 only, but Issue 6 showed single-seed ablation spreads sit
*below* seed noise and are undefendable. Locked baseline contract (5-D
`BiLSTMIntentPredictor` on `issue2_clean_protocol/sequences_clean/` — the 0.932
headline data; train=set01/02/04, val=set05/06, test=set03; train-only norm;
pos_weight=1.682; lr=1e-3; dropout 0.3; 2 layers; proj 64; patience 15). `hidden_dim`
is the only variable; hidden=128 reproduces the baseline.

## Result

| hidden | params | AUC | PR-AUC | F1 | Acc |
|---|---|---|---|---|---|
| 64 | 166 k | 0.927 ± 0.009 | 0.865 ± 0.021 | 0.809 ± 0.055 | 0.883 ± 0.022 |
| **128 (baseline)** | **595 k** | **0.933 ± 0.007** | 0.870 ± 0.012 | 0.828 ± 0.014 | 0.883 ± 0.012 |
| 256 | 2.24 M | 0.938 ± 0.003 | 0.878 ± 0.004 | 0.835 ± 0.003 | 0.889 ± 0.005 |

- **hidden=128 is justified — no size significantly beats it.** hidden=256 is
  nominally +0.0045 AUC but **not significant** (paired-t p=0.34) at **3.8× the
  parameters** (2.24 M vs 595 k); hidden=64 is no better (p=0.35) at lower capacity;
  Kruskal–Wallis p=0.12.
- **Honest nuance:** there is a *mild, non-significant* upward trend with capacity
  (0.927 → 0.933 → 0.938). The spread (0.010) slightly exceeds seed noise (±0.006),
  so we do **not** claim capacity is fully saturated — only that nothing beats 128
  significantly. We keep 128 as the **accuracy/cost compromise**: the smaller,
  faster model isn't significantly beaten by the 3.8×-larger one. That is the
  standard justification for a chosen capacity.
- **Cross-check:** the hidden=128 cell here (0.933 ± 0.007 on MPS) reproduces the
  existing baseline (0.932 ± 0.011).

These 15 runs are also the `hidden ∈ {64,128,256}` rows that **Issue 8** (grid
search) reuses, so it does not retrain them.

## Companion: network-depth ablation (`07b_`) — num_layers ∈ {1, 2, 3}

Width (hidden size) is only half the capacity question; `07b_num_layers_ablation.py`
varies **depth** at the baseline width (5 seeds, everything else locked; the grid
search tested 1 vs 2 but never 3). Result: **depth doesn't matter either.**

| num_layers | params | test AUC |
|---|---|---|
| 1 | 199 k | 0.930 ± 0.012 |
| **2 (baseline)** | **595 k** | **0.932 ± 0.006** |
| 3 | 990 k | 0.931 ± 0.010 |

Spread 0.0013 — *far below* seed noise (±0.0076); nl1 p=0.77, nl3 p=0.91, Kruskal
p=0.93 (none significant). **num_layers=2 is justified:** a 3rd layer adds 1.7× the
params for no measurable gain, and 1 layer is no worse. Same story as the width
sweep — the model is **small-data-limited, not capacity-limited** (N=2178 train), so
neither wider nor deeper helps. (Caveat: dropout is inter-layer LSTM dropout, inert
at 1 layer, so the 1-layer model runs without dropout — intrinsic to the model.)

## Files

```
07_hidden_size_ablation.py     harness (15 trainings + analysis)
07_hidden_size_results.csv     per (hidden, seed) test metrics
07_hidden_size_results.md      mean±std + params + significance + verdict
07_hidden_size_figure.png      AUC vs hidden size, params annotated, seed scatter
07b_num_layers_ablation.py     depth companion (num_layers 1/2/3, 15 trainings)
07b_num_layers_results.md / .csv / 07b_num_layers_figure.png
runs/h<H>/seed<k>.json + runs_layers/nl<L>/seed<k>.json   per-run metrics
```

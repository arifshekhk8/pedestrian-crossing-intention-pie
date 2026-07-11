# Issue 7 — Hidden-size ablation (multi-seed)

5-D baseline BiLSTM on the clean baseline data (`issue2_clean_protocol/sequences_clean/`, obs16 / TTE band [30,60] — the 0.932 headline data), hidden_dim ∈ [64, 128, 256], 5 seeds [42, 0, 1, 2, 3], MPS. Everything else locked (train=set01/02/04, val=set05/06, test=set03; train-only norm; pos_weight=1.682; lr=0.001; dropout 0.3; 2 layers; proj 64; patience 15). Test (set03) touched once per (config, seed); hidden_dim is the only variable.

| hidden | params | best ep | AUC | PR-AUC | F1 | Acc |
|---|---|---|---|---|---|---|
| 64 | 166,401 | 4 | 0.927 ± 0.009 | 0.865 ± 0.021 | 0.809 ± 0.055 | 0.883 ± 0.022 |
| 128 (baseline) | 594,561 | 8 | 0.933 ± 0.007 | 0.870 ± 0.012 | 0.828 ± 0.014 | 0.883 ± 0.012 |
| 256 | 2,237,313 | 10 | 0.938 ± 0.003 | 0.878 ± 0.004 | 0.835 ± 0.003 | 0.889 ± 0.005 |

**Between-size mean-AUC spread = 0.0104**, vs average within-size seed std = ±0.0063.

Pairwise vs hidden=128 (paired t-test, matched seeds; Mann-Whitney U):

| pair | ΔAUC | paired-t p | Mann-Whitney p |
|---|---|---|---|
| h64 vs h128 | -0.0059 | 0.347 | 0.310 |
| h256 vs h128 | +0.0045 | 0.338 | 0.222 |

Kruskal–Wallis across the three sizes: p = 0.121.

## Verdict

**hidden=128 is justified.** No size differs significantly from it: hidden=256 is nominally +0.0045 AUC but **not significant** (paired-t p=0.338) at **3.8× the parameters** (2,237,313 vs 594,561), and hidden=64 is no better (p=0.347) at lower capacity; Kruskal–Wallis p=0.121. There is a *mild, non-significant* upward trend with capacity (0.927 → 0.933 → 0.938): the spread (0.0104) slightly exceeds seed noise (±0.0063) but no pairwise test is significant, so we do **not** claim capacity is fully saturated — only that nothing beats 128 significantly. We keep **hidden=128 as the accuracy/cost compromise** — the smaller, faster model is not significantly beaten by the 3.8×-larger one, which is the standard justification for the chosen capacity. The hidden=128 cell reproduces the baseline (this run 0.933 ± 0.007 vs the existing 0.932 ± 0.011).

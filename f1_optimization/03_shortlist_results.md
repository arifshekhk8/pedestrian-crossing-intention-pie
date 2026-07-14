# 03 — LSTM shortlist at 5-seed granularity (AUC protocol, val-only, fresh runs)

**Amendment (see docstring + PROGRESS_LOG):** the current torch/MPS environment no longer reproduces Issue-8's cached grid values (env drift AUC 6.02e-03 / F1 2.25e-02 on the baseline cell), while being bit-deterministic run-to-run (determinism gate PASS, two same-seed runs identical). The cached grid therefore only *nominated* the shortlist (02); every measurement below is a fresh run under the current environment.

| rank | cfg | val F1 (5-seed) | val acc | val AUC |
|---|---|---|---|---|
| 1 | `lr1e-03_do0.5_h128_nl2` | 0.8368 ± 0.0170 | 0.9151 | 0.9660 |
| 2 | `lr1e-03_do0.3_h256_nl2` | 0.8269 ± 0.0290 | 0.9050 | 0.9685 |
| 3 | `lr1e-04_do0.2_h256_nl2` | 0.8239 ± 0.0234 | 0.9183 | 0.9699 |
| 4 | `lr1e-03_do0.5_h64_nl2` | 0.8230 ± 0.0218 | 0.9117 | 0.9621 |
| 5 | `lr1e-04_do0.3_h128_nl2` | 0.8106 ± 0.0633 | 0.9044 | 0.9666 |
| 6 | `lr1e-03_do0.3_h128_nl2` **baseline** | 0.8077 ± 0.0320 | 0.8946 | 0.9631 |
| 7 | `lr1e-03_do0.2_h128_nl2` | 0.8053 ± 0.0249 | 0.9028 | 0.9620 |
| 8 | `lr1e-04_do0.2_h128_nl2` | 0.7735 ± 0.1372 | 0.8975 | 0.9678 |

**Top-2 advancing to the F1-protocol confirm (04, part 4a): `lr1e-03_do0.5_h128_nl2`, `lr1e-03_do0.3_h256_nl2`.**

Selection key: 5-seed mean val F1 -> mean val acc -> mean val AUC (PLAN.md §2). All rows: select='auc' protocol, MPS, current environment.

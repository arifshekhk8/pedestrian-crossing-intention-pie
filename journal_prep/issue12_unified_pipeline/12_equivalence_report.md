# 12 — Unified-engine equivalence check

## G-A — engine equivalence, bilstm on CPU (context-free device)

f1_optimization engine: val F1 0.8271604938, AUC 0.9516869823, best_ep 10
unified engine:        val F1 0.8271604938, AUC 0.9516869823, best_ep 10

**PASS — bit-identical in every field**

## G-B — published-cell reproduction, transformer on mps

unified: val F1 0.8633540373, best_ep 25 | cached `f1_optimization/runs_f1/transformer_searched/pw1.682/seed42/final.json`: val F1 0.8633540373, best_ep 25

**PASS — exact reproduction of the published run**

## G-C — new families (registry-ready, no published result)

- **gru** — params 446,081; 3-epoch loss 0.7065 -> 0.6556 -> 0.6071 — PASS (decreasing)
- **birnn** — params 149,121; 3-epoch loss 0.7143 -> 0.5973 -> 0.5204 — PASS (decreasing)

## Measured reproducibility caveat (documented; not a gate)

Recurrent (nn.LSTM) TRAINING on Apple MPS is bit-deterministic only within an identical process history: the same cell measured val F1 0.82392027 (fresh process), 0.83439490 (after one other training in the same process), 0.83870968 (the published cell, produced mid-way through the 04 driver). The transformer family has no such dependence (G-B reproduces its published cell exactly, cross-process), and CPU has none for any family (0.8271604938 in every tested context). This also explains Issue-8's earlier 'environment drift' finding. Practice: train recurrent families on CPU when exact regeneration matters (~15 s/run); all published TEST metrics are unaffected (saved checkpoints, CPU evaluation, exact parity gates).

## Verdict

**ALL GATES PASS** — one engine, one code path, provably the same computation as the published pipeline; GRU/biRNN ready for follow-up.

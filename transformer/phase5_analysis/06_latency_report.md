# Phase T5 — Isolated inference-latency benchmark (Apple M4)

Issue-9 protocol applied to the Phase-T3 winner (`transformer_searched`, 794,241 params, num_layers=4, d_model=128) on this M4 — the same hardware the live demo runs on. Inference only, no training. MPS timings synchronise inside each timed call (`torch.mps.synchronize`), 50 warmup + 1000 timed forwards per cell. Loads the real `transformer_searched/seed42/best.pt` (latency is weight-independent, but this matches Issue 9's own convention).

## Transformer latency vs BiLSTM baseline

| device | batch | ms / forward | ms / **window** | windows / s | p99 ms | BiLSTM ms/window (ref) |
|---|---|---|---|---|---|---|
| CPU | 1 | 0.459 | **0.4592** | 2,178 | 0.507 | 0.5755 |
| CPU | 8 | 1.087 | **0.1358** | 7,363 | 1.148 | 0.1879 |
| CPU | 32 | 2.673 | **0.0835** | 11,969 | 3.030 | 0.1348 |
| MPS | 1 | 1.388 | **1.3878** | 721 | 1.903 | 1.6469 |
| MPS | 8 | 2.181 | **0.2727** | 3,668 | 2.620 | 0.3108 |
| MPS | 32 | 3.712 | **0.1160** | 8,620 | 5.762 | 0.0834 |

**The transformer is 1.25x faster than the BiLSTM per window** at batch 1 CPU (0.4592 ms vs 0.5755 ms) — despite having ~1.3x the parameters (794,241 vs 594,561), the fully parallel self-attention forward pass over T=16 tokens apparently outruns the BiLSTM's inherently sequential recurrence (which must step through the sequence one timestep at a time and can't parallelize across time), on this hardware and batch size. **In absolute terms both are effectively free**: fastest single-window latency is **0.459 ms** (CPU, batch 1) = 2,178 windows/s — about **73x inside** a 30 fps frame budget (33.3 ms). As with the BiLSTM, **CPU beats MPS at batch 1** (0.459 vs 1.388 ms) — GPU kernel-dispatch overhead dominates a model this small at batch 1. At batch 32 MPS reaches 0.1160 ms/window vs CPU 0.0835. At T=16, self-attention's O(T²) cost (16²=256 pairs) is trivial next to the linear projections and FFN either way — sequence length is not the deciding factor here.

## Verdict

The transformer is 1.25x faster than the BiLSTM per window (0.4592 vs 0.5755 ms CPU batch-1), but **neither model is a latency concern on this hardware**: both are roughly 2 orders of magnitude inside the 30 fps budget, and the live pipeline remains detection-bound (Issue 9: YOLO26-M is the actual bottleneck; the intent model — either architecture — is a rounding error next to it). If deploying the transformer in place of the BiLSTM, expect no perceptible change in end-to-end frame rate.

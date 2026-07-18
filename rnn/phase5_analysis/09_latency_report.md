# RNN study — Phase R5 isolated inference latency (Apple M4)

RNN F1-winner (560,001 params), Issue-9 protocol: 50 warmup + 1000 timed forwards per cell, MPS synced inside each timed call. Latency is weight-independent; the real F1-winner checkpoint is loaded for fidelity.

| device | batch | ms/forward | ms/**window** | windows/s | p99 ms |
|---|---|---|---|---|---|
| CPU | 1 | 0.316 | **0.3156** | 3,168 | 0.340 |
| CPU | 8 | 1.004 | **0.1254** | 7,971 | 1.055 |
| CPU | 32 | 2.069 | **0.0646** | 15,470 | 2.573 |
| MPS | 1 | 4.293 | **4.2932** | 233 | 8.421 |
| MPS | 8 | 4.530 | **0.5663** | 1,766 | 6.322 |
| MPS | 32 | 6.396 | **0.1999** | 5,003 | 7.443 |

**RNN single-window latency = 0.316 ms** (CPU, batch 1) = ~106× inside a 30 fps frame budget (33.3 ms). vs BiLSTM 0.575 ms, GRU 0.721 ms, and Transformer 0.459 ms (all M4 CPU batch-1). The vanilla RNN is the smallest family (560,001 params), so a low latency is expected; either way it is ~2 orders of magnitude inside the frame budget and the live pipeline stays detection-bound (Issue 9). Latency is not a deployment discriminator among the four model families.

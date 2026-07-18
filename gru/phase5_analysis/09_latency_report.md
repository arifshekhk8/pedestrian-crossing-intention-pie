# GRU study — Phase G5 isolated inference latency (Apple M4)

GRU F1-winner (h256, 1,678,209 params), Issue-9 protocol: 50 warmup + 1000 timed forwards per cell, MPS synced inside each timed call. Latency is weight-independent; the real F1-winner checkpoint is loaded for fidelity.

| device | batch | ms/forward | ms/**window** | windows/s | p99 ms |
|---|---|---|---|---|---|
| CPU | 1 | 0.721 | **0.7211** | 1,387 | 0.764 |
| CPU | 8 | 2.346 | **0.2932** | 3,411 | 2.422 |
| CPU | 32 | 4.867 | **0.1521** | 6,575 | 5.181 |
| MPS | 1 | 5.240 | **5.2396** | 191 | 5.903 |
| MPS | 8 | 15.618 | **1.9523** | 512 | 22.441 |
| MPS | 32 | 11.760 | **0.3675** | 2,721 | 16.788 |

**GRU single-window latency = 0.721 ms** (CPU, batch 1) = ~46× inside a 30 fps frame budget (33.3 ms). vs BiLSTM 0.575 ms and Transformer 0.459 ms (both M4 CPU batch-1). The GRU F1-winner is a bigger model (h256, 1,678,209 params vs the BiLSTM's 594,561), so a somewhat higher single-window latency is expected; either way it is ~2 orders of magnitude inside the frame budget and the live pipeline stays detection-bound (Issue 9). Latency is not a deployment discriminator among the three model families.

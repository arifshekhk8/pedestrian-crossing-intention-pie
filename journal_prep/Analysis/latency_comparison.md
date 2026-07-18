# Latency comparison (Apple M4, classifier forward only)

Issue-9 protocol: 50 warmup + 1000 timed forwards, `torch.mps.synchronize()` inside each timed MPS call. **Quote CPU batch-1** — the honest single-window latency (GPU launch overhead dominates a sub-million-param model at batch 1). 30 fps budget = 33.3 ms/frame.

| family (measured model) | **CPU batch-1** (ms/win) | GPU batch-1 | CPU batch-32 | ×inside 30 fps | source |
|---|---|---|---|---|---|
| BiLSTM (h128) | **0.575** | 1.647 | 0.135 | ~58× | Issue 9 |
| Transformer (d128/ff512/L4) | **0.459** | 1.388 | 0.084 | ~73× | transformer/phase5 |
| GRU (h256) | **0.721** | — | — | ~46× | gru/phase5 |
| RNN (h256) | **0.316** | 4.293 | 0.065 | ~105× | rnn/phase5 |

One representative model per family was timed (latency is weight-driven, so it is reported per family, not per variant). All four are ~2 orders of magnitude inside the frame budget — latency is not a deployment discriminator; the live YOLO+ByteTrack pipeline is detection-bound (Issue 9). The **vanilla RNN is fastest** (un-gated cell = smallest); the GRU is slowest only because its F1-winner is the largest model.

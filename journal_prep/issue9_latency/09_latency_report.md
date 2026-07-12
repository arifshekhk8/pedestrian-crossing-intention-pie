# Issue 9 — Isolated inference-latency benchmark (Apple M4)

All numbers measured on this MacBook Air **M4**; inference only (no training). MPS timings synchronise inside each timed call (torch.mps.synchronize), 50 warmup + 1000 timed forwards per cell. BiLSTM = the locked baseline (hidden 128, 2 layers, 0.6 M params, input 16×5); latency is weight-independent, but we load the real `bilstm_baseline/best.pt` for fidelity.

## 1. Isolated BiLSTM latency (the previously-unreported number)

| device | batch | ms / forward | ms / **window** | windows / s | p99 ms |
|---|---|---|---|---|---|
| CPU | 1 | 0.575 | **0.5755** | 1,738 | 0.606 |
| CPU | 8 | 1.503 | **0.1879** | 5,322 | 1.567 |
| CPU | 32 | 4.312 | **0.1348** | 7,421 | 5.186 |
| MPS | 1 | 1.647 | **1.6469** | 607 | 3.186 |
| MPS | 8 | 2.486 | **0.3108** | 3,218 | 5.230 |
| MPS | 32 | 2.669 | **0.0834** | 11,989 | 6.639 |

**The BiLSTM is effectively free.** Fastest single-window latency is **0.575 ms** (CPU, batch 1) = 1,738 windows/s — about **58× inside** a 30 fps frame budget (33.3 ms). Batching 32 parallel tracks amortises to 0.0834 ms/window (MPS). A notable detail: **CPU beats MPS at batch 1** (0.575 vs 1.647 ms) — GPU kernel-dispatch overhead dominates a model this small, so the GPU only pays off once many parallel tracks are batched. At batch 32 MPS reaches 0.0834 ms/window vs CPU 0.1348. For single-track real-time use, CPU is the better backend for this network. The 0.6 M-param model is not a latency concern on any backend.

## 2. Observation-window length vs latency (batch=1)

| obs_len | CPU ms | MPS ms |
|---|---|---|
| 8 | 0.321 | 1.214 |
| 16 | 0.562 | 1.401 |
| 30 | 1.005 | 2.675 |

"Shorter window = lower latency" is **confirmed**: on CPU, obs_len 8/16/30 → 0.321 / 0.562 / 1.005 ms (the LSTM unrolls over the sequence, so cost scales with length). The effect is real but ≤1 ms in absolute terms — immaterial next to detection (below).

## 3. Pipeline breakdown — where the time actually goes

Measured on real PIE frames (1920×1080, imgsz 640, person class). BiLSTM shown at its on-MPS single-track cost (conservative; on CPU it is 0.575 ms):

| stage | ms / frame | share |
|---|---|---|
| YOLO26-M detect (MPS) | 33.7 | 92.7% |
| ByteTrack assoc. (est.) | ~1.0 | 2.7% |
| **BiLSTM intent** (per track) | **1.647** | **4.53%** |
| **total / frame** | **36.4** | → **27.5 fps** |

**The pipeline is detection-bound, not prediction-bound.** YOLO26-M is ~20× the BiLSTM; the intent model is only **4.5%** of per-frame cost. Honest framing: the BiLSTM adds negligible latency on top of whatever detector is used, and the headline ~27 fps is set by YOLO26-M — a lighter detector would raise it without touching this thesis's contribution.

(YOLO26-M on CPU: 69 ms/frame = 14.4 fps — MPS is the deployment target.)

## Verdict

The previously-missing number: **isolated BiLSTM latency = 0.575 ms/window** (CPU, batch 1; 1.647 ms on MPS — slower for a model this small because of GPU dispatch overhead). That is **~58× inside** a 30 fps frame budget (33.3 ms) and a negligible part of the full perception→prediction pipeline, which is detection-bound (BiLSTM ≈ 5% of per-frame cost). Real-time operation is justified for the prediction model itself; the pipeline fps is a property of the chosen detector (YOLO26-M), not the BiLSTM.
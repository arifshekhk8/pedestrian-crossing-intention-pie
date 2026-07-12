# Issue 9 — Isolated inference-latency benchmark ✅

The only timing number in the project was "~900 frames in ~50 s on MPS" — the
*full* pipeline, dominated by YOLO26-M. The BiLSTM's own latency (this thesis's
contribution) was never measured in isolation, so "real-time / ~18 fps" conflated
three very different components. This measures each on the actual M4.

## How to run

```bash
source .venv/bin/activate
python journal_prep/issue9_latency/09_inference_latency.py            # ~3–5 min, inference only
python journal_prep/issue9_latency/09_inference_latency.py --report-only   # rebuild report/figure from saved JSON
```

**No training** — loads `runs/bilstm_baseline/best.pt` and `yolo26m.pt` and times
forward passes. MPS timings call `torch.mps.synchronize()` inside each timed call
(MPS is async — otherwise you'd time dispatch, not compute); 50 warmup + 1000 timed
forwards per cell.

## Headline numbers (Apple M4)

**Isolated BiLSTM latency:**

| device | batch | ms / window | windows / s |
|---|---|---|---|
| CPU | 1 | **0.575** | 1,738 |
| CPU | 32 | 0.135 | 7,421 |
| MPS | 1 | 1.647 | 607 |
| MPS | 32 | **0.083** | 11,989 |

- A single intent prediction is **0.575 ms (CPU)** — about **58× inside** a 30 fps
  frame budget (33.3 ms). The 0.6 M-param model is not a latency concern.
- **CPU beats MPS at batch 1** (0.575 vs 1.647 ms): GPU kernel-dispatch overhead
  dominates a model this small, so the GPU only pays off when batching many parallel
  tracks (batch 32: MPS 0.083 vs CPU 0.135 ms/window). For single-track real-time
  use, **CPU is the better backend** for this network — an honest, slightly
  counter-intuitive finding.
- **"Shorter window = lower latency" confirmed:** obs_len 8/16/30 → 0.321 / 0.562 /
  1.005 ms on CPU (the LSTM unrolls over the sequence). Real but ≤1 ms — immaterial.

**Pipeline breakdown (MPS, real PIE frames, person class):**

| stage | ms / frame | share |
|---|---|---|
| YOLO26-M detect | 33.7 | 92.7% |
| ByteTrack (est.) | ~1.0 | 2.7% |
| **BiLSTM intent** | **1.6** | **4.5%** |
| **total** | **36.4** | **→ 27.5 fps** |

**The pipeline is detection-bound, not prediction-bound.** YOLO26-M is ~20× the
BiLSTM. The honest framing for the paper: the BiLSTM adds negligible latency on top
of any detector, and the headline fps is set by YOLO26-M — a lighter detector would
raise it without touching this thesis's contribution. (YOLO on CPU: 69 ms/frame =
14.4 fps; MPS is the deployment target.)

## Files

```
09_inference_latency.py     benchmark (BiLSTM + YOLO + pipeline); --report-only to rebuild
09_latency_results.json     raw measured numbers
09_latency_report.md        tables + verdict
09_latency_figure.png       BiLSTM latency vs batch (CPU/MPS crossover) + pipeline breakdown
```

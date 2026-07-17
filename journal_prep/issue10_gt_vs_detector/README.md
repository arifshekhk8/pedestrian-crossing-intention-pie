# Issue 10 — GT-box vs YOLO-box prediction degradation ✅

The Phase-4 demo was qualitative ("AUC 1.000 on 10 peds" — not a result). The real
science of the live pipeline: **how much does feeding the BiLSTM noisy detector boxes
(YOLO26-M + ByteTrack) instead of ground-truth boxes actually cost?** This quantifies
the perception→prediction gap on the two demo clips.

> **Metric note (F1-first).** This is an **AUC-scoped** robustness measurement (GT-box
> vs YOLO-box AUC drop, an oracle-matched lower bound — see `10_gt_vs_detector_results.md`).
> Under the project's **F1 → acc → AUC** hierarchy the F1-first headline lives in
> [`../../f1_optimization/`](../../f1_optimization/) and
> [`../issue3_baseline_comparison/`](../issue3_baseline_comparison/).

## How to run

```bash
source .venv/bin/activate
python journal_prep/issue10_gt_vs_detector/10_gt_vs_detector_auc.py   # ~4 min first run; cached after
```

**No training** — inference only. YOLO+ByteTrack is run over just the *union of
segments* the GT windows need (~4.8 k frames, not the full 33 k), then cached to
`cache_dets.pkl` so the matching/scoring can be re-run instantly.

## Method (what makes it honest)

For each clean GT window (Issue-2 protocol) in video_0012 + video_0016:
1. Run YOLO26-M + ByteTrack over the needed frames.
2. **Assemble the YOLO-box window from the best-IoU detection per frame** — this
   isolates *box-localisation* noise (ego-speed is unchanged; it comes from the
   vehicle, not vision, so only bbox features differ between paths).
3. Score both the GT-box and YOLO-box windows through the **clean** BiLSTM
   (`runs_clean/bilstm_baseline_clean`).
4. Measure ByteTrack **identity** quality *separately* (purity, switches,
   fragmentation) so detection-box noise and tracking errors are never conflated.

## Result (indicative — 98 peds / 311 windows, 2 clips)

**Prediction is robust to detector box noise:**

| path | per-window AUC | per-pedestrian AUC |
|---|---|---|
| GT boxes (offline) | 0.962 | 0.958 |
| YOLO boxes (pipeline) | 0.953 | 0.948 |
| **drop** | **+0.009** | **+0.010** |

Decisions flip across the 0.5 threshold in only **10/311 (3%)** of windows; mean
IoU(GT, YOLO) = 0.75. So the offline AUC is broadly indicative of live performance.

**The pipeline's weak links are perception, not prediction:**
- **Detector recall:** 88% of pedestrians are detected (≥50% window coverage) — the
  other ~12% are never tracked well enough to predict at all (a safety gap worth
  stating). 71% of all windows are usable.
- **Tracker fragmentation:** a single ByteTrack ID covers a mean of only **39%** of a
  pedestrian's frames, and **59%** of windows carry a competing ID — heavy identity
  fragmentation in these crowded scenes. A deployment would need stronger re-ID.

Neither weakens the BiLSTM's tolerance to box noise — they are detector/tracker
engineering gaps, separate from this thesis's prediction model. This replaces the old
"N=10, AUC 1.000" with a real, indicative perception→prediction measurement.

## Files

```
10_gt_vs_detector_auc.py        harness (segment YOLO+ByteTrack, IoU match, dual scoring)
10_gt_vs_detector_results.md    detector/tracker quality + AUC table + verdict
10_gt_vs_detector.csv           per-window: detected, cov, IoU, purity, switch, frag, GT/YOLO prob
10_gt_vs_detector_figure.png    GT-vs-YOLO prob scatter + AUC bars
cache_dets.pkl                  cached YOLO+ByteTrack detections (re-run matching without YOLO)
```

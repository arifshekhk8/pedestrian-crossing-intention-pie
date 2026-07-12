# Issue 10 — GT-box vs YOLO-box prediction degradation (indicative)

Demo clips video_0012 + video_0016 (set03). For each clean GT window we run YOLO26-M + ByteTrack over the frames it needs and assemble the YOLO-box window from the **best-IoU detection per frame** (this isolates box-localisation noise; ego-speed is unchanged — it comes from the vehicle, not vision). Both paths are scored through the clean BiLSTM (`runs_clean/bilstm_baseline_clean`). ByteTrack **identity** quality is measured separately (below) so detection-box noise and tracking errors are not conflated. Indicative subset, not a full benchmark.

## Detector / tracker quality

- **Windows:** 439 total · **311 detected** (71% — the BiLSTM gets a usable track) · 128 missed (detector never covers ≥50% of the window).
- **Pedestrians:** 112 total · 98 with ≥1 detected window (88% detector recall).
- **Box quality (matched):** mean IoU(GT, YOLO) = **0.750**; the pedestrian's dominant ByteTrack ID covers a mean **39%** of its matched frames (track purity).
- **ID switches:** 183/311 detected windows (59%) have a *second* ByteTrack ID also substantially covering the same pedestrian (a genuine switch). **Fragmentation:** 138/311 (44%) have ≥1 frame where the dominant track drops out (gap-filled from its nearest box).

## Prediction: GT-box vs YOLO-box (matched subset)

| path | per-window AUC | per-pedestrian AUC |
|---|---|---|
| **GT boxes** (offline) | 0.962 | 0.958 |
| **YOLO boxes** (full pipeline) | 0.953 | 0.948 |
| **drop (GT − YOLO)** | **+0.009** | **+0.010** |

On the matched windows the two probability streams agree closely: a decision flips across the 0.5 threshold in **10/311 (3%)** of windows.

## Verdict

**The prediction model is robust to detector box noise.** On 311 matched windows / 98 pedestrians, replacing ground-truth boxes with YOLO26-M boxes (mean IoU 0.75) moves AUC by only **+0.009 per window / +0.010 per pedestrian**, and the decision flips in just 10/311 (3%) of windows — so the offline AUC is broadly indicative of live performance under realistic box noise.

**The pipeline's weak links are perception, not prediction:** (1) **detector recall** — 88% of pedestrians are detected, so ~12% are never covered well enough to predict at all (a safety gap worth stating); (2) **tracker fragmentation** — a single ByteTrack ID covers a mean of only **39%** of a pedestrian's frames and 59% of windows carry a competing ID, so a deployment would need stronger re-identification. Neither weakens the BiLSTM's tolerance to box noise above — they are detector/tracker engineering gaps, separate from this thesis's prediction model. Numbers are indicative (N=98 peds, two clips).

_AUC computed via the rank/Mann–Whitney estimator (no sklearn locally)._
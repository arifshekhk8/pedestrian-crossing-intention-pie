# Phase 9 — Detector-in-the-loop on IDD-PeD: assessed, feasible, **not run**

## The PIE precedent

The PIE study does contain a detector-derived bounding-box experiment
(`journal_prep/issue10_gt_vs_detector/`): for each clean ground-truth window, YOLO26-M +
ByteTrack is run over the frames the window needs, the window is reassembled from the
best-IoU detection per frame, and both paths are scored through the same BiLSTM. Headline
result on 311 matched windows / 98 pedestrians from two set03 clips: AUC drops only
**0.962 → 0.953** (+0.009), decisions flip in 3 % of windows, and the pipeline's weak links
are detector recall (88 %) and tracker fragmentation, not the predictor. The authors of that
note call it *indicative, not a full benchmark*.

## Is an equivalent IDD-PeD experiment possible?

**Yes, technically.** Everything required is available:

| requirement | status |
|---|---|
| raw video | ✅ 9 tars on the CVIT host, direct download, CC BY 4.0 |
| total video size | 41.9 GB (test sets `0003`+`0005`+`0008`+`0009` alone: **14.6 GB**) |
| local free disk | 50 GB — sufficient |
| detector weights | ✅ `yolo26m.pt` already present at the repo root |
| tracker | ✅ ByteTrack via ultralytics, already used by `pipeline/10_yolo_bytetrack_demo.py` |
| ground-truth windows to match against | ✅ the 2,357 strict test windows built here |

## Why it was not run

1. **It answers a different question.** This brief's primary objective is whether the PIE
   *findings* generalize to a different traffic environment. A detector-in-the-loop run
   measures **perception robustness**, which is orthogonal — and on PIE it was already shown
   to be a small effect (+0.009 AUC) relative to the domain gap measured here (−0.23 AUC).
2. **Comparability would be poor.** The PIE result comes from **two hand-picked clips**
   (439 windows, 112 pedestrians). Matching that scope on IDD-PeD would give an equally
   indicative number; matching it *properly* (all four test sets) is a different, larger
   experiment. Reporting a two-clip IDD-PeD number beside a two-clip PIE number would invite
   exactly the cross-dataset claim the brief forbids.
3. **Cost.** ~80 min to download 14.6 GB, plus YOLO26-M inference over 30 fps footage for
   every frame any test window touches, on a CPU/MPS laptop — comparable to the entire rest
   of this experiment, for a secondary question.
4. **A confound specific to IDD-PeD.** 29 of 33 videos are 1920×1440, and detections would
   land in that frame; the coordinate-mapping ambiguity already documented in
   `temporal_protocol_IDD_PeD.md` §6 would compound with detector noise, making the result
   hard to attribute.

## What this means for the paper

**Do not claim detector robustness across datasets.** The paper may state that the PIE
pipeline is robust to detector box noise **on PIE** (Issue 10), and must not extend that
claim to IDD-PeD. The honest sentence is: *"Detector-in-the-loop robustness was measured on
PIE only; the IDD-PeD videos are available and the experiment is feasible, but it was out of
scope for this cross-dataset study."*

## Recipe, if it is run later

```bash
# 1. test-set videos only (14.6 GB) — reuse the parallel range downloader
for s in 0003 0005 0008 0009; do
  # same 16-way range fetch as scripts/00_download_iddped.sh, URL:
  # https://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_${s}.tar
done

# 2. for each strict test window, run YOLO26-M + ByteTrack over its 16 frames,
#    take the best-IoU detection per frame against the GT box (isolates localisation
#    noise from association error), leave ego-speed untouched — it comes from OBD, not vision.

# 3. score both the GT-box and YOLO-box windows through the SAME Experiment-B checkpoints
#    and report per-window and per-pedestrian AUC, exactly as
#    journal_prep/issue10_gt_vs_detector/10_gt_vs_detector_auc.py does.
```

Report detector recall and ByteTrack purity separately from the AUC delta, as Issue 10 does,
so perception and prediction failures are never conflated.

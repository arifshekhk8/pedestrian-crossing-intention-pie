# Pedestrian Crossing-Intention Prediction — Supervisor Review Pack

> **⚠ SUPERSEDED SNAPSHOT (banner added 2026-07-13).** This pack is a **2026-06-16
> presentation snapshot from BEFORE the leakage fix and BEFORE the F1-first directive.**
> Every number in this file and in `results_summary.md` is the **leaky-era** result
> (test = 587 sequences, N = 1,389, `pos_weight` 1.44, **AUC-first**, "AUC 0.931").
> **Do not quote these numbers.** The current, leak-free, F1-first results are:
> **F1 0.844 (BiLSTM) / 0.847 (Transformer)**, Acc 0.897/0.896, AUC 0.940/0.947, on the
> clean test set03 (2,094 windows). See `../../journal_prep/issue2_clean_protocol/`,
> `../../journal_prep/issue3_baseline_comparison/`, and `../../f1_optimization/README.md`.
> This pack is retained only as a historical artifact / figure source; to reuse it,
> regenerate the figures + CSVs from the clean run outputs (do not hand-edit).

**Student:** Arif
**Date:** 2026-06-16 (snapshot — see banner above)
**One-line summary:** A bidirectional LSTM predicts whether a pedestrian will
cross in front of the ego-vehicle from a short history of their bounding box +
the car's speed, trained and evaluated on the PIE dataset, then demonstrated
end-to-end on raw video with a YOLO26 + ByteTrack perception front-end.

> **How to use this pack:** read this file top-to-bottom — it is written so you
> can explain the whole project to your supervisor in order. Section 9 is a
> ready-to-use talking script and likely-questions cheat sheet. All numbers here
> come from the actual run outputs (`runs/*/final.json`), not estimates.

---

## 1. The problem (why this matters)

Autonomous vehicles must anticipate **what a pedestrian is about to do**, not just
where they are now. The key safety question is: *will this person step into the
road in front of me?* Predicting this ~1.5 seconds early gives the planner time
to slow down. This thesis builds and evaluates a model for that prediction, and
shows it running on real dash-cam video.

**Task definition.** Given the last **16 frames (~0.5 s)** of a pedestrian's
bounding box and the ego-vehicle speed, predict a binary label: **will they be
crossing 45 frames (~1.5 s) later?** This is the standard "crossing intention
with a time-to-event (TTE) horizon" formulation used on PIE.

---

## 2. The data (PIE dataset)

PIE (Pedestrian Intention Estimation, York University) — on-board dash-cam video
in Toronto with per-frame pedestrian bounding boxes, crossing labels, and
synchronized vehicle OBD data (speed).

| Item | Value |
|---|---|
| Raw annotation rows | 582,376 frame-level boxes |
| Unique annotated pedestrians | 1,374 |
| Final training **sequences** (16-frame windows) | 1,389 |
| Class balance (sequences) | not-crossing 819 (59%) / crossing 570 (41%) |
| Split (by recording set, no leakage) | Train: set01/02/04 · Val: set05/06 · **Test: set03** |

**Pre-processing pipeline (scripts `01`→`02`):**
1. `01_parse_annotations.py` — parse PIE XML (boxes + attributes + OBD) into one
   table → `pie_annotations.pkl`.
2. `02_build_sequences.py` — for each pedestrian, cut one 16-frame observation
   window ending exactly 45 frames (TTE) before their last annotated frame; label
   = their crossing outcome. → `sequences/X.npy (1389,16,5)`, `y.npy`.

**Input features (5-D per frame):** `[x1, y1, x2, y2, ego_speed]` — bounding-box
corners (pixels) + vehicle speed. Standardized with **train-only** mean/std
(saved as `norm_mean.npy` / `norm_std.npy` so inference is identical).

> **Talking point:** splitting by *recording set* (not random) means train and
> test pedestrians/scenes never overlap — the AUC is an honest generalization
> number, not memorization.

---

## 3. The model (architecture)

A **2-layer stacked bidirectional LSTM** — bidirectional because, within the
observed window, both the approach trajectory and its recent dynamics matter.

```
[16 × 5]  ──Linear(5→64)+ReLU──►  BiLSTM(hidden=128, 2 layers, dropout=0.3, bi)
          ──► last timestep (256-d) ──► Linear(256→1) ──► sigmoid ──► P(cross)
```

- Params: **594,561**
- Loss: `BCEWithLogitsLoss` with `pos_weight=1.44` (handles the 59/41 imbalance)
- Optimizer: Adam, lr 1e-3, weight_decay 1e-5
- Early stopping on **validation AUC** (patience 15); test touched once, on the
  best-val checkpoint (no test-set tuning).

Two model **variants** were also built for comparison:
- **bbox-only (4-D):** drops ego-speed — isolates the value of the speed signal.
- **+ temporal attention:** replaces "use last timestep" with attention over all
  16 frames (Bahdanau-style), so the model can weight the most informative frames
  and we can *visualize* which frames it used.

---

## 4. Main result

On the held-out **test set (set03, 587 sequences)**, the 5-D baseline:

| Metric | Value |
|---|---|
| **AUC-ROC** | **0.931** |
| F1 | 0.844 |
| Accuracy | 0.874 |
| Precision | 0.820 |
| Recall | 0.870 |
| Confusion `[[TN,FP],[FN,TP]]` | `[[313,44],[30,200]]` |

**Reading the confusion matrix:** FP (44) > FN (30) → when the model is wrong, it
more often *predicts a crossing that doesn't happen* than *misses a real one*.
For an AV that is the **safe** failure mode (a false slow-down beats a missed
pedestrian). See `figures/01_baseline_training.png`.

---

## 5. Comparisons (what we learned by changing one thing)

All trained with identical settings; only the named factor changes.

| Model | AUC | F1 | Precision | Recall | Takeaway |
|---|---|---|---|---|---|
| **BiLSTM 5-D (baseline)** | **0.931** | 0.844 | 0.820 | 0.870 | reference |
| BiLSTM 4-D (bbox-only) | 0.889 | 0.797 | 0.712 | 0.904 | **ego-speed adds +0.042 AUC** |
| BiLSTM 5-D + attention | 0.933 | 0.845 | 0.779 | 0.922 | AUC on par, **+recall** |

- **Ego-speed matters:** removing it drops AUC 0.931→0.889 and precision
  0.82→0.71 (without speed the model over-predicts crossings).
- **Attention:** ~same AUC but shifts the error profile toward higher recall
  (fewer missed crossings) — again the AV-safe direction.

Figures: `figures/02_all_models_comparison.png`,
`figures/03_precision_recall_tradeoff.png`.

---

## 6. Ablations (robustness checks)

**Observation window** (how much history; TTE fixed at 45):

| obs_len | AUC | F1 | Recall |
|---|---|---|---|
| 8 (0.27 s) | 0.936 | 0.862 | 0.922 |
| 16 (0.53 s) | 0.931 | 0.844 | 0.870 |
| 30 (1.00 s) | 0.935 | 0.839 | 0.921 |

→ AUC is **insensitive** to window length (spread ≤0.005). Even 0.27 s of history
is enough — good for low-latency deployment. (`figures/04_window_ablation.png`)

**Prediction horizon (TTE)** (how far ahead; obs fixed at 16):

| TTE | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| 30 (1.0 s) | 0.959 | 0.863 | 0.849 | 0.878 |
| 45 (1.5 s) | 0.931 | 0.844 | 0.820 | 0.870 |
| 60 (2.0 s) | 0.944 | 0.846 | 0.789 | 0.913 |

→ AUC stays high out to **2 s**; the real effect is a precision/recall trade
(longer horizon → higher recall, lower precision). (`figures/05_tte_ablation.png`)

> **Talking point:** the model isn't brittle to the exact framing of the problem —
> it works across reasonable choices of history length and prediction horizon.

---

## 7. The live demo (Phase 4 — perception + prediction end-to-end)

The offline model proves the *idea* works. The demo proves it works on **raw
video with no ground-truth boxes** — a realistic AV pipeline:

```
dash-cam video → YOLO26-M (detect people) → ByteTrack (track IDs)
   → per-person rolling 16-frame [bbox + ego-speed] buffer
   → BiLSTM → P(cross) → colored overlay (green=safe ··· red=crossing)
```

- Detector: **YOLO26-M** (Ultralytics). Tracker: **ByteTrack**.
- Ego-speed taken from PIE's OBD record for the demo clip (a real car would read
  its own speedometer).
- Runs on a MacBook Air **M4 (MPS GPU)** at ~18 fps on 1080p.

**Demo clip A — `demo_videos/demo_video_0016.mp4`** (set03/video_0016, 30 s):
the car approaches an intersection (speed 0→27→0) and a crowd crosses while it is
stopped. The crossing crowd is boxed **red (P≈0.75–0.88)**; people on the
sidewalk get low scores. See `figures/06_demo_crossing_crowd.png` (the money
shot) and `07_demo_approach.png`.

**Demo clip B — `demo_videos/demo_video_0012.mp4`** (set03/video_0012, moving
vehicle, **mixed** crossing + not-crossing pedestrians): this one is used for a
*quantitative* check — see `results_summary.md` for the track-level numbers and
`figures/08_intent_over_time.png` for P(cross) rising for a crosser vs staying
low for a non-crosser.

> **Important honesty point (say this to your supervisor):** the demo is a
> **qualitative/sanity demonstration** of the deployed pipeline. The headline
> *quantitative* result is the **AUC 0.931 on the held-out test set** (Section 4).
> The demo clips show the same model running on pixels, end to end.

---

## 8. Honest limitations (raise these *before* your supervisor does)

1. ~~**Single training seed (42).**~~ **ADDRESSED.** Re-ran all 3 model configs
   over 5 seeds [42,0,1,7,123]. Results (`results_summary.md` §G):
   - Baseline: **AUC 0.948 ± 0.013** — mean is *higher* than seed-42 alone (0.931);
     seed 42 was a slightly below-average run.
   - Bbox-only: AUC 0.887 ± 0.011. Attention: AUC 0.942 ± 0.007.
   - All stds small vs inter-model gaps → ranking is stable, not a lucky seed.
2. **No comparison to published PIE baselines yet.** We should cite and tabulate
   the original PIE paper's intention numbers next to ours (the `PIEPredict`
   repo is included locally for this).
3. **Demo identity matching is approximate.** ByteTrack IDs are not PIE ped-IDs;
   the quantitative demo check matches detections to GT boxes by IoU, so treat it
   as indicative, not a benchmark.
4. **AUC is high (0.93) and converges fast (epoch 3).** Worth a sentence on why:
   PIE crossing labels are strongly signaled by trajectory + speed; this is
   consistent with other PIE results, but flag it rather than overclaim novelty.
5. **PIE only.** Single dataset/city. Cross-dataset generalization (e.g. JAAD) is
   future work.

---

## 9. How to present this (talking script + Q&A)

**60-second pitch:**
> "I predict pedestrian crossing intention 1.5 s ahead with a bidirectional LSTM
> over 16 frames of bounding box plus ego-speed, on the PIE dataset. It reaches
> 0.93 AUC on a held-out set of scenes. I showed ego-speed contributes +0.04 AUC,
> that attention trades a little precision for recall, and that results are stable
> across observation windows and prediction horizons. Finally I deployed it on raw
> video with YOLO26 + ByteTrack, so it runs end-to-end from pixels to a crossing
> probability at ~18 fps."

**Order to walk through the artifacts:** Section 1 (problem) → Fig 01 (it trains)
→ Section 4 table (main AUC) → Fig 02 (comparisons) → Figs 04/05 (ablations) →
play `demo_video_0012.mp4` then `demo_video_0016.mp4` → Fig 08 (intent over time)
→ Section 8 (limitations + next steps).

**Likely questions & answers:**
- *"Is 0.93 too good?"* → Honest: PIE intention is well-signaled by trajectory +
  speed; split is by scene so no leakage; converges fast because the signal is
  strong. Multi-seed (5 seeds) confirms: mean AUC 0.948±0.013 — not a lucky run.
  Next step: published-baseline comparison.
- *"Why BiLSTM not a Transformer?"* → 1,389 sequences is small; a 0.6 M-param
  BiLSTM is the right capacity and is interpretable (attention variant included).
- *"What does ego-speed add?"* → +0.042 AUC and +0.11 precision; without it the
  model over-predicts crossings (Section 5).
- *"Does the demo prove accuracy?"* → It proves the *pipeline* works on pixels;
  accuracy is the test-set AUC. (Don't conflate the two.)

---

## 10. File map of this pack

```
supervisor_review/
├── 00_README_START_HERE.md     ← this file (the full explanation)
├── results_summary.md          ← every table + the demo's quantitative check
├── figures/
│   ├── 01_baseline_training.png        training curves + test metrics
│   ├── 02_all_models_comparison.png    baseline vs bbox-only vs attention
│   ├── 03_precision_recall_tradeoff.png
│   ├── 04_window_ablation.png          obs_len 8/16/30
│   ├── 05_tte_ablation.png             TTE 30/45/60
│   ├── 06_demo_crossing_crowd.png      crowd crossing, boxed red (the demo shot)
│   ├── 07_demo_approach.png            car approaching, sparse detections
│   ├── 08_intent_over_time.png         P(cross): crosser vs non-crosser
│   └── 09_demo_v0012_crossing.png      video_0012 frame, crossers boxed red
├── demo_videos/
│   ├── demo_video_0016.mp4     crowd crossing at a stop
│   └── demo_video_0012.mp4     moving vehicle, mixed crossing/not-crossing
└── data/
    ├── demo_video_0016_predictions.csv
    ├── demo_video_0012_predictions.csv
    ├── multiseed_results.csv       ← 15 rows: one per (model, seed)
    ├── multiseed_summary.csv       ← 3 rows: mean ± std per model
    └── multiseed_summary.md        ← paste-ready thesis table
```

**To reproduce** (code lives one level up in the project root): scripts
`01`→`02` build data; `04_train_bilstm.py` trains the baseline; `04b`/`07_train`
the variants; `08`/`09` the ablations; `10_yolo_bytetrack_demo.py` runs the demo.
See `../CODE_STATE.md` and `../PROGRESS_LOG.md` for the full per-file log.

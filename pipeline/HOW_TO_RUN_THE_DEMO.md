# Running the crossing-intention demo

Everything here runs the **paper's headline model**: `BiLSTM-F1`, a five-seed
probability ensemble trained on the leakage-free crossing-point protocol, at its
validation-tuned threshold `tau* = 0.5164`. Same checkpoints, same feature order,
same normalization as Table 3 of the manuscript.

There is an older demo, `10_yolo_bytetrack_demo.py`. **Do not use it for anything
you will show or publish.** It loads the legacy leaky-protocol checkpoint, which
is the result the paper retracts. It is kept only as a historical record.

---

## 0. One-time setup

```bash
cd /Users/arif/Developer/pedestrian-thesis
source .venv/bin/activate
```

You need `PIE_clips/set03/video_0012.mp4` and `video_0016.mp4` (about 1.5 GB each)
and `pie_annotations.pkl` in the repo root. Both are already present on this
machine. `yolo26m.pt` is in the repo root and is found automatically.

---

## 1. Always run the parity gate first

```bash
python pipeline/11_demo_clean_ensemble.py --stage verify
```

This scores the whole clean test set through the assembled ensemble and checks it
reproduces the published numbers to within `1e-4`:

```
[verify] OK  auc  got 0.946739  expected 0.946739  |d| 0.00e+00
[verify] OK  f1   got 0.855685  expected 0.855685  |d| 0.00e+00
[verify] OK  acc  got 0.905444  expected 0.905444  |d| 0.00e+00
[verify] PASS — this ensemble is the paper's headline model. Safe to render.
```

It takes about 20 s and needs no video. If it fails, stop: a silent feature-order
or normalization mismatch would make every frame you render wrong in a way you
could not see. Do not skip this before a meeting.

---

## 2. Run it live, in a window

This is the one to use in front of your supervisor. It plays at video speed while
computing, so what they see is the system running, not a file being played back.

```bash
python pipeline/12_supervisor_demo.py --scene anticipation --live
```

**Controls:** `space` pauses and resumes (useful for stopping on the moment the
probability crosses the threshold), `q` quits.

Useful flags:

| Flag | Effect |
|---|---|
| `--no-blur-faces` | skips the privacy blur. **Use this for a live demo:** the blur is a second detector pass and roughly halves throughput. PIE is a published dataset, so blurring is a publication requirement, not a meeting one. |
| `--fast` | run flat out instead of pacing to 30 fps, to show peak throughput |
| `--display-width 1600` | bigger window (default 1280) |
| `--label-min-height 0` | label every detection, however small. The default is 95 px for a verdict and 171 px (1.8x) for a `buffering` plate, which only says a track is new. Below those the box is still drawn, just without text. Down the street the detector finds a dozen people whose tracks keep breaking, and their plates bury the pedestrian you are pointing at. Nothing is hidden: the boxes stay. |
| `--device cpu` | force CPU if MPS misbehaves |

The fastest, smoothest live demo:

```bash
python pipeline/12_supervisor_demo.py --scene anticipation --live --no-blur-faces
```

---

## 3. Write a video file to send or embed

```bash
python pipeline/12_supervisor_demo.py --scene anticipation --write-video
```

Writes to `pipeline/demo_videos/anticipation.mp4`, H.264, 1280 px wide, faces
blurred, plus a `.json` beside it recording the model, threshold, frame range and
measured timings. H.264 matters: OpenCV can only write `mp4v`, which will not play
in Keynote, PowerPoint or a browser, so the script transcodes with ffmpeg
afterwards. Pass `--no-transcode` to keep the raw file.

Do both at once with `--live --write-video`.

---

## 4. The five scenes

```bash
python pipeline/12_supervisor_demo.py --scene <name> --live
```

| Scene | Clip, frames | What it shows |
|---|---|---|
| `anticipation` | 0016, 4270 + 190 | The Figure 9a crossing. A pedestrian is flagged while still on the kerb, 1.5 s before stepping out. |
| `bystander` | 0012, 460 + 200 | The Figure 9b case. A worker stands at a kerb beside a marked crosswalk and does not cross. Same position as a crosser, opposite verdict. |
| `driving` | 0016, 11950 + 350 | The car actually moving, 6 to 24 km/h, staying quiet past pedestrians on the pavement. |
| `busy` | 0012, 5560 + 780 | 26 s of a busy corner, several tracks at once, probabilities firming up as each approach resolves. |
| `uncertainty` | 0016, 4415 + 340 | Where the model is worst. Probabilities sit near the threshold and several non-crossers cross it. |

**Show `uncertainty`.** A reel of nothing but successes invites the question of
what was left out. Answering it before it is asked is worth more than the extra
minute it costs, and the failure mode is the cheap one: the model gets uncertain
about a group waiting at a kerb and calls some of them crossers. Erring toward
"might cross" is the right direction for a driver-assistance system.

Arbitrary segments, no preset:

```bash
python pipeline/12_supervisor_demo.py \
  --video-id video_0012 --start-frame 7676 --max-frames 600 \
  --name my_clip --title "Whatever you want on screen" --live --write-video
```

---

## 5. Reading the overlay

**Per pedestrian.** Blue box means the model predicts a crossing, grey means it
does not, dim grey means the 16-frame window is still filling and no prediction
exists yet. The bar under each label is the probability; the dark tick on it is
the 0.516 threshold, so you can see at a glance whether a call was comfortable or
marginal. When two pedestrians stand close together their labels are nudged apart
and joined to their box by a thin line.

**Header.** Which model is running, the threshold, the current frame, the ego
speed being fed to the model, and live throughput.

**`(held)` next to the ego speed** means PIE has no annotation on that frame and
the last known value is being carried forward, which is what a real vehicle does
between CAN messages. It is marked so a held value is never mistaken for a reading.

---

## 6. About "real time"

The end-of-run summary separates the system from the demo scaffolding:

```
[demo] --- the system ---
[demo]   detector + tracker       37.9 ms/frame
[demo]   intention ensemble       1.96 ms/window (0.39 ms per single model)
[demo]   together                 21.7 FPS  = 0.72x real time at 30 fps
[demo]   the detector is 82% of that, so the pipeline is detection-bound
```

Two honest points to make with this, because a supervisor will ask:

**The intention model is not the bottleneck, and it is not close.** It costs about
0.4 ms per window per model. YOLO26-M costs about 38 ms per frame. The detector is
roughly 80 to 90% of the pipeline, which is exactly what Section 4.9 of the paper
reports. If you need more speed, you change the detector, not the model. A smaller
YOLO variant would clear 30 fps on this laptop without touching the predictor.

**This machine is an M4 Air, and it reaches about 0.7 to 0.8x real time with
YOLO26-M at 1920x1080.** Do not claim real time on this hardware without
qualification. What is fair to say: the prediction step is sub-millisecond and the
system is detection-bound, so real time is a detector choice.

The privacy blur, the overlay drawing and the file write are reported separately
and excluded from that figure. Including them would measure the presentation
rather than the model.

---

## 7. What the demo cannot do

**It needs the ego-vehicle speed.** That is one of the model's two inputs, and
removing it collapses AUC from 0.932 to 0.753. In these runs the speed is read
from PIE's annotations. Pointing the script at a webcam or a dashcam clip with no
speed channel would feed it a constant, and the predictions would not mean what
they mean here. On a real vehicle the speed comes off the CAN bus, which is why
the paper argues the input is cheap: every car already has it.

**Three of the five scenes have the car stopped**, at 0 km/h. That is not a bug.
Pedestrians cross when traffic is stopped, so scenes containing crossings tend to
be scenes where the car is waiting. `driving` is the counterexample, and it is
mostly correct negatives, which is the operationally important behaviour: no false
alarms while cruising.

---

## 8. Files

| Path | What |
|---|---|
| `pipeline/12_supervisor_demo.py` | this demo (presentation overlay, `--live`, scene presets) |
| `pipeline/11_demo_clean_ensemble.py` | the workhorse: parity gate, raw frames, prediction CSV |
| `pipeline/10_yolo_bytetrack_demo.py` | ⚠️ legacy leaky model, historical record only |
| `pipeline/demo_videos/` | generated videos and their `.json` provenance (gitignored) |
| `paper_and_artifacts/Journal_writing/supplementary/video_s1.mp4` | the clip submitted with the paper |

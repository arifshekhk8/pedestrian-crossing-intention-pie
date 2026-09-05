"""prep_fig2_track.py — the three PIE thumbnails used inside Figure 2.

Figure 2 opens on the dataset rather than on a box labelled "dataset": three
frames of one annotated pedestrian, with PIE's own bounding box drawn on each.
The crop rectangle is the same in all three, so the pedestrian visibly moves
across the strip; that motion is the input the model actually receives.

  video_0016, pedestrian 3_16_919, crossing_point frame 2567 (PIE annotation).
  Frames 2467, 2492 and 2517, i.e. 50 to 100 frames before the event, all of
  them inside the region a leakage-free window may observe.

Same pedestrian as the opening figure, for continuity, and the same privacy
routine: head regions of every person the detector finds are blurred in the
pixels before cropping, so nothing recoverable survives into the figure.

Writes `_track1.png` … `_track3.png` next to this file. They are small and are
committed, so make_fig2_system.py never needs the 1.5 GB clip.

Run once:  python prep_fig2_track.py
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CLIP = ROOT / "PIE_clips" / "set03" / "video_0016.mp4"
ANN = ROOT / "pie_annotations.pkl"
WEIGHTS = ROOT / "yolo26m.pt"

VIDEO, PED, CP = "video_0016", "3_16_919", 2567
FRAMES = [2467, 2492, 2517]
PAD_X, PAD_Y = 48, 40           # context kept around the union of the boxes
OUT_W = 340                     # stored width; the figure prints it near 125 px
ACCENT = (214, 120, 42)         # BGR of #2a78d6, the figure's single accent
GAMMA = 0.62                    # brightness lift; the clip is shot at night


def blur_heads(img, conf=0.05):
    """Blur the head region of every person the detector finds.

    Not a face detector: one that misses a turned head fails silently, and at
    thumbnail scale the failure is invisible. Blurring the top slice of every
    detected person box covers heads whichever way they face. The confidence
    floor is far below what detection would use, so a spurious box costs a
    blurred patch of pavement and a missed one costs a published face.
    """
    from ultralytics import YOLO
    res = YOLO(str(WEIGHTS)).predict(img, classes=[0], conf=conf, verbose=False)[0]
    out, n = img.copy(), 0
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        hy2 = y1 + max(12, int(0.24 * (y2 - y1)))
        x1, y1 = max(0, x1), max(0, y1)
        x2, hy2 = min(out.shape[1], x2), min(out.shape[0], hy2)
        if x2 - x1 < 4 or hy2 - y1 < 4:
            continue
        m = max(4, int(0.12 * (x2 - x1)))
        ex1, ey1 = max(0, x1 - m), max(0, y1 - m)
        ex2, ey2 = min(out.shape[1], x2 + m), min(out.shape[0], hy2 + m)
        roi = out[ey1:ey2, ex1:ex2]
        k = max(11, (min(roi.shape[:2]) // 2) * 2 + 1)
        mask = cv2.GaussianBlur(np.ones(roi.shape[:2], np.float32),
                                (max(3, (m // 2) * 2 + 1),) * 2, 0)[..., None]
        out[ey1:ey2, ex1:ex2] = (cv2.GaussianBlur(roi, (k, k), 0) * mask
                                 + roi * (1.0 - mask)).astype(np.uint8)
        n += 1
    return out, n


def main():
    ann = pd.read_pickle(ANN)
    trk = ann[(ann.video_id == VIDEO) & (ann.ped_id == PED)].set_index("frame")
    rows = [trk.loc[f] for f in FRAMES]
    assert all(f < CP - 30 for f in FRAMES), "a thumbnail reaches the event"

    # One crop rectangle for all three frames, so the pedestrian moves inside it.
    x0 = int(min(r.x1 for r in rows)) - PAD_X
    x1 = int(max(r.x2 for r in rows)) + PAD_X
    y0 = int(min(r.y1 for r in rows)) - PAD_Y
    y1 = int(max(r.y2 for r in rows)) + PAD_Y
    print(f"crop {x1 - x0} x {y1 - y0} at ({x0}, {y0})")

    cap = cv2.VideoCapture(str(CLIP))
    for i, (frame, r) in enumerate(zip(FRAMES, rows), start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
        ok, img = cap.read()
        assert ok, f"could not read frame {frame} of {CLIP}"
        img, n = blur_heads(img)
        cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)),
                      ACCENT, 3)
        crop = img[y0:y1, x0:x1]
        # The clip is shot at night and goes to mud at thumbnail size. One
        # gamma lift, identical for all three, applied after the box is drawn
        # so the annotation keeps its colour. Illustrative crops only; no
        # measurement in the paper is taken from these pixels.
        crop = np.clip(((crop / 255.0) ** GAMMA) * 255.0, 0, 255).astype(np.uint8)
        scale = OUT_W / crop.shape[1]
        crop = cv2.resize(crop, (OUT_W, int(round(crop.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
        out = HERE / f"_track{i}.png"
        cv2.imwrite(str(out), crop)
        print(f"frame {frame} (CP-{CP - frame}), {n} heads blurred -> {out.name}"
              f"  {crop.shape[1]}x{crop.shape[0]}")
    cap.release()


if __name__ == "__main__":
    main()

"""make_fig9_qualitative.py — Figure 9: the system running on real video.

Two panels, both correct calls, both verified against PIE's ground-truth crossing
labels rather than chosen by eye:

  (a) video_0016 frame 4345, pedestrian 3_16_942. A genuine crosser, flagged at
      p = 0.71 a second and a half before the annotated crossing point. This is
      the leakage-free protocol made visible: the model commits while the
      pedestrian is still on the kerb.

      Chosen on tracking quality as much as on looks. ByteTrack holds this
      identity for 88% of the 16-frame window; the alternative scenes in
      video_0012 sit at 62%, meaning the buffer that produced their probability
      was over a third some other person. A panel captioned "boxes come from the
      detector and tracker" should show a window the tracker actually kept
      together. The one candidate with better purity still (94%, video_0012
      f6889) puts a large unrelated pedestrian in the foreground, where the eye
      lands on them instead of on the subject.

  (b) video_0012 frame 523, pedestrian 3_12_725. A hard negative, a worker
      standing at the kerb, at a crosswalk, who does not cross. Correctly left
      unflagged at p = 0.31.

Other people are visible in both panels and carry no box. The panel boxes its own
subject; it does not claim to annotate every person on screen. In (a) the large
figure to the right has already begun crossing at this instant and so falls
outside the observation protocol entirely, which is precisely the mid-crossing
case Section 3.3 excludes.

Boxes come from YOLO26 + ByteTrack, not from annotations, so this is the
deployed path end to end. Probabilities are the five-seed BiLSTM-F1 ensemble at
tau* = 0.5164, the same predictor Table 3 reports.

PRIVACY. Every person the detector finds anywhere in the source frame has their
head region blurred before cropping, whether or not they end up visible in the
panel. The blur is applied to the pixels, not drawn over them, so it cannot be
undone from the published file.

Inputs (produced by pipeline/11_demo_clean_ensemble.py):
    pipeline/demo_out_clean/A3/raw/video_0016_f04345.png
    pipeline/demo_out_clean/altB/raw/video_0012_f00523.png
    ... and the matching *_predictions.csv for the box and probability.
Candidate scenes were ranked, not eyeballed; see QUALITATIVE_FIGURE_PLAN.md.

Run:  python make_fig9_qualitative.py  ->  fig9_qualitative.{pdf,png}
"""

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from figstyle import BILSTM, CONTEXT, INK, INK_2, MUTED, use_style

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEMO = ROOT / "pipeline" / "demo_out_clean"

TAU = 0.5164303779602051

PANELS = [
    dict(tag="a", frame=4345, ped="3_16_942", video="video_0016",
         raw=DEMO / "A3" / "raw" / "video_0016_f04345.png",
         csv=DEMO / "A3" / "demo_video_0016_predictions.csv",
         truth=1, lead_s=1.47, gamma=0.95, crop_scale=1.85,
         title="Correctly anticipates a crossing",
         note="Flagged 1.5 s before this pedestrian stepped into the road."),
    dict(tag="b", frame=523, ped="3_12_725", video="video_0012",
         raw=DEMO / "altB" / "raw" / "video_0012_f00523.png",
         csv=DEMO / "altB" / "demo_video_0012_predictions.csv",
         truth=0, lead_s=None, gamma=0.78, crop_scale=2.15,
         title="Correctly ignores a bystander",
         note="Standing at the kerb by a crosswalk, but not about to cross."),
]

CROP_AR = 4 / 3       # panel aspect ratio, applied to both


# --------------------------------------------------------------------- privacy
def blur_heads(img, conf=0.05):
    """Blur the head region of every person the detector finds in the frame.

    Deliberately not a face detector: a face detector that misses a profile or a
    turned head fails silently, and the failure is invisible at figure scale.
    Blurring the top slice of every detected person box covers heads whichever
    way they are facing, and covers people in the background too.

    The confidence floor is set far below what detection would use. A spurious
    box costs a blurred patch of pavement; a missed one costs a published face.
    """
    from ultralytics import YOLO
    # Absolute path on purpose. A bare filename makes ultralytics re-download a
    # 42 MB copy into whatever directory the script happens to be run from.
    model = YOLO(str(ROOT / "yolo26m.pt"))
    res = model.predict(img, classes=[0], conf=conf, verbose=False)[0]
    out, n = img.copy(), 0
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        h = y2 - y1
        hy2 = y1 + max(12, int(0.24 * h))            # top 24%: head with margin,
                                             # a head is ~1/7 of body height
        x1, y1 = max(0, x1), max(0, y1)
        x2, hy2 = min(out.shape[1], x2), min(out.shape[0], hy2)
        if x2 - x1 < 4 or hy2 - y1 < 4:
            continue
        # Blur a slightly larger patch than the head, then feather the blur back
        # to sharp across that added margin only. A hard-edged rectangle reads as
        # a censor bar and drags the eye to whichever person is nearest the
        # camera, which in a figure about a different pedestrian is exactly wrong.
        # The margin is what feathers; the head box itself stays fully blurred.
        m = max(4, int(0.12 * (x2 - x1)))
        ex1, ey1 = max(0, x1 - m), max(0, y1 - m)
        ex2, ey2 = min(out.shape[1], x2 + m), min(out.shape[0], hy2 + m)
        roi = out[ey1:ey2, ex1:ex2]
        k = max(11, (min(roi.shape[:2]) // 2) * 2 + 1)   # kernel scales with the head
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        mask = np.ones(roi.shape[:2], np.float32)
        fk = max(3, (m // 2) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (fk, fk), 0)[..., None]
        out[ey1:ey2, ex1:ex2] = (blurred * mask + roi * (1.0 - mask)).astype(np.uint8)
        n += 1
    print(f"  blurred {n} head regions")
    return out


# ------------------------------------------------------------------ box lookup
def target_box(panel):
    """The tracked box for this panel's pedestrian, matched to the annotation by IoU."""
    import pandas as pd
    df = pd.read_pickle(ROOT / "pie_annotations.pkl")
    g = df[(df.set_id == "set03") & (df.video_id == panel["video"])
           & (df.frame == panel["frame"]) & (df.ped_id == panel["ped"])].iloc[0]
    gt = [g.x1, g.y1, g.x2, g.y2]

    def iou(a, b):
        xa, ya = max(a[0], b[0]), max(a[1], b[1])
        xb, yb = min(a[2], b[2]), min(a[3], b[3])
        i = max(0, xb - xa) * max(0, yb - ya)
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
        return i / u if u > 0 else 0

    best, bi = None, 0.0
    with open(panel["csv"]) as fh:
        for r in csv.DictReader(fh):
            if int(r["frame"]) != panel["frame"]:
                continue
            pb = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
            v = iou(gt, pb)
            if v > bi:
                bi, best = v, (pb, float(r["prob_cross"]), r["track_id"])
    assert best is not None and bi > 0.5, f"{panel['ped']}: no tracked box (best IoU {bi:.2f})"
    box, prob, tid = best
    pred = int(prob >= TAU)
    assert pred == panel["truth"], \
        f"{panel['ped']}: model says {pred}, ground truth is {panel['truth']} — this panel is not a success"
    print(f"  {panel['ped']}: track {tid}, IoU {bi:.2f}, p={prob:.3f} -> "
          f"{'will cross' if pred else 'will not cross'} (truth matches)")
    return box, prob


def crop_window(box, W, H, scale):
    """A crop centred on the subject, same aspect ratio for both panels."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    ch = min(H, (box[3] - box[1]) * scale)
    cw = min(W, ch * CROP_AR)
    ch = cw / CROP_AR
    x0 = int(np.clip(cx - cw / 2, 0, W - cw))
    y0 = int(np.clip(cy - ch / 2, 0, H - ch))
    return x0, y0, int(cw), int(ch)


# ---------------------------------------------------------------------- figure
def main():
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45))

    for ax, panel in zip(axes, PANELS):
        print(f"panel ({panel['tag']}) {panel['video']} f{panel['frame']}")
        img = cv2.imread(str(panel["raw"]))
        assert img is not None, f"missing frame {panel['raw']} — run 11_demo_clean_ensemble.py first"
        box, prob = target_box(panel)
        img = blur_heads(img)

        H, W = img.shape[:2]
        x0, y0, cw, ch = crop_window(box, W, H, panel['crop_scale'])
        crop = cv2.cvtColor(img[y0:y0 + ch, x0:x0 + cw], cv2.COLOR_BGR2RGB)
        if panel["gamma"] != 1.0:
            # Tone lift only; the scene is backlit and prints almost black otherwise.
            crop = np.clip(((crop / 255.0) ** panel["gamma"]) * 255.0, 0, 255).astype(np.uint8)
        ax.imshow(crop)

        crosses = prob >= TAU
        color = BILSTM if crosses else CONTEXT
        bx, by = box[0] - x0, box[1] - y0
        bw, bh = box[2] - box[0], box[3] - box[1]
        ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor=color,
                               linewidth=2.0, zorder=4))

        # Label above the box. Text sits in ink on a solid plate so it stays
        # readable over whatever the street happens to be doing behind it.
        txt = f"{'will cross' if crosses else 'will not cross'}   $p$ = {prob:.2f}"
        lx = min(bx, cw - 0.42 * cw)
        ax.text(lx, by - 0.035 * ch, txt, fontsize=8.4, color=INK,
                va="bottom", ha="left", zorder=6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                          edgecolor=color, linewidth=1.2, alpha=0.95))

        ax.set_xlim(0, cw); ax.set_ylim(ch, 0)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d8d7d0"); s.set_linewidth(0.8)

        ax.text(0, -0.055, f"({panel['tag']})  {panel['title']}",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color=INK, va="top")
        ax.text(0, -0.145, panel["note"], transform=ax.transAxes,
                fontsize=8, color=MUTED, va="top")
        ax.text(0, -0.235,
                f"{panel['video'].replace('_', ' ')}, frame {panel['frame']}   ·   "
                f"ground truth: {'crosses' if panel['truth'] else 'does not cross'}",
                transform=ax.transAxes, fontsize=7.6, color=INK_2, va="top")

    fig.subplots_adjust(left=0.008, right=0.992, top=0.995, bottom=0.155, wspace=0.045)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = HERE / f"fig9_qualitative.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

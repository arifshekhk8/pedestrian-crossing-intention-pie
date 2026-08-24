"""make_fig_scenario.py — the opening figure: the judgement, in time and distance.

Three real PIE frames from one crossing, plus the distance the ego vehicle
covered between them, taken from the recorded speed rather than assumed.

  video_0016, pedestrian 3_16_919, crossing_point frame 2567 (PIE annotation).
  Panels at t = -1.5 s, -0.5 s and +0.5 s relative to that event, 30 fps.

Why this pedestrian. It is one of only six annotated crossers in the two set03
clips held locally whose ego speed is non-zero at the crossing point; the rest
cross in front of a vehicle already stopped at a signal, which is not the
scenario the Introduction describes. Of the six it has the clearest view and the
largest subject. Nothing was chosen by eye beyond that.

The point the panels make is the one the text makes: at -1.5 s the subject is
one of many people standing on that footway and nothing distinguishes them from
the ones who stay put. The boxes are PIE's own annotations, not detections; the
detector-in-the-loop figure later in the paper is the one that shows estimated
boxes.

Distances are integrated from the per-frame `vehicle_speed` channel of
pie_annotations.pkl, so the metre values are measured, not modelled.

PRIVACY. Head regions of every person the detector finds are blurred in the
pixels before cropping, using the same routine as the qualitative figure.

Run from anywhere:  python make_fig_scenario.py  ->  ../figures/fig_scenario.{pdf,png}
"""

import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from figstyle import ACCENT, CONTEXT, INK, INK_2, MUTED, RULE, use_style

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # repo root
CLIP = ROOT / "PIE_clips" / "set03" / "video_0016.mp4"
ANN = ROOT / "pie_annotations.pkl"
WEIGHTS = ROOT / "yolo26m.pt"

VIDEO, PED, CP, FPS = "video_0016", "3_16_919", 2567, 30.0
CROP_W, CROP_H = 760, 620                    # pixels in the source frame
PANELS = [
    dict(dt=-1.5, tag="a", title="1.5 s before the step",
         note="One of several people on the footway."),
    dict(dt=-0.5, tag="b", title="0.5 s before the step",
         note="Still on the kerb, facing the road."),
    dict(dt=+0.5, tag="c", title="0.5 s after the step",
         note="In the roadway, and now too late."),
]


def blur_heads(img, conf=0.05):
    """Blur the head region of every person the detector finds in the frame.

    Not a face detector: one that misses a turned head fails silently, and the
    failure is invisible at figure scale. Blurring the top slice of every
    detected person box covers heads whichever way they face, background
    included. The confidence floor is far below what detection would use, so a
    spurious box costs a blurred patch of pavement and a missed one costs a face.
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
        # Feather the blur across an added margin so it does not read as a
        # censor bar and pull the eye off the subject.
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
    print(f"  blurred {n} head regions")
    return out


def main():
    ann = pd.read_pickle(ANN)
    trk = ann[(ann.video_id == VIDEO) & (ann.ped_id == PED)].sort_values("frame")

    f0 = CP + int(PANELS[0]["dt"] * FPS)
    span = trk[(trk.frame >= f0) & (trk.frame <= CP + int(PANELS[-1]["dt"] * FPS))].copy()
    span["d"] = (span.vehicle_speed / 3.6 / FPS).cumsum()
    d0 = span[span.frame == f0].iloc[0].d

    def at(frame):
        r = span[span.frame == frame].iloc[0]
        return r, r.d - d0

    use_style()
    fig = plt.figure(figsize=(7.1, 3.30))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.30], hspace=0.42, wspace=0.035,
                          left=0.008, right=0.992, top=0.995, bottom=0.045)

    cap = cv2.VideoCapture(str(CLIP))
    for col, p in enumerate(PANELS):
        frame = CP + int(p["dt"] * FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
        ok, img = cap.read()
        assert ok, f"could not read frame {frame} of {CLIP}"
        print(f"panel ({p['tag']}) frame {frame}")
        img = blur_heads(img)

        r, _ = at(frame)
        cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
        x0 = int(np.clip(cx - CROP_W / 2, 0, img.shape[1] - CROP_W))
        y0 = int(np.clip(cy - CROP_H / 2 + 40, 0, img.shape[0] - CROP_H))
        crop = cv2.cvtColor(img[y0:y0 + CROP_H, x0:x0 + CROP_W], cv2.COLOR_BGR2RGB)

        ax = fig.add_subplot(gs[0, col])
        ax.imshow(crop)
        ax.add_patch(Rectangle((r.x1 - x0, r.y1 - y0), r.x2 - r.x1, r.y2 - r.y1,
                               fill=False, edgecolor=ACCENT, linewidth=1.8, zorder=4))
        ax.text(r.x1 - x0, r.y1 - y0 - 0.028 * CROP_H,
                f"{r.vehicle_speed:.0f} km/h", fontsize=8.0, color=INK,
                va="bottom", ha="left", zorder=6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor=ACCENT, linewidth=1.1, alpha=0.95))
        ax.set_xlim(0, CROP_W); ax.set_ylim(CROP_H, 0)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d8d7d0"); s.set_linewidth(0.8)
        ax.text(0, -0.055, f"({p['tag']})  {p['title']}", transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=INK, va="top")
        ax.text(0, -0.150, textwrap.fill(p["note"], 40), transform=ax.transAxes,
                fontsize=8.0, color=MUTED, va="top", linespacing=1.35)

    cap.release()

    # ---------------------------------------------------------- distance ruler
    _, dB = at(CP + int(PANELS[1]["dt"] * FPS))
    _, dCP = at(CP)
    _, dC = at(CP + int(PANELS[-1]["dt"] * FPS))

    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(-0.6, dC + 0.6); ax.set_ylim(-1.15, 1.30)
    ax.axis("off")
    # the half second that separates panel (b) from the step itself
    ax.add_patch(Rectangle((dB, -0.26), dCP - dB, 0.52, facecolor="#dce8f8",
                           edgecolor="none", zorder=1))
    ax.plot([0, dC], [0, 0], color=RULE, lw=1.0, zorder=2)
    for x, tag in ((0, "(a)"), (dB, "(b)"), (dC, "(c)")):
        ax.plot([x], [0], marker="|", ms=10, mew=1.6, color=INK, zorder=3)
        ax.text(x, 0.30, f"{x:.1f} m", ha="center", va="bottom",
                fontsize=8, color=INK_2, zorder=3)
        ax.text(x, -0.40, tag, ha="center", va="top", fontsize=7.8,
                color=INK_2, zorder=3)
    ax.plot([dCP], [0], marker="|", ms=10, mew=1.8, color=ACCENT, zorder=3)
    ax.text(dCP, -0.40, "steps into\nthe road", ha="center", va="top",
            fontsize=7.8, color=ACCENT, fontweight="bold", zorder=3,
            linespacing=1.3)
    ax.annotate("", xy=(dB, 0.80), xytext=(dCP, 0.80),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.0))
    ax.text((dB + dCP) / 2, 0.94,
            f"half a second later = {dCP - dB:.1f} m less road to stop in",
            ha="center", va="bottom", fontsize=8, color=INK, fontweight="bold")
    ax.text(0, -1.02, "Distance covered by the vehicle, integrated from its recorded speed.",
            ha="left", va="top", fontsize=7.6, color=MUTED)

    outdir = HERE.parent / "figures"
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = outdir / f"fig_scenario.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

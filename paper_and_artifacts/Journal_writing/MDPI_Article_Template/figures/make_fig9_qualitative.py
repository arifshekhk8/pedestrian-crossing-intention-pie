"""make_fig9_qualitative.py — Figure 9: the system running on real video.

A 2x3 grid of test-set street scenes. Every pedestrian the system is currently
predicting for carries a box and a verdict, so a panel shows several simultaneous
calls rather than one call in isolation.

WHY THIS REPLACED THE TWO-PANEL VERSION
---------------------------------------
The 2026-07-28 figure was two tightly cropped panels, one pedestrian boxed in
each: one "will cross", one "will not cross". It showed that the model can be
right twice. It could not show the thing a reader actually wants to check, which
is what the system does with a whole street at once — several people, some of
them about to cross and some not, all scored from the same frame. The grid shows
that, and it lets the figure carry a failure case without giving up any of the
successes.

WHAT EACH BOX MEANS, AND WHAT CAN BE CHECKED
--------------------------------------------
The model emits a probability for any ByteTrack identity with 16 frames buffered,
including people PIE never annotated. Ground truth exists only for annotated
pedestrians, and only where a clean-protocol window is anchored near the frame
shown. Boxes therefore fall into two classes and the figure keeps them apart:

  verified   the box IoU-matches an annotated pedestrian with a protocol window
             anchored within +-1 s, so the verdict is scored right or wrong and
             carries a check or a cross;
  unmarked   a real prediction with no nearby ground truth, shown as output only.

The +-1 s rule comes from the 2026-07-28 post-mortem: judging a pedestrian by a
window anchored seconds away is not a fair test, and doing it once already caused
a good scene to be rejected for a miss that was not one.

SCENE SELECTION
---------------
Ranked by `pipeline/11c_select_grid_scenes.py` over sixteen candidate segments,
not chosen by eye, on: simultaneous live predictions, whether the verdicts differ,
box height after cropping (a box under about a fifth of the panel height is
unreadable at 5.5 cm), and how many boxes ground truth can score. The shortlist
was then rendered and looked at, because the 2026-07-28 run showed a metric that
is invisible in the image cannot outvote what the image shows.

Panel (f) is a failure, and it is in the figure on purpose. It is the kerb cluster
at video_0016 frames 4659-4712, where the ensemble returns 14 false positives over
16 windows with the non-crossers scoring 0.49-0.63 against a threshold of 0.516 —
uncertain rather than wrong, and the cheap error rather than the expensive one.
A hand-picked all-successes figure in a paper whose contribution is a leakage audit
invites exactly the scepticism the audit is meant to disarm.

One caveat the caption states: only two PIE clips are downloaded locally, both from
the test split, so panels (a), (b) and (f) come from the same long intersection
approach in video_0016 at different times.

Boxes come from YOLO26 + ByteTrack, not from annotations, so this is the deployed
path end to end. Probabilities are the five-seed BiLSTM-F1 ensemble at
tau* = 0.5164, the same predictor Table 3 reports.

PRIVACY. Every person the detector finds anywhere in the source frame has their
head region blurred before cropping, whether or not they end up visible in the
panel. The blur is applied to the pixels, not drawn over them, so it cannot be
undone from the published file.

Inputs (produced by pipeline/11_demo_clean_ensemble.py, one folder per segment):
    pipeline/demo_out_grid/<seg>/demo_<video>_predictions.csv
Frames are read straight from the clips, since a raw frame is just a video frame.

Run:  python make_fig9_qualitative.py  ->  fig9_qualitative.{pdf,png}
"""

import pickle
import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from figstyle import BILSTM, CLEAN, CONTEXT, INK, INK_2, LEAK, MUTED, use_style

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GRID = ROOT / "pipeline" / "demo_out_grid"
CLEAN_SEQ = ROOT / "journal_prep" / "issue2_clean_protocol" / "sequences_clean"

TAU = 0.5164303779602051
VERIFY_TOL = 30       # frames: a window must be anchored within 1 s to score a box
IOU_MATCH = 0.5
CROP_AR = 3 / 2
MIN_CROP_W = 760      # source px; below this the panel is upscaled and goes soft

# `crop` is (x0, y0, width) in source pixels, height following from CROP_AR. Each
# was found by searching crop windows that hold the most boxes at the largest box
# size while still containing a ground-truth-scoreable pedestrian; a box that the
# crop would cut through is dropped rather than drawn broken.
#
# `expect` is the correctness of every ground-truth-verifiable box in the panel.
# Asserted at render time, so the figure cannot silently drift into showing a
# failure where it claims a success, or the reverse, if a checkpoint ever changes.
PANELS = [
    dict(tag="a", seg="s_16_2746", video="video_0016", frame=2740, expect="ok",
         crop=(40, 146, 1400), gamma=1.0, lead=[1.40],
         title="A group crosses in front",
         note="Nine calls from one frame. The checked crosser is flagged 1.4 s early."),
    dict(tag="b", seg="s_16_4345", video="video_0016", frame=4324, expect="ok",
         crop=(1120, 460, 760), gamma=0.95, lead=[2.17],
         title="Flagged before the step",
         note="The checked pedestrian is flagged 2.2 s before stepping into the road."),
    dict(tag="c", seg="s_12_5869", video="video_0012", frame=5864, expect="ok",
         crop=(80, 200, 1050), gamma=0.92, lead=[1.33, 1.73],
         title="Two crossers at a corner",
         note="Both are flagged, 1.3 s and 1.7 s before they stepped out."),
    dict(tag="d", seg="s_12_15869", video="video_0012", frame=15867, expect="ok",
         crop=(1160, 380, 760), gamma=0.90, lead=[],
         title="A busy footway, nobody flagged",
         note="People moving along the road rather than into it, cleared at 22 km/h."),
    dict(tag="e", seg="s_12_523", video="video_0012", frame=499, expect="ok",
         crop=(1160, 380, 760), gamma=0.88, lead=[],
         title="Two verdicts in one frame",
         note="A worker at the kerb is cleared; a pedestrian beside them is flagged."),
    dict(tag="f", seg="s_16_4691", video="video_0016", frame=4710, expect="err",
         crop=(0, 360, 760), gamma=1.0, lead=[],
         title="Where it fails: waiting at a kerb",
         note="Four people waiting are all flagged. The checked one never crosses."),
]


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
        hy2 = y1 + max(12, int(0.24 * h))        # top 24%: head with margin,
                                                 # a head is ~1/7 of body height
        x1, y1 = max(0, x1), max(0, y1)
        x2, hy2 = min(out.shape[1], x2), min(out.shape[0], hy2)
        if x2 - x1 < 4 or hy2 - y1 < 4:
            continue
        # Blur a slightly larger patch than the head, then feather the blur back
        # to sharp across that added margin only. A hard-edged rectangle reads as
        # a censor bar and drags the eye to whichever person is nearest the
        # camera, which in a figure about a different pedestrian is exactly wrong.
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


# ---------------------------------------------------------------- ground truth
def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, xb - xa) * max(0.0, yb - ya)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load_truth():
    ann = pd.read_pickle(ROOT / "pie_annotations.pkl")
    ann = ann[ann.set_id == "set03"][["video_id", "ped_id", "frame", "x1", "y1", "x2", "y2"]]
    meta = pickle.load(open(CLEAN_SEQ / "meta.pkl", "rb"))
    y = np.load(CLEAN_SEQ / "y.npy")
    win = pd.DataFrame([
        dict(video_id=m["video_id"], ped_id=m["ped_id"], anchor=int(m["anchor_frame"]),
             cp=int(m["crossing_point"]), label=int(lab))
        for m, lab in zip(meta, y) if m["set_id"] == "set03"
    ])
    return ann, win


def read_panel(panel, ann, win):
    """Every live prediction in the frame, each tagged with ground truth if scoreable."""
    df = pd.read_csv(GRID / panel["seg"] / f"demo_{panel['video']}_predictions.csv")
    g = df[df.frame == panel["frame"]]
    assert len(g), f"{panel['seg']}: no predictions at frame {panel['frame']}"
    gt_here = ann[(ann.video_id == panel["video"]) & (ann.frame == panel["frame"])]
    w_vid = win[win.video_id == panel["video"]]

    boxes = []
    for _, r in g.iterrows():
        box = [r.x1, r.y1, r.x2, r.y2]
        ped, bi = None, IOU_MATCH
        for _, t in gt_here.iterrows():
            v = iou(box, [t.x1, t.y1, t.x2, t.y2])
            if v > bi:
                bi, ped = v, t.ped_id
        rec = dict(box=box, prob=float(r.prob_cross), ego=float(r.ego_speed),
                   crosses=float(r.prob_cross) >= TAU, ped=ped,
                   truth=None, correct=None, lead_s=None)
        if ped is not None:
            w = w_vid[w_vid.ped_id == ped]
            d = (w.anchor - panel["frame"]).abs()
            if len(w) and d.min() <= VERIFY_TOL:
                row = w.loc[d.idxmin()]
                rec["truth"] = int(row.label)
                rec["correct"] = rec["crosses"] == bool(row.label)
                if row.label:
                    rec["lead_s"] = (int(row.cp) - panel["frame"]) / 30.0
        boxes.append(rec)
    return boxes


def crop_window(panel, W, H):
    """The panel's crop, clipped to the frame."""
    x0, y0, cw = panel["crop"]
    ch = cw / CROP_AR
    assert cw >= MIN_CROP_W, f"panel {panel['tag']}: {cw} px crop would be upscaled into softness"
    assert x0 + cw <= W and y0 + ch <= H, f"panel {panel['tag']}: crop runs off the frame"
    return int(x0), int(y0), int(cw), int(ch)


def inside(boxes, x0, y0, cw, ch, keep=0.85):
    """Boxes the crop shows whole. A box clipped by the panel edge reads as a
    detection failure rather than as framing, so it is dropped instead."""
    out = []
    for b in boxes:
        bx1, by1, bx2, by2 = b["box"]
        ix = max(0.0, min(bx2, x0 + cw) - max(bx1, x0))
        iy = max(0.0, min(by2, y0 + ch) - max(by1, y0))
        area = (bx2 - bx1) * (by2 - by1)
        if area > 0 and (ix * iy) / area >= keep:
            out.append(b)
    return out


def place_labels(boxes, cw, ch):
    """Label positions that do not sit on top of each other.

    Seven boxes in one panel is the point of the figure and also its main drawing
    problem. Each label starts above its box; if it collides with one already
    placed, it drops below the box, and failing that it steps up in fixed
    increments until it is clear.
    """
    lw, lh = 0.105 * cw, 0.055 * ch       # approximate plate size in crop pixels
    placed = []
    for b in sorted(boxes, key=lambda b: b["box"][1]):
        bx, by = b["box"][0], b["box"][1]
        bot = b["box"][3]
        options = [(bx, by - 0.012 * ch)] + \
                  [(bx, bot + 0.012 * ch + lh)] + \
                  [(bx, by - 0.012 * ch - k * lh) for k in range(1, 5)]
        for lx, ly in options:
            lx = float(np.clip(lx, 0, cw - lw))
            if ly - lh < 0 or ly > ch:
                continue
            if all(abs(lx - px) > lw * 0.92 or abs(ly - py) > lh * 0.95
                   for px, py in placed):
                break
        placed.append((lx, ly))
        b["label_xy"] = (lx, ly)
    return boxes


# ---------------------------------------------------------------------- figure
def main():
    use_style()
    ann, win = load_truth()
    caps, report, marks = {}, [], []

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.05))
    # Fixed before drawing, because the ground-truth marks are positioned from
    # measured text extents and those depend on the final axes geometry. The
    # bottom margin holds three lines of panel caption plus the legend beneath.
    fig.subplots_adjust(left=0.006, right=0.994, top=0.995, bottom=0.175,
                        wspace=0.045, hspace=0.40)

    for ax, panel in zip(axes.ravel(), PANELS):
        print(f"panel ({panel['tag']}) {panel['video']} f{panel['frame']}")
        boxes = read_panel(panel, ann, win)
        x0, y0, cw, ch = crop_window(panel, 1920, 1080)
        n_all = len(boxes)
        boxes = inside(boxes, x0, y0, cw, ch)
        assert boxes, f"panel {panel['tag']}: the crop holds no whole box"
        print(f"  {n_all} live predictions in the frame, {len(boxes)} inside the crop")

        ver = [b for b in boxes if b["correct"] is not None]
        assert ver, f"panel {panel['tag']}: no ground-truth-verifiable box"
        if panel["expect"] == "ok":
            assert all(b["correct"] for b in ver), \
                (f"panel {panel['tag']}: expected every verifiable box to be correct, got "
                 + ", ".join(f"{b['ped']} p={b['prob']:.2f} truth={b['truth']}"
                             for b in ver if not b["correct"]))
        else:
            assert any(not b["correct"] for b in ver), \
                f"panel {panel['tag']}: declared a failure panel but every verifiable box is correct"
        # The lead times are quoted in the panel notes, so they are pinned too.
        got_lead = sorted(round(b["lead_s"], 2) for b in ver if b["lead_s"] is not None)
        assert got_lead == sorted(panel["lead"]), \
            f"panel {panel['tag']}: lead times moved, note says {panel['lead']} but got {got_lead}"

        if panel["video"] not in caps:
            caps[panel["video"]] = cv2.VideoCapture(
                str(ROOT / "PIE_clips" / "set03" / f"{panel['video']}.mp4"))
        cap = caps[panel["video"]]
        cap.set(cv2.CAP_PROP_POS_FRAMES, panel["frame"])
        ok, img = cap.read()
        assert ok, f"cannot read {panel['video']} frame {panel['frame']}"
        img = blur_heads(img)

        crop = cv2.cvtColor(img[y0:y0 + ch, x0:x0 + cw], cv2.COLOR_BGR2RGB)
        if panel["gamma"] != 1.0:
            # Tone lift only; several scenes are backlit and print almost black.
            crop = np.clip(((crop / 255.0) ** panel["gamma"]) * 255.0, 0, 255).astype(np.uint8)
        ax.imshow(crop)

        for b in boxes:
            b["box"] = [b["box"][0] - x0, b["box"][1] - y0,
                        b["box"][2] - x0, b["box"][3] - y0]
        place_labels(boxes, cw, ch)

        for b in boxes:
            color = BILSTM if b["crosses"] else CONTEXT
            bx, by, bx2, by2 = b["box"]
            ax.add_patch(Rectangle((bx, by), bx2 - bx, by2 - by, fill=False,
                                   edgecolor=color, linewidth=1.1, zorder=4))
            lx, ly = b["label_xy"]
            txt = f"{'P' if b['crosses'] else 'N'} {b['prob']:.2f}"
            t = ax.text(lx, ly, txt, fontsize=6.3, color=INK, va="bottom", ha="left",
                        zorder=6, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                  edgecolor=color, linewidth=0.7, alpha=0.94))
            if b["correct"] is not None:
                # Drawn as mathtext, since the sans fonts the manuscript uses have
                # no check or ballot glyph and matplotlib silently substitutes a
                # missing-character box. Placed in a second pass, once the label's
                # rendered width is known.
                marks.append((ax, t, b["correct"]))

        ax.set_xlim(0, cw); ax.set_ylim(ch, 0)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d8d7d0"); s.set_linewidth(0.7)

        ego = boxes[0]["ego"]
        # Wrapped to the panel, not to the page: an unwrapped note runs under the
        # neighbouring column and the grid stops reading as six separate scenes.
        note = textwrap.fill(panel["note"], 48)
        ax.text(0, -0.05, f"({panel['tag']})  {panel['title']}", transform=ax.transAxes,
                fontsize=7.8, fontweight="bold", color=INK, va="top")
        ax.text(0, -0.155, note, transform=ax.transAxes,
                fontsize=6.9, color=MUTED, va="top", linespacing=1.35)
        ax.text(0, -0.155 - 0.105 * (note.count("\n") + 1),
                f"{panel['video'].replace('_', ' ')}, frame {panel['frame']}   ·   "
                f"ego {ego:.0f} km/h", transform=ax.transAxes,
                fontsize=6.5, color=INK_2, va="top")

        report.append((panel, boxes))

    for c in caps.values():
        c.release()

    # Second pass: the check or cross sits just past the end of its label, which
    # means measuring the label after it exists.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax, t, correct in marks:
        bb = t.get_window_extent(renderer=rend).transformed(ax.transData.inverted())
        pad = 0.16 * (bb.x1 - bb.x0)
        ax.text(bb.x1 + pad, bb.y0, "$\\checkmark$" if correct else "$\\times$",
                fontsize=7.4, color=CLEAN if correct else LEAK,
                va="bottom", ha="left", zorder=7, fontweight="bold")

    handles = [
        Line2D([], [], color=BILSTM, lw=1.6, label="will cross ($p \\geq \\tau^*$)"),
        Line2D([], [], color=CONTEXT, lw=1.6, label="will not cross"),
        Line2D([], [], color=CLEAN, marker="$\\checkmark$", lw=0, markersize=5.5,
               label="matches ground truth"),
        Line2D([], [], color=LEAK, marker="$\\times$", lw=0, markersize=5.5,
               label="contradicts ground truth"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                     fontsize=7.2, bbox_to_anchor=(0.5, -0.005),
                     handlelength=1.5, columnspacing=1.9, handletextpad=0.5)
    for t in leg.get_texts():
        t.set_color(INK_2)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        out = HERE / f"fig9_qualitative.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {out}")

    # ------------------------------------------------------- numbers for the text
    print("\n--- per-panel record (quote these in Section 4.11) ---")
    n_box = n_ver = n_ok = 0
    for panel, boxes in report:
        v = [b for b in boxes if b["correct"] is not None]
        n_box += len(boxes); n_ver += len(v); n_ok += sum(b["correct"] for b in v)
        lead = [b["lead_s"] for b in v if b["lead_s"] is not None]
        print(f"({panel['tag']}) {panel['video']} f{panel['frame']}: "
              f"{len(boxes)} live ({sum(b['crosses'] for b in boxes)}P/"
              f"{sum(not b['crosses'] for b in boxes)}N), "
              f"{len(v)} verifiable, {sum(b['correct'] for b in v)} correct, "
              f"ego {boxes[0]['ego']:.0f} km/h"
              + (f", lead {min(lead):.2f}-{max(lead):.2f} s" if lead else ""))
        for b in v:
            print(f"      {b['ped']}: p={b['prob']:.3f} truth={b['truth']} "
                  f"{'correct' if b['correct'] else 'WRONG'}"
                  + (f" lead {b['lead_s']:.2f} s" if b["lead_s"] is not None else ""))
    print(f"total: {n_box} boxes, {n_ver} verifiable, {n_ok} correct")


if __name__ == "__main__":
    main()

"""11c_select_grid_scenes.py — rank candidate frames for the multi-panel Figure 9.

Figure 9 was originally two cropped panels, each boxing a single pedestrian. The
replacement is a 2x3 grid in which *every* pedestrian the system is currently
predicting for carries a box and a verdict, which is what a reader wants to see:
several simultaneous calls in one street scene, not one call in isolation.

That change moves the selection problem. `11a_select_demo_scenes.py` ranked
individual pedestrians on their own confidence and tracking quality. Here the unit
is the frame, and the criteria are how many live predictions it carries, whether
those verdicts differ, whether the boxes survive being printed at ~5.7 cm wide, and
how much of it we can check against ground truth.

WHAT CAN AND CANNOT BE VERIFIED
-------------------------------
The model emits a probability for any track that has 16 frames buffered, including
people PIE never annotated. Ground truth exists only for annotated pedestrians, and
only where a clean-protocol window is anchored near the frame in question. So each
box falls into one of two classes, and the figure must not blur them:

  verifiable  the box IoU-matches an annotated pedestrian that has a protocol
              window anchored within +-VERIFY_TOL frames, so the verdict can be
              scored right or wrong;
  live only   a real prediction with no nearby ground truth, shown as output.

The +-1 s rule is inherited from the 2026-07-28 selection post-mortem: judging a
pedestrian at a frame by a window anchored seconds away is not a fair test, and
doing it once already caused a good scene to be rejected for a miss that was not
one.

Reads the per-segment CSVs written by 11_demo_clean_ensemble.py under
pipeline/demo_out_grid/, plus pie_annotations.pkl and the clean-protocol metadata.
Writes a ranked table; renders nothing.

Run:  python pipeline/11c_select_grid_scenes.py [--min-live 3] [--top 30]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "pipeline" / "demo_out_grid"
CLEAN = ROOT / "journal_prep" / "issue2_clean_protocol" / "sequences_clean"

TAU = 0.5164303779602051
VERIFY_TOL = 30      # frames; a window must be anchored within 1 s to score a box
IOU_MATCH = 0.5      # a detection is "the same person" as an annotation above this


def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, xb - xa) * max(0.0, yb - ya)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load_truth():
    """Per (video, ped, frame): the annotated box, and the label if a window is near."""
    ann = pd.read_pickle(ROOT / "pie_annotations.pkl")
    ann = ann[ann.set_id == "set03"][["video_id", "ped_id", "frame", "x1", "y1", "x2", "y2"]]

    meta = pickle.load(open(CLEAN / "meta.pkl", "rb"))
    y = np.load(CLEAN / "y.npy")
    win = pd.DataFrame([
        dict(video_id=m["video_id"], ped_id=m["ped_id"], anchor=int(m["anchor_frame"]),
             cp=int(m["crossing_point"]), tte=int(m["tte"]), label=int(lab))
        for m, lab in zip(meta, y) if m["set_id"] == "set03"
    ])
    return ann, win


def nearest_window(win, video, ped, frame):
    """The clean-protocol window anchored closest to `frame`, if within VERIFY_TOL."""
    w = win[(win.video_id == video) & (win.ped_id == ped)]
    if w.empty:
        return None
    d = (w.anchor - frame).abs()
    if d.min() > VERIFY_TOL:
        return None
    return w.loc[d.idxmin()]


def score_frames(min_live):
    ann, win = load_truth()
    out = []
    for seg in sorted(GRID.iterdir()):
        if not seg.is_dir():
            continue
        csvs = list(seg.glob("demo_*_predictions.csv"))
        if not csvs:
            continue
        video = csvs[0].name.replace("demo_", "").replace("_predictions.csv", "")
        df = pd.read_csv(csvs[0])
        a_v = ann[ann.video_id == video]

        for frame, g in df.groupby("frame"):
            if len(g) < min_live:
                continue
            gt_here = a_v[a_v.frame == frame]
            n_ok = n_bad = n_ver = 0
            peds = []
            for _, r in g.iterrows():
                box = [r.x1, r.y1, r.x2, r.y2]
                best, bi = None, IOU_MATCH
                for _, t in gt_here.iterrows():
                    v = iou(box, [t.x1, t.y1, t.x2, t.y2])
                    if v > bi:
                        bi, best = v, t.ped_id
                if best is None:
                    continue
                w = nearest_window(win, video, best, frame)
                if w is None:
                    continue
                n_ver += 1
                correct = int(r.prob_cross >= TAU) == int(w.label)
                n_ok += correct
                n_bad += not correct
                peds.append(f"{best}{'+' if w.label else '-'}{'ok' if correct else 'ERR'}")

            h = (g.y2 - g.y1)
            out.append(dict(
                seg=seg.name, video=video, frame=int(frame), n_live=len(g),
                n_P=int((g.prob_cross >= TAU).sum()), n_N=int((g.prob_cross < TAU).sum()),
                med_h=round(h.median(), 1), min_h=round(h.min(), 1), max_h=round(h.max(), 1),
                ego=round(g.ego_speed.iloc[0], 1),
                n_ver=n_ver, n_ok=n_ok, n_bad=n_bad, peds=" ".join(peds),
            ))
    d = pd.DataFrame(out)
    d["mixed"] = (d.n_P > 0) & (d.n_N > 0)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-live", type=int, default=3,
                    help="minimum simultaneous live predictions in the frame")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-med-h", type=float, default=110.0,
                    help="median box height in px; below this the panel is unreadable in print")
    args = ap.parse_args()

    d = score_frames(args.min_live)
    print(f"{len(d)} frames with >= {args.min_live} live predictions "
          f"across {d.seg.nunique()} segments")
    print(f"  mixed verdicts: {int(d.mixed.sum())}   "
          f"with >=1 ground-truth-verifiable box: {int((d.n_ver > 0).sum())}   "
          f"with >=2: {int((d.n_ver > 1).sum())}")

    legible = d[d.med_h >= args.min_med_h]
    print(f"  legible (median box >= {args.min_med_h:.0f} px): {len(legible)}")

    # One row per segment, so the shortlist spans scenes rather than 60 views of one.
    rank = legible.sort_values(["n_ver", "mixed", "n_live", "med_h"], ascending=False)
    best = rank.groupby("seg").head(3).sort_values(
        ["n_ver", "mixed", "n_live", "med_h"], ascending=False).head(args.top)
    pd.set_option("display.width", 250)
    print("\nshortlist (one scene per segment, best 3 frames each):")
    print(best[["seg", "frame", "n_live", "n_P", "n_N", "mixed", "med_h", "min_h",
                "ego", "n_ver", "n_ok", "n_bad", "peds"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""
11a_select_demo_scenes.py — pick the qualitative-figure scenes on evidence.

The one thing a hand-picked figure must not be is hand-picked by eye. This
script ranks every candidate frame in the two locally available set03 clips
against explicit criteria and prints a shortlist, so the final choice is made
from a table.

What it joins together
----------------------
  clean windows + labels   journal_prep/issue2_clean_protocol/sequences_clean/
                           (via the unified engine's load_splits, so the feature
                           order matches training exactly)
  crossing_point / tte     the same folder's meta.pkl
  BiLSTM-F1 probabilities  scored here, live, through the 5-seed ensemble
                           (NOT the gt_prob column in the issue-10 CSV, which
                           came from the clean h128 baseline, a different model)
  detection quality        journal_prep/issue10_gt_vs_detector/10_gt_vs_detector.csv
                           (detected, miou, purity, frag)

Panels it looks for
-------------------
  A  a crosser correctly flagged well before onset
  B  a non-crosser near the kerb correctly left unflagged
  C  ONE frame holding both, which is the panel worth having

Run:  python pipeline/11a_select_demo_scenes.py
"""

import csv
import importlib.util
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CLIPS = {"video_0012", "video_0016"}          # the only clips present locally
TAU = 0.5164303779602051
MIN_TTE = 30                                   # frames; >= 1 s of look-ahead
GOOD_IOU = 0.65                                # detector must actually find them
GOOD_PURITY = 0.50                             # ByteTrack must hold the identity


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    demo = _load("demo", ROOT / "pipeline" / "11_demo_clean_ensemble.py")
    eng = _load("eng", ROOT / "journal_prep" / "issue12_unified_pipeline" / "12_unified_engine.py")

    *_, X_test, y_test = eng.load_splits()
    y_test = y_test.astype(int)
    meta_all = pickle.load(
        open(ROOT / "journal_prep" / "issue2_clean_protocol" / "sequences_clean" / "meta.pkl", "rb"))
    meta_test = [m for m in meta_all if m["set_id"] == "set03"]
    assert len(meta_test) == len(y_test), f"{len(meta_test)} meta vs {len(y_test)} labels"

    device = torch.device("cpu")
    models, mean, std = demo.load_ensemble(
        [Path(d) for d in demo.default_weights_dirs().split(",")], demo.F1_HIDDEN, device)
    probs = demo.predict_batch(models, X_test, mean, std, device)
    print(f"[scenes] scored {len(probs)} set03 windows through the BiLSTM-F1 ensemble\n")

    # ---- detection quality from the issue-10 study, keyed (video, ped, anchor)
    detq = {}
    with open(ROOT / "journal_prep" / "issue10_gt_vs_detector" / "10_gt_vs_detector.csv") as fh:
        for r in csv.DictReader(fh):
            key = (r["video"], r["ped"], int(r["anchor"]))
            detq[key] = {
                "detected": int(r["detected"]),
                "miou": float(r["miou"]) if r["miou"] else 0.0,
                "purity": float(r["purity"]) if r["purity"] else 0.0,
                "frag": int(r["frag"]) if r["frag"] else 0,
            }

    rows = []
    for i, m in enumerate(meta_test):
        if m["video_id"] not in CLIPS:
            continue
        q = detq.get((m["video_id"], m["ped_id"], int(m["anchor_frame"])))
        if q is None:
            continue
        rows.append(dict(
            video=m["video_id"], ped=m["ped_id"], anchor=int(m["anchor_frame"]),
            cp=int(m["crossing_point"]) if m["crossing_point"] else None,
            tte=int(m["tte"]), label=int(y_test[i]), prob=float(probs[i]),
            pred=int(probs[i] >= TAU), **q))

    print(f"[scenes] {len(rows)} windows in the two local clips have detector stats")
    usable = [r for r in rows if r["detected"] == 1 and r["miou"] >= GOOD_IOU
              and r["purity"] >= GOOD_PURITY]
    print(f"[scenes] {len(usable)} of those are cleanly detected and tracked "
          f"(IoU>={GOOD_IOU}, purity>={GOOD_PURITY})\n")

    def show(title, cand, n=8):
        print(f"--- {title} " + "-" * max(0, 66 - len(title)))
        if not cand:
            print("    (none)\n")
            return
        print(f"    {'video':11s} {'ped':12s} {'frame':>6s} {'tte':>4s} {'s':>5s} "
              f"{'lab':>3s} {'prob':>6s} {'IoU':>5s} {'pur':>5s}")
        for r in cand[:n]:
            secs = r["tte"] / 30.0
            print(f"    {r['video']:11s} {r['ped']:12s} {r['anchor']:6d} {r['tte']:4d} "
                  f"{secs:5.2f} {r['label']:3d} {r['prob']:6.3f} {r['miou']:5.2f} {r['purity']:5.2f}")
        print()

    # ---- panel A: correct, confident, and genuinely ahead of the event
    a = [r for r in usable if r["label"] == 1 and r["pred"] == 1 and r["tte"] >= MIN_TTE]
    a.sort(key=lambda r: (-r["prob"], -r["tte"]))
    show("PANEL A  crosser, correctly flagged before onset", a)

    # ---- panel B: correct negatives, most confident first
    b = [r for r in usable if r["label"] == 0 and r["pred"] == 0]
    b.sort(key=lambda r: r["prob"])
    show("PANEL B  non-crosser, correctly not flagged", b)

    # ---- panel C: one frame, two correct and opposite verdicts
    by_frame = defaultdict(list)
    for r in usable:
        by_frame[(r["video"], r["anchor"])].append(r)
    c = []
    for (video, frame), rs in by_frame.items():
        pos = [r for r in rs if r["label"] == 1 and r["pred"] == 1 and r["tte"] >= MIN_TTE]
        neg = [r for r in rs if r["label"] == 0 and r["pred"] == 0]
        if pos and neg:
            c.append((video, frame, max(pos, key=lambda r: r["prob"]),
                      min(neg, key=lambda r: r["prob"]), len(rs)))
    c.sort(key=lambda t: -(t[2]["prob"] - t[3]["prob"]))

    print("--- PANEL C  one frame, two opposite correct verdicts " + "-" * 14)
    if not c:
        print("    (none — fall back to two separate panels; do not stage one)\n")
    else:
        print(f"    {'video':11s} {'frame':>6s} {'n':>3s}  {'crosser':12s} {'p':>6s} {'s_ahead':>7s}"
              f"   {'non-crosser':12s} {'p':>6s}")
        for video, frame, p, n_, k in c[:10]:
            print(f"    {video:11s} {frame:6d} {k:3d}  {p['ped']:12s} {p['prob']:6.3f} "
                  f"{p['tte']/30.0:7.2f}   {n_['ped']:12s} {n_['prob']:6.3f}")
        print()

    # ---- honest accounting
    #
    # Report BOTH pools. The IoU/purity filter above is a scene-quality gate, and
    # quoting its confusion counts as "how the model does on these clips" flatters
    # the model badly: the filter removes exactly the pedestrians the detector and
    # tracker struggled with, which are the ones most likely to be misclassified.
    # The manuscript quotes the unfiltered numbers.
    def counts(pool):
        tp = sum(1 for r in pool if r["label"] == 1 and r["pred"] == 1)
        fn = sum(1 for r in pool if r["label"] == 1 and r["pred"] == 0)
        tn = sum(1 for r in pool if r["label"] == 0 and r["pred"] == 0)
        fp = sum(1 for r in pool if r["label"] == 0 and r["pred"] == 1)
        acc = (tp + tn) / len(pool) if pool else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        return tp, fp, fn, tn, acc, f1

    for name, pool in (("ALL windows        ", rows), ("cleanly-tracked only", usable)):
        tp, fp, fn, tn, acc, f1 = counts(pool)
        print(f"[scenes] {name} n={len(pool):4d}:  TP {tp:3d}  FP {fp:3d}  FN {fn:3d}  "
              f"TN {tn:3d}   acc {acc:.3f}  F1 {f1:.3f}")
    print("[scenes] QUOTE THE FIRST LINE. The second is the pool the figure is drawn "
          "from, not a performance claim.")
    print("[scenes] NOTE when judging a whole frame by eye: a pedestrian is only right "
          "or wrong at frame f if a window is anchored within +/-1 s of f. Anyone else "
          "is out of protocol at that instant, usually because they are already "
          "mid-crossing, and must not be counted as an error.")


if __name__ == "__main__":
    main()

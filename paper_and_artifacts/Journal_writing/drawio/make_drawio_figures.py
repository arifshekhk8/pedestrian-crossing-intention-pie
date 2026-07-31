"""make_drawio_figures.py — the manuscript's figures as editable draw.io files.

Every figure in the paper, rebuilt as native draw.io shapes: boxes, lines, ticks
and labels you can select and edit in the desktop app. No image is embedded.

The two schematics, Figures 2 and 3, are hand-laid-out here, and draw.io is the
better tool for them than matplotlib ever was. The rest are data plots, and for
those this script loads the same source files as the matplotlib generators in
../MDPI_Article_Template/figures/ and computes every bar length and curve point
from the real numbers. Redrawing a chart by eye and nudging bars until they look
about right is how a figure quietly stops agreeing with its own results.

Figure 9 is not here. It is two photographs from PIE with detector boxes drawn
on them; there is nothing in it to redraw as vector shapes.

Run:  python paper_and_artifacts/Journal_writing/drawio/make_drawio_figures.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIGS = ROOT / "paper_and_artifacts" / "Journal_writing" / "MDPI_Article_Template" / "figures"
sys.path.insert(0, str(HERE))

from drawio_lib import (ACCENT, BILSTM, CLEAN, CONTEXT, GRID, GRU, INK, INK_2,  # noqa: E402
                        LEAK, MUTED, NONE, PANEL, RNN, RULE, TRANSF, WHITE,
                        Axes, Doc, legend, panel_title)

PARTIAL = "#e0a24b"
TIE = "#9b9a93"


# ===========================================================================
# Figure 1 — why pedestrians
# ===========================================================================

WHO_LABELS = ["Four-wheel vehicle occupants", "Pedestrians",
              "Powered 2-/3-wheeler users", "Other / unknown", "Cyclists"]
WHO_SHARES = [30, 23, 21, 20, 6]
WHO_EMPH = 1
YEARS = list(range(2014, 2024))
PED_DEATHS = [4910, 5494, 6080, 6075, 6374, 6272, 6565, 7470, 7593, 7314]


def fig1():
    d = Doc("Figure 1 - pedestrian statistics", 1240, 560)
    panel_title(d, 40, 26, "(a)  Global road traffic deaths by road user type",
                "1.19 million deaths worldwide (2021). Source: WHO Global Status "
                "Report on Road Safety 2023.", width=520)

    ax = Axes(d, 260, 100, 250, 330, (0, 36), (-0.7, 4.7))
    for i, (lab, v) in enumerate(zip(WHO_LABELS, WHO_SHARES)):
        emph = i == WHO_EMPH
        y = 4 - i
        ax.hbar(y, v, 34, fill=ACCENT if emph else CONTEXT)
        d.text(40, ax.py(y) - 10, 210, 20, lab, font=11,
               color=INK if emph else INK_2, bold=emph, align="right")
        d.text(ax.px(v) + 8, ax.py(y) - 10, 60, 20, f"{v}%", font=11,
               color=INK if emph else INK_2, bold=emph)

    panel_title(d, 620, 26, "(b)  Pedestrians killed in traffic crashes, United States",
                "+49% over the decade; 15% to 18% of all traffic deaths. "
                "Source: NHTSA DOT HS 813 727, Table 1.", width=580)

    bx = Axes(d, 690, 110, 500, 320, (2013.4, 2023.6), (0, 8600))
    bx.hgrid([2000, 4000, 6000, 8000])
    bx.yticks([0, 2000, 4000, 6000, 8000],
              ["0", "2,000", "4,000", "6,000", "8,000"], width=54)
    for yr, v in zip(YEARS, PED_DEATHS):
        bx.vbar(yr, v, 30, fill=ACCENT)
    bx.xticks(YEARS, [str(y)[-2:] for y in YEARS], width=34)
    bx.spine_bottom()
    for idx in (0, len(YEARS) - 1):
        d.text(bx.px(YEARS[idx]) - 30, bx.py(PED_DEATHS[idx]) - 22, 60, 18,
               f"{PED_DEATHS[idx]:,}", font=11, color=INK, bold=True,
               align="center")
    d.text(690, bx.tick_bottom + 4, 500, 22 * d.fk, "Year (20xx)", font=11,
           color=INK_2, align="center")
    d.save(HERE / "fig1_pedestrian_statistics.drawio")


# ===========================================================================
# Figure 2 — the model and the live pipeline   (a true schematic)
# ===========================================================================

def fig2():
    d = Doc("Figure 2 - system", 1480, 850)

    # ---------------------------------------------------------- panel (a)
    panel_title(d, 40, 24, "(a)  Two-stream crossing-intention model",
                "Only the dashed encoder block changes across the four families; "
                "every other component and the whole training recipe are frozen.",
                width=1400)
    mid = 250
    d.rect(40, mid - 116, 220, 86,
           "Pedestrian box\nx₁, y₁, x₂, y₂",
           fill="#e9f0fa", stroke="#a9c4e8", font=12)
    d.rect(40, mid + 30, 220, 86, "Ego-vehicle speed\nv  (OBD)",
           fill="#e9f0fa", stroke="#a9c4e8", font=12)
    d.arrow(260, mid - 73, 298, mid - 18)
    d.arrow(260, mid + 73, 298, mid + 18)

    d.rect(300, mid - 62, 215, 124,
           "16 × 5 window\n(0.5 s @ 30 fps)\ntrain-only z-score", font=12)
    d.arrow(515, mid, 553, mid)
    d.rect(555, mid - 62, 165, 124, "Linear\n5 → 64\n+ ReLU", font=12)
    d.arrow(720, mid, 758, mid)

    # the swappable encoder
    d.rect(760, mid - 132, 360, 264, "", fill="#fbfaf6", stroke=INK_2,
           dashed=True, lw=2)
    d.text(760, mid - 120, 360, 24, "Temporal encoder", font=13, color=INK,
           bold=True, align="center")
    for i, (name, col) in enumerate([("Bidirectional LSTM", BILSTM),
                                     ("Transformer encoder", TRANSF),
                                     ("Bidirectional GRU", GRU),
                                     ("Vanilla RNN (un-gated)", RNN)]):
        y = mid - 78 + i * 50
        d.rect(786, y, 20, 28, fill=col, stroke=NONE, rounded=False)
        d.text(816, y + 2, 290, 24, name, font=12, color=INK_2)
    d.arrow(1120, mid, 1158, mid)

    d.rect(1160, mid - 62, 165, 124, "Read-out\n+ linear\nhead", font=12)
    d.arrow(1325, mid, 1358, mid)
    d.rect(1360, mid - 42, 72, 84, "σ", fill="#e9f0fa", stroke="#a9c4e8",
           font=18)
    d.text(1336, mid + 50, 120, 24, "p(cross)", font=12, color=INK,
           align="center")

    # ---------------------------------------------------------- panel (b)
    panel_title(d, 40, 462, "(b)  Live perception-to-prediction pipeline",
                "Used for the latency and detector-in-the-loop experiments; the "
                "benchmark results use annotated boxes, as is standard.",
                width=1400)
    m = 672
    stages = [
        (40, 205, "Video frame\n1920 × 1080\n30 fps", None),
        (285, 215, "YOLO detector", "93% of per-frame cost"),
        (540, 215, "ByteTrack\nidentity\nassociation", None),
        (795, 215, "Per-track\n16-frame\nring buffer", None),
        (1050, 235, "Crossing-intention\nmodel  (panel a)", "4.5%"),
        (1320, 160, "Alert /\noverlay", None),
    ]
    for x, w, text, note in stages:
        emph = note is not None
        d.rect(x, m - 70, w, 140, text, font=12,
               fill="#eef4fc" if emph else PANEL,
               stroke=BILSTM if emph else RULE, lw=2 if emph else 1)
        if note:
            d.text(x, m + 78, w, 24, note, font=11, color=BILSTM, bold=True,
                   align="center")
    for i in range(len(stages) - 1):
        x, w = stages[i][0], stages[i][1]
        d.arrow(x + w + 4, m, stages[i + 1][0] - 4, m)
    # Sits in the band between the wrapped subtitle and the stage row; any
    # higher and it lands on the subtitle's second line.
    d.text(830, m - 128, 200, 26, "ego speed v", font=11, color=MUTED,
           align="center")
    d.arrow(930, m - 100, 930, m - 74)
    d.save(HERE / "fig2_system.drawio")


# ===========================================================================
# Figure 3 — the two windowing protocols   (a true schematic)
# ===========================================================================

def fig3():
    """Every vertical offset is a multiple of d.fk.

    The first version used fixed offsets, which were fine until the fonts began
    scaling with the page fit: the subtitle grew from one line to two and the TTE
    arrow, the onset label and the callout all ended up sitting inside it. Bands
    expressed in fk stay clear of each other whatever the scale turns out to be.
    """
    d = Doc("Figure 3 - protocol", 1320, 840)
    k = d.fk
    T0, ONSET, T_END = 60, 640, 1160        # page x of track start / onset / end
    OBS_W = 86                              # 16 frames, to scale with TTE below
    TTE_W = 242                             # 45 frames

    # Vertical bands within a panel, in units of k from the panel's top.
    B_TITLE, B_SUB = 0, 24
    B_ANNOT = 70          # TTE label and the onset label share this band
    B_ARROW = 98
    B_BAR = 112
    H_BAR = 46
    B_TRACK = B_BAR + H_BAR + 6
    B_CALL = B_BAR + H_BAR + 30
    PANEL_H = 250

    def panel(y0, title, subtitle, badge, badge_color, truncated, seg_label_x):
        d.text(40, y0 + B_TITLE * k, 900, 22 * k, title, font=14, color=INK,
               bold=True)
        d.text(1000, y0 + B_TITLE * k, 280, 22 * k, badge, font=13,
               color=badge_color, bold=True, align="right")
        d.text(40, y0 + B_SUB * k, 1240, 36 * k, subtitle, font=11, color=MUTED)

        bar_y, bar_h = y0 + B_BAR * k, H_BAR * k
        d.rect(T0, bar_y, ONSET - T0, bar_h, "", fill="#e8e7e1", stroke=NONE,
               rounded=False)
        d.rect(ONSET, bar_y, T_END - ONSET, bar_h, "",
               fill="#f6e3e1" if not truncated else "#f2f1ec", stroke=NONE,
               rounded=False)
        # Segment labels are placed by hand, clear of wherever the window block
        # lands, rather than centred in their segment where the block covers them.
        d.text(seg_label_x[0], bar_y, 260, bar_h, "approaching the kerb",
               font=12, color=MUTED, align="center")
        d.text(seg_label_x[1], bar_y, 260, bar_h,
               "truncated: never observed" if truncated else "already crossing",
               font=12, color="#a89a97" if truncated else LEAK, align="center")

        d.line(ONSET, y0 + B_ANNOT * k, ONSET, bar_y + bar_h + 14 * k,
               stroke=LEAK, lw=2)
        d.text(ONSET + 10, y0 + B_ANNOT * k, 220, 20 * k, "crossing onset",
               font=12, color=LEAK, bold=True)
        d.text(T0, y0 + B_TRACK * k, 240, 20 * k, "annotated track", font=11,
               color=MUTED)
        return bar_y, bar_h

    def window_block(anchor, bar_y, bar_h, y0, fill, stroke):
        d.rect(anchor - OBS_W, bar_y - 6 * k, OBS_W, bar_h + 12 * k, "",
               fill=fill, stroke=stroke, lw=2, rounded=False, opacity=55)
        d.line(anchor - OBS_W / 2, bar_y + bar_h + 8 * k,
               anchor - OBS_W / 2, y0 + B_CALL * k, stroke=RULE, lw=1)
        d.text(anchor - OBS_W - 90, y0 + B_CALL * k, OBS_W + 180, 40 * k,
               "16-frame\nobservation window", font=12, color=INK,
               align="center")

    def tte(x0, x1, y0, label):
        d.line(x0, y0 + B_ARROW * k, x1, y0 + B_ARROW * k, stroke=INK_2,
               lw=1.5, arrow="blockThin", start_arrow="blockThin")
        # font 11, and the text kept short: at 12 the range label wrapped
        # inside the arrow span and its second line landed on the arrow.
        d.text(x0, y0 + B_ANNOT * k, x1 - x0, 20 * k, label, font=11,
               color=INK_2, align="center")

    # ---- (a) track-end anchor. The window sits inside the crossing region, so
    # "already crossing" is pushed right of it.
    y0 = 26
    anchor = T_END - TTE_W
    bar_y, bar_h = panel(
        y0, "(a)  Track-end anchor: the common protocol",
        "The window is offset from the end of the track, so nothing stops it "
        "overlapping the crossing.",
        "leaks for 67.9% of crossers", LEAK, False, (220, 900))
    window_block(anchor, bar_y, bar_h, y0, "#efd3d0", LEAK)
    tte(anchor, T_END, y0, "TTE = 45 frames")
    d.line(T_END, y0 + B_ARROW * k, T_END, bar_y + bar_h, stroke=RULE, lw=1)
    d.text(T_END + 8, bar_y - 4 * k, 140, 40 * k, "last annotated\nframe",
           font=11, color=MUTED)

    # ---- (b) crossing-point anchor. Here the window sits in the pre-crossing
    # region, so that segment's label moves right instead.
    y0 = 26 + PANEL_H * k + 30
    anchor = ONSET - TTE_W
    bar_y, bar_h = panel(
        y0, "(b)  Crossing-point anchor: this work",
        "The window must end at least a second before the annotated crossing "
        "point, so leakage is impossible by construction.",
        "leaks for 0.0% of windows", CLEAN, True, (400, 760))
    window_block(anchor, bar_y, bar_h, y0, "#d5ece4", CLEAN)
    tte(anchor, ONSET, y0, "TTE 30–60 frames (1–2 s)")
    d.save(HERE / "fig3_protocol.drawio")


# ===========================================================================
# Figure 4 — the leakage audit   (computed from the audit CSVs)
# ===========================================================================

OBS = 16
FEATURES = [("bbox_area", "Box area"), ("bbox_height", "Box height"),
            ("bbox_bottom_y", "Box bottom edge"), ("bbox_xcenter", "Box center x")]


def _audit(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _composition(rows):
    n_cross = [int(r["n_crossing_in_window"]) for r in rows if r["label"] == "1"]
    n = len(n_cross)
    full = sum(1 for v in n_cross if v >= OBS)
    part = sum(1 for v in n_cross if 0 < v < OBS)
    return n, np.array([n - full - part, part, full]) / n * 100.0


def _rank_biserial(rows, field):
    from scipy.stats import mannwhitneyu
    a = np.array([float(r[field]) for r in rows if r["label"] == "1"])
    b = np.array([float(r[field]) for r in rows if r["label"] == "0"])
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return abs(2.0 * u / (len(a) * len(b)) - 1.0)


def fig4():
    leaky = _audit(ROOT / "journal_prep/issue1_leakage_audit/leakage_per_sequence.csv")
    clean = _audit(ROOT / "journal_prep/issue2_clean_protocol/leakage_per_sequence.csv")
    n_l, comp_l = _composition(leaky)
    n_c, comp_c = _composition(clean)

    d = Doc("Figure 4 - leakage audit", 1300, 560)
    panel_title(d, 40, 26, "(a)  What the model actually sees",
                "Crossing pedestrians grouped by how much of the 16-frame window "
                "is spent already crossing.", width=560)

    segs = [("entirely pre-crossing", CLEAN), ("partly crossing", PARTIAL),
            ("entirely crossing", LEAK)]
    ax = Axes(d, 250, 130, 340, 190, (0, 100), (-0.6, 1.6))
    for (label, comp, y) in [
            (f"Track-end anchor\nn = {n_l:,} crossers", comp_l, 1.0),
            (f"Crossing-point anchor\nn = {n_c:,} crossers", comp_c, 0.0)]:
        left = 0.0
        for (name, col), val in zip(segs, comp):
            w = ax.px(left + val) - ax.px(left)
            d.rect(ax.px(left), ax.py(y) - 26, max(w, 0.5), 52,
                   f"{val:.0f}%" if val >= 10 else "", fill=col, stroke=WHITE,
                   lw=2, rounded=False, font=12, color=WHITE, bold=True)
            left += val
        d.text(40, ax.py(y) - 20, 200, 40, label, font=11, color=INK_2,
               align="right")
    d.text(ax.px(comp_l[0] + comp_l[1] / 2) - 30, ax.py(1.0) - 62, 60, 18,
           f"{comp_l[1]:.0f}%", font=11, color=INK_2, align="center")
    d.line(ax.px(comp_l[0] + comp_l[1] / 2), ax.py(1.0) - 42,
           ax.px(comp_l[0] + comp_l[1] / 2), ax.py(1.0) - 28, stroke=CONTEXT)
    legend(d, 250, 360, [(c, n) for n, c in segs])

    panel_title(d, 700, 26, "(b)  Can one frame give it away?",
                "Rank-biserial effect of the single box at the last observed "
                "frame, crossers vs non-crossers.", width=560)
    r_l = [_rank_biserial(leaky, f) for f, _ in FEATURES]
    r_c = [_rank_biserial(clean, f) for f, _ in FEATURES]
    bx = Axes(d, 880, 130, 340, 230, (0, 0.75), (-0.6, 3.6))
    bx.vgrid([0.2, 0.4, 0.6])
    for i, (_, name) in enumerate(FEATURES):
        y = 3 - i
        bx.hbar(y + 0.20, r_l[i], 20, fill=LEAK)
        bx.hbar(y - 0.20, r_c[i], 20, fill=CLEAN)
        d.text(700, bx.py(y) - 10, 170, 20, name, font=11, color=INK_2,
               align="right")
        d.text(bx.px(r_l[i]) + 6, bx.py(y + 0.20) - 9, 54, 18, f"{r_l[i]:.2f}",
               font=10, color=INK_2)
        d.text(bx.px(r_c[i]) + 6, bx.py(y - 0.20) - 9, 54, 18, f"{r_c[i]:.2f}",
               font=10, color=INK_2)
    bx.xticks([0, 0.2, 0.4, 0.6], ["0", "0.2", "0.4", "0.6"])
    bx.spine_bottom()
    d.text(880, bx.tick_bottom + 4, 340, 22 * d.fk,
           "Rank-biserial correlation (absolute)", font=11, color=INK_2,
           align="center")
    legend(d, 880, bx.tick_bottom + 34 * d.fk, [(LEAK, "Track-end anchor"),
                         (CLEAN, "Crossing-point anchor")])
    d.save(HERE / "fig4_leakage.drawio")


# ===========================================================================
# Figure 5 — ROC and precision-recall   (computed from _probs.npz)
# ===========================================================================

SERIES = [("bilstm_f1", "BiLSTM", BILSTM, 2, False),
          ("transformer_f1", "Transformer", TRANSF, 2, False),
          ("gru_f1", "GRU", GRU, 2, False),
          ("rnn_f1", "Vanilla RNN", RNN, 2, False),
          ("bilstm_bbox_only", "Box only (no ego-speed)", CONTEXT, 2, True)]


def _thin(xs, ys, n=44):
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int)
    return [(float(xs[i]), float(ys[i])) for i in idx]


def fig5():
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    z = np.load(FIGS / "_probs.npz")
    y = z["y"]

    d = Doc("Figure 5 - ROC and PR curves", 1240, 640)
    panel_title(d, 40, 26, "(a)  ROC",
                "Five-seed ensembles on the 2,094 leakage-free test windows.",
                width=520)
    ax = Axes(d, 110, 110, 420, 420, (0, 1), (0, 1))
    ax.hgrid([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.vgrid([0.2, 0.4, 0.6, 0.8, 1.0])
    d.line(ax.px(0), ax.py(0), ax.px(1), ax.py(1), stroke=GRID, lw=1, dashed=True)
    aucs = {}
    for key, _, col, lw, dash in SERIES:
        fpr, tpr, _ = roc_curve(y, z[key])
        ax.curve(_thin(fpr, tpr), stroke=col, lw=lw, dashed=dash)
        aucs[key] = roc_auc_score(y, z[key])
    ax.xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yticks([0, 0.25, 0.5, 0.75, 1.0], width=40)
    ax.spine_bottom()
    ax.spine_left()
    d.text(110, ax.tick_bottom + 4, 420, 22 * d.fk, "False positive rate",
           font=11, color=INK_2, align="center")
    d.text(-42, 310, 160, 18, "True positive rate", font=11, color=INK_2,
           align="center")

    panel_title(d, 660, 26, "(b)  Precision-recall",
                "The four families overlap; dropping ego-speed pulls the curve "
                "away.", width=520)
    bx = Axes(d, 730, 110, 420, 420, (0, 1), (0, 1))
    bx.hgrid([0.2, 0.4, 0.6, 0.8, 1.0])
    bx.vgrid([0.2, 0.4, 0.6, 0.8, 1.0])
    aps = {}
    for key, _, col, lw, dash in SERIES:
        pr, rc, _ = precision_recall_curve(y, z[key])
        bx.curve(_thin(rc, pr), stroke=col, lw=lw, dashed=dash)
        aps[key] = average_precision_score(y, z[key])
    bx.xticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.yticks([0, 0.25, 0.5, 0.75, 1.0], width=40)
    bx.spine_bottom()
    bx.spine_left()
    d.text(730, bx.tick_bottom + 4, 420, 22 * d.fk, "Recall", font=11,
           color=INK_2, align="center")
    d.text(578, 310, 160, 18, "Precision", font=11, color=INK_2, align="center")

    for i, (key, name, col, _, _) in enumerate(SERIES):
        yy = ax.tick_bottom + 34 * d.fk + (i // 3) * 22 * d.fk
        xx = 110 + (i % 3) * 350
        d.rect(xx, yy + 3, 16, 10, fill=col, stroke=NONE, rounded=False)
        d.text(xx + 23, yy, 330, 16,
               f"{name}   AUC {aucs[key]:.3f}   AP {aps[key]:.3f}",
               font=11, color=INK_2)
    d.save(HERE / "fig5_curves.drawio")


# ===========================================================================
# Figure 6 — forest plot of paired contrasts
# ===========================================================================

F1_ROWS = [
    ("GRU vs. BiLSTM", 0.007118, -0.008911, 0.022339),
    ("Vanilla RNN vs. BiLSTM", 0.003336, -0.012996, 0.018743),
    ("Transformer vs. BiLSTM", 0.000827, -0.019553, 0.019984),
    ("GRU vs. Transformer", 0.006290, -0.008522, 0.021594),
    ("Vanilla RNN vs. Transformer", 0.002509, -0.011085, 0.017062),
    ("Vanilla RNN vs. GRU", -0.003781, -0.012754, 0.004874),
    (None, None, None, None),
    ("BiLSTM vs. AUC-selected baseline", 0.018674, 0.004312, 0.034935),
    ("GRU vs. AUC-selected baseline", 0.025792, 0.014795, 0.038050),
    ("Vanilla RNN vs. AUC-selected baseline", 0.022011, 0.009681, 0.035370),
]
AUC_ROWS = [
    ("Vanilla RNN vs. searched Transformer", -0.001344, -0.006077, 0.003312),
    ("GRU vs. searched Transformer", -0.006953, -0.012904, -0.001755),
    (None, None, None, None),
    ("Searched Transformer vs. baseline †", 0.013453, 0.009705, 0.017368),
    ("Un-searched Transformer vs. baseline †", 0.000468, -0.003424, 0.004261),
    ("Vanilla RNN (h256) vs. baseline", 0.012125, 0.006307, 0.018737),
    ("Vanilla RNN (h128) vs. baseline", 0.005931, 0.001225, 0.010992),
    ("GRU (h128) vs. baseline", -0.000799, -0.006728, 0.004538),
]
WIN = "#2a78d6"


def _forest(d, rows, y0, xlim, title, subtitle, xlabel, ticks):
    panel_title(d, 40, y0, title, subtitle, width=1100)
    n = sum(1 for r in rows if r[0] is not None)
    breaks = sum(1 for r in rows if r[0] is None)
    h = n * 30 + breaks * 16
    ax = Axes(d, 460, y0 + 66, 560, h, xlim, (0, 1))
    ax.vgrid(ticks)
    d.line(ax.px(0), ax.py(0), ax.px(0), ax.py(1), stroke=RULE, lw=1.5)
    yy = y0 + 66 + 15
    for label, delta, lo, hi in rows:
        if label is None:
            yy += 16
            continue
        col = TIE if (lo <= 0.0 <= hi) else WIN
        d.line(ax.px(lo), yy, ax.px(hi), yy, stroke=col, lw=3)
        ms = 10 * d.fk
        d.ellipse(ax.px(delta) - ms / 2, yy - ms / 2, ms, ms, fill=col,
                  stroke=WHITE)
        d.text(180, yy - 10, 270, 20, label, font=11, color=INK_2, align="right")
        d.text(ax.px(xlim[1]) - 74, yy - 10, 70, 20, f"{delta:+.4f}", font=11,
               color=INK if col == WIN else INK_2, bold=col == WIN, align="right")
        yy += 30
    ax.xticks(ticks, [f"{t:+.3f}".replace("+0.000", "0") for t in ticks])
    d.text(460, y0 + 66 + h + 26, 560, 18, xlabel, font=11, color=INK_2,
           align="center")
    return y0 + 66 + h + 52


def fig6():
    d = Doc("Figure 6 - forest plot", 1180, 1020)
    y = _forest(d, F1_ROWS, 26, (-0.030, 0.048),
                "(a)  F1: the primary metric",
                "Every cross-family contrast spans zero: the four families are "
                "indistinguishable.",
                "ΔF1 (95% pedestrian-cluster bootstrap interval)",
                [-0.02, 0.0, 0.02, 0.04])
    legend(d, 460, y, [(WIN, "interval excludes zero: a real difference"),
                       (TIE, "interval spans zero: a tie")])
    y += 62
    y = _forest(d, AUC_ROWS, y, (-0.016, 0.026),
                "(b)  ROC-AUC: the corroborating metric",
                "Here the same comparisons do separate, so the ties above are "
                "not a test without power.",
                "ΔAUC (95% interval)",
                [-0.01, 0.0, 0.01, 0.02])
    d.text(40, y + 6, 1100, 18,
           "† seed-paired window-level bootstrap; all other rows resample "
           "the 541 pedestrian clusters.", font=10, color=MUTED)
    d.save(HERE / "fig6_forest.drawio")


# ===========================================================================
# Figure 7 — ablations
# ===========================================================================

EGO = {"F1": [(0.828, 0.012), (0.551, 0.028)],
       "Accuracy": [(0.883, 0.009), (0.744, 0.007)],
       "ROC-AUC": [(0.932, 0.011), (0.753, 0.020)]}
WINDOWS = [16, 32, 64]
OW = {"BiLSTM": ([0.844, 0.837, 0.818], BILSTM),
      "Transformer": ([0.847, 0.838, 0.819], TRANSF),
      "GRU": ([0.849, 0.834, 0.822], GRU),
      "Vanilla RNN": ([0.852, 0.834, 0.802], RNN)}


def _horizons():
    acc = defaultdict(lambda: defaultdict(list))
    p = ROOT / "journal_prep/issue6_window_tte_ablation/06b_matched_tte_results.csv"
    with open(p) as fh:
        for r in csv.DictReader(fh):
            t = float(r["tte_seconds"])
            acc[t]["auc"].append(float(r["auc"]))
            acc[t]["f1"].append(float(r["f1"]))
    ts = sorted(acc)
    return ts, {k: [float(np.mean(acc[t][k])) for t in ts] for k in ("auc", "f1")}


def fig7():
    ts, hor = _horizons()
    d = Doc("Figure 7 - ablations", 1420, 620)

    # -------- (a) ego-speed ablation
    panel_title(d, 40, 26, "(a)  Removing ego-speed", None, width=440)
    ax = Axes(d, 90, 100, 300, 280, (-0.6, 2.6), (0, 1.0))
    ax.hgrid([0.25, 0.5, 0.75, 1.0])
    ax.yticks([0, 0.25, 0.5, 0.75, 1.0], width=40)
    for i, m in enumerate(EGO):
        (v2, _), (v1, _) = EGO[m]
        ax.vbar(i - 0.19, v2, 38, fill=BILSTM)
        ax.vbar(i + 0.19, v1, 38, fill=CONTEXT)
        d.text(ax.px(i - 0.19) - 30, ax.py(v2) - 20, 60, 18, f"{v2:.2f}",
               font=11, color=INK, bold=True, align="center")
        d.text(ax.px(i + 0.19) - 30, ax.py(v1) - 20, 60, 18, f"{v1:.2f}",
               font=11, color=INK_2, align="center")
    ax.xticks([0, 1, 2], list(EGO), width=90)
    ax.spine_bottom()
    legend(d, 90, ax.tick_bottom + 10, [(BILSTM, "Box + ego-speed"),
                                       (CONTEXT, "Box only")])

    # -------- (b) prediction horizon
    panel_title(d, 500, 26, "(b)  Predicting further ahead", None, width=430)
    bx = Axes(d, 560, 100, 280, 280, (0.85, 2.15), (0.72, 1.0))
    bx.hgrid([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    bx.yticks([0.75, 0.85, 0.95], width=44)
    for key, col in (("auc", CONTEXT), ("f1", BILSTM)):
        bx.curve(list(zip(ts, hor[key])), stroke=col, lw=2)
        for t, v in zip(ts, hor[key]):
            bx.marker(t, v, fill=col)
        d.text(bx.px(ts[-1]) + 6, bx.py(hor[key][-1]) - 9, 60, 18,
               f"{hor[key][-1]:.3f}", font=10,
               color=INK if key == "f1" else INK_2, bold=key == "f1")
    bx.xticks(ts, [f"{t:.1f}" for t in ts])
    bx.spine_bottom()
    d.text(560, bx.tick_bottom + 4, 280, 22 * d.fk, "Prediction horizon (s)",
           font=11, color=INK_2, align="center")
    legend(d, 560, bx.tick_bottom + 34 * d.fk,
           [(BILSTM, "F1"), (CONTEXT, "ROC-AUC")])

    # -------- (c) observation window
    panel_title(d, 960, 26, "(c)  Observing for longer", None, width=440)
    cx = Axes(d, 1020, 100, 300, 280, (-0.25, 2.25), (0.765, 0.885))
    cx.hgrid([0.78, 0.80, 0.82, 0.84, 0.86])
    cx.yticks([0.78, 0.82, 0.86], width=44)
    for name, (vals, col) in OW.items():
        cx.curve(list(zip(range(3), vals)), stroke=col, lw=2)
        for i, v in enumerate(vals):
            cx.marker(i, v, fill=col)
    d.text(1020 - 96 * d.fk, 232, 40 * d.fk, 20 * d.fk, "F1", font=11,
           color=INK_2, align="center")
    cx.xticks([0, 1, 2], [f"{w}\n({w / 30:.1f} s)" for w in WINDOWS], width=80)
    cx.spine_bottom()
    d.text(1020, cx.tick_bottom + 4, 300, 22 * d.fk,
           "Observation window (frames)", font=11, color=INK_2, align="center")
    legend(d, 1020, cx.tick_bottom + 34 * d.fk,
           [(c, n) for n, (_, c) in OW.items()], gap=18)
    d.save(HERE / "fig7_ablations.drawio")


# ===========================================================================
# Figure 8 — latency
# ===========================================================================

FRAME_BUDGET_MS = 1000.0 / 30.0
LATENCY = [("Vanilla RNN", 0.316, RNN), ("Transformer", 0.459, TRANSF),
           ("BiLSTM", 0.575, BILSTM), ("GRU", 0.721, GRU)]
PIPELINE = [("YOLO detector", 33.7, "#7d7c76"),
            ("ByteTrack association", 1.0, "#b9b8b2"),
            ("Crossing-intention model", 1.647, BILSTM)]


def fig8():
    d = Doc("Figure 8 - latency", 1260, 520)
    panel_title(d, 40, 26, "(a)  Cost of one prediction",
                "Apple M4 CPU, batch 1, 50 warm-up + 1,000 timed forwards.",
                width=520)
    k = d.fk
    ax = Axes(d, 250, 110, 240, 230, (0, 0.95), (-0.6, 3.6))
    ax.vgrid([0.25, 0.5, 0.75])
    # The value sits just past its bar, so the longest bar's label reaches
    # furthest right. Park the "x budget" column past that worst case rather
    # than at the axis edge, where 0.721 ms and 46x collided.
    val_w = 78 * k
    mult_x = ax.px(max(v for _, v, _ in LATENCY)) + val_w + 14 * k
    for i, (name, v, col) in enumerate(LATENCY):
        y = 3 - i
        ax.hbar(y, v, 34, fill=col)
        d.text(40, ax.py(y) - 10 * k, 200, 20 * k, name, font=11, color=INK_2,
               align="right")
        d.text(ax.px(v) + 8, ax.py(y) - 10 * k, val_w, 20 * k, f"{v:.3f} ms",
               font=11, color=INK, bold=True)
        d.text(mult_x, ax.py(y) - 10 * k, 56 * k, 20 * k,
               f"{FRAME_BUDGET_MS / v:.0f}×", font=11, color=MUTED)
    ax.xticks([0, 0.25, 0.5, 0.75])
    ax.spine_bottom()
    d.text(250, ax.tick_bottom + 4, 260, 22 * d.fk, "ms per window", font=11,
           color=INK_2, align="center")
    d.text(250, ax.tick_bottom + 34 * d.fk, 340, 40 * d.fk,
           "× = how many predictions fit in one 33.3 ms frame budget",
           font=10, color=MUTED)

    panel_title(d, 700, 26, "(b)  Where a frame's time actually goes",
                "The same machine, live pipeline. The answer to (a) is "
                "irrelevant next to this.", width=520)
    total = sum(v for _, v, _ in PIPELINE)
    bx = Axes(d, 730, 130, 460, 90, (0, total), (0, 1))
    left = 0.0
    for name, v, col in PIPELINE:
        w = bx.px(left + v) - bx.px(left)
        d.rect(bx.px(left), 130, max(w, 1.0), 90, "", fill=col, stroke=WHITE,
               lw=2, rounded=False)
        left += v
    d.text(730, 150, bx.px(PIPELINE[0][1]) - 730, 50,
           f"YOLO detector\n{PIPELINE[0][1]:.1f} ms  ({100 * PIPELINE[0][1] / total:.0f}%)",
           font=12, color=WHITE, bold=True, align="center")
    d.text(730, 240, 460, 18, f"total {total:.1f} ms per frame", font=11,
           color=INK_2)
    legend(d, 730, 280, [(c, f"{n}   {v:.3f} ms   {100 * v / total:.1f}%")
                         for n, v, c in PIPELINE])
    d.text(730, 380, 460, 40,
           "The intention model is 4.5% of the frame. Making it faster changes "
           "nothing; the pipeline is detection-bound.", font=11, color=MUTED)
    d.save(HERE / "fig8_latency.drawio")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print("\nFigure 9 is two photographs with detector boxes drawn on them; "
          "there is nothing in it to redraw as vector shapes.")

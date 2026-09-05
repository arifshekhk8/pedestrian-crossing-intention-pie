"""make_fig_pipeline.py — the methodological pipeline figure.

Six stages, left to right: datasets, temporal window construction, frame-level
temporal audit, phase-controlled sampling, input representation and models,
evaluation and comparative findings.

The point of the figure is that the contribution is a method, not an
architecture. Nothing here shows pose, RGB, segmentation, flow or depth: the
input contract is five numbers per frame, and the figure says so.

Palette and ink tokens come from figstyle.py so this reads as the same document
as Figure 1.

Run:  python make_fig_pipeline.py   ->  ../figures/fig_pipeline.{pdf,png}
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle, FancyArrow

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"

# ------------------------------------------------------------------ palette
BLUE = "#2a78d6"      # data and inputs
ORANGE = "#d1622b"
TEAL = "#1f9b7a"
PURPLE = "#8b5cc7"
LEAK = "#b4433a"      # contamination / the problem
CLEAN = "#1f9b7a"     # corrected / controlled

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
RULE = "#c3c2b7"
PANEL = "#f4f3ee"     # model blocks: light cream
GRID = "#e1e0d9"

BLUE_F = "#dfeaf9"
LEAK_F = "#f6e0de"
CLEAN_F = "#dcece6"
GREY_F = "#f0efea"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

W, H = 1350, 432
LW = 0.7                      # thin dark-grey outline everywhere

# stage bands (x0, x1)
S = {
    1: (12, 142),
    2: (150, 410),
    3: (418, 552),
    4: (560, 812),
    5: (820, 1065),
    6: (1073, 1338),
}
HEAD_Y = 424


# ------------------------------------------------------------------ helpers
def box(ax, x, y, w, h, fc="white", ec=INK_2, lw=LW, r=3.0, z=2, ls="-"):
    p = FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r,
                       boxstyle=f"round,pad={r}", facecolor=fc, edgecolor=ec,
                       linewidth=lw, zorder=z, linestyle=ls)
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, size=7.5, color=INK_2, weight="normal", ha="left",
        va="center", z=5, style="normal"):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=z, style=style, linespacing=1.35)


ARROW = "$\\rightarrow$"


def tw(ax, s, size, weight="normal", style="normal"):
    """Width of a string in data units, measured on the real renderer."""
    t = ax.text(0, -500, s, fontsize=size, fontweight=weight, style=style)
    ax.figure.canvas.draw()
    w = t.get_window_extent(ax.figure.canvas.get_renderer()).width
    t.remove()
    return w / ax.figure.dpi * 100


def fit(ax, x, y, s, maxw, size=7.5, color=INK_2, weight="normal", ha="left",
        va="center", z=5, style="normal", floor=4.8):
    """Place text, shrinking the size until it fits maxw data units."""
    while size > floor and tw(ax, s, size, weight, style) > maxw:
        size -= 0.15
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=z, style=style, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=INK_2, lw=0.9, z=4, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=z,
                arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.34",
                                color=color, linewidth=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0))


def strip(ax, x, y, n, cw, gap, fills, ec=INK_2, h=9, lw=0.45, z=3):
    """A run of n frame cells. `fills` is a colour or a list of length n."""
    if isinstance(fills, str):
        fills = [fills] * n
    for i in range(n):
        ax.add_patch(Rectangle((x + i * (cw + gap), y), cw, h, facecolor=fills[i],
                               edgecolor=ec, linewidth=lw, zorder=z))
    return x + n * (cw + gap) - gap


def check(ax, x, y, s=4.2, color=CLEAN, lw=1.3):
    ax.plot([x - s, x - s * 0.25, x + s], [y, y - s * 0.75, y + s * 0.8],
            color=color, lw=lw, solid_capstyle="round", zorder=6)


def warn(ax, x, y, s=4.6, color=LEAK):
    ax.add_patch(Polygon([[x, y + s], [x - s * 0.92, y - s * 0.72],
                          [x + s * 0.92, y - s * 0.72]], closed=True,
                         facecolor="white", edgecolor=color, lw=1.0, zorder=6))
    ax.plot([x, x], [y + s * 0.42, y - s * 0.14], color=color, lw=1.0, zorder=7,
            solid_capstyle="round")
    ax.plot([x], [y - s * 0.40], marker=".", ms=1.9, color=color, zorder=7)


def magnifier(ax, x, y, r=5.2, color=INK_2):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color, lw=1.0, zorder=6))
    ax.plot([x + r * 0.72, x + r * 1.55], [y - r * 0.72, y - r * 1.55],
            color=color, lw=1.3, solid_capstyle="round", zorder=6)


def balance(ax, x, y, w=13, color=INK_2):
    ax.plot([x - w, x + w], [y, y], color=color, lw=1.1, solid_capstyle="round", zorder=6)
    ax.plot([x, x], [y, y - 7], color=color, lw=1.1, zorder=6)
    ax.add_patch(Polygon([[x - 4.5, y - 7], [x + 4.5, y - 7], [x, y - 1.5]],
                         closed=True, facecolor=color, edgecolor="none", zorder=6))
    for sx in (-w, w):
        ax.plot([x + sx, x + sx], [y, y - 3.4], color=color, lw=0.8, zorder=6)
        ax.add_patch(Circle((x + sx, y - 4.6), 1.5, facecolor="white",
                            edgecolor=color, lw=0.8, zorder=6))


def clip_icon(ax, x, y, w=13, h=10, color=BLUE):
    """A compact video/road-clip glyph."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color,
                           lw=0.8, zorder=6))
    ax.plot([x + w * 0.5, x + w * 0.5], [y + 1.4, y + h - 1.4], color=color,
            lw=0.6, ls=(0, (1.4, 1.4)), zorder=6)
    for fy in (y + h * 0.30, y + h * 0.70):
        ax.plot([x + 1.6, x + w - 1.6], [fy, fy], color=GRID, lw=0.5, zorder=6)


def netglyph(ax, cx, cy, color):
    """Three-node abstract network mark. Deliberately minimal."""
    xs = [cx - 6.5, cx, cx + 6.5]
    ys = [[cy + 3.6, cy - 3.6], [cy + 5.0, cy, cy - 5.0], [cy + 3.6, cy - 3.6]]
    for i in range(2):
        for a in ys[i]:
            for b in ys[i + 1]:
                ax.plot([xs[i], xs[i + 1]], [a, b], color=color, lw=0.32,
                        alpha=0.55, zorder=5)
    for x, col in zip(xs, ys):
        for y in col:
            ax.add_patch(Circle((x, y), 1.25, facecolor=color, edgecolor="none",
                                zorder=6))


def stage_head(ax, i, label):
    x0, x1 = S[i]
    ax.add_patch(Circle((x0 + 6.5, HEAD_Y), 6.0, facecolor=INK, edgecolor="none",
                        zorder=6))
    txt(ax, x0 + 6.5, HEAD_Y, str(i), 7.0, "white", "bold", "center", "center", 7)
    txt(ax, x0 + 17, HEAD_Y, label, 8.6, INK, "bold", "left", "center")
    ax.plot([x0, x1], [HEAD_Y - 11, HEAD_Y - 11], color=RULE, lw=0.7, zorder=2)


# ------------------------------------------------------------------ canvas
fig, ax = plt.subplots(figsize=(13.5, 4.32))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

for i, lab in [(1, "Datasets"), (2, "Window Construction"),
               (3, "Temporal Audit"), (4, "Phase Control"),
               (5, "Inputs and Models"), (6, "Evaluation and Findings")]:
    stage_head(ax, i, lab)

# ============================================================ 1  DATASETS
x0, x1 = S[1]
bw = x1 - x0

box(ax, x0, 268, bw, 128, BLUE_F, BLUE, 1.1)
clip_icon(ax, x0 + 9, 372, 13, 10, BLUE)
txt(ax, x0 + 27, 377, "PIE", 10.5, INK, "bold")
txt(ax, x0 + 9, 359, "Primary controlled benchmark", 6.6, INK_2)
ax.plot([x0 + 9, x1 - 9], [352, 352], color=BLUE, lw=0.5, alpha=0.5, zorder=3)
txt(ax, x0 + 9, 341, "Partition by recording set", 6.3, MUTED)
for k, (lab, sets) in enumerate([("Train", "set01, set02, set04"),
                                 ("Val", "set05, set06"),
                                 ("Test", "set03")]):
    txt(ax, x0 + 9, 328 - k * 11, lab, 6.5, INK_2, "bold")
    txt(ax, x0 + 36, 328 - k * 11, sets, 6.5, INK_2)
fit(ax, x0 + 9, 282, "4,906 windows · 1,267 pedestrians", bw - 18, 6.3, MUTED)

box(ax, x0, 196, bw, 58, GREY_F, INK_2)
clip_icon(ax, x0 + 9, 230, 11, 8.5, INK_2)
txt(ax, x0 + 25, 234, "JAAD", 9.0, INK, "bold")
txt(ax, x0 + 9, 213, "Temporal audit", 6.6, INK_2)
txt(ax, x0 + 9, 203, "no ego-speed channel", 6.1, MUTED, style="italic")

box(ax, x0, 118, bw, 64, GREY_F, INK_2)
clip_icon(ax, x0 + 9, 158, 11, 8.5, INK_2)
txt(ax, x0 + 25, 162, "IDD-PeD", 9.0, INK, "bold")
txt(ax, x0 + 9, 141, "Temporal audit +", 6.6, INK_2)
txt(ax, x0 + 9, 132, "external evaluation", 6.6, INK_2)
txt(ax, x0 + 9, 122, "unstructured traffic", 6.1, MUTED, style="italic")

arrow(ax, x1 + 1, 332, S[2][0] - 2, 332, INK_2)
arrow(ax, x1 + 1, 225, S[2][0] - 2, 300, MUTED, 0.7)
arrow(ax, x1 + 1, 150, S[2][0] - 2, 292, MUTED, 0.7)

# =============================== 2  TEMPORAL WINDOW CONSTRUCTION
x0, x1 = S[2]

# --- (A) historical track-end extraction
box(ax, x0, 288, x1 - x0, 108, LEAK_F, LEAK, 0.9)
txt(ax, x0 + 10, 383, "A   Historical track-end extraction", 7.6, LEAK, "bold")

tx, ty = x0 + 14, 338
onset = tx + 27 * 6.2
txt(ax, tx, 356, "pedestrian track", 6.1, MUTED)
strip(ax, tx, ty, 27, 5.0, 1.2, GREY_F, MUTED, 9, 0.4)
strip(ax, onset, ty, 12, 5.0, 1.2, LEAK_F, LEAK, 9, 0.4)
ax.plot([onset - 1.5, onset - 1.5], [ty - 7, ty + 17], color=LEAK, lw=1.2, zorder=6)
txt(ax, onset - 1.5, ty + 21, "crossing onset", 6.1, LEAK, "bold", "center")

# the observation window, straddling the onset
wx = onset - 8 * 6.2
ax.add_patch(Rectangle((wx - 1.6, ty - 3.2), 16 * 6.2 - 1.2 + 3.2, 15.4,
                       facecolor="none", edgecolor=INK, lw=0.9, zorder=7))
strip(ax, wx, ty, 8, 5.0, 1.2, BLUE_F, BLUE, 9, 0.45, z=8)
strip(ax, onset, ty, 8, 5.0, 1.2, "#e9a9a2", LEAK, 9, 0.45, z=8)
txt(ax, wx - 2, ty - 9.5, "16-frame observation window", 6.1, INK, "bold")

warn(ax, x0 + 16, 310, 4.2)
fit(ax, x0 + 25, 311, "Crossing-frame contamination", x1 - x0 - 45, 6.9, LEAK, "bold")
fit(ax, x0 + 14, 299, "67.9% of positive PIE sequences affected", x1 - x0 - 28, 6.3, INK_2)

# --- (B) event-anchored extraction
box(ax, x0, 150, x1 - x0, 118, CLEAN_F, CLEAN, 0.9)
txt(ax, x0 + 10, 255, "B   Event-anchored extraction", 7.6, CLEAN, "bold")

ty2 = 208
onset2 = x0 + 14 + 34 * 6.2
strip(ax, x0 + 14, ty2, 34, 5.0, 1.2, GREY_F, MUTED, 9, 0.4)
ax.plot([onset2 - 1.5, onset2 - 1.5], [ty2 - 7, ty2 + 17], color=LEAK, lw=1.2, zorder=6)
txt(ax, onset2 - 1.5, ty2 + 21, "crossing onset", 6.1, LEAK, "bold", "center")

wx2 = onset2 - 26 * 6.2
ax.add_patch(Rectangle((wx2 - 1.6, ty2 - 3.2), 16 * 6.2 - 1.2 + 3.2, 15.4,
                       facecolor="none", edgecolor=INK, lw=0.9, zorder=7))
strip(ax, wx2, ty2, 16, 5.0, 1.2, BLUE_F, BLUE, 9, 0.45, z=8)

# the prediction horizon, between window end and onset
hx0, hx1 = wx2 + 16 * 6.2 - 1.2, onset2 - 1.5
ax.annotate("", xy=(hx1, ty2 + 4.5), xytext=(hx0 + 2, ty2 + 4.5), zorder=8,
            arrowprops=dict(arrowstyle="<|-|>,head_width=0.11,head_length=0.26",
                            color=INK, linewidth=0.7, shrinkA=0, shrinkB=0))
txt(ax, (hx0 + hx1) / 2, ty2 - 9.5, "1–2 s horizon", 6.1, INK, "bold", "center")
txt(ax, wx2 - 2, ty2 - 9.5, "observation", 6.1, INK, "bold")

check(ax, x0 + 16, 176)
txt(ax, x0 + 25, 176, "Pre-crossing observations only", 6.9, CLEAN, "bold")
txt(ax, x0 + 14, 163, "16 frames  ·  stride 8  ·  30–60 frames before the event",
    6.3, INK_2)

arrow(ax, x1 + 1, 332, S[3][0] - 2, 352, INK_2)
arrow(ax, x1 + 1, 205, S[3][0] - 2, 300, INK_2)

# ==================================== 3  FRAME-LEVEL TEMPORAL AUDIT
x0, x1 = S[3]
bw = x1 - x0
box(ax, x0, 226, bw, 170, "white", INK_2, 0.9)

magnifier(ax, x0 + 16, 380)
txt(ax, x0 + 29, 381, "Audit the frames,", 7.2, INK, "bold")
txt(ax, x0 + 29, 372, "not the sampling rule", 7.2, INK, "bold")

box(ax, x0 + 9, 334, bw - 18, 24, BLUE_F, BLUE, 0.6, 2.5)
fit(ax, x0 + 15, 346, "generated observation frames", bw - 30, 6.3, INK_2)
txt(ax, x0 + 15, 338, "compared against", 6.0, MUTED, style="italic")
box(ax, x0 + 9, 306, bw - 18, 20, GREY_F, INK_2, 0.6, 2.5)
fit(ax, x0 + 15, 316, "frame-level crossing annotations", bw - 30, 6.3, INK_2)

fit(ax, x0 + 9, 295, "Contaminated if \u2265 1 observed", bw - 18, 6.2, LEAK, "bold")
fit(ax, x0 + 9, 287, "frame is already crossing", bw - 18, 6.2, LEAK, "bold")

ax.plot([x0 + 9, x1 - 9], [278, 278], color=RULE, lw=0.6, zorder=3)
rows = [("PIE", "67.9%", "0.0%"), ("JAAD", "93.0%", "0.0%"), ("IDD-PeD", "81.3%", "0.0%")]
txt(ax, x0 + 72, 269, "track-end", 5.6, MUTED, ha="center")
txt(ax, x0 + 116, 269, "event", 5.6, MUTED, ha="center")
for k, (d, bad, good) in enumerate(rows):
    yy = 258 - k * 12
    fit(ax, x0 + 9, yy, d, 40, 6.2, INK_2, "bold")
    warn(ax, x0 + 54, yy, 3.2)
    txt(ax, x0 + 60, yy, bad, 6.0, LEAK)
    check(ax, x0 + 103, yy, 2.8)
    txt(ax, x0 + 109, yy, good, 6.0, CLEAN)

arrow(ax, x1 + 1, 316, S[4][0] - 2, 330, INK_2)

# ================================ 4  PHASE-CONTROLLED SAMPLING
x0, x1 = S[4]
bw = x1 - x0

# --- before: the residual phase difference
box(ax, x0, 288, bw, 108, LEAK_F, LEAK, 0.9)
txt(ax, x0 + 10, 383, "Event-anchored, but still phase-separated", 7.4, LEAK, "bold")

for k, (lab, wstart, col) in enumerate([("crossing", 6, BLUE), ("non-crossing", 30, BLUE)]):
    yy = 352 - k * 26
    fit(ax, x0 + 10, yy + 4.5, lab, 52, 6.2, INK_2)
    bxs = x0 + 66
    strip(ax, bxs, yy, 32, 4.2, 0.9, GREY_F, MUTED, 9, 0.35)
    strip(ax, bxs + wstart * 5.1, yy, 8, 4.2, 0.9, BLUE_F, col, 9, 0.4, z=8)
    ax.plot([bxs + 32 * 5.1 - 0.9, bxs + 32 * 5.1 - 0.9], [yy - 3, yy + 12],
            color=INK_2, lw=0.8, zorder=6)

fit(ax, x0 + 10, 311, "distance from observation end to track termination",
    bw - 20, 6.1, INK, "bold")
fit(ax, x0 + 10, 302, "separates the classes at AUC 1.0000", bw - 20, 6.0, INK_2)
fit(ax, x0 + 10, 294, "diagnostic only — never supplied to the models",
    bw - 20, 6.0, MUTED, style="italic")

# --- the control
arrow(ax, x0 + bw / 2, 286, x0 + bw / 2, 268, INK_2, 1.0)
balance(ax, x0 + bw / 2 + 46, 277)
txt(ax, x0 + bw / 2 + 62, 277, "match the phase", 6.2, INK_2, "bold")

box(ax, x0, 150, bw, 108, CLEAN_F, CLEAN, 0.9)
txt(ax, x0 + 10, 245, "Phase-matched control", 7.6, CLEAN, "bold")

for k, (lab, wstart) in enumerate([("crossing", 6), ("non-crossing", 9)]):
    yy = 214 - k * 26
    fit(ax, x0 + 10, yy + 4.5, lab, 52, 6.2, INK_2)
    bxs = x0 + 66
    strip(ax, bxs, yy, 32, 4.2, 0.9, GREY_F, MUTED, 9, 0.35)
    strip(ax, bxs + wstart * 5.1, yy, 8, 4.2, 0.9, CLEAN_F, CLEAN, 9, 0.4, z=8)
    ax.plot([bxs + 32 * 5.1 - 0.9, bxs + 32 * 5.1 - 0.9], [yy - 3, yy + 12],
            color=INK_2, lw=0.8, zorder=6)
ax.plot([x0 + 66 + 7 * 5.1, x0 + 66 + 7 * 5.1], [178, 226], color=CLEAN,
        lw=0.7, ls=(0, (2.2, 1.8)), zorder=9)

fit(ax, x0 + 10, 172, "positives unchanged  ·  negatives resampled earlier",
    bw - 20, 6.2, INK_2)
fit(ax, x0 + 10, 163, "rule fitted on training positives only, then frozen",
    bw - 20, 6.2, INK_2)
fit(ax, x0 + 10, 154, "separation falls to AUC 0.79 — reduced, not removed",
    bw - 20, 6.1, LEAK, "bold")

arrow(ax, x1 + 1, 330, S[5][0] - 2, 300, INK_2)
arrow(ax, x1 + 1, 200, S[5][0] - 2, 262, INK_2)

# ============================ 5  INPUT REPRESENTATION AND MODELS
x0, x1 = S[5]
bw = x1 - x0

box(ax, x0, 330, bw, 66, BLUE_F, BLUE, 0.9)
txt(ax, x0 + 10, 384, "Input contract — five numbers per frame", 7.2, INK, "bold")
txt(ax, x0 + 10, 370, "pedestrian box", 6.3, INK_2)
txt(ax, x1 - 10, 370, "$x_1,\\,y_1,\\,x_2,\\,y_2$", 6.8, BLUE, "bold", ha="right")
txt(ax, x0 + 10, 358, "ego-vehicle speed", 6.3, INK_2)
txt(ax, x1 - 10, 358, "$v_{\\mathrm{ego}}$", 6.8, BLUE, "bold", ha="right")
ax.plot([x0 + 10, x1 - 10], [350, 350], color=BLUE, lw=0.5, alpha=0.5, zorder=3)
txt(ax, x0 + 10, 340, "16-frame sequence  ×  5 features", 7.0, INK, "bold")

box(ax, x0, 296, bw, 26, GREY_F, INK_2, 0.7)
txt(ax, x0 + 10, 313, "Standardisation", 6.6, INK, "bold")
txt(ax, x0 + 10, 303, "$\\mu,\\sigma$ from the training partition only", 6.2, INK_2)
arrow(ax, x0 + bw / 2, 329, x0 + bw / 2, 323, INK_2, 0.8)

# input configurations
txt(ax, x0, 284, "Input configurations", 6.4, MUTED)
cfgs = ["Box + speed", "Box only", "Speed only", "Final frame"]
cw = (bw - 3 * 4) / 4
for k, c in enumerate(cfgs):
    cx = x0 + k * (cw + 4)
    box(ax, cx, 258, cw, 18, "white", BLUE, 0.6, 2.2)
    fit(ax, cx + cw / 2, 267, c, cw - 4, 5.9, INK_2, ha="center")
arrow(ax, x0 + bw / 2, 295, x0 + bw / 2, 277, INK_2, 0.8)

# --- A: temporal sequence models
box(ax, x0, 178, bw, 70, PANEL, INK_2, 0.9)
txt(ax, x0 + 10, 238, "A   Temporal sequence models", 7.0, INK, "bold")
fams = [("BiLSTM", BLUE), ("GRU", TEAL), ("Vanilla RNN", PURPLE), ("Transformer", ORANGE)]
mw = (bw - 20 - 3 * 5) / 4
for k, (name, col) in enumerate(fams):
    mx = x0 + 10 + k * (mw + 5)
    box(ax, mx, 188, mw, 40, "white", col, 0.8, 2.5)
    netglyph(ax, mx + mw / 2, 211, col)
    fit(ax, mx + mw / 2, 193, name, mw - 3, 5.9, INK_2, "bold", ha="center")
arrow(ax, x0 + bw / 2, 257, x0 + bw / 2, 249, INK_2, 0.8)

# --- B: simple raw-feature baselines
box(ax, x0, 118, bw, 52, "white", INK_2, 0.9, ls=(0, (3, 2)))
txt(ax, x0 + 10, 160, "B   Simple raw-feature baselines", 7.0, INK, "bold")
box(ax, x0 + 10, 129, bw * 0.56, 24, GREY_F, INK_2, 0.7, 2.5)
txt(ax, x0 + 16, 145, "Logistic regression", 6.4, INK, "bold")
txt(ax, x0 + 16, 135, "flattened raw sequence", 6.0, INK_2)
box(ax, x0 + 16 + bw * 0.56, 129, bw * 0.38, 24, GREY_F, INK_2, 0.7, 2.5)
txt(ax, x0 + 22 + bw * 0.56, 145, "Always-positive", 6.4, INK, "bold")
txt(ax, x0 + 22 + bw * 0.56, 135, "trivial reference", 6.0, INK_2)

arrow(ax, x1 + 1, 210, S[6][0] - 2, 250, INK_2)
arrow(ax, x1 + 1, 142, S[6][0] - 2, 230, INK_2)

# ==================== 6  EVALUATION AND COMPARATIVE FINDINGS
x0, x1 = S[6]
bw = x1 - x0

box(ax, x0, 322, bw, 74, "white", INK_2, 0.9)
txt(ax, x0 + 10, 386, "Selection and scoring", 7.2, INK, "bold")
fit(ax, x0 + 10, 374, "validation-fitted configuration and threshold", bw - 20, 6.2, INK_2)
fit(ax, x0 + 10, 365, "five random seeds for every neural model", bw - 20, 6.2, INK_2)
txt(ax, x0 + 10, 355, "test partition scored once", 6.2, INK, "bold")
ax.plot([x0 + 10, x1 - 10], [347, 347], color=RULE, lw=0.6, zorder=3)
mets = ["ROC-AUC", "PR-AUC", "F1", "Accuracy", "Precision", "Recall"]
for k, m in enumerate(mets):
    txt(ax, x0 + 12 + (k % 3) * 80, 338 - (k // 3) * 9, "· " + m, 5.8, INK_2)

box(ax, x0, 288, bw, 34, GREY_F, INK_2, 0.7)
txt(ax, x0 + 10, 312, "Uncertainty and multiplicity", 6.6, INK, "bold")
fit(ax, x0 + 10, 302, "pedestrian-cluster bootstrap  ·  10,000 resamples", bw - 20, 6.1, INK_2)
fit(ax, x0 + 10, 293, "Holm–Bonferroni correction within each test family", bw - 20, 6.1, INK_2)
arrow(ax, x0 + bw / 2, 321, x0 + bw / 2, 323, INK_2, 0.8)

# --- the findings panel
box(ax, x0, 150, bw, 128, "white", INK, 1.1)
txt(ax, x0 + 10, 267, "Protocol-sensitive evaluation", 8.2, INK, "bold")
ax.plot([x0 + 10, x1 - 10], [259, 259], color=INK, lw=0.6, zorder=3)

finds = [
    ("Temporal validity", "are the observations genuinely pre-crossing?"),
    ("Feature sensitivity", "bounding box vs. ego-vehicle speed"),
    ("Model complexity", "neural families vs. a linear baseline"),
]
for k, (t, s) in enumerate(finds):
    yy = 246 - k * 19
    ax.add_patch(Circle((x0 + 15, yy - 1), 4.6, facecolor=PANEL, edgecolor=INK_2,
                        lw=0.6, zorder=5))
    txt(ax, x0 + 15, yy - 1, str(k + 1), 5.7, INK, "bold", "center", "center", 6)
    txt(ax, x0 + 24, yy + 2.5, t, 6.5, INK, "bold")
    fit(ax, x0 + 24, yy - 5.5, s, bw - 36, 6.0, INK_2)

# the two protocols, side by side
ax.plot([x0 + 10, x1 - 10], [186, 186], color=RULE, lw=0.6, zorder=3)
pw = (bw - 24) / 2
box(ax, x0 + 10, 156, pw, 26, LEAK_F, LEAK, 0.7, 2.5)
fit(ax, x0 + 10 + pw / 2, 173, "Event-anchored PIE", pw - 6, 6.3, LEAK, "bold", ha="center")
fit(ax, x0 + 10 + pw / 2, 163, "families separable", pw - 6, 5.9, INK_2, ha="center")
box(ax, x0 + 14 + pw, 156, pw, 26, CLEAN_F, CLEAN, 0.7, 2.5)
fit(ax, x0 + 14 + pw + pw / 2, 173, "Phase-matched PIE", pw - 6, 6.3, CLEAN, "bold", ha="center")
fit(ax, x0 + 14 + pw + pw / 2, 163, "ranking not resolvable", pw - 6, 5.9, INK_2, ha="center")
ax.annotate("", xy=(x0 + 13 + pw, 169), xytext=(x0 + 8 + pw, 169), zorder=8,
            arrowprops=dict(arrowstyle="-|>,head_width=0.13,head_length=0.28",
                            color=INK_2, linewidth=0.8, shrinkA=0, shrinkB=0))

# --- external branch
box(ax, x0, 108, bw, 34, "white", MUTED, 0.7, ls=(0, (3, 2)))
txt(ax, x0 + 10, 133, "External evaluation", 6.5, INK_2, "bold")
txt(ax, x0 + 10, 123, "PIE " + ARROW + " IDD-PeD zero-shot transfer", 6.0, INK_2)
txt(ax, x0 + 10, 114, "IDD-PeD in-domain training", 6.0, INK_2)
arrow(ax, x0 + bw / 2, 149, x0 + bw / 2, 143, MUTED, 0.7)

# ------------------------------------------------------- the core message
ax.plot([S[1][0], S[6][1]], [80, 80], color=RULE, lw=0.7, zorder=2)
txt(ax, S[1][0], 64, "The contribution is the protocol, not the architecture.",
    7.8, INK, "bold")
chain = ["sample construction", "temporal audit", "phase control",
         "controlled input and model comparison", "clustered, corrected evaluation"]
cx = S[1][0]
for k, c in enumerate(chain):
    t = txt(ax, cx, 46, c, 6.8, INK_2)
    fig.canvas.draw()
    wpx = t.get_window_extent(fig.canvas.get_renderer()).width
    wdata = wpx / fig.dpi * 100
    cx += wdata + 6
    if k < len(chain) - 1:
        arrow(ax, cx - 4, 46, cx + 5, 46, MUTED, 0.7)
        cx += 11

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_pipeline.{ext}", dpi=400, facecolor="white",
                bbox_inches="tight", pad_inches=0.06)
print("wrote", OUT / "fig_pipeline.pdf")
print("wrote", OUT / "fig_pipeline.png")

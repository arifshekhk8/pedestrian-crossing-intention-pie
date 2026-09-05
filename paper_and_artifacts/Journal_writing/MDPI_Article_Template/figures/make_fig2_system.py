"""make_fig2_system.py — Figure 2: how a window is built, encoded, and served.

Panel (a) is one continuous path. The upper lane cuts an observation window out
of a pedestrian track so that the crossing it predicts falls outside it; the
lower lane encodes exactly that window. They are one panel because the contract
between them is the argument of the paper: the extraction rule fixes what a
window may contain, and every family then sees the same window.

Panel (b) is the detector-in-the-loop path, used for the latency and
detector-in-the-loop experiments only.

Design rules, kept deliberately tight:
  * One accent (#2a78d6) and nothing else coloured. It marks what the model
    observes: the annotated box in the thumbnails and the sixteen-frame window.
    The crossing event is drawn in ink, because it is the thing never observed.
  * White boxes, hairline borders, no fills or tints carrying meaning of their
    own; type does the work instead.
  * Nothing on the canvas that the caption or the text already says.

The layout is one axes in square units, x in [0, 100] and y increasing downward,
so a rounding radius is the same size wherever it is drawn. Every coordinate and
every label is a named constant in the LAYOUT and WORDING blocks; the editable
draw.io twin imports both rather than re-typing them.

Numbers on the figure and where they come from:
    16 frames, TTE 30-60 frames        journal_prep/issue2_clean_protocol
    5 -> 64 projection, four families  Section 3.4 of the manuscript
    92.7% / 4.5% of per-frame cost     journal_prep/issue9_latency
The three thumbnails come from prep_fig2_track.py (PIE video_0016, pedestrian
3_16_919, frames 2467/2492/2517; heads blurred in the pixels).

Run:  python make_fig2_system.py  ->  fig2_system.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figstyle import ACCENT, INK, INK_2, RULE, use_style

HERE = Path(__file__).resolve().parent

HAIR = "#c7c6bc"       # box borders
WHITE = "#ffffff"
# Every label on this figure is set solid black; grey is for rules and
# borders only. The one exception is the accent on "16 observed frames".
BLACK = "#000000"

# Two variants of the same figure. WITHOUT icons is what the manuscript uses;
# WITH icons is the alternative asked for separately and written elsewhere, so
# the paper is never changed by building it. Everything below that differs
# between the two is chosen here and nowhere else.
WITH_ICONS = "--icons" in sys.argv

W = 100.0              # canvas width in layout units, shared by both figures
H_MODEL = 39.0 if WITH_ICONS else 36.0    # each figure's own height
H_PIPE = 17.0 if WITH_ICONS else 15.6
FIG_W_IN = 7.1         # scaled to ~394 pt of MDPI text width, as the other figures
L, R = 2.5, 97.1       # the column both figures are set in

# --------------------------------------------------------------------- LAYOUT
# Figure "system": the window, then the model that encodes it. No heading — the
# caption is the title, which is what a journal figure wants.
SRC_TOP, SRC_BOT = 2.0, 18.4              # the PIE track panel
SRC_R = 34.0
SRC_TITLE_Y, SRC_CAP_Y = 4.4, 16.7
THUMB_TOP, THUMB_W, THUMB_GAP = 6.4, 9.7, 0.7
SRC_MID = (SRC_TOP + SRC_BOT) / 2

EX_LABEL_X, EX_LABEL_Y = 39.5, 3.8        # the extraction lane
TRACK_L, TRACK_Y = 38.5, 10.4
WIN_L, WIN_R, WIN_TOP, WIN_BOT = 39.5, 61.0, 8.4, 12.4
EVENT_X, EVENT_TOP, EVENT_BOT = 88.0, 6.8, 14.0
EVENT_LBL_Y, WIN_LBL_Y = 5.4, 14.0        # above the event, below the window
TTE_Y, TTE_LBL_Y = 15.4, 16.9

DROP_X, WRAP_Y, WRAP_X = 41.1, 21.4, 11.5  # the window drops into the lower lane

ROW2_TOP = 24.4
if WITH_ICONS:
    # Each stage carries an icon over its title, on one band, so the row reads
    # as a row; ICON_H is the drawn height and the width follows the aspect.
    ROW2_BOT = 37.0
    ICON_TOP, ICON_H = 25.9, 3.4
    STAGE_TITLE_Y = 30.7
else:
    ROW2_BOT = 34.0
    ICON_TOP, ICON_H = None, None
    STAGE_TITLE_Y = None
MID_B = (ROW2_TOP + ROW2_BOT) / 2
# (x0, x1) of the five stages of the lower lane
STREAMS_BOX = (2.5, 20.5)
PREP_BOX = (23.4, 43.4)
ENC_BOX = (46.3, 70.6)
READ_BOX = (73.5, 87.0)
OUT_BOX = (89.9, 97.1)
ENC_TITLE_Y = 30.7 if WITH_ICONS else 26.6
FAM_ROWS = (32.6, 35.4) if WITH_ICONS else (29.6, 32.3)  # family rows
FAM_COL, FAM_PAD = 9.0, 5.1               # column pitch and left inset
CHIP_H, CHIP_PAD = 2.6, 1.2               # a stream chip, on the family rows

OBD_Y = 2.8                               # figure "pipeline", its own canvas
STAGE_TOP = 6.2
STAGE_BOT = 15.4 if WITH_ICONS else 13.5
PIPE_ICON_TOP = 7.1 if WITH_ICONS else None
PIPE_TITLE_Y = 11.9 if WITH_ICONS else None

FS_BOX, FS_SUB = 7.6, 6.8                 # a box title / its detail
FS_MARK, FS_FAM = 7.0, 6.7                # an annotation on the lane / a family

# -------------------------------------------------------------------- WORDING
# Held out of the drawing code because the editable draw.io twin of this figure
# (../../drawio/make_drawio_figures.py) imports these strings. Two copies of one
# figure that disagree about their own labels are worse than no second copy.
T_SRC, T_SRC_CAP = "PIE pedestrian track", "annotated box + ego speed, per frame"
T_EX = "Event-anchored extraction"
T_WIN, T_EVENT, T_TTE = "16 observed frames", "crossing event", "30–60 frames (1–2 s)"
T_STREAMS = "Two input streams"
T_STREAM_BOX, T_STREAM_SPD = "Box trajectory  16 × 4", "Ego speed  v   16 × 1"
T_PREP, T_PREP_SUB = "Preprocessing", "train-only z-score\nLinear 5 → 64  +  ReLU"
T_ENC = "Temporal encoder"
# Exactly the names the other figures use, so a family reads the same wherever
# it appears (figstyle.FAMILY_COLOR, make_fig5_curves.py, make_fig8_latency.py).
FAMILIES = ["BiLSTM", "Transformer",
            "GRU", "Vanilla RNN"]
T_READ, T_READ_SUB = "Read-out + head", "linear → logit → σ"
T_OUT = "p(cross)"

T_OBD = "OBD ego speed"
# (x0, x1, title, detail, icon), gaps a uniform 3.0.
STAGES = [
    (2.5, 16.0, "Video frame", "1920 × 1080, 30 fps", "video"),
    (19.0, 32.5, "YOLO26-M", "person detector", "scan"),
    (35.5, 48.0, "ByteTrack", "track identity", "route"),
    (51.0, 65.5, "16-frame ring buffer", "per track", "stack-2"),
    (68.5, 83.5, "Crossing-intention\nmodel", None, "binary-tree"),
    (86.5, 97.1, "p(cross)", None, "walk"),
]


def _mt(s):
    """Helvetica has no U+2192, so the shared label strings carry a real arrow
    and matplotlib gets the mathtext one. Left alone it prints a tofu box."""
    return s.replace("→", "$\\rightarrow$")


def frame(ax, x0, y0, x1, y1, *, ec=HAIR, lw=0.7, ls="solid", fc=WHITE, r=0.6,
          z=2):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z))


def icon(ax, name, cx, top, h=ICON_H):
    """One stage icon, centred on cx and hanging from `top`.

    The PNGs are square and already black on transparent (prep_fig2_icons.py),
    so there is nothing to recolour here.
    """
    img = plt.imread(HERE / f"_icon_{name}.png")
    w = h * img.shape[1] / img.shape[0]
    ax.imshow(img, extent=(cx - w / 2, cx + w / 2, top + h, top),
              aspect="auto", zorder=4, interpolation="antialiased")


def box(ax, x0, x1, title, sub=None, *, y0=ROW2_TOP, y1=ROW2_BOT, fs=FS_BOX,
        sub_fs=FS_SUB, ls="solid", ec=HAIR, lw=0.7, bold=True, ico=None,
        ico_top=None, title_y=None):
    """A stage: a hairline box with a title and, optionally, an icon over it and
    muted detail under it.

    With no icon, title and detail are stacked as one block and centred
    together; centring each separately is what leaves a two-line caption sitting
    low in its box. With an icon, the icon and the title sit on fixed bands
    instead, so every stage in a row lines up with its neighbours.
    """
    frame(ax, x0, y0, x1, y1, ec=ec, lw=lw, ls=ls)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    weight = "bold" if bold else "normal"
    unit_pt = 72 * FIG_W_IN / W                   # layout unit in points

    if ico:
        icon(ax, ico, cx, ico_top if ico_top is not None else ICON_TOP)
        ty = title_y if title_y is not None else STAGE_TITLE_Y
        ax.text(cx, ty, _mt(title), ha="center", va="center", fontsize=fs,
                color=BLACK, fontweight=weight, zorder=4, linespacing=1.3)
        if sub:
            ht = (title.count("\n") + 1) * fs * 1.30 / unit_pt
            hs = (sub.count("\n") + 1) * sub_fs * 1.30 / unit_pt
            ax.text(cx, ty + ht / 2 + 0.45 + hs / 2, _mt(sub), ha="center",
                    va="center", fontsize=sub_fs, color=BLACK, zorder=4,
                    linespacing=1.4)
        return

    if not sub:
        ax.text(cx, cy, _mt(title), ha="center", va="center", fontsize=fs,
                color=BLACK, fontweight=weight, zorder=4)
        return
    ht = (title.count("\n") + 1) * fs * 1.30 / unit_pt
    hs = (sub.count("\n") + 1) * sub_fs * 1.30 / unit_pt
    top = cy - (ht + 0.5 + hs) / 2
    ax.text(cx, top + ht / 2, _mt(title), ha="center", va="center", fontsize=fs,
            color=BLACK, fontweight=weight, zorder=4, linespacing=1.3)
    ax.text(cx, top + ht + 0.5 + hs / 2, _mt(sub), ha="center", va="center",
            fontsize=sub_fs, color=BLACK, zorder=4, linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, *, color=RULE, lw=0.8, z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def seg(ax, xs, ys, *, color=RULE, lw=0.8, ls="solid", z=2):
    ax.add_line(Line2D(xs, ys, color=color, linewidth=lw, linestyle=ls,
                       solid_capstyle="butt", zorder=z))


def draw_model(ax):
    """Figure "system": the window is cut, then encoded."""
    # ------------------------------------------------- the dataset, as pixels
    frame(ax, L, SRC_TOP, SRC_R, SRC_BOT)
    ax.text((L + SRC_R) / 2, SRC_TITLE_Y, T_SRC, ha="center", va="center",
            fontsize=FS_BOX, fontweight="bold", color=BLACK, zorder=4)
    strip = 3 * THUMB_W + 2 * THUMB_GAP
    x = (L + SRC_R) / 2 - strip / 2
    for i in range(3):
        img = plt.imread(HERE / f"_track{i + 1}.png")
        h = THUMB_W * img.shape[0] / img.shape[1]
        ax.imshow(img, extent=(x, x + THUMB_W, THUMB_TOP + h, THUMB_TOP),
                  aspect="auto", zorder=3, interpolation="antialiased")
        frame(ax, x, THUMB_TOP, x + THUMB_W, THUMB_TOP + h, ec=HAIR, lw=0.5,
              fc="none", r=0, z=4)
        x += THUMB_W + THUMB_GAP
    ax.text((L + SRC_R) / 2, SRC_CAP_Y, T_SRC_CAP, ha="center", va="center",
            fontsize=FS_SUB, color=BLACK, zorder=4)
    arrow(ax, SRC_R + 0.5, SRC_MID, TRACK_L - 0.9, SRC_MID)

    # ------------------------------------------ the rule that cuts the window
    ax.text(EX_LABEL_X, EX_LABEL_Y, T_EX, ha="left", va="center",
            fontsize=FS_BOX, fontweight="bold", color=BLACK)

    seg(ax, [TRACK_L, EVENT_X], [TRACK_Y, TRACK_Y], color=RULE, lw=0.9)

    frame(ax, WIN_L, WIN_TOP, WIN_R, WIN_BOT, ec=ACCENT, lw=1.0, r=0.4, z=3)
    for i in range(16):
        x = WIN_L + (i + 0.5) * (WIN_R - WIN_L) / 16
        seg(ax, [x, x], [WIN_TOP + 0.5, WIN_BOT - 0.5], color=ACCENT, lw=0.5,
            z=4)
    ax.text((WIN_L + WIN_R) / 2, WIN_LBL_Y, T_WIN, ha="center", va="center",
            fontsize=FS_MARK, fontweight="bold", color=ACCENT)

    seg(ax, [EVENT_X, EVENT_X], [EVENT_TOP, EVENT_BOT], color=INK, lw=1.4, z=4)
    ax.text(EVENT_X, EVENT_LBL_Y, T_EVENT, ha="center", va="center",
            fontsize=FS_MARK, fontweight="bold", color=BLACK)

    ax.add_patch(FancyArrowPatch((WIN_R, TTE_Y), (EVENT_X, TTE_Y),
                                 arrowstyle="<|-|>", mutation_scale=6,
                                 color=BLACK, lw=0.8, shrinkA=0, shrinkB=0,
                                 zorder=4))
    ax.text((WIN_R + EVENT_X) / 2, TTE_LBL_Y, T_TTE, ha="center", va="center",
            fontsize=FS_SUB, color=BLACK)

    # the window drops out of the lane above and is encoded below
    seg(ax, [DROP_X, DROP_X], [WIN_BOT, WRAP_Y], color=RULE)
    seg(ax, [DROP_X, WRAP_X], [WRAP_Y, WRAP_Y], color=RULE)
    arrow(ax, WRAP_X, WRAP_Y, WRAP_X, ROW2_TOP - 0.5)

    # ------------------------------------------------------- the model itself
    frame(ax, STREAMS_BOX[0], ROW2_TOP, STREAMS_BOX[1], ROW2_BOT)
    # the two streams are two icons, side by side, over the one title
    cxs = sum(STREAMS_BOX) / 2
    if WITH_ICONS:
        for dx, name in ((-3.4, "crop"), (3.4, "gauge")):
            icon(ax, name, cxs + dx, ICON_TOP)
    ax.text(cxs, STAGE_TITLE_Y if WITH_ICONS else ENC_TITLE_Y, T_STREAMS,
            ha="center", va="center",
            fontsize=FS_BOX, fontweight="bold", color=BLACK, zorder=4)
    for row, label in enumerate((T_STREAM_BOX, T_STREAM_SPD)):
        box(ax, STREAMS_BOX[0] + CHIP_PAD, STREAMS_BOX[1] - CHIP_PAD, label,
            y0=FAM_ROWS[row] - CHIP_H / 2, y1=FAM_ROWS[row] + CHIP_H / 2,
            fs=FS_SUB, bold=False)
    box(ax, *PREP_BOX, T_PREP, T_PREP_SUB, ico="adjustments" if WITH_ICONS else None)
    box(ax, *READ_BOX, T_READ, T_READ_SUB, ico="arrows-join" if WITH_ICONS else None)
    box(ax, *OUT_BOX, T_OUT, fs=8.2, ico="walk" if WITH_ICONS else None)

    frame(ax, *ENC_BOX[:1], ROW2_TOP, ENC_BOX[1], ROW2_BOT, ec=INK_2, lw=0.9,
          ls=(0, (3.4, 2.2)))
    if WITH_ICONS:
        icon(ax, "binary-tree", sum(ENC_BOX) / 2, ICON_TOP)
    ax.text(sum(ENC_BOX) / 2, ENC_TITLE_Y, T_ENC, ha="center", va="center",
            fontsize=FS_BOX, fontweight="bold", color=BLACK, zorder=4)
    for i, name in enumerate(FAMILIES):
        ax.text(ENC_BOX[0] + FAM_PAD + (i % 2) * FAM_COL, FAM_ROWS[i // 2], name,
                ha="left", va="center", fontsize=FS_FAM, color=BLACK, zorder=4)

    for a, b in ((STREAMS_BOX, PREP_BOX), (PREP_BOX, ENC_BOX),
                 (ENC_BOX, READ_BOX), (READ_BOX, OUT_BOX)):
        arrow(ax, a[1] + 0.5, MID_B, b[0] - 0.5, MID_B)


def draw_pipeline(ax):
    """Figure "pipeline": the same predictor behind a detector and a tracker."""
    mid = (STAGE_TOP + STAGE_BOT) / 2
    for x0, x1, title, sub, ico in STAGES:
        box(ax, x0, x1, title, sub, y0=STAGE_TOP, y1=STAGE_BOT,
            ico=ico if WITH_ICONS else None, ico_top=PIPE_ICON_TOP,
            title_y=PIPE_TITLE_Y)
    for a, b in zip(STAGES, STAGES[1:]):
        arrow(ax, a[1] + 0.5, mid, b[0] - 0.5, mid)

    ring = (STAGES[3][0] + STAGES[3][1]) / 2
    ax.text(ring, OBD_Y, T_OBD, ha="center", va="center", fontsize=FS_SUB,
            color=BLACK)
    arrow(ax, ring, OBD_Y + 1.1, ring, STAGE_TOP - 0.4)


def canvas(height):
    """A figure of the shared width, in square layout units, y increasing down."""
    fig = plt.figure(figsize=(FIG_W_IN, FIG_W_IN * height / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(height, 0)     # y downward: the layout reads like the figure
    ax.set_axis_off()
    return fig, ax


def main():
    """Default: the manuscript figures, into this folder. With `--icons`, the
    icon variant, and with `--out-dir DIR` into DIR instead, so building one
    never overwrites the other."""
    out_dir = HERE
    if "--out-dir" in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1]).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    use_style()
    for name, height, draw in (("fig2_system", H_MODEL, draw_model),
                               ("fig2b_pipeline", H_PIPE, draw_pipeline)):
        fig, ax = canvas(height)
        draw(ax)
        for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
            out = out_dir / f"{name}.{ext}"
            fig.savefig(out, facecolor="white", **kw)
            print(f"wrote {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()

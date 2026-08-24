"""figstyle.py — shared visual language for every figure in the manuscript.

One palette, one set of ink tokens, one chrome treatment, so the figures read as
a single document rather than a folder of unrelated plots.

Palette validated with the dataviz validator (light mode, surface #fcfcfb):
    #2a78d6, #d1622b, #1f9b7a, #8b5cc7
    lightness band PASS - chroma floor PASS - CVD separation PASS (worst
    adjacent pair dE 10.7 deutan) - normal-vision floor PASS (dE 24.1) -
    contrast vs surface PASS (all >= 3:1).

Rules kept throughout: one axis per panel (never a second y-scale), recessive
grid and axes, selective direct labels (never a number on every mark), text in
ink tokens rather than the series color, and a legend whenever two or more
series share a panel.
"""

import matplotlib as mpl

# ------------------------------------------------------------------ series
# Fixed order; a family keeps its hue in every figure it appears in.
BILSTM = "#2a78d6"
TRANSF = "#d1622b"
GRU = "#1f9b7a"
RNN = "#8b5cc7"

FAMILY_COLOR = {
    "BiLSTM": BILSTM,
    "Transformer": TRANSF,
    "GRU": GRU,
    "RNN": RNN,
    "Vanilla RNN": RNN,
}

ACCENT = BILSTM       # the single-emphasis hue
CONTEXT = "#b9b8b2"   # de-emphasis gray for context marks
LEAK = "#b4433a"      # status: the leaked region (paired with an explicit label)
CLEAN = "#1f9b7a"     # status: the leakage-free region

# -------------------------------------------------------------------- ink
INK = "#0b0b0b"       # primary ink: titles, emphasized values
INK_2 = "#52514e"     # secondary ink: labels, ordinary values
MUTED = "#898781"     # tertiary ink: axis ticks, captions, units
GRID = "#e1e0d9"      # hairline gridline
RULE = "#c3c2b7"      # axis spine
PANEL = "#f4f3ee"     # a filled panel behind a schematic block


def use_style():
    """Base rcParams. Type is sized so that, after the figure is scaled to the
    MDPI text width (~16 cm), on-figure text lands near 8 pt."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def style_axes(ax, left=True, bottom=True):
    """Recessive chrome: drop the box, mute what remains."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(colors=MUTED, length=0, pad=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def panel_title(ax, text, subtitle=None, pad=8, y=-0.155):
    """Left-aligned panel title in primary ink, optional muted subtitle."""
    ax.set_title(text, loc="left", color=INK, fontweight="bold", pad=pad)
    if subtitle:
        ax.text(0, y, subtitle, transform=ax.transAxes,
                fontsize=8.5, color=MUTED, va="top")


def hide_frame(ax):
    """For schematic panels that carry no scale at all."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

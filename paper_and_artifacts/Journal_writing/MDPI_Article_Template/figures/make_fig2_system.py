"""make_fig2_system.py — Figure 2: the model and the live pipeline.

Panel (a) is the two-stream crossing-intention model. Everything outside the
dashed block is held fixed across the whole study; only the temporal encoder is
swapped, which is what makes the four-family comparison an isolation of one
design choice rather than a comparison of four different systems.

Panel (b) is the deployed perception-to-prediction pipeline used for the latency
and detector-in-the-loop experiments, annotated with the measured share of
per-frame cost (journal_prep/issue9_latency).

Run:  python make_fig2_system.py  ->  fig2_system.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figstyle import (BILSTM, GRU, INK, INK_2, MUTED, PANEL, RNN, RULE, TRANSF,
                      hide_frame, use_style)

HERE = Path(__file__).resolve().parent


def box(ax, x, y, w, h, text, *, fc=PANEL, ec=RULE, tc=INK, fs=7.8,
        weight="normal", lw=0.8, ls="solid"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=weight, zorder=4, linespacing=1.35)
    return x + w


def arrow(ax, x0, y0, x1, y1, color=RULE):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=8, color=color, lw=0.9,
                                 shrinkA=0, shrinkB=0, zorder=2))


def main():
    use_style()
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(7.1, 4.5), gridspec_kw={"height_ratios": [1.22, 1.0]})

    # ================================================================ panel (a)
    ax_a.text(0, 1.02, "(a)  Two-stream crossing-intention model",
              transform=ax_a.transAxes, fontsize=9.5, fontweight="bold", color=INK)
    ax_a.text(0, 0.905,
              "Only the dashed encoder block changes across the four families; every "
              "other component and the whole training recipe are frozen.",
              transform=ax_a.transAxes, fontsize=8.3, color=MUTED)

    mid = 0.50
    # Inputs -------------------------------------------------------------
    box(ax_a, 0.005, mid + 0.055, 0.135, 0.16,
        "Pedestrian box\n$x_1, y_1, x_2, y_2$", fc="#e9f0fa", ec="#a9c4e8")
    box(ax_a, 0.005, mid - 0.215, 0.135, 0.16,
        "Ego-vehicle speed\n$v$  (OBD)", fc="#e9f0fa", ec="#a9c4e8")
    ax_a.text(0.0725, mid + 0.245, "two streams", ha="center", fontsize=7.6,
              color=MUTED, style="italic")

    arrow(ax_a, 0.145, mid + 0.135, 0.178, mid + 0.06)
    arrow(ax_a, 0.145, mid - 0.135, 0.178, mid - 0.06)

    # Window -------------------------------------------------------------
    box(ax_a, 0.182, mid - 0.10, 0.145, 0.20,
        "16 $\\times$ 5 window\n(0.5 s @ 30 fps)\ntrain-only $z$-score")
    arrow(ax_a, 0.331, mid, 0.362, mid)

    box(ax_a, 0.366, mid - 0.10, 0.115, 0.20, "Linear\n5 $\\rightarrow$ 64\n+ ReLU")
    arrow(ax_a, 0.485, mid, 0.516, mid)

    # Encoder ------------------------------------------------------------
    ex, ew = 0.520, 0.235
    ax_a.add_patch(FancyBboxPatch(
        (ex, mid - 0.245), ew, 0.49,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor="#fbfaf6", edgecolor=INK_2, linewidth=1.0,
        linestyle=(0, (4, 2)), zorder=3))
    ax_a.text(ex + ew / 2, mid + 0.195, "Temporal encoder", ha="center",
              va="center", fontsize=8.2, color=INK, fontweight="bold", zorder=4)
    fam = [("Bidirectional LSTM", BILSTM), ("Transformer encoder", TRANSF),
           ("Bidirectional GRU", GRU), ("Vanilla RNN (un-gated)", RNN)]
    for i, (name, col) in enumerate(fam):
        y = mid + 0.098 - i * 0.086
        ax_a.add_patch(Rectangle((ex + 0.016, y - 0.028), 0.021, 0.056,
                                 facecolor=col, edgecolor="none", zorder=5))
        ax_a.text(ex + 0.047, y, name, va="center", fontsize=7.5,
                  color=INK_2, zorder=5)
    arrow(ax_a, ex + ew + 0.004, mid, ex + ew + 0.035, mid)

    box(ax_a, 0.794, mid - 0.10, 0.118, 0.20, "Read-out\n+ linear\nhead")
    arrow(ax_a, 0.916, mid, 0.947, mid)
    box(ax_a, 0.951, mid - 0.075, 0.048, 0.15, "$\\sigma$", fs=10,
        fc="#e9f0fa", ec="#a9c4e8")
    ax_a.text(0.975, mid - 0.115, "$p(\\mathrm{cross})$", ha="center",
              va="top", fontsize=7.8, color=INK)

    # ================================================================ panel (b)
    ax_b.text(0, 1.02, "(b)  Live perception-to-prediction pipeline",
              transform=ax_b.transAxes, fontsize=9.5, fontweight="bold", color=INK)
    ax_b.text(0, 0.885,
              "Used for the latency and detector-in-the-loop experiments; the "
              "benchmark results use annotated boxes, as is standard.",
              transform=ax_b.transAxes, fontsize=8.3, color=MUTED)

    m = 0.45
    stages = [
        (0.005, 0.128, "Video frame\n1920 $\\times$ 1080\n30 fps", None),
        (0.178, 0.140, "YOLO detector", "93% of per-frame cost"),
        (0.351, 0.152, "ByteTrack\nidentity\nassociation", None),
        (0.536, 0.152, "Per-track\n16-frame\nring buffer", None),
        (0.721, 0.150, "Crossing-intention\nmodel  (panel a)", "4.5%"),
        (0.904, 0.095, "Alert /\noverlay", None),
    ]
    for x, w, text, note in stages:
        emph = note is not None
        box(ax_b, x, m - 0.135, w, 0.27, text,
            fc="#eef4fc" if emph else PANEL,
            ec=BILSTM if emph else RULE,
            lw=1.0 if emph else 0.8)
        if note:
            ax_b.text(x + w / 2, m - 0.175, note, ha="center", va="top",
                      fontsize=7.4, color=BILSTM, fontweight="bold")
    for i in range(len(stages) - 1):
        x, w = stages[i][0], stages[i][1]
        arrow(ax_b, x + w + 0.004, m, stages[i + 1][0] - 0.004, m)

    ax_b.annotate("ego speed $v$", xy=(0.612, m + 0.14), xytext=(0.612, m + 0.34),
                  ha="center", va="bottom", fontsize=7.5, color=MUTED,
                  arrowprops=dict(arrowstyle="-|>", color=RULE, lw=0.9,
                                  mutation_scale=8))

    for ax in (ax_a, ax_b):
        hide_frame(ax)
        ax.set_xlim(-0.005, 1.005)
    ax_a.set_ylim(0.17, 1.10)
    ax_b.set_ylim(0.14, 1.10)

    fig.subplots_adjust(left=0.008, right=0.992, top=0.97, bottom=0.01, hspace=0.14)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = HERE / f"fig2_system.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

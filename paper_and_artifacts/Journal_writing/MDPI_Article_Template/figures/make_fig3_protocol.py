"""make_fig3_protocol.py — Figure 3: the two windowing protocols, side by side.

A schematic (not a data plot) of where the 16-frame observation window lands on a
single crossing pedestrian's track under each anchoring rule:

  (a) Track-end anchor (the common protocol). The window is placed at the last
      annotated frame minus a fixed TTE, which carries no guarantee about the
      crossing onset. On PIE this lands the window inside the crossing for
      67.9% of crossers.

  (b) Crossing-point anchor (ours). The window is required to end TTE in
      [30, 60] frames BEFORE the annotated crossing point, so the model never
      observes a frame in which the pedestrian is already crossing.

Frame positions are illustrative; the TTE values and the leakage rates are the
real ones (journal_prep/issue1_leakage_audit, journal_prep/issue2_clean_protocol).

Run:  python make_fig3_protocol.py  ->  fig3_protocol.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from figstyle import (CLEAN, INK, INK_2, LEAK, MUTED, RULE, hide_frame,
                      use_style)

HERE = Path(__file__).resolve().parent

# Illustrative track geometry, in frames.
T0, T_END = 0, 300      # annotated track extent
ONSET = 150             # crossing_point (first frame with cross == crossing)
OBS = 16                # observation-window length
TTE_LEAKY = 45          # the legacy fixed offset from the track end
TTE_CLEAN = 45          # a representative draw from the [30, 60] clean range

BAR_Y, BAR_H = 0.46, 0.20
BAR_TOP = BAR_Y + BAR_H


def draw_track(ax, truncated=False):
    """The pedestrian's annotated track: pre-crossing in gray, crossing shaded."""
    ax.add_patch(Rectangle((T0, BAR_Y), ONSET - T0, BAR_H,
                           facecolor="#e8e7e1", edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((ONSET, BAR_Y), T_END - ONSET, BAR_H,
                           facecolor=LEAK, alpha=0.10 if truncated else 0.17,
                           edgecolor="none", zorder=2))
    ax.plot([ONSET, ONSET], [BAR_Y - 0.09, 0.755], color=LEAK, lw=1.4, zorder=6)
    ax.text(ONSET + 5, 0.765, "crossing onset", fontsize=9.0, color=LEAK,
            fontweight="bold", va="bottom", ha="left")
    ax.text((T0 + ONSET) / 2, BAR_Y + BAR_H / 2, "approaching the curb",
            fontsize=8.6, color=MUTED, va="center", ha="center", zorder=5)
    if truncated:
        ax.text((ONSET + T_END) / 2, BAR_Y + BAR_H / 2,
                "truncated: never observed", fontsize=8.6, color="#a89a97",
                style="italic", va="center", ha="center", zorder=5)
    else:
        ax.text((ONSET + T_END) / 2, BAR_Y + BAR_H / 2, "already crossing",
                fontsize=8.6, color=LEAK, va="center", ha="center", zorder=5)
    ax.text(T0, BAR_Y - 0.10, "annotated track", fontsize=8.4, color=MUTED,
            va="top", ha="left")


def draw_window(ax, end_frame, color, hatch=None):
    """The 16-frame observation window, drawn as an outlined block."""
    start = end_frame - OBS
    ax.add_patch(Rectangle((start, BAR_Y - 0.02), OBS, BAR_H + 0.04,
                           facecolor=color, alpha=0.32, edgecolor=color,
                           lw=1.3, hatch=hatch, zorder=7))
    ax.annotate("16-frame\nobservation window",
                xy=(end_frame - OBS / 2, BAR_Y - 0.04),
                xytext=(end_frame - OBS / 2, BAR_Y - 0.16),
                fontsize=9.0, color=INK, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=RULE, lw=0.8))


def tte_arrow(ax, x0, x1, label):
    y = 0.885
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="<->",
                                 mutation_scale=8, color=INK_2, lw=1.0, zorder=6))
    ax.text((x0 + x1) / 2, y + 0.025, label, fontsize=8.6, color=INK_2,
            ha="center", va="bottom")


def header(ax, title, subtitle, badge, badge_color):
    ax.text(0, 1.20, title, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", color=INK, va="bottom")
    ax.text(1.0, 1.20, badge, transform=ax.transAxes, fontsize=9.0,
            color=badge_color, fontweight="bold", ha="right", va="bottom")
    ax.text(0, 1.06, subtitle, transform=ax.transAxes, fontsize=8.8,
            color=MUTED, va="bottom")


def main():
    use_style()
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.1, 3.9))

    # ------------------------------------------------------------- panel (a)
    draw_track(ax_a)
    anchor_leaky = T_END - TTE_LEAKY
    draw_window(ax_a, anchor_leaky, LEAK, hatch="////")
    tte_arrow(ax_a, anchor_leaky, T_END, f"TTE = {TTE_LEAKY} frames")
    ax_a.plot([T_END, T_END], [BAR_Y, 0.865], color=RULE, lw=1.0, zorder=5)
    ax_a.text(T_END + 4, BAR_Y + BAR_H + 0.03, "last annotated\nframe",
              fontsize=8.4, color=MUTED, ha="left", va="bottom")
    header(ax_a, "(a)  Track-end anchor: the common protocol",
           "The window is offset from the end of the track, so nothing stops it "
           "overlapping the crossing.",
           "leaks for 67.9% of crossers", LEAK)

    # ------------------------------------------------------------- panel (b)
    draw_track(ax_b, truncated=True)
    anchor_clean = ONSET - TTE_CLEAN
    draw_window(ax_b, anchor_clean, CLEAN)
    tte_arrow(ax_b, anchor_clean, ONSET, "TTE $\\in$ [30, 60] frames (1–2 s)")
    header(ax_b, "(b)  Crossing-point anchor: this work",
           "The window must end at least a second before the annotated crossing "
           "point, so leakage is impossible by construction.",
           "leaks for 0.0% of windows", CLEAN)

    for ax in (ax_a, ax_b):
        hide_frame(ax)
        ax.set_xlim(-12, T_END + 62)
        ax.set_ylim(0.19, 0.98)

    fig.subplots_adjust(left=0.012, right=0.988, top=0.86, bottom=0.02, hspace=0.62)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = HERE / f"fig3_protocol.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

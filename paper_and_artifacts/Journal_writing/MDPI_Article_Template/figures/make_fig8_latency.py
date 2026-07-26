"""make_fig8_latency.py — Figure 8: is any of this fast enough?

  (a) Isolated cost of one crossing-intention prediction, per family, measured
      on an Apple M4 CPU at batch 1 (50 warm-up + 1,000 timed forwards).
      Source: journal_prep/Analysis/latency_comparison.csv.
  (b) Where a frame's time actually goes in the live pipeline, on the same
      machine. Source: journal_prep/issue9_latency/09_latency_report.md.

The two panels answer different questions and so carry different scales; the
point of putting them together is that the answer to (a) is irrelevant next
to (b).

Run:  python make_fig8_latency.py  ->  fig8_latency.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figstyle import (BILSTM, CONTEXT, GRID, GRU, INK, INK_2, MUTED, RNN,
                      TRANSF, style_axes, use_style)

HERE = Path(__file__).resolve().parent

FRAME_BUDGET_MS = 1000.0 / 30.0   # 33.3 ms at 30 fps

# family, ms per window (CPU, batch 1), colour  -- fastest first
LATENCY = [
    ("Vanilla RNN", 0.316, RNN),
    ("Transformer", 0.459, TRANSF),
    ("BiLSTM", 0.575, BILSTM),
    ("GRU", 0.721, GRU),
]

# stage, ms per frame, colour
PIPELINE = [
    ("YOLO detector", 33.7, "#7d7c76"),
    ("ByteTrack association", 1.0, "#b9b8b2"),
    ("Crossing-intention model", 1.647, BILSTM),
]


def main():
    use_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.1, 2.6), gridspec_kw={"width_ratios": [1.0, 1.12]})

    # ------------------------------------------------------------- panel (a)
    names = [n for n, _, _ in LATENCY]
    vals = [v for _, v, _ in LATENCY]
    cols = [c for _, _, c in LATENCY]
    y = np.arange(len(names))[::-1]
    ax_a.barh(y, vals, height=0.56, color=cols, zorder=3)
    for yy, v in zip(y, vals):
        ax_a.text(v + 0.022, yy, f"{v:.3f} ms", va="center", fontsize=8.4,
                  color=INK, fontweight="bold")
        ax_a.text(0.99, yy, f"{FRAME_BUDGET_MS / v:.0f}$\\times$",
                  transform=ax_a.get_yaxis_transform(), va="center",
                  ha="right", fontsize=8.3, color=MUTED)
    ax_a.text(0.99, len(names) - 0.62, "inside\nbudget",
              transform=ax_a.get_yaxis_transform(), va="bottom", ha="right",
              fontsize=8.0, color=MUTED, linespacing=1.3)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(names)
    ax_a.set_xlim(0, 1.30)
    ax_a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_ylim(-0.62, len(names) - 0.15)
    ax_a.set_xlabel("Milliseconds per window (CPU, batch 1)", color=INK_2,
                    labelpad=2)
    ax_a.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    style_axes(ax_a, left=False)
    ax_a.set_title("(a)  One prediction", loc="left", color=INK,
                   fontweight="bold", pad=7, fontsize=8.8)
    ax_a.text(0, -0.30, "A 30 fps frame allows 33.3 ms; every family fits "
              "tens of times over.", transform=ax_a.transAxes, fontsize=8.2,
              color=MUTED, va="top")

    # ------------------------------------------------------------- panel (b)
    total = sum(v for _, v, _ in PIPELINE)
    left = 0.0
    for name, v, col in PIPELINE:
        ax_b.barh(0, v, left=left, height=0.44, color=col, edgecolor="white",
                  linewidth=1.6, zorder=3, label=f"{name}  ({v / total:.1%})")
        left += v
    ax_b.axvline(FRAME_BUDGET_MS, color="#b4433a", lw=1.2, ls=(0, (4, 2)),
                 zorder=5)
    ax_b.text(FRAME_BUDGET_MS - 0.7, 0.40, "30 fps budget", fontsize=8.2,
              color="#b4433a", ha="right", va="bottom", fontweight="bold")
    ax_b.text(total, -0.30, f"{total:.1f} ms per frame  =  {1000 / total:.1f} fps",
              ha="right", va="top", fontsize=8.3, color=INK, fontweight="bold")
    ax_b.set_xlim(0, 39)
    ax_b.set_ylim(-1.30, 0.72)
    ax_b.set_yticks([])
    ax_b.set_xticks([0, 10, 20, 30])
    ax_b.set_xlabel("Milliseconds per frame", color=INK_2, labelpad=2)
    ax_b.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    style_axes(ax_b, left=False)
    ax_b.legend(loc="lower left", bbox_to_anchor=(-0.015, -0.02), frameon=False,
                ncol=1, fontsize=8.1, handlelength=0.95, handleheight=0.95,
                handletextpad=0.45, borderpad=0, labelspacing=0.3,
                labelcolor=INK_2)
    ax_b.set_title("(b)  One frame of the live pipeline", loc="left",
                   color=INK, fontweight="bold", pad=7, fontsize=8.8)

    fig.subplots_adjust(left=0.115, right=0.985, top=0.86, bottom=0.235, wspace=0.30)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = HERE / f"fig8_latency.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

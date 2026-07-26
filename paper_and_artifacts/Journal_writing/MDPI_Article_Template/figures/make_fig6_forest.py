"""make_fig6_forest.py — Figure 6: paired contrasts with pedestrian-cluster CIs.

The paper's central architectural claim in one picture. Every row is a paired
bootstrap on the identical 2,094 test windows, resampled over the 541 pedestrian
clusters (B = 10,000). A contrast whose interval crosses zero is a tie; one that
clears zero is a real difference. Colouring by that verdict rather than by
family keeps the reader's eye on the question being asked.

Panel (a) is F1 - the primary metric. Panel (b) is ROC-AUC, where the same
models do separate, which is what makes the F1 ties informative rather than a
symptom of a test with no power.

Numbers are transcribed from the stored bootstrap outputs:
  f1_optimization/07_cluster_bootstrap.json
  gru/phase5_analysis/08_cluster_bootstrap.json
  rnn/phase5_analysis/08_cluster_bootstrap.json
  transformer/phase5_analysis/05_comparison_results.json   (seed-paired, marked)

Run:  python make_fig6_forest.py  ->  fig6_forest.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figstyle import (CONTEXT, GRID, INK, INK_2, MUTED, RULE, style_axes,
                      use_style)

HERE = Path(__file__).resolve().parent

WIN = "#2a78d6"    # interval clears zero
TIE = "#9b9a93"    # interval spans zero

# (label, delta, lo, hi, footnote marker)
F1_ROWS = [
    ("GRU vs. BiLSTM", 0.007118, -0.008911, 0.022339, ""),
    ("Vanilla RNN vs. BiLSTM", 0.003336, -0.012996, 0.018743, ""),
    ("Transformer vs. BiLSTM", 0.000827, -0.019553, 0.019984, ""),
    ("GRU vs. Transformer", 0.006290, -0.008522, 0.021594, ""),
    ("Vanilla RNN vs. Transformer", 0.002509, -0.011085, 0.017062, ""),
    ("Vanilla RNN vs. GRU", -0.003781, -0.012754, 0.004874, ""),
    (None, None, None, None, None),  # group break
    ("BiLSTM vs. AUC-selected baseline", 0.018674, 0.004312, 0.034935, ""),
    ("GRU vs. AUC-selected baseline", 0.025792, 0.014795, 0.038050, ""),
    ("Vanilla RNN vs. AUC-selected baseline", 0.022011, 0.009681, 0.035370, ""),
]

AUC_ROWS = [
    ("Vanilla RNN vs. searched Transformer", -0.001344, -0.006077, 0.003312, ""),
    ("GRU vs. searched Transformer", -0.006953, -0.012904, -0.001755, ""),
    (None, None, None, None, None),
    ("Searched Transformer vs. baseline", 0.013453, 0.009705, 0.017368, "†"),
    ("Un-searched Transformer vs. baseline", 0.000468, -0.003424, 0.004261, "†"),
    ("Vanilla RNN (h256) vs. baseline", 0.012125, 0.006307, 0.018737, ""),
    ("Vanilla RNN (h128) vs. baseline", 0.005931, 0.001225, 0.010992, ""),
    ("GRU (h128) vs. baseline", -0.000799, -0.006728, 0.004538, ""),
]


def draw(ax, rows, xlim, title, subtitle, xlabel):
    ypos, labels = [], []
    y = 0.0
    for label, d, lo, hi, mark in rows:
        if label is None:
            y -= 0.55
            continue
        color = TIE if (lo <= 0.0 <= hi) else WIN
        ax.plot([lo, hi], [y, y], color=color, lw=1.7, solid_capstyle="round",
                zorder=4)
        ax.plot([d], [y], marker="o", ms=5.2, color=color, mec="white",
                mew=1.0, zorder=5)
        ax.text(xlim[1] * 0.985, y, f"{d:+.4f}", ha="right", va="center",
                fontsize=8.2, color=INK if color == WIN else INK_2,
                fontweight="bold" if color == WIN else "normal", zorder=6)
        ypos.append(y)
        labels.append(label + mark)
        y -= 1.0

    ax.axvline(0.0, color=RULE, lw=1.0, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_ylim(y + 0.45, 0.8)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, color=INK_2, labelpad=2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax, left=False)
    ax.text(0, 1.20, title, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", color=INK, va="bottom")
    ax.text(0, 1.06, subtitle, transform=ax.transAxes, fontsize=8.6,
            color=MUTED, va="bottom")


def main(compact=False):
    """compact=True emits a shorter variant (fig6_forest_compact) for main_short.tex,
    where this figure was leaving a quarter of its page empty. Same data, same rows,
    tighter vertical spacing; the full-height version main.tex uses is untouched."""
    use_style()
    height, hspace, top, bottom = (4.45, 0.60, 0.895, 0.115) if compact else \
                                  (5.15, 0.72, 0.900, 0.105)
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(7.1, height), gridspec_kw={"height_ratios": [1.0, 0.9]})

    draw(ax_a, F1_ROWS, (-0.030, 0.048),
         "(a)  F1: the primary metric",
         "Every cross-family contrast spans zero: the four families are "
         "indistinguishable.",
         "$\\Delta$F1 (95% pedestrian-cluster bootstrap interval)")

    draw(ax_b, AUC_ROWS, (-0.016, 0.026),
         "(b)  ROC-AUC: the corroborating metric",
         "Here the same comparisons do separate, so the ties above are not a "
         "test without power.",
         "$\\Delta$AUC (95% interval)")

    fig.text(0.012, 0.012,
             "† seed-paired window-level bootstrap; all other rows resample the "
             "541 pedestrian clusters.",
             fontsize=7.9, color=MUTED)

    handles = [plt.Line2D([], [], color=WIN, lw=1.7, marker="o", ms=5.2,
                          mec="white", mew=1.0, label="interval excludes zero: a real difference"),
               plt.Line2D([], [], color=TIE, lw=1.7, marker="o", ms=5.2,
                          mec="white", mew=1.0, label="interval spans zero: a tie")]
    ax_a.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.2,
                handlelength=1.7, borderpad=0, labelspacing=0.3,
                labelcolor=INK_2)

    fig.subplots_adjust(left=0.315, right=0.985, top=top, bottom=bottom,
                        hspace=hspace)

    stem = "fig6_forest_compact" if compact else "fig6_forest"
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = HERE / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {out}")


if __name__ == "__main__":
    import sys
    main(compact="--compact" in sys.argv)

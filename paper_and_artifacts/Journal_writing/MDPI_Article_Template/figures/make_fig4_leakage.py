"""make_fig4_leakage.py — Figure 4: the temporal-leakage audit, before and after.

Both panels are computed live from the two per-sequence audit files, so the
figure cannot drift from the reported numbers:

  journal_prep/issue1_leakage_audit/leakage_per_sequence.csv   (track-end anchor)
  journal_prep/issue2_clean_protocol/leakage_per_sequence.csv  (crossing-point anchor)

  (a) How much of each crossing pedestrian's observation window is already
      spent crossing, under each protocol.
  (b) Whether the single bounding box at the last observed frame can separate
      crossers from non-crossers on its own — a static shortcut that a genuine
      predictor should not have. Effect size is the rank-biserial correlation
      from a two-sided Mann-Whitney U test.

Run:  python make_fig4_leakage.py  ->  fig4_leakage.{pdf,png}
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from figstyle import (CLEAN, CONTEXT, GRID, INK, INK_2, LEAK, MUTED,
                      style_axes, use_style)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEAKY_CSV = ROOT / "journal_prep" / "issue1_leakage_audit" / "leakage_per_sequence.csv"
CLEAN_CSV = ROOT / "journal_prep" / "issue2_clean_protocol" / "leakage_per_sequence.csv"

OBS = 16
PARTIAL = "#e0a24b"
FEATURES = [("bbox_area", "Box area"), ("bbox_height", "Box height"),
            ("bbox_bottom_y", "Box bottom edge"), ("bbox_xcenter", "Box center $x$")]


def read_audit(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def window_composition(rows):
    """Split crossers into clean / partly crossing / fully crossing windows."""
    n_cross = [int(r["n_crossing_in_window"]) for r in rows if r["label"] == "1"]
    n = len(n_cross)
    full = sum(1 for v in n_cross if v >= OBS)
    part = sum(1 for v in n_cross if 0 < v < OBS)
    return n, np.array([n - full - part, part, full]) / n * 100.0


def rank_biserial(rows, field):
    a = np.array([float(r[field]) for r in rows if r["label"] == "1"])
    b = np.array([float(r[field]) for r in rows if r["label"] == "0"])
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return abs(2.0 * u / (len(a) * len(b)) - 1.0)


def head(ax, title, sub1, sub2):
    ax.text(0, 1.34, title, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", color=INK, va="bottom")
    ax.text(0, 1.18, sub1, transform=ax.transAxes, fontsize=8.6,
            color=MUTED, va="bottom")
    ax.text(0, 1.05, sub2, transform=ax.transAxes, fontsize=8.6,
            color=MUTED, va="bottom")


def main():
    use_style()
    leaky, clean = read_audit(LEAKY_CSV), read_audit(CLEAN_CSV)
    n_leaky, comp_leaky = window_composition(leaky)
    n_clean, comp_clean = window_composition(clean)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.1, 3.25), gridspec_kw={"width_ratios": [1.0, 1.0]})

    # ------------------------------------------------------------- panel (a)
    segs = [("entirely pre-crossing", CLEAN), ("partly crossing", PARTIAL),
            ("entirely crossing", LEAK)]
    rows = [(f"Track-end anchor\n$n$ = {n_leaky:,} crossers", comp_leaky),
            (f"Crossing-point anchor\n$n$ = {n_clean:,} crossers", comp_clean)]
    ypos = [1.0, 0.0]
    for (label, comp), y in zip(rows, ypos):
        left = 0.0
        for (name, col), val in zip(segs, comp):
            ax_a.barh(y, val, left=left, height=0.44, color=col,
                      edgecolor="white", linewidth=1.6, zorder=3,
                      label=name if y == 1.0 else None)
            if val >= 10:
                ax_a.text(left + val / 2, y, f"{val:.0f}%", ha="center",
                          va="center", fontsize=8.5, color="white",
                          fontweight="bold", zorder=5)
            left += val
    ax_a.annotate(f"{comp_leaky[1]:.0f}%",
                  xy=(comp_leaky[0] + comp_leaky[1] / 2, 1.23),
                  xytext=(comp_leaky[0] + comp_leaky[1] / 2, 1.62),
                  ha="center", va="bottom", fontsize=8.4, color=INK_2,
                  arrowprops=dict(arrowstyle="-", color=CONTEXT, lw=0.8))
    ax_a.set_yticks(ypos)
    ax_a.set_yticklabels([r[0] for r in rows])
    ax_a.set_ylim(-0.62, 1.95)
    ax_a.set_xlim(0, 100)
    ax_a.get_xaxis().set_visible(False)
    style_axes(ax_a, left=False, bottom=False)
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.46, -0.055), frameon=False,
                ncol=3, handlelength=0.95, handleheight=0.95, columnspacing=1.1,
                handletextpad=0.45, borderpad=0, fontsize=8.2, labelcolor=INK_2)
    head(ax_a, "(a) What the model actually sees",
         "Crossing pedestrians grouped by how much of the",
         "16-frame window is spent already crossing.")

    # ------------------------------------------------------------- panel (b)
    r_leaky = [rank_biserial(leaky, f) for f, _ in FEATURES]
    r_clean = [rank_biserial(clean, f) for f, _ in FEATURES]
    y = np.arange(len(FEATURES))[::-1]
    h = 0.34
    ax_b.barh(y + h / 2 + 0.02, r_leaky, height=h, color=LEAK, zorder=3,
              label="Track-end anchor")
    ax_b.barh(y - h / 2 - 0.02, r_clean, height=h, color=CLEAN, zorder=3,
              label="Crossing-point anchor")
    for yy, v in zip(y + h / 2 + 0.02, r_leaky):
        ax_b.text(v + 0.016, yy, f"{v:.2f}", va="center", fontsize=8.4,
                  color=INK, fontweight="bold")
    for yy, v in zip(y - h / 2 - 0.02, r_clean):
        ax_b.text(v + 0.016, yy, f"{v:.2f}", va="center", fontsize=8.4, color=INK_2)
    ax_b.axvline(0.30, color="#a9a89f", lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax_b.text(0.315, 3.92, "medium effect", fontsize=8.2, color=MUTED,
              ha="left", va="top")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([n for _, n in FEATURES])
    ax_b.set_xlim(0, 0.82)
    ax_b.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax_b.set_xlabel("Rank-biserial correlation $|r|$", color=INK_2, labelpad=2)
    ax_b.set_ylim(-0.62, 3.95)
    ax_b.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    style_axes(ax_b, left=False)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.46, -0.19), frameon=False,
                ncol=2, handlelength=0.95, handleheight=0.95, columnspacing=1.4,
                handletextpad=0.45, borderpad=0, fontsize=8.2, labelcolor=INK_2)
    head(ax_b, "(b) The static-geometry shortcut",
         "How well the last observed box alone separates",
         "the classes, with no temporal information at all.")

    fig.subplots_adjust(left=0.165, right=0.985, top=0.70, bottom=0.155, wspace=0.60)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = HERE / f"fig4_leakage.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {out}")

    print(f"  (a) leaky  n={n_leaky} composition={np.round(comp_leaky, 1)}")
    print(f"  (a) clean  n={n_clean} composition={np.round(comp_clean, 1)}")
    print(f"  (b) leaky |r|={np.round(r_leaky, 3)}  clean |r|={np.round(r_clean, 3)}")


if __name__ == "__main__":
    main()

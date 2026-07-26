"""make_intro_figure.py — Figure 1 of the MTI manuscript (Introduction).

Two panels, both from PRIMARY sources documented in
../../STATISTICS_SOURCES.md — do not edit the numbers here without updating
that file.

  (a) Global distribution of road traffic deaths by road user type.
      Source: WHO, Global Status Report on Road Safety 2023 (data year 2021),
      Section 1 "Fatalities by road user type". Verbatim shares.

  (b) Pedestrian fatalities in the United States, 2014-2023.
      Source: NHTSA Traffic Safety Facts "Pedestrians: 2023 Data",
      DOT HS 813 727 (June 2025), Table 1. Verbatim counts.

Design follows the dataviz method: EMPHASIS form (the one series that is the
point in an accent hue, the rest in de-emphasis gray), no dual axis anywhere,
recessive grid/axes, selective direct labels, text in ink tokens (never the
series color). Palette validated for CVD separation and >=3:1 contrast on a
white print surface.

Run:  python make_intro_figure.py      ->  fig1_pedestrian_statistics.{pdf,png}
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------- palette
ACCENT = "#2a78d6"   # categorical slot 1 (blue) - the pedestrian series
CONTEXT = "#b9b8b2"  # de-emphasis gray for context categories
INK = "#0b0b0b"      # primary ink
INK_2 = "#52514e"    # secondary ink
MUTED = "#898781"    # axis / tick labels
GRID = "#e1e0d9"     # hairline gridline

# ----------------------------------------------------------------- data (a)
# WHO Global Status Report on Road Safety 2023, data year 2021.
WHO_LABELS = [
    "Four-wheel vehicle\noccupants",
    "Pedestrians",
    "Powered 2-/3-wheeler\nusers",
    "Other / unknown",
    "Cyclists",
]
WHO_SHARES = [30, 23, 21, 20, 6]
WHO_EMPHASIS = 1  # index of "Pedestrians"

# ----------------------------------------------------------------- data (b)
# NHTSA DOT HS 813 727, Table 1 (FARS). Counts of pedestrians killed.
YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
PED_DEATHS = [4910, 5494, 6080, 6075, 6374, 6272, 6565, 7470, 7593, 7314]

# ----------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    # Sized so that, after the figure is scaled to the MDPI \textwidth
    # (~16 cm) in the manuscript, on-figure text lands near 8 pt.
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,  # embed TrueType so the PDF is editable/portable
    "ps.fonttype": 42,
})


def style_axes(ax):
    """Recessive chrome: no box, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, length=0, pad=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def main():
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.1, 2.95), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    # ---------------------------------------------------------- panel (a)
    y_pos = range(len(WHO_LABELS))
    colors = [ACCENT if i == WHO_EMPHASIS else CONTEXT for i in range(len(WHO_LABELS))]
    ax_a.barh(list(y_pos), WHO_SHARES, color=colors, height=0.62, zorder=3)
    ax_a.set_yticks(list(y_pos))
    ax_a.set_yticklabels(WHO_LABELS)
    ax_a.invert_yaxis()  # largest at top

    # Direct value labels replace the x-axis entirely.
    for i, v in enumerate(WHO_SHARES):
        ax_a.text(
            v + 0.9, i, f"{v}%",
            va="center", ha="left", fontsize=9,
            color=INK if i == WHO_EMPHASIS else INK_2,
            fontweight="bold" if i == WHO_EMPHASIS else "normal",
        )
    ax_a.set_xlim(0, 36)
    ax_a.get_xaxis().set_visible(False)
    for side in ("top", "right", "bottom", "left"):
        ax_a.spines[side].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    for lbl in ax_a.get_yticklabels():
        lbl.set_color(INK_2)
    ax_a.get_yticklabels()[WHO_EMPHASIS].set_color(INK)
    ax_a.get_yticklabels()[WHO_EMPHASIS].set_fontweight("bold")
    ax_a.set_title(
        "(a) Global road traffic deaths by road user type",
        loc="left", color=INK, fontweight="bold", pad=8,
    )
    ax_a.text(
        0, -0.28, "1.19 million deaths worldwide (2021)",
        transform=ax_a.transAxes, fontsize=8.5, color=MUTED, va="top",
    )

    # ---------------------------------------------------------- panel (b)
    ax_b.bar(YEARS, PED_DEATHS, color=ACCENT, width=0.68, zorder=3)
    ax_b.set_ylim(0, 8600)
    ax_b.set_yticks([0, 2000, 4000, 6000, 8000])
    ax_b.set_yticklabels(["0", "2,000", "4,000", "6,000", "8,000"])
    ax_b.set_xticks(YEARS)
    ax_b.set_xticklabels([str(y)[-2:] for y in YEARS])
    ax_b.set_xlabel("Year (20xx)", color=INK_2, labelpad=2)
    ax_b.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    style_axes(ax_b)
    ax_b.spines["left"].set_visible(False)

    # Selective direct labels: first and last only.
    for idx in (0, len(YEARS) - 1):
        ax_b.text(
            YEARS[idx], PED_DEATHS[idx] + 190, f"{PED_DEATHS[idx]:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK,
        )

    ax_b.set_title(
        "(b) Pedestrians killed in traffic crashes, United States",
        loc="left", color=INK, fontweight="bold", pad=8,
    )
    ax_b.text(
        0, -0.28, "+49% over the decade; 15% to 18% of all traffic deaths",
        transform=ax_b.transAxes, fontsize=8.5, color=MUTED, va="top",
    )

    fig.subplots_adjust(left=0.155, right=0.985, top=0.86, bottom=0.20, wspace=0.42)

    out_pdf = HERE / "fig1_pedestrian_statistics.pdf"
    out_png = HERE / "fig1_pedestrian_statistics.png"
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()

"""make_fig7_ablations.py — Figure 7: what the model needs, and how far ahead.

  (a) The ego-speed ablation. Dropping the second stream from the otherwise
      identical BiLSTM (journal_prep/issue2_clean_protocol/04_multiseed_summary.md).
  (b) Prediction horizon, on the matched pedestrian cohort that holds the test
      population fixed across horizons
      (journal_prep/issue6_window_tte_ablation/06b_matched_tte_results.csv,
      recomputed here from the per-seed rows).
  (c) Observation-window length for all four F1-optimized families
      (journal_prep/obs_window_extension/).

Bars and markers show the 5-seed mean; whiskers are +/- 1 SD across seeds.

Run:  python make_fig7_ablations.py  ->  fig7_ablations.{pdf,png}
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figstyle import (BILSTM, CONTEXT, GRID, GRU, INK, INK_2, MUTED, RNN,
                      TRANSF, style_axes, use_style)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TTE_CSV = (ROOT / "journal_prep" / "issue6_window_tte_ablation"
           / "06b_matched_tte_results.csv")

# 5-seed mean +/- SD, clean protocol, identical recipe apart from the input.
EGO = {
    "F1": [(0.828, 0.012), (0.551, 0.028)],
    "Accuracy": [(0.883, 0.009), (0.744, 0.007)],
    "ROC-AUC": [(0.932, 0.011), (0.753, 0.020)],
}

# F1, 5-seed mean, per family per observation window (16 / 32 / 64 frames).
WINDOWS = [16, 32, 64]
OW = {
    "BiLSTM": ([0.844, 0.837, 0.818], [0.008, 0.014, 0.018], BILSTM),
    "Transformer": ([0.847, 0.838, 0.819], [0.017, 0.022, 0.030], TRANSF),
    "GRU": ([0.849, 0.834, 0.822], [0.011, 0.019, 0.029], GRU),
    "Vanilla RNN": ([0.852, 0.834, 0.802], [0.012, 0.012, 0.020], RNN),
}


def read_horizon():
    acc = defaultdict(lambda: defaultdict(list))
    with open(TTE_CSV) as fh:
        for r in csv.DictReader(fh):
            t = float(r["tte_seconds"])
            acc[t]["auc"].append(float(r["auc"]))
            acc[t]["f1"].append(float(r["f1"]))
    ts = sorted(acc)
    out = {}
    for k in ("auc", "f1"):
        out[k] = (np.array([np.mean(acc[t][k]) for t in ts]),
                  np.array([np.std(acc[t][k], ddof=1) for t in ts]))
    return ts, out


def main():
    use_style()
    ts, hor = read_horizon()

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(7.1, 2.85), gridspec_kw={"width_ratios": [1.0, 0.92, 1.0]})

    # ------------------------------------------------------------- panel (a)
    metrics = list(EGO)
    x = np.arange(len(metrics))
    w = 0.34
    two = [EGO[m][0] for m in metrics]
    one = [EGO[m][1] for m in metrics]
    ax_a.bar(x - w / 2 - 0.015, [v for v, _ in two], w,
             yerr=[s for _, s in two], color=BILSTM, zorder=3,
             error_kw=dict(ecolor=INK_2, elinewidth=0.8, capsize=2.2),
             label="Box + ego-speed")
    ax_a.bar(x + w / 2 + 0.015, [v for v, _ in one], w,
             yerr=[s for _, s in one], color=CONTEXT, zorder=3,
             error_kw=dict(ecolor=INK_2, elinewidth=0.8, capsize=2.2),
             label="Box only")
    for xi, (v, _), (u, _) in zip(x, two, one):
        ax_a.text(xi - w / 2 - 0.015, v + 0.035, f"{v:.2f}", ha="center",
                  fontsize=8.3, color=INK, fontweight="bold")
        ax_a.text(xi + w / 2 + 0.015, u + 0.035, f"{u:.2f}", ha="center",
                  fontsize=8.3, color=INK_2)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(metrics)
    ax_a.set_ylim(0, 1.10)
    ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    style_axes(ax_a, left=False)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), frameon=False,
                ncol=2, fontsize=8.1, handlelength=0.95, handleheight=0.95,
                columnspacing=1.0, handletextpad=0.4, borderpad=0,
                labelcolor=INK_2)
    ax_a.set_title("(a)  Removing the ego-speed stream", loc="left",
                   color=INK, fontweight="bold", pad=7, fontsize=8.8)

    # ------------------------------------------------------------- panel (b)
    for key, name, color, marker in (("auc", "ROC-AUC", CONTEXT, "s"),
                                     ("f1", "F1", BILSTM, "o")):
        m, s = hor[key]
        ax_b.errorbar(ts, m, yerr=s, color=color, lw=1.6, marker=marker, ms=4.6,
                      mec="white", mew=0.9, capsize=2.2, elinewidth=0.8,
                      zorder=4, label=name)
        ax_b.text(ts[-1] + 0.035, m[-1], f"{m[-1]:.3f}", va="center",
                  fontsize=8.3, color=INK if key == "f1" else INK_2,
                  fontweight="bold" if key == "f1" else "normal")
    ax_b.set_xticks(ts)
    ax_b.set_xticklabels([f"{t:.1f}" for t in ts])
    ax_b.set_xlim(0.87, 2.30)
    ax_b.set_ylim(0.72, 1.0)
    ax_b.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    ax_b.set_xlabel("Prediction horizon (s)", color=INK_2, labelpad=2)
    ax_b.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    style_axes(ax_b, left=False)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False,
                ncol=2, fontsize=8.1, handlelength=1.5, columnspacing=1.0,
                handletextpad=0.4, borderpad=0, labelcolor=INK_2)
    ax_b.set_title("(b)  Predicting further ahead", loc="left", color=INK,
                   fontweight="bold", pad=7, fontsize=8.8)

    # ------------------------------------------------------------- panel (c)
    for name, (m, s, color) in OW.items():
        ax_c.errorbar(WINDOWS, m, yerr=s, color=color, lw=1.5, marker="o",
                      ms=4.2, mec="white", mew=0.9, capsize=2.0,
                      elinewidth=0.8, zorder=4, label=name)
    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks(WINDOWS)
    ax_c.set_xticklabels([f"{w}\n({w / 30:.1f} s)" for w in WINDOWS])
    ax_c.minorticks_off()
    ax_c.set_xlim(13.5, 76)
    ax_c.set_ylim(0.765, 0.885)
    ax_c.set_yticks([0.78, 0.80, 0.82, 0.84, 0.86])
    ax_c.set_xlabel("Observation window (frames)", color=INK_2, labelpad=2)
    ax_c.set_ylabel("F1", color=INK_2, labelpad=3)
    ax_c.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax_c.set_axisbelow(True)
    style_axes(ax_c, left=False)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False,
                ncol=2, fontsize=8.1, handlelength=1.4, columnspacing=1.0,
                handletextpad=0.4, borderpad=0, labelspacing=0.28,
                labelcolor=INK_2)
    ax_c.set_title("(c)  Observing for longer", loc="left", color=INK,
                   fontweight="bold", pad=7, fontsize=8.8)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.87, bottom=0.245, wspace=0.34)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = HERE / f"fig7_ablations.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {out}")

    print(f"  (b) horizons {ts}")
    print(f"      AUC {np.round(hor['auc'][0], 4)} +/- {np.round(hor['auc'][1], 4)}")
    print(f"      F1  {np.round(hor['f1'][0], 4)} +/- {np.round(hor['f1'][1], 4)}")


if __name__ == "__main__":
    main()

"""make_fig5_curves.py — Figure 5: ROC and precision-recall curves.

The four F1-optimized family headliners, plus the bounding-box-only ablation as
the context series. Curves are the 5-seed probability ensemble on the 2,094
leakage-free test windows; run prep_probs.py first to build `_probs.npz`.

The visual claim is deliberately twofold: the four families lie on top of one
another, and dropping the ego-speed stream pulls the curve down and away.

Run:  python make_fig5_curves.py  ->  fig5_curves.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from figstyle import (BILSTM, CONTEXT, GRID, GRU, INK, INK_2, MUTED, RNN,
                      TRANSF, style_axes, use_style)

HERE = Path(__file__).resolve().parent

SERIES = [
    ("bilstm_f1", "BiLSTM", BILSTM, 1.5, "solid"),
    ("transformer_f1", "Transformer", TRANSF, 1.5, "solid"),
    ("gru_f1", "GRU", GRU, 1.5, "solid"),
    ("rnn_f1", "Vanilla RNN", RNN, 1.5, "solid"),
    ("bilstm_bbox_only", "Box only (no ego-speed)", CONTEXT, 1.6, (0, (4, 2))),
]


def main():
    use_style()
    d = np.load(HERE / "_probs.npz")
    y = d["y"]
    base_rate = y.mean()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.1, 3.35))

    for key, label, color, lw, ls in SERIES:
        p = d[key]
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax_a.plot(fpr, tpr, color=color, lw=lw, ls=ls, zorder=4,
                  label=f"{label}  ({auc:.3f})")

        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax_b.plot(rec, prec, color=color, lw=lw, ls=ls, zorder=4,
                  label=f"{label}  ({ap:.3f})")
        print(f"  {label:26s} ROC-AUC {auc:.4f}   PR-AUC {ap:.4f}")

    # ------------------------------------------------------------- panel (a)
    ax_a.plot([0, 1], [0, 1], color="#cfcec6", lw=0.9, ls=(0, (3, 3)), zorder=2)
    ax_a.text(0.62, 0.545, "chance", fontsize=8.2, color=MUTED, rotation=33,
              rotation_mode="anchor")
    ax_a.set_xlabel("False positive rate", color=INK_2, labelpad=2)
    ax_a.set_ylabel("True positive rate", color=INK_2, labelpad=3)
    ax_a.set_xlim(-0.012, 1.012)
    ax_a.set_ylim(-0.012, 1.012)
    ax_a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    style_axes(ax_a)
    leg_a = ax_a.legend(loc="lower right", frameon=False, fontsize=8.1,
                        handlelength=1.5, borderpad=0, labelspacing=0.32,
                        title="ROC-AUC", alignment="left")
    leg_a.get_title().set_fontsize(8.1)
    leg_a.get_title().set_color(MUTED)
    for t in leg_a.get_texts():
        t.set_color(INK_2)
    ax_a.set_title("(a)  ROC", loc="left", color=INK, fontweight="bold", pad=7)

    # ------------------------------------------------------------- panel (b)
    ax_b.axhline(base_rate, color="#cfcec6", lw=0.9, ls=(0, (3, 3)), zorder=2)
    ax_b.text(0.985, base_rate + 0.018, f"chance ({base_rate:.2f} positive)",
              fontsize=8.2, color=MUTED, va="bottom", ha="right")
    ax_b.set_xlabel("Recall", color=INK_2, labelpad=2)
    ax_b.set_ylabel("Precision", color=INK_2, labelpad=3)
    ax_b.set_xlim(-0.012, 1.012)
    ax_b.set_ylim(-0.012, 1.012)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    style_axes(ax_b)
    leg_b = ax_b.legend(loc="lower left", frameon=False, fontsize=8.1,
                        handlelength=1.5, borderpad=0, labelspacing=0.32,
                        title="PR-AUC", alignment="left")
    leg_b.get_title().set_fontsize(8.1)
    leg_b.get_title().set_color(MUTED)
    for t in leg_b.get_texts():
        t.set_color(INK_2)
    ax_b.set_title("(b)  Precision–recall", loc="left", color=INK,
                   fontweight="bold", pad=7)

    fig.subplots_adjust(left=0.075, right=0.988, top=0.90, bottom=0.125, wspace=0.24)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = HERE / f"fig5_curves.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

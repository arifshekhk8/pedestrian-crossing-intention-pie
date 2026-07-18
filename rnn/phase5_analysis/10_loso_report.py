"""10_loso_report.py — RNN study Phase R5: LOSO fold table vs BiLSTM / GRU / Transformer.

Reads phase4_final/06_loso_summary.json (the RNN F1-winner's 6-fold LOSO, produced by
06_rnn_loso.py) and formats a comparison against the BiLSTM's LOSO (Issue 5: 0.928 ± 0.041),
the GRU's LOSO (0.946), and the transformer's LOSO (0.939). 6 folds is descriptive, not a
hypothesis test — the fixed-split paired bootstrap (07/08) is the actual evidence; LOSO is a
generalization sanity check that "set03 isn't just an easy fold".

Outputs: 10_loso_report.md
Run from the repo root:  python rnn/phase5_analysis/10_loso_report.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LOSO = json.loads((ROOT / "rnn" / "phase4_final" / "06_loso_summary.json").read_text())

# published references (ddof=1), for orientation
BILSTM_LOSO = dict(mean=0.928, std=0.041, excl05=0.915, src="Issue 5")
GRU_LOSO = dict(mean=0.946, std=0.036, excl05=0.935, src="gru/phase5")
TF_LOSO = dict(mean=0.939, std=0.044, excl05=0.927, src="transformer/phase5")


def main():
    folds = LOSO["folds"]
    m, s = LOSO["auc_mean"], LOSO["auc_std"]
    me, se = LOSO["auc_mean_excl_set05"], LOSO["auc_std_excl_set05"]
    f1m = LOSO["f1_mean"]
    set03_auc = [f["auc"] for f in folds if f["test_set"] == "set03"][0]
    print(f"RNN LOSO: AUC {m:.3f} ± {s:.3f} (excl set05 {me:.3f}), F1 {f1m:.3f}")

    L = ["# RNN study — Phase R5 Leave-One-Set-Out CV (generalization check)", "",
         f"RNN F1-winner, 6 folds, seed 42, Issue-5 protocol (per-fold pedestrian-grouped "
         "85/15 val split, per-fold pos_weight + train-only norm, AUC-selected so folds are "
         "comparable to the BiLSTM/GRU/Transformer LOSO). 6 folds is descriptive, not a "
         "hypothesis test — the fixed-split paired + cluster bootstrap (07/08) is the actual "
         "evidence.", "",
         "| Fold (test set) | test N | pos % | AUC | PR-AUC | F1 | Acc | pos_w | best ep |",
         "|---|---|---|---|---|---|---|---|---|"]
    for f in folds:
        L.append(f"| {f['test_set']} | {f['test_n']} | {f['test_pos']*100:.1f} | "
                 f"{f['auc']:.3f} | {f['pr_auc']:.3f} | {f['f1']:.3f} | {f['acc']:.3f} | "
                 f"{f['pos_weight']:.2f} | {f['best_epoch']} |")
    L += ["",
          f"**RNN LOSO (6-fold, unweighted): AUC {m:.3f} ± {s:.3f}, F1 {f1m:.3f}.** "
          f"Excluding the tiny set05 fold (N=47, near-perfect and uninterpretable): "
          f"AUC {me:.3f} ± {se:.3f}.", "",
          "## Comparison (LOSO mean AUC, all AUC-selected, ddof=1)", "",
          "| model | 6-fold AUC | excl. set05 | source |", "|---|---|---|---|",
          f"| **Vanilla RNN (this study)** | **{m:.3f} ± {s:.3f}** | {me:.3f} | this folder |",
          f"| BiLSTM | {BILSTM_LOSO['mean']:.3f} ± {BILSTM_LOSO['std']:.3f} | "
          f"{BILSTM_LOSO['excl05']:.3f} | {BILSTM_LOSO['src']} |",
          f"| GRU | {GRU_LOSO['mean']:.3f} ± {GRU_LOSO['std']:.3f} | "
          f"{GRU_LOSO['excl05']:.3f} | {GRU_LOSO['src']} |",
          f"| Transformer | {TF_LOSO['mean']:.3f} ± {TF_LOSO['std']:.3f} | "
          f"{TF_LOSO['excl05']:.3f} | {TF_LOSO['src']} |", "",
          f"The vanilla RNN's fold-average AUC ({m:.3f}) sits in the same band as the BiLSTM "
          f"({BILSTM_LOSO['mean']:.3f}), GRU ({GRU_LOSO['mean']:.3f}), and Transformer "
          f"({TF_LOSO['mean']:.3f}) — the cross-set generalization is consistent with the "
          f"fixed-split finding, and set03 is not an unusually easy fold for the RNN (its set03 "
          f"AUC {set03_auc:.3f} ≈ its fixed-split number). Individual folds vary (set04 is the "
          f"hardest for all families); 6 folds is too few for a significance test, so this is "
          f"reported as a generalization sanity check only.", ""]
    (HERE / "10_loso_report.md").write_text("\n".join(L))
    print("wrote 10_loso_report.md")


if __name__ == "__main__":
    main()

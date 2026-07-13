"""07_loso_report.py — Phase T5: aggregate Phase T4's runs_loso/ vs the BiLSTM's LOSO.

Reads `../phase4_kaggle_final/runs_loso/<fold>.json` (winner config, seed 42, Issue-5
protocol: per-fold grouped 85/15 val split, per-fold pos_weight + train-only norm) and
reports the same table shape as journal_prep/issue5_loso_cv/05_loso_results.md so the
two sit side by side.

    python transformer/phase5_analysis/07_loso_report.py
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS_LOSO = HERE.parent / "phase4_kaggle_final" / "runs_loso"
LSTM_LOSO_CSV = HERE.parent.parent / "journal_prep" / "issue5_loso_cv" / "05_loso_results.csv"

SETS = ["set01", "set02", "set03", "set04", "set05", "set06"]


def load_lstm_loso():
    """BiLSTM's own LOSO (journal_prep/issue5_loso_cv) -- read directly from its CSV,
    not hand-transcribed, so there's no risk of a copy error in the reference numbers."""
    rows = {r["test_set"]: r for r in csv.DictReader(open(LSTM_LOSO_CSV))}
    for s in SETS:
        for k in ("test_n", "pos_weight", "auc", "pr_auc", "f1", "acc"):
            rows[s][k] = float(rows[s][k])
        rows[s]["test_n"] = int(rows[s]["test_n"])
    return rows


LSTM_LOSO = load_lstm_loso()
_lstm_aucs = np.array([LSTM_LOSO[s]["auc"] for s in SETS])
_lstm_big = np.array([LSTM_LOSO[s]["auc"] for s in SETS if LSTM_LOSO[s]["test_n"] >= 100])
LSTM_MEAN, LSTM_STD = float(_lstm_aucs.mean()), float(_lstm_aucs.std(ddof=1))
LSTM_BIG_MEAN, LSTM_BIG_STD = float(_lstm_big.mean()), float(_lstm_big.std(ddof=1))


def load_rows():
    rows = []
    for s in SETS:
        r = json.loads((RUNS_LOSO / f"{s}.json").read_text())
        required = {"test_set", "train_n", "val_n", "test_n", "test_pos", "pos_weight",
                   "val", "test", "test_confusion_matrix", "best_epoch", "seed"}
        assert required.issubset(r.keys()), f"{s}.json missing keys: {required - r.keys()}"
        assert r["test_set"] == s and r["seed"] == 42
        rows.append(r)
    return rows


def main():
    rows = load_rows()
    aucs = np.array([r["test"]["auc"] for r in rows])
    prs = np.array([r["test"]["pr_auc"] for r in rows])
    f1s = np.array([r["test"]["f1"] for r in rows])
    big = [r for r in rows if r["test_n"] >= 100]
    big_aucs = np.array([r["test"]["auc"] for r in big])

    print(f"{'fold':8s} {'test_n':>7s} {'pos%':>5s} {'AUC':>7s} {'PR':>7s} {'F1':>7s} "
         f"{'pw':>6s} {'ep':>3s}  |  LSTM AUC (ref)")
    print("-" * 78)
    for r in rows:
        lstm_ref = LSTM_LOSO[r["test_set"]]["auc"]
        print(f"{r['test_set']:8s} {r['test_n']:7d} {r['test_pos']*100:5.1f} "
             f"{r['test']['auc']:7.4f} {r['test']['pr_auc']:7.4f} {r['test']['f1']:7.4f} "
             f"{r['pos_weight']:6.3f} {r['best_epoch']:3d}  |  {lstm_ref:.4f}")
    print("-" * 78)
    print(f"{'MEAN±STD':8s} {'':>7s} {'':>5s} {aucs.mean():.4f}±{aucs.std(ddof=1):.4f}"
         f"                                    |  {LSTM_MEAN:.3f}±{LSTM_STD:.3f}")
    print(f"excl. set05 (N<100): {big_aucs.mean():.4f} ± {big_aucs.std(ddof=1):.4f}  "
         f"(n={len(big)})  |  LSTM: {LSTM_BIG_MEAN:.3f}±{LSTM_BIG_STD:.3f}")

    # --- CSV ---
    cols = ["test_set", "test_n", "test_pos", "train_n", "val_n", "pos_weight",
           "best_epoch", "val_auc", "auc", "pr_auc", "f1", "acc", "prec", "rec"]
    with open(HERE / "07_loso_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(dict(test_set=r["test_set"], test_n=r["test_n"], test_pos=r["test_pos"],
                            train_n=r["train_n"], val_n=r["val_n"], pos_weight=r["pos_weight"],
                            best_epoch=r["best_epoch"], val_auc=r["val"]["auc"],
                            auc=r["test"]["auc"], pr_auc=r["test"]["pr_auc"], f1=r["test"]["f1"],
                            acc=r["test"]["acc"], prec=r["test"]["prec"], rec=r["test"]["rec"]))

    # --- Markdown ---
    L = ["# Phase T5 — LOSO 6-fold CV (transformer_searched, seed 42)", "",
        "Issue-5 protocol applied to the Phase-T3 winner "
        "(`d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05`): per-fold test = "
        "one PIE set, train+val = the other 5 (85/15 pedestrian-grouped split), "
        "per-fold pos_weight + train-only norm. Identical architecture/recipe across "
        "folds. Aggregated from `../phase4_kaggle_final/runs_loso/<fold>.json` "
        "(downloaded Kaggle output, independently re-verified in `PROGRESS_LOG.md`).", "",
        "| Fold (test set) | test N | pos % | AUC | PR-AUC | F1 | Acc | pos_w | best ep | LSTM AUC (ref, Issue 5) |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lstm_ref = LSTM_LOSO[r["test_set"]]
        L.append(f"| {r['test_set']} | {r['test_n']} | {r['test_pos']*100:.1f} | "
                 f"{r['test']['auc']:.4f} | {r['test']['pr_auc']:.4f} | {r['test']['f1']:.4f} | "
                 f"{r['test']['acc']:.4f} | {r['pos_weight']:.3f} | {r['best_epoch']} | "
                 f"{lstm_ref['auc']:.4f} |")
    L += ["",
         f"**Mean ± std over 6 folds (unweighted): AUC {aucs.mean():.4f} ± {aucs.std(ddof=1):.4f}, "
         f"PR-AUC {prs.mean():.4f} ± {prs.std(ddof=1):.4f}, F1 {f1s.mean():.4f} ± {f1s.std(ddof=1):.4f}.**",
         "",
         f"Excluding the tiny, uninterpretable set05 fold (N=47), the {len(big)} folds "
         f"with N≥100: AUC {big_aucs.mean():.4f} ± {big_aucs.std(ddof=1):.4f}.", "",
         f"**vs BiLSTM (Issue 5):** 6-fold {LSTM_MEAN:.3f} ± {LSTM_STD:.3f}, excl. set05 "
         f"{LSTM_BIG_MEAN:.3f} ± {LSTM_BIG_STD:.3f} — recomputed here from Issue 5's raw "
         f"`05_loso_results.csv` with `ddof=1` (sample std) for consistency with this "
         f"report's own convention; Issue 5's originally-published text quotes "
         f"`0.928 ± 0.041` / `0.915 ± 0.029` using `ddof=0` (population std) — same "
         f"underlying per-fold numbers, not a data discrepancy. Transformer LOSO mean is "
         f"{'above' if aucs.mean() > LSTM_MEAN else 'below'} the BiLSTM's by "
         f"{abs(aucs.mean() - LSTM_MEAN):.4f} (unweighted 6-fold), "
         f"{'above' if big_aucs.mean() > LSTM_BIG_MEAN else 'below'} by "
         f"{abs(big_aucs.mean() - LSTM_BIG_MEAN):.4f} excl. set05.", "",
         "**This is a descriptive fold table, not a hypothesis test** — 6 folds is too "
         "few for a paired test to have any power; it's reported as a generalization "
         "sanity check (does the win/loss on the fixed test03 split hold up when every "
         "set gets a turn as the held-out fold), not as the primary evidence. The "
         "primary evidence is `05_comparison_report.md`'s window-paired bootstrap on "
         "the fixed test03 split."]
    (HERE / "07_loso_report.md").write_text("\n".join(L) + "\n")

    print("\nwrote 07_loso_results.csv, 07_loso_report.md")


if __name__ == "__main__":
    main()

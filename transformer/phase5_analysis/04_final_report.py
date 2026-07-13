"""04_final_report.py — Phase T5: aggregate Phase T4's runs_final/ into 5-seed tables.

Reads the downloaded `../phase4_kaggle_final/runs_final/transformer_{searched,default}/
seed<k>/final.json` files (schema: {best_epoch, val, test, test_confusion_matrix} —
pipeline/05_compare_runs.py's schema) and the determinism-check rerun, and reports
5-seed mean±std val/test metrics for both configs next to the frozen BiLSTM row.

This is pure aggregation of already-existing numbers -- no new statistics, no model
loading. The actual comparison (paired bootstrap + verdict) is 05_compare_vs_lstm.py.

    python transformer/phase5_analysis/04_final_report.py
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PHASE4 = HERE.parent / "phase4_kaggle_final"
RUNS_FINAL = PHASE4 / "runs_final"

SEEDS = [42, 0, 1, 2, 3]
CONFIGS = ["transformer_searched", "transformer_default"]
METRICS = ["loss", "acc", "f1", "auc", "pr_auc", "prec", "rec"]

# Frozen BiLSTM baseline (journal_prep/issue2_clean_protocol, Issue 4) -- never retrained,
# reported here only for side-by-side context.
LSTM = dict(params=594561, val_auc_mean=0.9644, val_auc_std=0.0043,
           test_auc_mean=0.9324, test_auc_std=0.0114, test_pr_mean=0.876,
           test_f1_mean=0.8275, test_f1_std=0.0123, test_acc_mean=0.8827, test_acc_std=0.0091)
TRANSFORMER_PARAMS = {"transformer_searched": 794241, "transformer_default": 268417}


def load_rows():
    rows = []
    for tag in CONFIGS:
        for seed in SEEDS:
            d = RUNS_FINAL / tag / f"seed{seed}"
            fj = json.loads((d / "final.json").read_text())
            assert set(fj.keys()) == {"best_epoch", "val", "test", "test_confusion_matrix"}, \
                f"{tag} seed{seed}: unexpected final.json keys {fj.keys()}"
            row = dict(tag=tag, seed=seed, best_epoch=fj["best_epoch"],
                      n_params=TRANSFORMER_PARAMS[tag])
            for m in METRICS:
                row[f"val_{m}"] = fj["val"][m]
                row[f"test_{m}"] = fj["test"][m]
            rows.append(row)
    return rows


def load_determinism_check():
    primary = json.loads((RUNS_FINAL / "transformer_searched" / "seed42" / "final.json").read_text())
    rerun_dir = RUNS_FINAL / "_determinism_check" / "transformer_searched_seed42_rerun"
    rerun = json.loads((rerun_dir / "final.json").read_text())
    val_delta = abs(primary["val"]["auc"] - rerun["val"]["auc"])
    test_delta = abs(primary["test"]["auc"] - rerun["test"]["auc"])
    return dict(primary_val_auc=primary["val"]["auc"], rerun_val_auc=rerun["val"]["auc"],
               primary_test_auc=primary["test"]["auc"], rerun_test_auc=rerun["test"]["auc"],
               val_delta=val_delta, test_delta=test_delta, passed=(val_delta == 0.0 and test_delta == 0.0))


def summarize(rows, tag):
    sub = [r for r in rows if r["tag"] == tag]
    out = dict(tag=tag, n=len(sub), n_params=TRANSFORMER_PARAMS[tag])
    for m in METRICS:
        for split in ("val", "test"):
            vals = np.array([r[f"{split}_{m}"] for r in sub])
            out[f"{split}_{m}_mean"] = float(vals.mean())
            out[f"{split}_{m}_std"] = float(vals.std(ddof=1))
    return out


def main():
    rows = load_rows()
    det = load_determinism_check()

    print(f"Loaded {len(rows)} Stage D runs ({len(CONFIGS)} configs x {len(SEEDS)} seeds)\n")

    print("=== Determinism-of-record check (transformer_searched seed42) ===")
    print(f"  primary  val_auc={det['primary_val_auc']:.6f}  test_auc={det['primary_test_auc']:.6f}")
    print(f"  rerun    val_auc={det['rerun_val_auc']:.6f}  test_auc={det['rerun_test_auc']:.6f}")
    print(f"  |delta|  val={det['val_delta']:.2e}  test={det['test_delta']:.2e}  "
         f"[{'PASS' if det['passed'] else 'FAIL'}]")
    assert det["passed"], "determinism check failed -- do not trust the headline numbers"

    summaries = {tag: summarize(rows, tag) for tag in CONFIGS}

    print("\n=== 5-seed summary (val / test AUC) ===")
    print(f"{'config':<22} {'params':>8} {'val AUC':>18} {'test AUC':>18}")
    for tag in CONFIGS:
        s = summaries[tag]
        print(f"{tag:<22} {s['n_params']:>8,} "
             f"{s['val_auc_mean']:>7.4f} ± {s['val_auc_std']:<7.4f} "
             f"{s['test_auc_mean']:>7.4f} ± {s['test_auc_std']:<7.4f}")
    print(f"{'BiLSTM baseline (frozen)':<22} {LSTM['params']:>8,} "
         f"{LSTM['val_auc_mean']:>7.4f} ± {LSTM['val_auc_std']:<7.4f} "
         f"{LSTM['test_auc_mean']:>7.4f} ± {LSTM['test_auc_std']:<7.4f}")

    # --- CSV: per-seed rows ---
    cols = ["tag", "seed", "n_params", "best_epoch"] + \
          [f"{split}_{m}" for split in ("val", "test") for m in METRICS]
    with open(HERE / "04_final_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})

    # --- Markdown summary ---
    L = ["# Phase T5 — Final 5-seed results (Stage D)", "",
        "Aggregated from `../phase4_kaggle_final/runs_final/*/seed<k>/final.json` "
        "(downloaded Kaggle T4×2 output, independently re-verified in "
        "`PROGRESS_LOG.md`). This is pure aggregation — the formal comparison against "
        "the frozen LSTM (paired bootstrap + verdict) is `05_compare_vs_lstm.py`.", "",
        f"**Determinism-of-record check: {'PASS' if det['passed'] else 'FAIL'}** — "
        f"`transformer_searched` seed42 rerun reproduces val_auc="
        f"{det['rerun_val_auc']:.6f}, test_auc={det['rerun_test_auc']:.6f} exactly "
        f"(|delta| = {max(det['val_delta'], det['test_delta']):.1e}).", "",
        "## 5-seed mean ± std (test set03, N=2094)", "",
        "| config | params | val AUC | test AUC | test PR-AUC | test F1 | test Acc |",
        "|---|---|---|---|---|---|---|"]
    for tag in CONFIGS:
        s = summaries[tag]
        L.append(f"| `{tag}` | {s['n_params']:,} | "
                 f"{s['val_auc_mean']:.4f} ± {s['val_auc_std']:.4f} | "
                 f"**{s['test_auc_mean']:.4f} ± {s['test_auc_std']:.4f}** | "
                 f"{s['test_pr_auc_mean']:.4f} ± {s['test_pr_auc_std']:.4f} | "
                 f"{s['test_f1_mean']:.4f} ± {s['test_f1_std']:.4f} | "
                 f"{s['test_acc_mean']:.4f} ± {s['test_acc_std']:.4f} |")
    L.append(f"| BiLSTM baseline (frozen, Issue 4) | {LSTM['params']:,} | "
            f"{LSTM['val_auc_mean']:.4f} ± {LSTM['val_auc_std']:.4f} | "
            f"**{LSTM['test_auc_mean']:.4f} ± {LSTM['test_auc_std']:.4f}** | "
            f"{LSTM['test_pr_mean']:.3f} | "
            f"{LSTM['test_f1_mean']:.4f} ± {LSTM['test_f1_std']:.4f} | "
            f"{LSTM['test_acc_mean']:.4f} ± {LSTM['test_acc_std']:.4f} |")
    L += ["", "## Per-seed detail", "",
         "| config | seed | best_epoch | val AUC | test AUC | test F1 | test Acc |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['tag']}` | {r['seed']} | {r['best_epoch']} | "
                 f"{r['val_auc']:.4f} | {r['test_auc']:.4f} | {r['test_f1']:.4f} | "
                 f"{r['test_acc']:.4f} |")
    L += ["", "**This table alone is not the verdict** — raw mean-vs-mean comparisons "
         "don't account for sampling noise on the shared 2094 test windows. See "
         "`05_comparison_report.md` for the pre-registered paired-bootstrap comparison "
         "and Verdict (PLAN.md §6)."]
    (HERE / "04_final_summary.md").write_text("\n".join(L) + "\n")

    print("\nwrote 04_final_results.csv, 04_final_summary.md")


if __name__ == "__main__":
    main()

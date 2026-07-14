"""01_threshold_audit.py — Phase F1: frozen models, VAL-ONLY threshold audit (PLAN.md §9).

For each frozen model (BiLSTM baseline ×5 seeds, transformer_searched ×5,
transformer_default ×5): regenerate VAL probabilities from the frozen checkpoints (CPU),
select tau* per the pre-registered rule (00_common.best_threshold), and report val
F1/acc/precision/recall at 0.5 vs at tau*, per seed and for the 5-seed
averaged-probability ensemble. Also reproduces the checkpoint-headroom table (val F1 at
the AUC-best epoch vs the trajectory's max val F1) from each run's history.json.

TEST IS NEVER LOADED HERE — set03 rows are untouched (PLAN.md §8).

Outputs: 01_threshold_audit.csv, 01_threshold_audit.md
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("f1_common", HERE / "00_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

MODELS = {
    "lstm_baseline": dict(dirs=lambda s: C.LSTM_MULTISEED / f"seed{s}",
                          build=lambda: C.build_lstm(C.BASELINE_LSTM_CFG)),
    "transformer_searched": dict(dirs=lambda s: C.TF_RUNS_FINAL / "transformer_searched" / f"seed{s}",
                                 build=lambda: C.build_transformer(C.SEARCHED_TF_CFG)),
    "transformer_default": dict(dirs=lambda s: C.TF_RUNS_FINAL / "transformer_default" / f"seed{s}",
                                build=lambda: C.build_transformer(C.DEFAULT_TF_CFG)),
}


def headroom_rows():
    rows = []
    for name, spec in MODELS.items():
        for s in C.SEEDS:
            d = spec["dirs"](s)
            hist = json.loads((d / "history.json").read_text())
            fin = json.loads((d / "final.json").read_text())
            by_ep = {e["epoch"]: e for e in hist}
            be = fin["best_epoch"]
            f1_at_auc_best = by_ep[be]["val_f1"]
            mx = max(hist, key=lambda e: e["val_f1"])
            rows.append(dict(model=name, seed=s, auc_best_epoch=be,
                             val_f1_at_auc_best=f1_at_auc_best,
                             max_val_f1=mx["val_f1"], max_f1_epoch=mx["epoch"],
                             headroom=mx["val_f1"] - f1_at_auc_best))
    return rows


def main():
    X = C.load_splits()
    _, _, Xva, yva, _, _ = X
    print(f"val N={len(yva)} pos={int(yva.sum())}  (test untouched)")

    rows = []
    ens_rows = []
    for name, spec in MODELS.items():
        val_probs = {}
        for s in C.SEEDS:
            probs = C.prob_fn_from_run_dir(spec["dirs"](s), spec["build"]())(Xva)
            val_probs[s] = probs
            tau = C.best_threshold(yva, probs)
            m05 = C.metrics_at(yva, probs, 0.5)
            mt = C.metrics_at(yva, probs, tau)
            rows.append(dict(model=name, seed=s, tau=tau,
                             val_f1_05=m05["f1"], val_f1_tau=mt["f1"],
                             val_acc_05=m05["acc"], val_acc_tau=mt["acc"],
                             val_prec_05=m05["prec"], val_prec_tau=mt["prec"],
                             val_rec_05=m05["rec"], val_rec_tau=mt["rec"],
                             val_auc=m05["auc"]))
            print(f"  {name:22s} seed{s:<3d} tau*={tau:.4f}  val F1 {m05['f1']:.4f} -> {mt['f1']:.4f}"
                  f"  acc {m05['acc']:.4f} -> {mt['acc']:.4f}")
        ens = np.mean([val_probs[s] for s in C.SEEDS], axis=0)
        tau_e = C.best_threshold(yva, ens)
        e05 = C.metrics_at(yva, ens, 0.5)
        et = C.metrics_at(yva, ens, tau_e)
        ens_rows.append(dict(model=name, tau=tau_e, val_f1_05=e05["f1"], val_f1_tau=et["f1"],
                             val_acc_05=e05["acc"], val_acc_tau=et["acc"], val_auc=e05["auc"]))
        print(f"  {name:22s} ENSEMBLE tau*={tau_e:.4f}  val F1 {e05['f1']:.4f} -> {et['f1']:.4f}\n")

    hr = headroom_rows()

    with open(HERE / "01_threshold_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- markdown ----
    L = ["# 01 — Threshold audit on the frozen models (VAL ONLY; test untouched)", "",
         "tau* rule (pre-registered, PLAN.md §3.1): argmax val F1 over achievable cutoffs,",
         "tie-break higher val acc then |tau-0.5|, bounded [0.05, 0.95].", "",
         "## Per-seed: val metrics at 0.5 vs at tau*", "",
         "| model | seed | tau* | val F1 @0.5 | val F1 @tau* | val acc @0.5 | val acc @tau* |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['seed']} | {r['tau']:.4f} | {r['val_f1_05']:.4f} "
                 f"| {r['val_f1_tau']:.4f} | {r['val_acc_05']:.4f} | {r['val_acc_tau']:.4f} |")
    L += ["", "## 5-seed averaged-probability ensemble (the deployable form)", "",
          "| model | tau*_ens | val F1 @0.5 | val F1 @tau* | val acc @0.5 | val acc @tau* |",
          "|---|---|---|---|---|---|"]
    for r in ens_rows:
        L.append(f"| {r['model']} | {r['tau']:.4f} | {r['val_f1_05']:.4f} | {r['val_f1_tau']:.4f} "
                 f"| {r['val_acc_05']:.4f} | {r['val_acc_tau']:.4f} |")
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    L += ["", "## Mean per-seed gain from tau* alone (no retraining)", "",
          "| model | mean val F1 @0.5 | mean val F1 @tau* | mean gain |", "|---|---|---|---|"]
    for name, rs in by_model.items():
        g0 = np.mean([r["val_f1_05"] for r in rs])
        gt = np.mean([r["val_f1_tau"] for r in rs])
        L.append(f"| {name} | {g0:.4f} | {gt:.4f} | {gt - g0:+.4f} |")
    L += ["", "## Checkpoint headroom (history.json): val F1 at AUC-best epoch vs trajectory max", "",
          "| model | seed | AUC-best ep | val F1 there | max val F1 (ep) | headroom |",
          "|---|---|---|---|---|---|"]
    for r in hr:
        L.append(f"| {r['model']} | {r['seed']} | {r['auc_best_epoch']} | "
                 f"{r['val_f1_at_auc_best']:.4f} | {r['max_val_f1']:.4f} ({r['max_f1_epoch']}) | "
                 f"{r['headroom']:+.4f} |")
    mean_hr = np.mean([r["headroom"] for r in hr])
    L += ["", f"Mean headroom across the 15 frozen runs: **{mean_hr:+.4f} val F1** — the "
          "expected gain of the hybrid F1-checkpoint rule (PLAN.md §3.3), measured before "
          "any retraining.", ""]
    (HERE / "01_threshold_audit.md").write_text("\n".join(L))
    print("wrote 01_threshold_audit.csv, 01_threshold_audit.md")


if __name__ == "__main__":
    main()

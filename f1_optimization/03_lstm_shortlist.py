"""03_lstm_shortlist.py — Phase F3: measure the LSTM shortlist at 5-seed granularity
(PLAN.md §5 steps 2-3). VAL-ONLY; MPS; cached by JSON.

AMENDMENT (documented in PROGRESS_LOG.md, 2026-07-12; wording corrected 2026-07-13
after audit): the original fork-fidelity gate demanded exact reproduction of a cached
Issue-8 grid JSON. The baseline (dropout-0.3) cell could not be reproduced (6e-3 val
AUC drift) even though the forked loop is bit-deterministic run-to-run on BOTH mps and
cpu. Post-audit refinement: the irreproducibility is CONFIG-DEPENDENT, not universal —
this script's own runs_shortlist/ dropout-0.5 cells came out bit-identical to the
Issue-8 cache (all seeds, 16 digits: direct positive proof of fork fidelity), while
dropout-0.2/0.3 and lr1e-4 cells drift; LSTM training on MPS is additionally
process-history-dependent (see journal_prep/issue12_unified_pipeline/
12_equivalence_report.md — CPU is context-free). Consequences:
  (a) the gate is now a DETERMINISM GATE: the same (cfg, seed) trained twice must give
      bit-identical val metrics — abort otherwise; the drift vs the cached grid cell is
      reported for the record;
  (b) to avoid mixing drifted-environment cached measurements with fresh ones inside a
      ranking, ALL shortlist configs are re-measured fresh (x5 seeds, select="auc",
      current env, MPS) — the cached grid only NOMINATED the shortlist (02), it does not
      contribute measurements to the ranking below.

Steps:
1. DETERMINISM GATE (+ env-drift report vs the cached baseline cell).
2. All 8 shortlist configs x 5 seeds, select="auc", val-only
   -> runs_shortlist/<cfgid>/seed<k>.json (skip-if-exists).
3. Rank by 5-seed mean val F1 (tie: acc, then AUC). Top-2 advance to 04 (-> 03_top2.json).

Outputs: runs_shortlist/, 03_shortlist_results.csv, 03_shortlist_results.md, 03_top2.json
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_specc = importlib.util.spec_from_file_location("f1_common", HERE / "00_common.py")
C = importlib.util.module_from_spec(_specc)
_specc.loader.exec_module(C)
_spece = importlib.util.spec_from_file_location("f1_engines", HERE / "00_train_engines.py")
E = importlib.util.module_from_spec(_spece)
_spece.loader.exec_module(E)

RUNS = HERE / "runs_shortlist"


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = C.load_splits()
    sl = json.loads((HERE / "02_shortlist.json").read_text())
    print(f"device {device} | shortlist {len(sl['shortlist'])} cfgs x {len(C.SEEDS)} seeds "
          f"(all measured fresh; see AMENDMENT in docstring)\n")

    # ---------- 1. determinism gate + env-drift report ----------
    cached = json.loads((C.LSTM_GRID / C.lstm_cfg_id(C.BASELINE_LSTM_CFG) / "seed42.json").read_text())
    print("=== Determinism gate: baseline cfg @seed42 trained twice must be bit-identical ===")
    r = E.train_lstm(C.BASELINE_LSTM_CFG, 42, device, data, select="auc")
    r2 = E.train_lstm(C.BASELINE_LSTM_CFG, 42, device, data, select="auc")
    if not (r["val"] == r2["val"] and r["best_epoch"] == r2["best_epoch"]):
        raise RuntimeError("DETERMINISM GATE FAILED: two same-seed runs differ — "
                           "cached sweeps would be unreproducible. Aborting.")
    drift = abs(r["val"]["auc"] - cached["val"]["auc"])
    f1_drift = abs(r["val"]["f1"] - cached["val"]["f1"])
    print(f"  DETERMINISM GATE: PASS (two runs bit-identical)")
    print(f"  env drift vs Issue-8 cached cell (for the record): "
          f"AUC {drift:.2e}, F1 {f1_drift:.2e} — cached grid nominates only\n")

    # ---------- 2. fresh 5-seed measurement of the whole shortlist ----------
    for cfgid in sl["shortlist"]:
        cfg = sl["cfgs"][cfgid]
        for seed in C.SEEDS:
            jp = RUNS / cfgid / f"seed{seed}.json"
            if jp.exists():
                continue
            rr = E.train_lstm(cfg, seed, device, data, select="auc")
            jp.parent.mkdir(parents=True, exist_ok=True)
            jp.write_text(json.dumps(rr, indent=2))
            print(f"  {cfgid:26s} seed{seed:<3d} val F1 {rr['val']['f1']:.4f} "
                  f"auc {rr['val']['auc']:.4f} ({rr['seconds']:.0f}s)")

    # ---------- 3. rank shortlist by 5-seed mean val F1 ----------
    rows = []
    for cfgid in sl["shortlist"]:
        vals = [json.loads((RUNS / cfgid / f"seed{s}.json").read_text())["val"] for s in C.SEEDS]
        f1s = np.array([v["f1"] for v in vals])
        accs = np.array([v["acc"] for v in vals])
        aucs = np.array([v["auc"] for v in vals])
        rows.append(dict(cfg_id=cfgid, val_f1_mean=f1s.mean(), val_f1_std=f1s.std(ddof=1),
                         val_acc_mean=accs.mean(), val_auc_mean=aucs.mean()))
    rows.sort(key=lambda r: (round(r["val_f1_mean"], 10), round(r["val_acc_mean"], 10),
                             r["val_auc_mean"]), reverse=True)

    with open(HERE / "03_shortlist_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                     for r in rows])

    top2 = [r["cfg_id"] for r in rows[:2]]
    (HERE / "03_top2.json").write_text(json.dumps(dict(
        top2=top2, cfgs={c: sl["cfgs"][c] for c in top2},
        env_drift_vs_issue8=dict(auc_drift=drift, f1_drift=f1_drift),
        determinism_gate="PASS"), indent=2))

    L = ["# 03 — LSTM shortlist at 5-seed granularity (AUC protocol, val-only, fresh runs)", "",
         "**Amendment (see docstring + PROGRESS_LOG):** the current torch/MPS environment "
         "no longer reproduces Issue-8's cached grid values (env drift "
         f"AUC {drift:.2e} / F1 {f1_drift:.2e} on the baseline cell), while being "
         "bit-deterministic run-to-run (determinism gate PASS, two same-seed runs "
         "identical). The cached grid therefore only *nominated* the shortlist (02); "
         "every measurement below is a fresh run under the current environment.", "",
         "| rank | cfg | val F1 (5-seed) | val acc | val AUC |",
         "|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        tag = " **baseline**" if r["cfg_id"] == C.lstm_cfg_id(C.BASELINE_LSTM_CFG) else ""
        L.append(f"| {i} | `{r['cfg_id']}`{tag} | {r['val_f1_mean']:.4f} ± {r['val_f1_std']:.4f} "
                 f"| {r['val_acc_mean']:.4f} | {r['val_auc_mean']:.4f} |")
    L += ["", f"**Top-2 advancing to the F1-protocol confirm (04, part 4a): "
          f"`{top2[0]}`, `{top2[1]}`.**", "",
          "Selection key: 5-seed mean val F1 -> mean val acc -> mean val AUC "
          "(PLAN.md §2). All rows: select='auc' protocol, MPS, current environment.", ""]
    (HERE / "03_shortlist_results.md").write_text("\n".join(L))
    print(f"\ntop-2: {top2}")
    print("wrote runs_shortlist/, 03_shortlist_results.csv/.md, 03_top2.json")


if __name__ == "__main__":
    main()

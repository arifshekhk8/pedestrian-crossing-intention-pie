"""02_rnn_search.py — RNN study Phase R2: staged search (val-only, local CPU, resumable).

Applies the SAME search the BiLSTM received (Issue-8's 36-config grid + a pos_weight sweep),
now with --family birnn (bidirectional vanilla tanh RNN), to isolate gating. Every training run
routes through the ONE unified engine (journal_prep/issue12_unified_pipeline/12_unified_engine.py);
NO trainer code is duplicated here — this file is pure orchestration + caching.

VAL-ONLY BY CONSTRUCTION: the engine has no test code path. Every cached JSON is asserted to
carry no `test` key. Selection uses validation metrics only; test set03 is never read here.

Efficiency: every run uses select="f1", which records BOTH `val` (F1-best epoch) and
`val_at_auc_best` (AUC-best epoch). Early stopping is on val AUC and independent of select, so
one run per config yields both the val-F1 ranking and the val-AUC ranking.

INSTABILITY LEDGER (RNN-specific): vanilla tanh RNNs can vanish/explode. Every run is scored
for divergence (val AUC < 0.70 or a non-finite val metric) and, if flagged, RECORDED in
`_stage_summary.json["instability_ledger"]` — never silently dropped (mirrors transformer
PLAN §4's divergence contingency). Flagged configs still participate in the ranking; they just
rank last by construction.

Stages (PLAN.md §4):
  1. grid       36 configs (Issue-8 build_grid) x seed 42
  2. multiseed  {top-5 by val-F1} u {top-5 by val-AUC} u birnn_default, x 5 seeds
  3. pw sweep   F1-winner x pos_weight {1.0,1.3,1.682,2.1,2.5} x 5 seeds

Outputs:
  runs_search/<cfg_id>/seed<k>.json          stages 1-2 (pos_weight 1.682)
  runs_search/<cfg_id>/pw<w>/seed<k>.json    stage 3 sweep
  _stage_summary.json                        this script's own picks (R3 re-derives from raw)

Run from the repo root:  python rnn/phase2_search/02_rnn_search.py
"""
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = HERE / "runs_search"
CPU = torch.device("cpu")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


engine = _load("rnn_engine", ROOT / "journal_prep" / "issue12_unified_pipeline" / "12_unified_engine.py")
issue8 = _load("rnn_issue8", ROOT / "journal_prep" / "issue8_grid_search" / "08_grid_search.py")

SEEDS = engine.SEEDS                         # [42, 0, 1, 2, 3]
RNN_DEFAULT = dict(engine.PRESETS["birnn"])  # lr1e-3/do0.3/h128/nl2
PW_SWEEP = [1.0, 1.3, 1.682, 2.1, 2.5]
TOPK = 5
DIVERGE_AUC = 0.70   # val AUC below this = collapsed/diverged run (flag, don't drop)


def cfg_id(cfg):
    return issue8.cfg_id(cfg)


def val_f1(r):
    return r["val"]["f1"]


def val_auc(r):
    """Max val AUC over the trajectory (recorded via val_at_auc_best under select='f1')."""
    return r["val_at_auc_best"]["auc"]


def is_diverged(r):
    vf1, vauc = r["val"]["f1"], val_auc(r)
    finite = math.isfinite(vf1) and math.isfinite(vauc) and math.isfinite(r["val"]["acc"])
    return (not finite) or (vauc < DIVERGE_AUC)


def run_cached(cfg, seed, data, pw=None, subdir=None):
    """Train one (cfg, seed) via the engine, or load the cached JSON. VAL-ONLY."""
    base = RUNS / cfg_id(cfg)
    rdir = base / subdir if subdir else base
    jp = rdir / f"seed{seed}.json"
    if jp.exists():
        r = json.loads(jp.read_text())
    else:
        t0 = time.time()
        r = engine.train_run("birnn", cfg, seed, CPU, data, pos_weight=pw,
                             select="f1", out_dir=None)
        r["seconds"] = round(time.time() - t0, 1)
        rdir.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(r, indent=2))
    assert "test" not in r, f"LEAK: test key found in {jp}"
    return r


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main():
    data = engine.load_splits()
    grid = issue8.build_grid()
    ledger = []   # instability ledger: flagged (config, seed, subdir) runs
    print(f"device cpu | {len(grid)} grid configs | seeds {SEEDS} | family birnn\n")

    def note_divergence(cfg, seed, r, stage, subdir=None):
        if is_diverged(r):
            entry = dict(cfg_id=cfg_id(cfg), seed=seed, stage=stage, subdir=subdir,
                         val_f1=round(val_f1(r), 4), val_auc=round(val_auc(r), 4),
                         best_epoch=r["best_epoch"])
            ledger.append(entry)
            print(f"    ⚠ DIVERGED (recorded, not dropped): {cfg_id(cfg)} seed{seed} "
                  f"valAUC {val_auc(r):.4f}")

    # ---------- Stage 1: full grid at seed 42, VAL ONLY ----------
    print("=== Stage 1: 36-config grid (seed 42, val only) ===")
    grid_rows = []
    for i, cfg in enumerate(grid, 1):
        r = run_cached(cfg, 42, data)
        grid_rows.append((cfg, r))
        note_divergence(cfg, 42, r, "grid")
        print(f"  [{i:2d}/{len(grid)}] {cfg_id(cfg):22s} valF1 {val_f1(r):.4f}  "
              f"valAUC {val_auc(r):.4f}  ep {r['best_epoch']:2d} ({r.get('seconds',0):.0f}s)")

    by_f1 = sorted(grid_rows, key=lambda cr: (-val_f1(cr[1]), cfg_id(cr[0])))
    by_auc = sorted(grid_rows, key=lambda cr: (-val_auc(cr[1]), cfg_id(cr[0])))
    top_f1 = [cr[0] for cr in by_f1[:TOPK]]
    top_auc = [cr[0] for cr in by_auc[:TOPK]]

    # candidate union (dedup by cfg_id), birnn_default guaranteed in
    cand, seen = [], set()
    for cfg in top_f1 + top_auc + [RNN_DEFAULT]:
        if cfg_id(cfg) not in seen:
            seen.add(cfg_id(cfg)); cand.append(cfg)
    print(f"\ntop-5 val-F1 : {[cfg_id(c) for c in top_f1]}")
    print(f"top-5 val-AUC: {[cfg_id(c) for c in top_auc]}")
    print(f"multiseed candidates ({len(cand)}): {[cfg_id(c) for c in cand]}")

    # ---------- Stage 2: candidates x 5 seeds, VAL ONLY ----------
    print(f"\n=== Stage 2: {len(cand)} candidates x {len(SEEDS)} seeds (val only) ===")
    cand_stats = {}
    for cfg in cand:
        f1s, aucs, accs = [], [], []
        for seed in SEEDS:
            r = run_cached(cfg, seed, data)
            note_divergence(cfg, seed, r, "multiseed")
            f1s.append(val_f1(r)); aucs.append(val_auc(r)); accs.append(r["val"]["acc"])
        mf1, sf1 = mean_std(f1s)
        mauc, sauc = mean_std(aucs)
        macc, _ = mean_std(accs)
        cand_stats[cfg_id(cfg)] = dict(cfg=cfg, mean_f1=mf1, std_f1=sf1, mean_acc=macc,
                                       mean_auc=mauc, std_auc=sauc)
        print(f"  {cfg_id(cfg):22s} valF1 {mf1:.4f}±{sf1:.4f}  valAUC {mauc:.4f}±{sauc:.4f}")

    # F1-winner: max mean val F1 (tie: mean acc, then mean AUC). AUC-winner: max mean val AUC.
    f1_winner = max(cand_stats.values(),
                    key=lambda s: (s["mean_f1"], s["mean_acc"], s["mean_auc"]))
    auc_winner = max(cand_stats.values(), key=lambda s: (s["mean_auc"], s["mean_f1"]))
    f1w_id = cfg_id(f1_winner["cfg"]); aucw_id = cfg_id(auc_winner["cfg"])
    print(f"\nF1-winner : {f1w_id}  (mean valF1 {f1_winner['mean_f1']:.4f})")
    print(f"AUC-winner: {aucw_id}  (mean valAUC {auc_winner['mean_auc']:.4f})"
          f"{'  [same as F1-winner]' if aucw_id == f1w_id else ''}")

    # ---------- Stage 3: pos_weight sweep on the F1-winner, VAL ONLY ----------
    print(f"\n=== Stage 3: pos_weight sweep on {f1w_id} x {len(SEEDS)} seeds (val only) ===")
    sweep = {}
    for pw in PW_SWEEP:
        f1s = []
        for seed in SEEDS:
            r = run_cached(f1_winner["cfg"], seed, data, pw=pw, subdir=f"pw{pw:g}")
            note_divergence(f1_winner["cfg"], seed, r, "pw_sweep", subdir=f"pw{pw:g}")
            f1s.append(val_f1(r))
        mf1, sf1 = mean_std(f1s)
        sweep[f"{pw:g}"] = dict(pw=pw, mean_f1=mf1, std_f1=sf1)
        print(f"  pw {pw:<5g} valF1 {mf1:.4f}±{sf1:.4f}")
    best_pw = max(sweep.values(), key=lambda s: s["mean_f1"])["pw"]
    anchor_f1 = sweep["1.682"]["mean_f1"]
    # keep anchor 1.682 unless another pw beats it (strictly)
    chosen_pw = 1.682 if sweep[f"{best_pw:g}"]["mean_f1"] <= anchor_f1 + 1e-9 else best_pw
    print(f"\nbest sweep pw {best_pw:g} (mean valF1 {sweep[f'{best_pw:g}']['mean_f1']:.4f}); "
          f"anchor 1.682 = {anchor_f1:.4f} -> chosen pw {chosen_pw:g}")

    if ledger:
        print(f"\n⚠ instability ledger: {len(ledger)} diverged run(s) recorded (not dropped)")
    else:
        print("\ninstability ledger: 0 diverged runs (every config trained cleanly)")

    summary = dict(
        family="birnn",
        seeds=SEEDS,
        grid=[dict(cfg_id=cfg_id(c), cfg=c, seed42_val_f1=val_f1(r),
                   seed42_val_auc=val_auc(r), best_epoch=r["best_epoch"],
                   diverged=is_diverged(r))
              for c, r in grid_rows],
        top5_val_f1=[cfg_id(c) for c in top_f1],
        top5_val_auc=[cfg_id(c) for c in top_auc],
        candidates=cand_stats,
        f1_winner=f1w_id, auc_winner=aucw_id,
        pw_sweep=sweep, chosen_pw=chosen_pw,
        instability_ledger=ledger, diverge_auc_threshold=DIVERGE_AUC,
    )
    (HERE / "_stage_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote runs_search/ + _stage_summary.json  (R3 re-derives from raw JSONs)")


if __name__ == "__main__":
    main()

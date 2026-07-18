"""03_search_report.py — GRU study Phase G3: independent search review (human checkpoint).

Re-derives the ENTIRE search ranking from the raw runs_search/*.json files — it does NOT
trust phase2_search/_stage_summary.json, it recomputes everything and then asserts exact
agreement with it (the transformer/phase3 discipline). Also hard-asserts that zero files
anywhere under runs_search/ carry a `test` key (val-only-by-construction check).

Produces, for the user to confirm BEFORE any test-touching code exists:
  03_arch_grid.csv            36 configs, seed-42 val F1 + val AUC (the full grid)
  03_candidates_multiseed.csv candidates, 5-seed mean±std val F1 & val AUC
  03_pw_sweep.csv             F1-winner pos_weight sweep, 5-seed mean val F1
  03_search_summary.md        F1-winner (+ AUC-winner if different) + chosen pos_weight

Run from the repo root:  python gru/phase3_search_review/03_search_report.py
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SEARCH = ROOT / "gru" / "phase2_search"
RUNS = SEARCH / "runs_search"
SEEDS = [42, 0, 1, 2, 3]
PW_SWEEP = [1.0, 1.3, 1.682, 2.1, 2.5]
TOPK = 5


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


issue8 = _load("gru_issue8", ROOT / "journal_prep" / "issue8_grid_search" / "08_grid_search.py")
cfg_id = issue8.cfg_id


def load(cfg_id_str, seed, subdir=None):
    base = RUNS / cfg_id_str
    p = (base / subdir / f"seed{seed}.json") if subdir else (base / f"seed{seed}.json")
    r = json.loads(p.read_text())
    assert "test" not in r, f"LEAK: test key in {p}"
    return r


def vf1(r):
    return r["val"]["f1"]


def vauc(r):
    return r["val_at_auc_best"]["auc"]


def ms(vals):
    a = np.array(vals, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def assert_no_test_keys():
    n = 0
    for p in RUNS.rglob("seed*.json"):
        assert "test" not in json.loads(p.read_text()), f"LEAK: {p}"
        n += 1
    return n


def main():
    grid = issue8.build_grid()

    # ---- Stage 1: grid (seed 42) ----
    grid_rows = [(cfg, load(cfg_id(cfg), 42)) for cfg in grid]
    by_f1 = sorted(grid_rows, key=lambda cr: (-vf1(cr[1]), cfg_id(cr[0])))
    by_auc = sorted(grid_rows, key=lambda cr: (-vauc(cr[1]), cfg_id(cr[0])))
    top_f1 = [cfg_id(cr[0]) for cr in by_f1[:TOPK]]
    top_auc = [cfg_id(cr[0]) for cr in by_auc[:TOPK]]

    default_id = cfg_id(dict(lr=1e-3, dropout=0.3, hidden=128, num_layers=2))
    cand_ids, seen = [], set()
    for cid in top_f1 + top_auc + [default_id]:
        if cid not in seen:
            seen.add(cid); cand_ids.append(cid)

    # ---- Stage 2: candidates x 5 seeds ----
    cand = {}
    for cid in cand_ids:
        f1s = [vf1(load(cid, s)) for s in SEEDS]
        aucs = [vauc(load(cid, s)) for s in SEEDS]
        accs = [load(cid, s)["val"]["acc"] for s in SEEDS]
        mf1, sf1 = ms(f1s); mauc, sauc = ms(aucs); macc, _ = ms(accs)
        cand[cid] = dict(mean_f1=mf1, std_f1=sf1, mean_acc=macc, mean_auc=mauc, std_auc=sauc)

    f1w = max(cand, key=lambda c: (cand[c]["mean_f1"], cand[c]["mean_acc"], cand[c]["mean_auc"]))
    aucw = max(cand, key=lambda c: (cand[c]["mean_auc"], cand[c]["mean_f1"]))

    # ---- Stage 3: pos_weight sweep on F1-winner ----
    sweep = {}
    for pw in PW_SWEEP:
        f1s = [vf1(load(f1w, s, subdir=f"pw{pw:g}")) for s in SEEDS]
        mf1, sf1 = ms(f1s)
        sweep[f"{pw:g}"] = dict(mean_f1=mf1, std_f1=sf1)
    best_pw = max(PW_SWEEP, key=lambda pw: sweep[f"{pw:g}"]["mean_f1"])
    anchor = sweep["1.682"]["mean_f1"]
    chosen_pw = 1.682 if sweep[f"{best_pw:g}"]["mean_f1"] <= anchor + 1e-9 else best_pw

    n_files = assert_no_test_keys()

    # ---- cross-check vs the search's own summary (must match exactly) ----
    summ = json.loads((SEARCH / "_stage_summary.json").read_text())
    checks = {
        "top5_val_f1": summ["top5_val_f1"] == top_f1,
        "top5_val_auc": summ["top5_val_auc"] == top_auc,
        "f1_winner": summ["f1_winner"] == f1w,
        "auc_winner": summ["auc_winner"] == aucw,
        "chosen_pw": abs(summ["chosen_pw"] - chosen_pw) < 1e-9,
    }
    for cid, s in summ["candidates"].items():
        checks[f"cand_f1_{cid}"] = abs(s["mean_f1"] - cand[cid]["mean_f1"]) < 1e-9
    all_match = all(checks.values())
    assert all_match, f"MISMATCH vs _stage_summary.json: {[k for k,v in checks.items() if not v]}"

    write_outputs(grid_rows, by_f1, cand, f1w, aucw, sweep, best_pw, chosen_pw,
                  top_f1, top_auc, n_files)
    print(f"F1-winner : {f1w}  (val F1 {cand[f1w]['mean_f1']:.4f} ± {cand[f1w]['std_f1']:.4f})")
    print(f"AUC-winner: {aucw}  (val AUC {cand[aucw]['mean_auc']:.4f} ± {cand[aucw]['std_auc']:.4f})"
          f"{'  [SAME]' if aucw == f1w else '  [DIFFERENT]'}")
    print(f"chosen pos_weight: {chosen_pw:g}")
    print(f"cross-check vs _stage_summary.json: {'ALL MATCH' if all_match else 'MISMATCH'} "
          f"| no-test-key files scanned: {n_files}")
    print("wrote 03_arch_grid.csv, 03_candidates_multiseed.csv, 03_pw_sweep.csv, "
          "03_search_summary.md")


def write_outputs(grid_rows, by_f1, cand, f1w, aucw, sweep, best_pw, chosen_pw,
                  top_f1, top_auc, n_files):
    with open(HERE / "03_arch_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank_by_f1", "config", "lr", "dropout", "hidden", "num_layers",
                    "n_params", "best_epoch", "seed42_val_f1", "seed42_val_auc"])
        for i, (cfg, r) in enumerate(by_f1, 1):
            do = "inert" if cfg["num_layers"] == 1 else cfg["dropout"]
            w.writerow([i, cfg_id(cfg), f"{cfg['lr']:.0e}", do, cfg["hidden"],
                        cfg["num_layers"], r["n_params"], r["best_epoch"],
                        round(vf1(r), 4), round(vauc(r), 4)])

    with open(HERE / "03_candidates_multiseed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "val_f1_mean", "val_f1_std", "val_acc_mean",
                    "val_auc_mean", "val_auc_std", "is_f1_winner", "is_auc_winner"])
        for cid in sorted(cand, key=lambda c: -cand[c]["mean_f1"]):
            s = cand[cid]
            w.writerow([cid, round(s["mean_f1"], 4), round(s["std_f1"], 4),
                        round(s["mean_acc"], 4), round(s["mean_auc"], 4),
                        round(s["std_auc"], 4), cid == f1w, cid == aucw])

    with open(HERE / "03_pw_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos_weight", "val_f1_mean", "val_f1_std", "chosen"])
        for pw in PW_SWEEP:
            s = sweep[f"{pw:g}"]
            w.writerow([pw, round(s["mean_f1"], 4), round(s["std_f1"], 4), pw == chosen_pw])

    sf1, sacc, sauc = (cand[f1w]["mean_f1"], cand[f1w]["mean_acc"], cand[f1w]["mean_auc"])
    same = (aucw == f1w)
    L = [
        "# GRU study — Phase G3 search review (val-only; test UNTOUCHED)", "",
        "Independently re-derived from the raw `phase2_search/runs_search/*.json` files by "
        "`03_search_report.py`; every ranking recomputed from scratch and cross-checked against "
        "the search's own `_stage_summary.json` (exact agreement). "
        f"**{n_files} run files scanned — zero carry a `test` key** (val-only by construction).",
        "",
        "All numbers are **validation** metrics (set05/06, N=634), 5-seed mean ± std (ddof=1). "
        "`val F1` = best-val-F1-epoch F1; `val AUC` = max val AUC over the trajectory "
        "(`val_at_auc_best`). Everything through the unified engine, `--family gru`, CPU.",
        "",
        "## Winners (val-only selection)", "",
        f"- **F1-winner (primary, F1-first hierarchy): `{f1w}`** — "
        f"val F1 **{sf1:.4f} ± {cand[f1w]['std_f1']:.4f}**, val acc {sacc:.4f}, "
        f"val AUC {sauc:.4f}.",
        (f"- **AUC-winner: same config (`{f1w}`)** — the F1 and AUC rankings agree."
         if same else
         f"- **AUC-winner (secondary): `{aucw}`** — val AUC "
         f"**{cand[aucw]['mean_auc']:.4f} ± {cand[aucw]['std_auc']:.4f}**, "
         f"val F1 {cand[aucw]['mean_f1']:.4f}. *Differs from the F1-winner — both carried to G4.*"),
        "",
        f"- **pos_weight:** swept {{1.0, 1.3, 1.682, 2.1, 2.5}} on the F1-winner; best mean "
        f"val F1 at pw {best_pw:g} ({sweep[f'{best_pw:g}']['mean_f1']:.4f}), anchor 1.682 = "
        f"{sweep['1.682']['mean_f1']:.4f} → **chosen pw {chosen_pw:g}** "
        f"({'anchor retained' if abs(chosen_pw-1.682)<1e-9 else 'sweep value beats anchor'}).",
        "",
        "## Candidate multiseed (5-seed val, sorted by val F1)", "",
        "| config | val F1 | val acc | val AUC | note |", "|---|---|---|---|---|",
    ]
    for cid in sorted(cand, key=lambda c: -cand[c]["mean_f1"]):
        s = cand[cid]
        note = []
        if cid == f1w: note.append("**F1-winner**")
        if cid == aucw and not same: note.append("**AUC-winner**")
        if cid == cfg_id(dict(lr=1e-3, dropout=0.3, hidden=128, num_layers=2)):
            note.append("gru_default")
        L.append(f"| `{cid}` | {s['mean_f1']:.4f} ± {s['std_f1']:.4f} | {s['mean_acc']:.4f} | "
                 f"{s['mean_auc']:.4f} ± {s['std_auc']:.4f} | {' '.join(note)} |")

    L += ["", "## pos_weight sweep (F1-winner, 5-seed val F1)", "",
          "| pos_weight | val F1 | chosen |", "|---|---|---|"]
    for pw in PW_SWEEP:
        s = sweep[f"{pw:g}"]
        L.append(f"| {pw:g} | {s['mean_f1']:.4f} ± {s['std_f1']:.4f} | "
                 f"{'✅' if pw == chosen_pw else ''} |")

    L += ["", "## Top-5 grid rankings (seed 42)", "",
          f"- top-5 by val F1: {', '.join(f'`{c}`' for c in top_f1)}",
          f"- top-5 by val AUC: {', '.join(f'`{c}`' for c in top_auc)}",
          "", "Full 36-config grid in `03_arch_grid.csv` (ranked by seed-42 val F1).", "",
          "---", "",
          "## ⏸ HUMAN CHECKPOINT",
          "",
          "Test set03 is still **untouched**. The next phase (G4) trains these winners × 5 "
          "seeds and evaluates test **exactly once**. **Confirm the F1-winner"
          + ("" if same else " and AUC-winner")
          + f" (`{f1w}`{'' if same else f' / `{aucw}`'}) and pos_weight {chosen_pw:g} before G4 runs.**"]
    (HERE / "03_search_summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()

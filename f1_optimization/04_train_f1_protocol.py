"""04_train_f1_protocol.py — Phase F4: F1-protocol training + pos_weight sweep
(PLAN.md §3.3-3.4, §5 step 4, gates G1-G3). ALL VAL-ONLY; training on MPS; cached.

Parts (--part lstm | transformer | all):
  lstm:        4a confirm the 03 top-2 configs under the F1 protocol @pw 1.682 x5 seeds;
               the winner (5-seed mean val F1, tie acc then AUC) is the F1-selected
               config. 4b pos_weight sweep {1.0, 1.3, 1.682, 2.1, 2.5} x5 seeds on it.
               4c reference cell A2 = baseline cfg @1.682 (skipped if baseline is the
               winner — the cells collapse).
  transformer: 4b pos_weight sweep on the searched config (config re-confirmed by 02).
               4c reference cell B4 = default cfg @1.682.

F1 protocol (hybrid, PLAN.md §3.3): early stop + plateau on val AUC (frozen dynamics),
checkpoint = best val F1 (tie acc, then AUC); every final.json also records
val_at_auc_best for gate G1 (within-run criterion comparison).

Run dirs: runs_f1/<family_tag>/pw<w>/seed<k>/{best.pt, final.json, history.json,
norm_mean.npy, norm_std.npy} — skip-if-final.json-exists (interruption-safe).

Outputs: runs_f1/, 04_selection.json (gates + selected cfg/pw, consumed by 05),
04_sweep_results.csv, 04_sweep_results.md
"""
import argparse
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

RUNS = HERE / "runs_f1"
SEL = HERE / "04_selection.json"
PW_GRID = [1.0, 1.3, 1.682, 2.1, 2.5]
PW_ANCHOR = 1.682


def pw_tag(pw):
    return f"pw{pw:g}"


def run_cell(train_fn, family_tag, cfg, pw, device, data):
    """One (cfg, pw) cell x 5 seeds under the F1 protocol; returns the 5 results."""
    out = []
    for seed in C.SEEDS:
        d = RUNS / family_tag / pw_tag(pw) / f"seed{seed}"
        r = E.cached_run(train_fn, d, cfg, seed, device, data, pos_weight=pw, select="f1")
        out.append(r)
        print(f"  {family_tag:34s} {pw_tag(pw):8s} seed{seed:<3d} "
              f"val F1 {r['val']['f1']:.4f} (auc-ckpt would be {r['val_at_auc_best']['f1']:.4f}) "
              f"acc {r['val']['acc']:.4f} auc {r['val']['auc']:.4f} "
              f"ep {r['best_epoch']}/{r['auc_best_epoch']} ({r['seconds']:.0f}s)")
    return out


def cell_key(rs):
    """5-seed selection key: mean val F1 -> mean val acc -> mean val AUC."""
    return (round(float(np.mean([r["val"]["f1"] for r in rs])), 10),
            round(float(np.mean([r["val"]["acc"] for r in rs])), 10),
            float(np.mean([r["val"]["auc"] for r in rs])))


def gate_g1(rs):
    """G1: F1-checkpoint beats the same run's AUC-best-epoch val F1 in >=3/5 seeds."""
    wins = sum(1 for r in rs if r["val"]["f1"] > r["val_at_auc_best"]["f1"])
    return dict(wins=wins, n=len(rs), passed=wins >= 3)


def load_sel():
    return json.loads(SEL.read_text()) if SEL.exists() else {}


def save_sel(sel):
    SEL.write_text(json.dumps(sel, indent=2))


def part_lstm(device, data):
    top2 = json.loads((HERE / "03_top2.json").read_text())
    base_id = C.lstm_cfg_id(C.BASELINE_LSTM_CFG)
    print(f"=== 4a: LSTM top-2 confirm under F1 protocol @pw{PW_ANCHOR} ===")
    cells = {}
    for cid in top2["top2"]:
        cells[cid] = run_cell(E.train_lstm, f"lstm_{cid}", top2["cfgs"][cid], PW_ANCHOR,
                              device, data)
    win_id = max(cells, key=lambda c: cell_key(cells[c]))
    win_cfg = top2["cfgs"][win_id]
    print(f"\nF1-selected LSTM config: {win_id}  "
          f"(mean val F1 {np.mean([r['val']['f1'] for r in cells[win_id]]):.4f})")

    print(f"\n=== 4b: pos_weight sweep on {win_id} ===")
    sweep = {PW_ANCHOR: cells[win_id]}
    for pw in PW_GRID:
        if pw == PW_ANCHOR:
            continue
        sweep[pw] = run_cell(E.train_lstm, f"lstm_{win_id}", win_cfg, pw, device, data)

    a2_id = base_id
    if base_id != win_id and base_id not in cells:
        print(f"\n=== 4c: reference cell A2 = baseline cfg @pw{PW_ANCHOR} ===")
        cells[base_id] = run_cell(E.train_lstm, f"lstm_{base_id}", C.BASELINE_LSTM_CFG,
                                  PW_ANCHOR, device, data)

    best_pw = max(sweep, key=lambda p: cell_key(sweep[p]))
    g2 = dict(best_pw=best_pw, passed=best_pw != PW_ANCHOR and
              cell_key(sweep[best_pw]) > cell_key(sweep[PW_ANCHOR]),
              mean_f1={pw_tag(p): float(np.mean([r["val"]["f1"] for r in rs]))
                       for p, rs in sorted(sweep.items())})
    sel_pw = best_pw if g2["passed"] else PW_ANCHOR
    g3_base_cells = cells.get(base_id, cells.get(win_id))
    g3 = dict(winner=win_id, baseline=base_id,
              winner_f1=float(np.mean([r["val"]["f1"] for r in cells[win_id]])),
              baseline_f1=float(np.mean([r["val"]["f1"] for r in g3_base_cells])),
              passed=(win_id == base_id) or cell_key(cells[win_id]) > cell_key(g3_base_cells))
    headline_id = win_id if g3["passed"] else base_id
    g1 = gate_g1(sweep[sel_pw] if headline_id == win_id else cells[headline_id])

    sel = load_sel()
    sel["lstm"] = dict(headline_cfg_id=headline_id,
                       headline_cfg=top2["cfgs"].get(headline_id, C.BASELINE_LSTM_CFG),
                       confirm_winner=win_id, pw=sel_pw, baseline_cfg_id=base_id,
                       gates=dict(G1=g1, G2=g2, G3=g3))
    save_sel(sel)
    print(f"\nLSTM selection: cfg {headline_id} @pw{sel_pw:g} | "
          f"G1 {g1['wins']}/{g1['n']} {'PASS' if g1['passed'] else 'FAIL'} | "
          f"G2 best_pw={best_pw:g} {'PASS' if g2['passed'] else 'keep 1.682'} | "
          f"G3 {'PASS' if g3['passed'] else 'keep baseline cfg'}")


def part_transformer(device, data):
    print(f"=== transformer: pos_weight sweep on the searched config ===")
    sweep = {}
    for pw in PW_GRID:
        sweep[pw] = run_cell(E.train_transformer, "transformer_searched",
                             C.SEARCHED_TF_CFG, pw, device, data)
    print(f"\n=== reference cell B4 = default cfg @pw{PW_ANCHOR} ===")
    b4 = run_cell(E.train_transformer, "transformer_default", C.DEFAULT_TF_CFG,
                  PW_ANCHOR, device, data)

    best_pw = max(sweep, key=lambda p: cell_key(sweep[p]))
    g2 = dict(best_pw=best_pw, passed=best_pw != PW_ANCHOR and
              cell_key(sweep[best_pw]) > cell_key(sweep[PW_ANCHOR]),
              mean_f1={pw_tag(p): float(np.mean([r["val"]["f1"] for r in rs]))
                       for p, rs in sorted(sweep.items())})
    sel_pw = best_pw if g2["passed"] else PW_ANCHOR
    g1 = gate_g1(sweep[sel_pw])

    sel = load_sel()
    sel["transformer"] = dict(pw=sel_pw, gates=dict(
        G1=g1, G2=g2,
        G1_default=gate_g1(b4)))
    save_sel(sel)
    print(f"\ntransformer selection: pw{sel_pw:g} | "
          f"G1 {g1['wins']}/{g1['n']} {'PASS' if g1['passed'] else 'FAIL'} | "
          f"G2 best_pw={best_pw:g} {'PASS' if g2['passed'] else 'keep 1.682'}")


def write_summary():
    rows = []
    for fj in sorted(RUNS.glob("*/pw*/seed*/final.json")):
        r = json.loads(fj.read_text())
        family_tag, pwdir, seeddir = fj.parts[-4], fj.parts[-3], fj.parts[-2]
        rows.append(dict(family_tag=family_tag, pw=r["pos_weight"], seed=r["seed"],
                         best_epoch=r["best_epoch"], auc_best_epoch=r["auc_best_epoch"],
                         val_f1=r["val"]["f1"], val_acc=r["val"]["acc"],
                         val_auc=r["val"]["auc"],
                         val_f1_auc_ckpt=r["val_at_auc_best"]["f1"],
                         seconds=r["seconds"], device=r["device"]))
    with open(HERE / "04_sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows([{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
                     for r in rows])

    sel = load_sel()
    cells = {}
    for r in rows:
        cells.setdefault((r["family_tag"], r["pw"]), []).append(r)
    L = ["# 04 — F1-protocol training + pos_weight sweep (val-only)", "",
         "Hybrid rule per PLAN.md §3.3: stop/schedule on val AUC (frozen dynamics), "
         "checkpoint = best val F1 (tie acc, then AUC). `f1 (auc-ckpt)` shows what the "
         "frozen AUC-checkpoint rule would have scored on the same trajectory (gate G1).", "",
         "| cell | pw | val F1 (5-seed) | val F1 auc-ckpt | val acc | val AUC |",
         "|---|---|---|---|---|---|"]
    for (fam, pw), rs in sorted(cells.items()):
        f1s = np.array([r["val_f1"] for r in rs])
        f1a = np.array([r["val_f1_auc_ckpt"] for r in rs])
        L.append(f"| `{fam}` | {pw:g} | {f1s.mean():.4f} ± {f1s.std(ddof=1):.4f} | "
                 f"{f1a.mean():.4f} | {np.mean([r['val_acc'] for r in rs]):.4f} | "
                 f"{np.mean([r['val_auc'] for r in rs]):.4f} |")
    L += ["", "## Selections + gates (04_selection.json)", "", "```json",
          json.dumps(sel, indent=2), "```", ""]
    (HERE / "04_sweep_results.md").write_text("\n".join(L))
    print("wrote 04_sweep_results.csv, 04_sweep_results.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["lstm", "transformer", "all"])
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = C.load_splits()
    print(f"device {device} | pw grid {PW_GRID} | seeds {C.SEEDS}\n")
    if args.part in ("lstm", "all"):
        part_lstm(device, data)
    if args.part in ("transformer", "all"):
        part_transformer(device, data)
    write_summary()


if __name__ == "__main__":
    main()

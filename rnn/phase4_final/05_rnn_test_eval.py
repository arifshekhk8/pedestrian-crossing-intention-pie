"""05_rnn_test_eval.py — RNN study Phase R4: THE single designated test-touch.

This is the ONLY script in the RNN study that reads test set03. It runs exactly once, after
the R3 human checkpoint cleared. Mirrors f1_optimization/05_final_test_eval.py and
gru/phase4_final/05_gru_test_eval.py's discipline:

  1. PARITY GATE — regenerate the frozen BiLSTM's per-seed test AUC from its checkpoints and
     assert it matches the stored final.json values (0.9131...) BEFORE any RNN number is
     emitted. If the frozen baseline has drifted, abort.
  2. For each RNN arm (from runs_final/), regenerate val + test probabilities on CPU
     (common.prob_fn_from_run_dir), select tau* on that seed's own VAL probs (never test),
     and report F1/acc/AUC per-seed-mean AND 5-seed probability-ensemble, at BOTH 0.5 and tau*.
  3. Cache all prob vectors to phase4_final/probs_cache/ for the R5 paired bootstrap.

Reuses f1_optimization/00_common.py verbatim (best_threshold, metrics_at, prob_fn_from_run_dir,
load_splits). No metric code is reinvented.

Run from the repo root:  python rnn/phase4_final/05_rnn_test_eval.py
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = HERE / "runs_final"
SEARCH = ROOT / "rnn" / "phase2_search"
PCACHE = HERE / "probs_cache"
PCACHE.mkdir(exist_ok=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


common = _load("rnn_common", ROOT / "f1_optimization" / "00_common.py")
engine = _load("rnn_engine", ROOT / "journal_prep" / "issue12_unified_pipeline" / "12_unified_engine.py")

SEEDS = common.SEEDS
LSTM_MULTISEED = common.LSTM_MULTISEED
BASELINE_LSTM_CFG = common.BASELINE_LSTM_CFG

F1_WINNER_ID = json.loads((SEARCH / "_stage_summary.json").read_text())["f1_winner"]
F1_WINNER_CFG = json.loads(
    (SEARCH / "runs_search" / F1_WINNER_ID / "seed42.json").read_text())["cfg"]
DEFAULT_CFG = dict(engine.PRESETS["birnn"])

# arm -> birnn cfg used to build the model
ARMS = {
    "rnn_f1_winner": F1_WINNER_CFG,
    "rnn_winner_auc": F1_WINNER_CFG,
    "rnn_default_f1": DEFAULT_CFG,
    "rnn_default_auc": DEFAULT_CFG,
}


def parity_gate(Xte, yte):
    print("=== PARITY GATE: frozen BiLSTM test AUC (checkpoints vs stored final.json) ===")
    max_delta = 0.0
    for seed in SEEDS:
        rd = LSTM_MULTISEED / f"seed{seed}"
        model = common.build_lstm(BASELINE_LSTM_CFG)
        pf = common.prob_fn_from_run_dir(rd, model)
        recomputed = roc_auc_score(yte, pf(Xte))
        stored = json.loads((rd / "final.json").read_text())["test"]["auc"]
        d = abs(recomputed - stored)
        max_delta = max(max_delta, d)
        print(f"  seed {seed:2d}: recomputed {recomputed:.6f}  stored {stored:.6f}  |Δ| {d:.2e}")
    assert max_delta < 1e-4, f"PARITY GATE FAILED: frozen BiLSTM drifted (max |Δ|={max_delta:.2e})"
    print(f"  PARITY GATE PASS (max |Δ| = {max_delta:.2e})\n")
    return max_delta


def eval_arm(arm, cfg, Xva, yva, Xte, yte):
    per_val, per_test, taus = [], [], []
    rows = []
    for seed in SEEDS:
        model = engine.MODEL_REGISTRY["birnn"](cfg)
        pf = common.prob_fn_from_run_dir(RUNS / arm / f"seed{seed}", model)
        pv, pt = pf(Xva), pf(Xte)
        per_val.append(pv); per_test.append(pt)
        np.save(PCACHE / f"{arm}_seed{seed}_val.npy", pv)
        np.save(PCACHE / f"{arm}_seed{seed}_test.npy", pt)
        tau = common.best_threshold(yva, pv)
        taus.append(tau)
        m05 = common.metrics_at(yte, pt, 0.5)
        mt = common.metrics_at(yte, pt, tau)
        rows.append(dict(seed=seed, tau=tau, f1_05=m05["f1"], acc_05=m05["acc"],
                         auc=m05["auc"], f1_tau=mt["f1"], acc_tau=mt["acc"]))

    val_ens = np.mean(per_val, axis=0)
    test_ens = np.mean(per_test, axis=0)
    np.save(PCACHE / f"{arm}_ens_val.npy", val_ens)
    np.save(PCACHE / f"{arm}_ens_test.npy", test_ens)
    tau_ens = common.best_threshold(yva, val_ens)
    ens05 = common.metrics_at(yte, test_ens, 0.5)
    enst = common.metrics_at(yte, test_ens, tau_ens)

    def ms(key):
        a = np.array([r[key] for r in rows], float)
        return float(a.mean()), float(a.std(ddof=1))

    summary = dict(
        arm=arm, cfg=cfg,
        per_seed=dict(
            f1_05=ms("f1_05"), acc_05=ms("acc_05"), auc=ms("auc"),
            f1_tau=ms("f1_tau"), acc_tau=ms("acc_tau"),
            tau_mean=float(np.mean(taus)),
        ),
        ensemble=dict(
            tau=tau_ens, f1_05=ens05["f1"], acc_05=ens05["acc"], auc=ens05["auc"],
            f1_tau=enst["f1"], acc_tau=enst["acc"],
        ),
        rows=rows,
    )
    return summary


def main():
    Xtr, ytr, Xva, yva, Xte, yte = common.load_splits()
    np.save(PCACHE / "y_val.npy", yva)
    np.save(PCACHE / "y_test.npy", yte)
    print(f"loaded splits | val {len(yva)} | test {len(yte)} ({int(yte.sum())} pos)")
    print(f"F1-winner arm cfg: {F1_WINNER_CFG}\n")

    parity_max = parity_gate(Xte, yte)

    results = {}
    print("=== RNN arms — test set03 (single touch) ===")
    for arm, cfg in ARMS.items():
        s = eval_arm(arm, cfg, Xva, yva, Xte, yte)
        results[arm] = s
        ps, en = s["per_seed"], s["ensemble"]
        print(f"\n{arm}  (cfg={cfg})")
        print(f"  per-seed  F1@0.5 {ps['f1_05'][0]:.4f}±{ps['f1_05'][1]:.4f}  "
              f"F1@τ* {ps['f1_tau'][0]:.4f}±{ps['f1_tau'][1]:.4f}  "
              f"acc@τ* {ps['acc_tau'][0]:.4f}  AUC {ps['auc'][0]:.4f}±{ps['auc'][1]:.4f}  "
              f"(τ̄ {ps['tau_mean']:.3f})")
        print(f"  ensemble  F1@0.5 {en['f1_05']:.4f}  F1@τ* {en['f1_tau']:.4f}  "
              f"acc@τ* {en['acc_tau']:.4f}  AUC {en['auc']:.4f}  (τ* {en['tau']:.3f})")

    out = dict(parity_gate_max_delta=parity_max, seeds=SEEDS, f1_winner_id=F1_WINNER_ID,
               arms=results)
    (HERE / "05_final_arms.json").write_text(json.dumps(out, indent=2))
    write_csv_md(results, parity_max)
    print("\nwrote 05_final_arms.json/.csv, 05_final_summary.md, probs_cache/*.npy")


def write_csv_md(results, parity_max):
    with open(HERE / "05_final_arms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "stat", "f1_05", "f1_tau", "acc_05", "acc_tau", "auc", "tau"])
        for arm, s in results.items():
            ps, en = s["per_seed"], s["ensemble"]
            w.writerow([arm, "per_seed_mean", round(ps["f1_05"][0], 4), round(ps["f1_tau"][0], 4),
                        round(ps["acc_05"][0], 4), round(ps["acc_tau"][0], 4),
                        round(ps["auc"][0], 4), round(ps["tau_mean"], 3)])
            w.writerow([arm, "ensemble", round(en["f1_05"], 4), round(en["f1_tau"], 4),
                        round(en["acc_05"], 4), round(en["acc_tau"], 4),
                        round(en["auc"], 4), round(en["tau"], 3)])

    L = ["# RNN study — Phase R4 final test evaluation (set03, touched ONCE)", "",
         f"Single test pass through `05_rnn_test_eval.py` after the R3 sign-off. "
         f"**Parity gate PASS** (frozen BiLSTM per-seed test AUC recomputed from checkpoints "
         f"matches stored `final.json`, max |Δ| = {parity_max:.2e}). All probabilities "
         f"regenerated on CPU; τ\\* selected on each seed's own **validation** probs (never "
         f"test); ensemble = the 5 seeds' averaged probabilities.", "",
         "Per-seed = mean ± std (ddof=1) over seeds [42,0,1,2,3] at each seed's own τ\\*. "
         "`ens` = deployable 5-seed probability ensemble (a different statistic — always "
         "labeled). AUC is threshold-free.", "",
         "| arm | stat | F1@0.5 | F1@τ\\* | Acc@τ\\* | AUC |", "|---|---|---|---|---|---|"]
    for arm, s in results.items():
        ps, en = s["per_seed"], s["ensemble"]
        L.append(f"| `{arm}` | per-seed | {ps['f1_05'][0]:.4f} ± {ps['f1_05'][1]:.4f} | "
                 f"{ps['f1_tau'][0]:.4f} ± {ps['f1_tau'][1]:.4f} | {ps['acc_tau'][0]:.4f} | "
                 f"{ps['auc'][0]:.4f} ± {ps['auc'][1]:.4f} |")
        L.append(f"| `{arm}` | ensemble | {en['f1_05']:.4f} | {en['f1_tau']:.4f} | "
                 f"{en['acc_tau']:.4f} | {en['auc']:.4f} |")
    L += ["", "Reference (frozen, for orientation — formal deltas in Phase R5 "
          "`07_compare.py`): BiLSTM F1 0.828 / AUC 0.9324; BiLSTM-F1 0.844 / 0.940; "
          "Transformer-F1 0.847 / 0.947; searched Transformer 0.845 / 0.9497; "
          "GRU-F1 0.849 / 0.941.", ""]
    (HERE / "05_final_summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()

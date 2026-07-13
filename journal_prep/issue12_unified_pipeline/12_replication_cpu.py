"""12_replication_cpu.py — single-engine, single-device replication of the F1-first
endpoints (audit finding ENGINE-DEVICE-CONFOUND).

WHY (pre-registered before running): the original headline comparisons span engines
and devices — frozen BiLSTM trained by pipeline/04 on local CPU, frozen transformer by
the phase1 engine on Kaggle T4, F1-program arms by the forked engines on MPS — so
model family was confounded with training engine/device, and recurrent-on-MPS training
is process-history-dependent (see 12_equivalence_report.md). This annex retrains every
arm the three pre-registered endpoints need under ONE engine (12_unified_engine.py, a
proven-equivalent single code path) on ONE device (CPU — measured context-free and
bit-reproducible), and recomputes the endpoints. Selections are FROZEN from the
original program (configs, checkpoint rules, pos_weights: 04_selection.json) — this
replication tests engine/device sensitivity of the VERDICTS, it does not re-select.

Cells (x5 seeds each, CPU, cached by final.json):
  A0c bilstm  baseline cfg  select=auc pw1.682  @0.5   (frozen-protocol replica)
  A2c bilstm  baseline cfg  select=f1  pw1.682  @tau*
  A3c bilstm  h256 cfg      select=f1  pw1.682  @tau*  (LSTM-F1 headline)
  B0c transformer searched  select=auc pw1.682  @0.5   (frozen-protocol replica)
  B2c transformer searched  select=f1  pw1.682  @tau*
  B3c transformer searched  select=f1  pw2.5    @tau*  (Transformer-F1 headline)

TEST-TOUCH POLICY: this script is the designated single test-toucher for these NEW
runs (each checkpoint's test probabilities computed once, thresholds fitted on val
only, before test metrics are looked at). Original frozen artifacts are not touched.

Endpoints (10k paired bootstrap, rng(42), ensemble vectors, fixed val-fitted taus):
  (i')   A3c vs A0c — original verdict: IMPROVED (dF1 +0.0187 CI [+0.0073,+0.0300])
  (ii')  B3c vs B0c — original verdict: NO SIGNIFICANT CHANGE
  (iii') B3c vs A3c — original verdict: TIE

Outputs: runs_replication/, 12_replication_results.json, 12_replication_report.md
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
_specu = importlib.util.spec_from_file_location("unified", HERE / "12_unified_engine.py")
U = importlib.util.module_from_spec(_specu)
_specu.loader.exec_module(U)
_specc = importlib.util.spec_from_file_location("f1_common", ROOT / "f1_optimization" / "00_common.py")
C = importlib.util.module_from_spec(_specc)
_specc.loader.exec_module(C)

RUNS = HERE / "runs_replication"
B, RNG_SEED = 10000, 42

CELLS = {
    "A0c": ("bilstm", U.BILSTM_BASELINE_CFG, "auc", 1.682, "0.5"),
    "A2c": ("bilstm", U.BILSTM_BASELINE_CFG, "f1", 1.682, "tau"),
    "A3c": ("bilstm", U.BILSTM_F1_CFG, "f1", 1.682, "tau"),
    "B0c": ("transformer", U.TRANSFORMER_SEARCHED_CFG, "auc", 1.682, "0.5"),
    "B2c": ("transformer", U.TRANSFORMER_SEARCHED_CFG, "f1", 1.682, "tau"),
    "B3c": ("transformer", U.TRANSFORMER_SEARCHED_CFG, "f1", 2.5, "tau"),
    # G1 test-side counterfactuals (audit finding: the pre-registered G1 fallback was
    # never exercised — these measure on TEST what the F1-checkpoint rule actually
    # buys vs plain AUC checkpointing, same cfg/pw/device/engine):
    "A3f": ("bilstm", U.BILSTM_F1_CFG, "auc", 1.682, "tau"),
    "B3f": ("transformer", U.TRANSFORMER_SEARCHED_CFG, "auc", 2.5, "tau"),
}
ORIGINAL = {  # from f1_optimization/06_comparison_results.json (test, ensemble @tau*)
    "i": dict(pair="A3 vs A0", delta=+0.0187, ci=[+0.0073, +0.0300], verdict="IMPROVED"),
    "ii": dict(pair="B3 vs B0", delta=+0.0075, ci=[-0.0021, +0.0173], verdict="NO SIGNIFICANT CHANGE"),
    "iii": dict(pair="B3 vs A3", delta=+0.0008, ci=[-0.0124, +0.0142], verdict="TIE"),
}


@torch.no_grad()
def probs_from(run_dir, family, cfg, X):
    mean = np.load(run_dir / "norm_mean.npy")
    std = np.load(run_dir / "norm_std.npy")
    model = U.MODEL_REGISTRY[family](cfg)
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    Xn = ((X - mean) / std).astype(np.float32)
    return torch.sigmoid(model(torch.from_numpy(Xn)).squeeze(-1)).numpy()


def endpoint(name, yte, ea, eb, seeds_a, seeds_b):
    yb = yte.astype(bool)
    pa = ea["probs"] >= ea["tau"]
    pb = eb["probs"] >= eb["tau"]
    d = C.paired_bootstrap(yb, pa, pb, C.f1_from_preds, C.f1_from_preds, B, RNG_SEED)
    delta = C.f1_from_preds(yb, pa) - C.f1_from_preds(yb, pb)
    ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    t = stats.ttest_rel(seeds_a, seeds_b)
    excl = not (ci[0] <= 0 <= ci[1])
    if name == "iii":
        v = ("WIN" if delta > 0 else "LOSS") if (excl and t.pvalue < 0.05) else "TIE"
    else:
        v = "IMPROVED" if (excl and delta > 0) else \
            ("DEGRADED" if (excl and delta < 0) else "NO SIGNIFICANT CHANGE")
    return dict(name=name, delta_f1=float(delta), f1_ci=ci, t_p=float(t.pvalue), verdict=v)


def main():
    device = torch.device("cpu")
    data = U.load_splits()
    Xva, yva, Xte, yte = data[2], data[3], data[4], data[5]

    # ---- train all cells (cached) ----
    for cell, (family, cfg, select, pw, _) in CELLS.items():
        for seed in C.SEEDS:
            d = RUNS / cell / f"seed{seed}"
            if (d / "final.json").exists():
                continue
            r = U.train_run(family, cfg, seed, device, data, pos_weight=pw,
                            select=select, out_dir=d)
            print(f"  {cell} seed{seed}: val F1 {r['val']['f1']:.4f} "
                  f"auc {r['val']['auc']:.4f} ({r['seconds']:.0f}s)", flush=True)

    # ---- single test pass ----
    arms = {}
    for cell, (family, cfg, select, pw, rule) in CELLS.items():
        per_seed_f1, taus = [], []
        pv_list, pt_list = [], []
        for seed in C.SEEDS:
            d = RUNS / cell / f"seed{seed}"
            pv = probs_from(d, family, cfg, Xva)
            pt = probs_from(d, family, cfg, Xte)
            pv_list.append(pv)
            pt_list.append(pt)
            tau = 0.5 if rule == "0.5" else C.best_threshold(yva, pv)
            taus.append(tau)
            per_seed_f1.append(C.metrics_at(yte, pt, tau)["f1"])
        pv_e, pt_e = np.mean(pv_list, axis=0), np.mean(pt_list, axis=0)
        tau_e = 0.5 if rule == "0.5" else C.best_threshold(yva, pv_e)
        me = C.metrics_at(yte, pt_e, tau_e)
        arms[cell] = dict(rule=rule, taus=taus, tau_ens=float(tau_e),
                          per_seed_f1=per_seed_f1,
                          f1_mean=float(np.mean(per_seed_f1)),
                          f1_std=float(np.std(per_seed_f1, ddof=1)),
                          ens=dict(probs=pt_e, tau=tau_e), ens_metrics=me)
        print(f"{cell}: test F1 {np.mean(per_seed_f1):.4f} ± "
              f"{np.std(per_seed_f1, ddof=1):.4f} (ens {me['f1']:.4f} @tau {tau_e:.3f})",
              flush=True)

    eps = [endpoint("i", yte, arms["A3c"]["ens"], arms["A0c"]["ens"],
                    arms["A3c"]["per_seed_f1"], arms["A0c"]["per_seed_f1"]),
           endpoint("ii", yte, arms["B3c"]["ens"], arms["B0c"]["ens"],
                    arms["B3c"]["per_seed_f1"], arms["B0c"]["per_seed_f1"]),
           endpoint("iii", yte, arms["B3c"]["ens"], arms["A3c"]["ens"],
                    arms["B3c"]["per_seed_f1"], arms["A3c"]["per_seed_f1"])]

    out = dict(arms={k: {kk: vv for kk, vv in v.items() if kk != "ens"}
                     for k, v in arms.items()},
               endpoints=eps, original=ORIGINAL, B=B, rng_seed=RNG_SEED, device="cpu",
               engine="12_unified_engine.py")
    (HERE / "12_replication_results.json").write_text(json.dumps(out, indent=2))

    L = ["# 12 — Single-engine, single-device replication of the F1-first endpoints", "",
         "All six arms retrained under `12_unified_engine.py` on CPU (context-free, "
         "bit-reproducible — see 12_equivalence_report.md); selections frozen from the "
         "original program; test touched once per new arm, in this script only.", "",
         "| cell | test F1 (5-seed) | ens F1 @tau* | tau*_ens |", "|---|---|---|---|"]
    for cell in CELLS:
        a = arms[cell]
        L.append(f"| {cell} | {a['f1_mean']:.4f} ± {a['f1_std']:.4f} | "
                 f"{a['ens_metrics']['f1']:.4f} | {a['tau_ens']:.3f} |")
    L += ["", "## Endpoints — replication vs original", "",
          "| endpoint | original | replication (CPU, unified engine) | agrees? |",
          "|---|---|---|---|"]
    agree_all = True
    for e in eps:
        o = ORIGINAL[e["name"]]
        agree = (e["verdict"] == o["verdict"]) or \
                (e["name"] == "iii" and e["verdict"] == "TIE" == o["verdict"])
        agree_all &= agree
        L.append(f"| ({e['name']}) {o['pair']} | {o['verdict']} (dF1 {o['delta']:+.4f} "
                 f"CI [{o['ci'][0]:+.4f},{o['ci'][1]:+.4f}]) | **{e['verdict']}** "
                 f"(dF1 {e['delta_f1']:+.4f} CI [{e['f1_ci'][0]:+.4f},{e['f1_ci'][1]:+.4f}], "
                 f"p={e['t_p']:.3f}) | {'YES' if agree else '**NO**'} |")
    # ---- G1 counterfactual: F1-checkpoint vs AUC-checkpoint on TEST ----
    L += ["", "## G1 counterfactual — what the F1-checkpoint rule buys on test", "",
          "Same config, pos_weight, engine, device; only the checkpoint rule differs.",
          "", "| pair | F1-ckpt test F1 (5-seed) | AUC-ckpt test F1 (5-seed) | "
          "ens dF1 (F1-ckpt − AUC-ckpt) |", "|---|---|---|---|"]
    g1 = {}
    for main_c, cf_c in (("A3c", "A3f"), ("B3c", "B3f")):
        a, f = arms[main_c], arms[cf_c]
        yb = yte.astype(bool)
        d_ens = C.f1_from_preds(yb, a["ens"]["probs"] >= a["ens"]["tau"]) - \
                C.f1_from_preds(yb, f["ens"]["probs"] >= f["ens"]["tau"])
        g1[main_c] = float(d_ens)
        L.append(f"| {main_c} vs {cf_c} | {a['f1_mean']:.4f} ± {a['f1_std']:.4f} | "
                 f"{f['f1_mean']:.4f} ± {f['f1_std']:.4f} | {d_ens:+.4f} |")
    out["g1_counterfactual_ens_df1"] = g1
    (HERE / "12_replication_results.json").write_text(json.dumps(out, indent=2))

    key_agree = (eps[0]["verdict"] == ORIGINAL["i"]["verdict"]
                 and eps[2]["verdict"] == ORIGINAL["iii"]["verdict"])
    ii_stronger = (eps[1]["verdict"] == "IMPROVED" and eps[1]["delta_f1"] > 0
                   and ORIGINAL["ii"]["verdict"] == "NO SIGNIFICANT CHANGE")
    if agree_all:
        conc = ("**All three verdicts replicate under a single engine on a single "
                "context-free device — the engine/device confound does not drive the "
                "F1-first conclusions.**")
    elif key_agree and ii_stronger:
        conc = ("**The two headline verdicts replicate exactly — (i) the LSTM's F1 "
                "improvement is significant, (iii) the families TIE on F1 — and "
                "endpoint (ii) becomes STRONGER, not weaker: with the transformer's "
                "reference arm also trained by the unified engine on CPU, its "
                "F1-first improvement is significant here (it was a non-significant "
                "positive under the original mixed regime, where the reference was "
                "the Kaggle-trained frozen model). No published conclusion is "
                "weakened by this replication; the original, more conservative "
                "verdict for (ii) remains the one to cite, with this replication "
                "reported as the single-engine sensitivity analysis.**")
    else:
        conc = ("**At least one verdict did NOT replicate in a direction that "
                "weakens a published claim — the original conclusion is "
                "engine/device-sensitive and must be revised before submission.**")
    L += ["", "## Conclusion", "", conc, ""]
    (HERE / "12_replication_report.md").write_text("\n".join(L))
    print("\n" + " | ".join(f"({e['name']}) {e['verdict']}" for e in eps))
    print("wrote 12_replication_results.json, 12_replication_report.md")


if __name__ == "__main__":
    main()

"""08_cluster_bootstrap.py — RNN study Phase R5: pedestrian-cluster bootstrap (the honest CI).

The window-level bootstrap in 07 resamples the 2094 test windows as i.i.d., but they belong to
~541 pedestrians with 50%-overlap windows, so window CIs understate uncertainty. This recomputes
every endpoint's primary-metric delta by resampling PEDESTRIANS (each drawn ped contributes all
its windows), 10k resamples, paired clusters across both models — reusing
f1_optimization/07_cluster_bootstrap.py's machinery verbatim (load_ped_clusters,
cluster_paired_delta, cluster_ci) and 07_compare.py's already-built source registry.

Reports, for every endpoint, the window CI (07) next to the cluster CI and whether the verdict
survives clustering. Cluster intervals are the ones to quote in the manuscript.

Outputs: 08_cluster_bootstrap.json, 08_cluster_bootstrap.md
Run from the repo root:  python rnn/phase5_analysis/08_cluster_bootstrap.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("rnn_common", ROOT / "f1_optimization" / "00_common.py")
cb = _load("rnn_cluster", ROOT / "f1_optimization" / "07_cluster_bootstrap.py")
cmp7 = _load("rnn_cmp", ROOT / "rnn" / "phase5_analysis" / "07_compare.py")

yte, YB, SRC, ENDPOINTS = cmp7.yte, cmp7.YB, cmp7.SRC, cmp7.ENDPOINTS
RNN_ARMS = ("rnn_f1_winner", "rnn_winner_auc", "rnn_default_f1", "rnn_default_auc")


def main():
    uniq, groups = cb.load_ped_clusters()
    print(f"test: {len(yte)} windows in {len(uniq)} pedestrian clusters "
          f"(median {int(np.median([len(g) for g in groups]))} windows/ped)\n")

    window = {e["name"]: e for e in
              json.loads((HERE / "07_comparison_results.json").read_text())["endpoints"]}

    rows = []
    for name, left_key, right_key, metric in ENDPOINTS:
        L, R = SRC[left_key], SRC[right_key]
        pg, pt, tg, tt = L["probs"], R["probs"], L["tau"], R["tau"]
        if metric == "f1":
            predg, predt = pg >= tg, pt >= tt
            d = C.f1_from_preds(YB, predg) - C.f1_from_preds(YB, predt)
            ci = cb.cluster_paired_delta(YB, predg, predt, C.f1_from_preds, groups)
        else:
            d = C.auc_fast(yte, pg) - C.auc_fast(yte, pt)
            ci = cb.cluster_paired_delta(yte, pg, pt, C.auc_fast, groups)
        excl = not (ci[0] <= 0 <= ci[1])
        wci = window[name]["ci"]
        wverdict = window[name]["verdict"]
        cverdict = "WIN" if (excl and d > 0) else ("LOSS" if (excl and d < 0) else "TIE")
        survives = (cverdict == wverdict)
        rows.append(dict(name=name, rnn=left_key, target=right_key, metric=metric, delta=float(d),
                         window_ci=wci, cluster_ci=ci, window_verdict=wverdict,
                         cluster_verdict=cverdict, survives=survives, label=R["label"]))
        pm = "ΔF1" if metric == "f1" else "ΔAUC"
        print(f"({name}) {left_key} vs {right_key}: {pm}={d:+.4f}  "
              f"window [{wci[0]:+.4f},{wci[1]:+.4f}]={wverdict}  "
              f"CLUSTER [{ci[0]:+.4f},{ci[1]:+.4f}]={cverdict}  "
              f"{'✓ survives' if survives else '⚠ CHANGES'}")

    # absolute cluster CIs for the RNN arms
    print()
    absolute = {}
    for arm in RNN_ARMS:
        p, tau = SRC[arm]["probs"], SRC[arm]["tau"]
        f1ci = cb.cluster_ci(YB, p >= tau, C.f1_from_preds, groups)
        aucci = cb.cluster_ci(yte, p, C.auc_fast, groups)
        absolute[arm] = dict(tau=tau, f1_cluster_ci=f1ci, auc_cluster_ci=aucci)
        print(f"{arm}: ens F1 cluster CI [{f1ci[0]:.4f},{f1ci[1]:.4f}]  "
              f"AUC cluster CI [{aucci[0]:.4f},{aucci[1]:.4f}]")

    out = dict(n_clusters=int(len(uniq)), B=cb.B, endpoints=rows, absolute=absolute)
    (HERE / "08_cluster_bootstrap.json").write_text(json.dumps(out, indent=2))
    write_md(rows, absolute, len(uniq))
    print("\nwrote 08_cluster_bootstrap.json, 08_cluster_bootstrap.md")


def write_md(rows, absolute, n_clusters):
    L = ["# RNN study — Phase R5 pedestrian-cluster bootstrap (the honest CI)", "",
         f"Windows are pedestrian-correlated ({n_clusters} clusters for 2094 windows, 50% "
         "overlap), so i.i.d.-window CIs understate uncertainty. Every endpoint's primary-metric "
         "delta is recomputed by resampling PEDESTRIANS (all-windows-per-drawn-ped, 10k "
         "resamples, paired clusters across models) — machinery reused verbatim from "
         "`f1_optimization/07_cluster_bootstrap.py`. **Cluster intervals are the ones to quote.**",
         "",
         "| endpoint | metric | Δ | window CI (07) → verdict | cluster CI → verdict | survives? |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        pm = "F1" if r["metric"] == "f1" else "AUC"
        L.append(f"| ({r['name']}) {r['rnn'].replace('rnn_','')} vs {r['target']} | {pm} | "
                 f"{r['delta']:+.4f} | [{r['window_ci'][0]:+.4f}, {r['window_ci'][1]:+.4f}] "
                 f"{r['window_verdict']} | [{r['cluster_ci'][0]:+.4f}, {r['cluster_ci'][1]:+.4f}] "
                 f"{r['cluster_verdict']} | {'✓' if r['survives'] else '⚠ changes'} |")
    L += ["", "| RNN arm | ens F1 95% cluster CI | ens AUC 95% cluster CI |", "|---|---|---|"]
    for arm, r in absolute.items():
        L.append(f"| `{arm}` | [{r['f1_cluster_ci'][0]:.4f}, {r['f1_cluster_ci'][1]:.4f}] | "
                 f"[{r['auc_cluster_ci'][0]:.4f}, {r['auc_cluster_ci'][1]:.4f}] |")
    all_survive = all(r["survives"] for r in rows)
    # the scientific endpoints (gating/cell isolation): 2 (vs LSTM-F1), 4 (vs GRU), 6 (matched AUC)
    key = {r["name"]: r for r in rows}
    sci_survive = all(key[n]["survives"] for n in ("2", "4", "6"))
    L += ["", f"**All endpoint verdicts {'survive' if all_survive else 'do NOT all survive'} "
          "the cluster bootstrap** (wider, dependence-honest intervals). "
          + ("The gating/cell-isolation verdicts (RNN-F1 vs BiLSTM-F1; RNN-F1 vs GRU-F1; "
             "RNN-AUC vs frozen BiLSTM) are unchanged under clustering — the central finding is "
             "robust to the pedestrian correlation structure."
             if sci_survive else
             "Note: at least one gating/cell-isolation verdict changes under clustering — see "
             "the table; report both intervals."), ""]
    (HERE / "08_cluster_bootstrap.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()

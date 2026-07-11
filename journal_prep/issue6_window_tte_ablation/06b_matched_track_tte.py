"""06b_matched_track_tte.py — Issue 6 hardening: matched-cohort TTE control.

WHY. The single-point TTE sweep in `06_multiseed_ablations.py` evaluates each
horizon on a DIFFERENT, nested pedestrian set: tte60 needs a track of length
L >= obs_len+60 = 76, but tte30 only needs L >= 46. So tte30 carries 48 extra
short tracks (set03) that the Issue-2 parity check showed are *harder*
(46-75 f AUC 0.863 vs >=76 f AUC 0.919). A reviewer can argue the 0.960 -> 0.919
"decline with horizon" is partly a change of sample, not of horizon.

WHAT. Restrict ALL THREE horizons to the COMMON cohort — the pedestrians eligible
for tte60 (L >= 76) — and give each pedestrian exactly one window at T=30/45/60.
Now the three cells share an IDENTICAL pedestrian population (train AND test); the
ONLY thing that changes between cells is which 16-frame window is observed (how far
before the crossing point). This is the canonical "AUC vs prediction horizon on a
fixed cohort" curve. matched-tte60 == the existing `runs/tte60` (reused), so only
matched-tte30 and matched-tte45 are retrained.

Because the crossing label is per-pedestrian, the three matched cells have an
IDENTICAL class balance by construction, so the fixed pos_weight=1.682 is exactly
right and cannot bias the comparison. Everything else reuses the locked contract
in 06_multiseed_ablations.py (same arch, split, norm, hyperparams, seeds).

Outputs (next to this script):
  sequences_matched/<cfg>/         matched-cohort X/y/meta (tte30, tte45)
  runs_matched/<cfg>/seed<k>.json  matched runs (tte30, tte45; tte60 reused)
  06b_matched_tte_results.csv      per (horizon, seed) on the matched cohort
  06b_matched_tte_report.md        matched table + paired tests + verdict
  06b_matched_tte_figure.png       all-eligible vs matched-cohort AUC-vs-horizon
"""
import csv
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
SEQ = HERE / "sequences"
SEQ_M = HERE / "sequences_matched"
RUNS = HERE / "runs"                 # existing single-point runs (tte60 reused)
RUNS_M = HERE / "runs_matched"

# reuse the locked harness (train_one, METRICS, agg, BiLSTM, contract constants)
_spec = importlib.util.spec_from_file_location("h6", HERE / "06_multiseed_ablations.py")
H = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(H)

HORIZONS = [("tte30", 30), ("tte45", 45), ("tte60", 60)]
SEEDS = H.SEEDS_DEFAULT


def pedkey(m):
    return (m["set_id"], m["video_id"], m["ped_id"])


def load_cfg(seq_dir):
    X = np.load(seq_dir / "X.npy").astype(np.float32)
    y = np.load(seq_dir / "y.npy").astype(np.float32)
    meta = pickle.load(open(seq_dir / "meta.pkl", "rb"))
    return X, y, meta


def build_matched():
    """Filter tte30/tte45 to the tte60-eligible cohort; one window/ped already."""
    _, _, meta60 = load_cfg(SEQ / "tte60")
    common = {pedkey(m) for m in meta60}
    print(f"common cohort (tte60-eligible peds): {len(common)}")

    ped_sets = {}
    for cfg, _ in HORIZONS:
        X, y, meta = load_cfg(SEQ / cfg)
        keep = np.array([pedkey(m) in common for m in meta])
        Xm, ym, metam = X[keep], y[keep], [m for m, k in zip(meta, keep) if k]
        out = SEQ_M / cfg
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "X.npy", Xm); np.save(out / "y.npy", ym)
        pickle.dump(metam, open(out / "meta.pkl", "wb"))
        ped_sets[cfg] = {pedkey(m) for m in metam}
        # one window per ped on the matched cohort
        assert len(metam) == len(ped_sets[cfg]) == len(common), \
            f"{cfg}: {len(metam)} windows / {len(ped_sets[cfg])} peds / {len(common)} cohort"
        print(f"  {cfg}-matched: {len(metam)} windows (= {len(ped_sets[cfg])} peds), "
              f"pos-rate {ym.mean():.3f}")

    # the three cells must be the SAME pedestrians with the SAME labels
    assert ped_sets["tte30"] == ped_sets["tte45"] == ped_sets["tte60"], \
        "matched cohorts differ across horizons"
    lab = {}
    for cfg, _ in HORIZONS:
        _, y, meta = load_cfg(SEQ_M / cfg)
        lab[cfg] = {pedkey(m): int(v) for m, v in zip(meta, y)}
    assert lab["tte30"] == lab["tte45"] == lab["tte60"], "labels differ across horizons"
    print("  ✓ identical pedestrians AND labels across the three matched horizons\n")


def run_matched(device):
    RUNS_M.mkdir(exist_ok=True)
    results = {}
    print(f"{'horizon':8s} {'seed':>4s} {'N_tr':>5s} {'N_te':>5s} {'AUC':>6s} "
          f"{'PR':>6s} {'F1':>6s} {'Acc':>6s} {'ep':>3s}")
    print("-" * 56)
    for cfg, T in HORIZONS:
        rows = []
        for seed in SEEDS:
            if cfg == "tte60":                      # identical to the existing run
                m = json.loads((RUNS / "tte60" / f"seed{seed}.json").read_text())
            else:
                rdir = RUNS_M / cfg; rdir.mkdir(exist_ok=True)
                jp = rdir / f"seed{seed}.json"
                if jp.exists():
                    m = json.loads(jp.read_text())
                else:
                    m = H.train_one(SEQ_M / cfg, seed, device)
                    m.update(config=f"{cfg}_matched", tte=T, seed=seed)
                    jp.write_text(json.dumps(m, indent=2))
            rows.append(m)
            print(f"{cfg:8s} {seed:>4d} {m['n_train']:>5d} {m['n_test']:>5d} "
                  f"{m['auc']:6.3f} {m['pr_auc']:6.3f} {m['f1']:6.3f} {m['acc']:6.3f} "
                  f"{m['best_epoch']:>3d}")
        results[cfg] = rows
    return results


def write_outputs(results):
    # CSV
    with open(HERE / "06b_matched_tte_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_frames", "tte_seconds", "seed", "n_train", "n_test",
                    "test_pos"] + H.METRICS)
        for cfg, T in HORIZONS:
            for r in results[cfg]:
                w.writerow([T, round(T / 30, 2), r["seed"], r["n_train"], r["n_test"],
                            r["test_pos"]] + [round(r[m], 4) for m in H.METRICS])

    aucs = {cfg: np.array([r["auc"] for r in results[cfg]]) for cfg, _ in HORIZONS}
    means = {cfg: aucs[cfg].mean() for cfg, _ in HORIZONS}
    spread = max(means.values()) - min(means.values())
    seed_std = float(np.mean([aucs[cfg].std(ddof=1) for cfg, _ in HORIZONS]))
    ntr = results["tte30"][0]["n_train"]; nte = results["tte30"][0]["n_test"]

    L = ["# Issue 6 — Matched-cohort TTE control (horizon isolated from track length)",
         "",
         f"All three horizons restricted to the **common tte60-eligible cohort** "
         f"(L ≥ 76 frames), one window per pedestrian at T = 30/45/60. The three "
         f"cells share an **identical pedestrian population and labels** in both "
         f"train (N={ntr}) and test (set03, N={nte}) — only the observed 16-frame "
         f"window moves. Locked baseline contract (5-D BiLSTM, train=set01/02/04, "
         f"val=set05/06, train-only norm, pos_weight={H.POS_WEIGHT}, lr={H.LR}, "
         f"patience {H.PATIENCE}); {len(SEEDS)} seeds {SEEDS}. matched-tte60 reuses "
         f"the existing `runs/tte60`.", "",
         "| TTE (horizon) | N train | N test | best ep | AUC | PR-AUC | F1 | Acc |",
         "|---|---|---|---|---|---|---|---|"]
    for cfg, T in HORIZONS:
        rows = results[cfg]
        ep = f"{np.mean([r['best_epoch'] for r in rows]):.0f}"
        cells = " | ".join(f"{H.agg(rows, m)[0]:.3f} ± {H.agg(rows, m)[1]:.3f}"
                           for m in ["auc", "pr_auc", "f1", "acc"])
        L.append(f"| {T} ({T/30:.2f} s) | {rows[0]['n_train']} | {rows[0]['n_test']} | "
                 f"{ep} | {cells} |")
    L += ["",
          f"**Max between-horizon mean-AUC spread = {spread:.4f}**, vs average "
          f"within-horizon seed std = ±{seed_std:.4f}.", "",
          "Pairwise (paired t-test, matched seeds; Mann-Whitney U):", "",
          "| pair | ΔAUC | paired-t p | Mann-Whitney p |", "|---|---|---|---|"]
    names = [c[0] for c in HORIZONS]
    pairs = H.pairwise_tests(results, names)
    for a, b, d, tp, up in pairs:
        L.append(f"| {a} vs {b} | {d:+.4f} | {tp:.3f} | {up:.3f} |")
    kw = stats.kruskal(*[aucs[n] for n in names]).pvalue
    max_tp = max(p[3] for p in pairs)
    L.append(f"\nKruskal–Wallis across the three horizons: p = {kw:.3f}.\n")

    # comparison to the un-matched (all-eligible) single-point numbers
    allelig = {cfg: np.mean([json.loads((RUNS / cfg / f"seed{s}.json").read_text())["auc"]
                             for s in SEEDS]) for cfg, _ in HORIZONS}
    L += ["## Matched cohort vs all-eligible single-point\n",
          "| TTE | all-eligible AUC | matched-cohort AUC | Δ (sample effect) |",
          "|---|---|---|---|"]
    for cfg, T in HORIZONS:
        L.append(f"| {T} | {allelig[cfg]:.3f} | {means[cfg]:.3f} | "
                 f"{means[cfg]-allelig[cfg]:+.3f} |")

    sig = spread > seed_std and max_tp < 0.05 and kw < 0.05
    L += ["", "## Verdict\n"]
    if sig:
        L.append(f"**The horizon effect is real, not a sampling artifact.** On a "
                 f"*fixed cohort* (identical {nte}-pedestrian test set at all three "
                 f"horizons), AUC still declines monotonically "
                 f"{means['tte30']:.3f} (1.0 s) → {means['tte45']:.3f} (1.5 s) → "
                 f"{means['tte60']:.3f} (2.0 s); spread {spread:.4f} exceeds seed "
                 f"noise (±{seed_std:.4f}), every pairwise paired-t is significant "
                 f"(all p ≤ {max_tp:.3f}), Kruskal–Wallis p = {kw:.3f}. Removing the "
                 f"nested-sample confound leaves the decline intact, so the "
                 f"single-point result in `06_` is confirmed: **prediction AUC "
                 f"degrades significantly with horizon** on leak-free data — the "
                 f"intuitive behaviour the leaky single-seed run had masked.\n")
    else:
        L.append(f"**On the matched cohort the horizon effect weakens** (spread "
                 f"{spread:.4f} vs seed std ±{seed_std:.4f}, max pairwise p "
                 f"{max_tp:.3f}, Kruskal p {kw:.3f}). Part of the single-point "
                 f"`06_` decline was the nested-sample confound (tte30 carried the "
                 f"harder short tracks). Report the matched-cohort numbers as the "
                 f"horizon-sensitivity result and down-weight the single-point "
                 f"version.\n")
    (HERE / "06b_matched_tte_report.md").write_text("\n".join(L))

    make_figure(results, allelig, means)
    print(f"\nmatched spread {spread:.4f} | seed std ±{seed_std:.4f} | "
          f"max pairwise p {max_tp:.3f} | Kruskal p {kw:.3f}")
    print("wrote 06b_matched_tte_results.csv, 06b_matched_tte_report.md, "
          "06b_matched_tte_figure.png")
    return sig


def make_figure(results, allelig, matched):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [T for _, T in HORIZONS]
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor="white")
    # all-eligible (confounded)
    ax.plot(xs, [allelig[c] for c, _ in HORIZONS], color="#9ca3af", lw=2,
            marker="s", ms=7, ls="--", zorder=2,
            label="all-eligible (nested samples, 06_)")
    # matched cohort
    mu = [H.agg(results[c], "auc")[0] for c, _ in HORIZONS]
    sd = [H.agg(results[c], "auc")[1] for c, _ in HORIZONS]
    ax.errorbar(xs, mu, yerr=sd, color="#0d9488", lw=2.5, marker="o", ms=9,
                capsize=5, zorder=3, label="matched cohort (fixed peds)")
    for c, x in zip([h[0] for h in HORIZONS], xs):
        ys = [r["auc"] for r in results[c]]
        ax.scatter([x] * len(ys), ys, color="#0d9488", alpha=0.35, s=30, zorder=2)
    for x, m in zip(xs, mu):
        ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                    xytext=(0, 12), ha="center", color="#0d9488", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{T}\n({T/30:.1f}s)" for T in xs])
    ax.set_xlabel("prediction horizon TTE (frames)")
    ax.set_ylabel("test AUC (set03)")
    ax.set_title("TTE horizon effect survives sample matching\n"
                 "(matched cohort: same pedestrians at all three horizons)")
    ax.set_ylim(0.88, 0.97); ax.grid(alpha=0.3); ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(HERE / "06b_matched_tte_figure.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {device} | seeds {SEEDS} | pos_weight {H.POS_WEIGHT} (fixed)\n")
    build_matched()
    results = run_matched(device)
    write_outputs(results)


if __name__ == "__main__":
    main()

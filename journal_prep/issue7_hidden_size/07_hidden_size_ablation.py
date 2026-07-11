"""07_hidden_size_ablation.py — Issue 7: hidden-size ablation {64,128,256}, multi-seed.

WHY. THESIS_PLAN.md Day 11 promised a hidden-size sweep; it was never run, so the
central capacity choice (hidden=128) is *asserted*, not justified. A reviewer asks:
"did you pick 128 because it was best, or because you decided it first?" This sweep
trains hidden_dim ∈ {64,128,256} on the clean baseline data, everything else locked,
and reports mean ± std + parameter count + significance so the choice is defended.

MULTI-SEED (deviation from the plan, on purpose). The plan said seed 42 only, but
Issue 6 showed single-seed ablation spreads sit *below* the seed-to-seed std and are
undefendable as conclusions. So every config is run across 5 seeds and the verdict
is argued from effect size / equivalence, not one lucky seed.

LOCKED to the baseline (identical to 04_train_bilstm.py / 06_), hidden_dim the only
variable: 5-D BiLSTMIntentPredictor on issue2's `sequences_clean/` (obs16, TTE band
[30,60] — the data behind the 0.932 headline), train=set01/02/04 val=set05/06
test=set03, train-only norm, pos_weight=1.682, lr=1e-3, wd=1e-5, batch=32,
patience=15, dropout 0.3, 2 layers, proj_dim 64, threshold 0.5, best-on-val-AUC
checkpoint, test touched once per (config, seed). hidden=128 reproduces the existing
baseline (0.932 ± 0.011). Local MPS, ~13 s/training, 15 trainings.

Outputs (next to this script):
  runs/h<H>/seed<k>.json          per-run metrics
  07_hidden_size_results.csv      per (hidden, seed)
  07_hidden_size_results.md       mean±std + params + significance + verdict
  07_hidden_size_figure.png       AUC vs hidden size (params annotated), seed scatter
"""
import csv
import importlib.util
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SEQ_DIR = HERE.parent / "issue2_clean_protocol" / "sequences_clean"

# locked baseline architecture (root 03_bilstm_model.py; filename starts with a digit)
_spec = importlib.util.spec_from_file_location("m03", ROOT / "03_bilstm_model.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
BiLSTM = _m.BiLSTMIntentPredictor

TRAIN_SETS = {"set01", "set02", "set04"}
VAL_SETS   = {"set05", "set06"}
TEST_SETS  = {"set03"}
POS_WEIGHT = 1.682
EPOCHS, BATCH, LR, WD, PATIENCE, THR = 100, 32, 1e-3, 1e-5, 15, 0.5
HIDDEN_SIZES = [64, 128, 256]
SEEDS = [42, 0, 1, 2, 3]
METRICS = ["auc", "pr_auc", "f1", "acc", "prec", "rec"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_split():
    X = np.load(SEQ_DIR / "X.npy").astype(np.float32)
    y = np.load(SEQ_DIR / "y.npy").astype(np.float32)
    meta = pickle.load(open(SEQ_DIR / "meta.pkl", "rb"))
    sid = np.array([m["set_id"] for m in meta])
    tr, va, te = (np.isin(sid, list(s)) for s in (TRAIN_SETS, VAL_SETS, TEST_SETS))
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    p = torch.sigmoid(model(torch.from_numpy(X).to(device)).squeeze(-1)).cpu().numpy()
    pred = (p >= THR).astype(int)
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                f1=f1_score(y, pred, zero_division=0), acc=accuracy_score(y, pred),
                prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0))


def train_one(hidden, seed, device, data):
    set_seed(seed)
    Xtr, ytr, Xva, yva, Xte, yte = data
    mean = Xtr.reshape(-1, 5).mean(0); std = Xtr.reshape(-1, 5).std(0) + 1e-6
    Xtr, Xva, Xte = (Xtr - mean) / std, (Xva - mean) / std, (Xte - mean) / std

    model = BiLSTM(input_dim=5, hidden_dim=hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True)

    best_auc, best_state, best_ep, noimp = -1.0, None, 0, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb).squeeze(-1), yb).backward(); opt.step()
        vauc = evaluate(model, Xva, yva, device)["auc"]
        sched.step(vauc)
        if vauc > best_auc:
            best_auc, best_ep, noimp = vauc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            noimp += 1
            if noimp >= PATIENCE:
                break
    model.load_state_dict(best_state)
    m = evaluate(model, Xte, yte, device)
    m.update(hidden=hidden, seed=seed, n_params=int(n_params), best_epoch=best_ep,
             val_auc=round(best_auc, 4), n_train=int(len(ytr)), n_test=int(len(yte)))
    return m


def agg(rows, metric):
    v = np.array([r[metric] for r in rows]); return v.mean(), v.std(ddof=1)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {device} | hidden {HIDDEN_SIZES} | seeds {SEEDS} | "
          f"pos_weight {POS_WEIGHT} (fixed)\n")
    data = load_split()
    runs_root = HERE / "runs"; runs_root.mkdir(exist_ok=True)

    results = {h: [] for h in HIDDEN_SIZES}
    print(f"{'hidden':>6s} {'params':>9s} {'seed':>4s} {'AUC':>6s} {'PR':>6s} "
          f"{'F1':>6s} {'Acc':>6s} {'ep':>3s} {'s':>4s}")
    print("-" * 56)
    for h in HIDDEN_SIZES:
        rdir = runs_root / f"h{h}"; rdir.mkdir(exist_ok=True)
        for seed in SEEDS:
            jp = rdir / f"seed{seed}.json"
            if jp.exists():
                m = json.loads(jp.read_text())
            else:
                t0 = time.time()
                m = train_one(h, seed, device, data)
                m["seconds"] = round(time.time() - t0, 1)
                jp.write_text(json.dumps(m, indent=2))
            results[h].append(m)
            print(f"{h:>6d} {m['n_params']:>9,d} {m['seed']:>4d} {m['auc']:6.3f} "
                  f"{m['pr_auc']:6.3f} {m['f1']:6.3f} {m['acc']:6.3f} "
                  f"{m['best_epoch']:>3d} {m.get('seconds',0):>4.0f}")

    write_outputs(results)


def write_outputs(results):
    # CSV
    with open(HERE / "07_hidden_size_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hidden", "n_params", "seed", "best_epoch"] + METRICS)
        for h in HIDDEN_SIZES:
            for r in results[h]:
                w.writerow([h, r["n_params"], r["seed"], r["best_epoch"]]
                           + [round(r[m], 4) for m in METRICS])

    aucs = {h: np.array([r["auc"] for r in results[h]]) for h in HIDDEN_SIZES}
    means = {h: aucs[h].mean() for h in HIDDEN_SIZES}
    seed_std = float(np.mean([aucs[h].std(ddof=1) for h in HIDDEN_SIZES]))
    spread = max(means.values()) - min(means.values())
    params = {h: results[h][0]["n_params"] for h in HIDDEN_SIZES}
    best_h = max(means, key=means.get)

    L = ["# Issue 7 — Hidden-size ablation (multi-seed)", "",
         f"5-D baseline BiLSTM on the clean baseline data (`issue2_clean_protocol/"
         f"sequences_clean/`, obs16 / TTE band [30,60] — the 0.932 headline data), "
         f"hidden_dim ∈ {HIDDEN_SIZES}, {len(SEEDS)} seeds {SEEDS}, MPS. Everything "
         f"else locked (train=set01/02/04, val=set05/06, test=set03; train-only norm; "
         f"pos_weight={POS_WEIGHT}; lr={LR}; dropout 0.3; 2 layers; proj 64; patience "
         f"{PATIENCE}). Test (set03) touched once per (config, seed); hidden_dim is "
         f"the only variable.", "",
         "| hidden | params | best ep | AUC | PR-AUC | F1 | Acc |",
         "|---|---|---|---|---|---|---|"]
    for h in HIDDEN_SIZES:
        rows = results[h]
        ep = f"{np.mean([r['best_epoch'] for r in rows]):.0f}"
        cells = " | ".join(f"{agg(rows, m)[0]:.3f} ± {agg(rows, m)[1]:.3f}"
                           for m in ["auc", "pr_auc", "f1", "acc"])
        L.append(f"| {h}{' (baseline)' if h==128 else ''} | {params[h]:,} | {ep} | {cells} |")
    L += ["",
          f"**Between-size mean-AUC spread = {spread:.4f}**, vs average within-size "
          f"seed std = ±{seed_std:.4f}.", "",
          "Pairwise vs hidden=128 (paired t-test, matched seeds; Mann-Whitney U):", "",
          "| pair | ΔAUC | paired-t p | Mann-Whitney p |", "|---|---|---|---|"]
    pvals = {}
    for h in [64, 256]:
        a, b = aucs[h], aucs[128]
        tp = stats.ttest_rel(a, b).pvalue
        up = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        pvals[h] = tp
        L.append(f"| h{h} vs h128 | {a.mean()-b.mean():+.4f} | {tp:.3f} | {up:.3f} |")
    kw = stats.kruskal(*[aucs[h] for h in HIDDEN_SIZES]).pvalue
    L.append(f"\nKruskal–Wallis across the three sizes: p = {kw:.3f}.\n")

    # data-driven verdict — anchored on SIGNIFICANCE (not the spread heuristic):
    # hidden=128 is defensible iff no other size significantly beats it.
    no_sig_diff = pvals[64] > 0.05 and pvals[256] > 0.05 and kw > 0.05
    mild_trend = spread > seed_std            # nominal spread exceeds seed noise, though n.s.
    L.append("## Verdict\n")
    if no_sig_diff:
        trend_note = (
            f" There is a *mild, non-significant* upward trend with capacity "
            f"({means[64]:.3f} → {means[128]:.3f} → {means[256]:.3f}): the spread "
            f"({spread:.4f}) slightly exceeds seed noise (±{seed_std:.4f}) but no "
            f"pairwise test is significant, so we do **not** claim capacity is fully "
            f"saturated — only that nothing beats 128 significantly."
            if mild_trend else
            f" All three sizes are statistically equivalent (spread {spread:.4f} "
            f"within seed noise ±{seed_std:.4f}).")
        L.append(f"**hidden=128 is justified.** No size differs significantly from it: "
                 f"hidden=256 is nominally {means[256]-means[128]:+.4f} AUC but **not "
                 f"significant** (paired-t p={pvals[256]:.3f}) at "
                 f"**{params[256]/params[128]:.1f}× the parameters** ({params[256]:,} vs "
                 f"{params[128]:,}), and hidden=64 is no better (p={pvals[64]:.3f}) at "
                 f"lower capacity; Kruskal–Wallis p={kw:.3f}.{trend_note} We keep "
                 f"**hidden=128 as the accuracy/cost compromise** — the smaller, faster "
                 f"model is not significantly beaten by the 3.8×-larger one, which is "
                 f"the standard justification for the chosen capacity. The hidden=128 "
                 f"cell reproduces the baseline (this run {means[128]:.3f} ± "
                 f"{aucs[128].std(ddof=1):.3f} vs the existing 0.932 ± 0.011).\n")
    else:
        sig = [h for h in (64, 256) if pvals[h] < 0.05]
        L.append(f"**hidden={best_h} significantly outperforms hidden=128** "
                 f"({means[best_h]:.3f} vs {means[128]:.3f}; significant sizes vs 128: "
                 f"{sig}; h64 p={pvals[64]:.3f}, h256 p={pvals[256]:.3f}, Kruskal {kw:.3f}). "
                 f"Report the best size and its accuracy/cost trade-off explicitly "
                 f"(params: 64={params[64]:,}, 128={params[128]:,}, 256={params[256]:,}); "
                 f"reconsider whether the paper should adopt hidden={best_h}.\n")
    (HERE / "07_hidden_size_results.md").write_text("\n".join(L))

    make_figure(results, means, params)
    print(f"\nspread {spread:.4f} | seed std ±{seed_std:.4f} | best hidden={best_h} "
          f"| h64 p={pvals[64]:.3f} | h256 p={pvals[256]:.3f} | Kruskal {kw:.3f}")
    print("wrote 07_hidden_size_results.csv, 07_hidden_size_results.md, "
          "07_hidden_size_figure.png")


def make_figure(results, means, params):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(len(HIDDEN_SIZES)))
    mu = [means[h] for h in HIDDEN_SIZES]
    sd = [agg(results[h], "auc")[1] for h in HIDDEN_SIZES]
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor="white")
    ax.errorbar(xs, mu, yerr=sd, color="#7c3aed", lw=2.5, marker="o", ms=9,
                capsize=5, zorder=3, label="mean ± std (5 seeds)")
    for h, x in zip(HIDDEN_SIZES, xs):
        ys = [r["auc"] for r in results[h]]
        ax.scatter([x] * len(ys), ys, color="#7c3aed", alpha=0.35, s=30, zorder=2)
    for x, h, m in zip(xs, HIDDEN_SIZES, mu):
        ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                    xytext=(0, 12), ha="center", color="#7c3aed", fontweight="bold")
        ax.annotate(f"{params[h]/1000:.0f}k params", (x, m), textcoords="offset points",
                    xytext=(0, -20), ha="center", color="#6b7280", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"hidden={h}{' (baseline)' if h==128 else ''}" for h in HIDDEN_SIZES])
    ax.set_ylabel("test AUC (set03)")
    ax.set_title("Hidden-size ablation — capacity vs accuracy\n"
                 "(clean baseline data, 5 seeds)")
    ax.set_ylim(0.90, 0.96); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(HERE / "07_hidden_size_figure.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

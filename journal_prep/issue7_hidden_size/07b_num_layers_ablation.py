"""07b_num_layers_ablation.py — depth ablation: num_layers ∈ {1, 2, 3}, multi-seed.

Companion to the hidden-size (width) ablation: this varies network DEPTH while
holding everything else at the baseline (lr=1e-3, dropout=0.3, hidden=128). The grid
search (Issue 8) searched num_layers ∈ {1,2} but never 3 — this adds the 3-layer
point and reports a clean, multi-seed 1-vs-2-vs-3 comparison.

Caveat (intrinsic to the architecture): dropout in the locked model is *inter-layer*
LSTM dropout, so it is inert at num_layers=1 (a 1-layer LSTM has nothing between
layers to drop). So num_layers=1 runs with no dropout; 2 and 3 use dropout=0.3. This
is a property of the model, not a bug — flagged in the report.

Locked to the baseline otherwise: 5-D BiLSTMIntentPredictor on `sequences_clean/`,
train=set01/02/04 val=set05/06 test=set03, train-only norm, pos_weight=1.682, lr=1e-3,
wd=1e-5, batch=32, patience=15, threshold 0.5, best-on-val-AUC, 100 max epochs. 5
seeds [42,0,1,2,3]. num_layers=2 reproduces the baseline. Local MPS.

Outputs: runs_layers/nl<L>/seed<k>.json, 07b_num_layers_results.csv,
07b_num_layers_results.md, 07b_num_layers_figure.png
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

_spec = importlib.util.spec_from_file_location("m03", ROOT / "03_bilstm_model.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
BiLSTM = _m.BiLSTMIntentPredictor

TRAIN_SETS = {"set01", "set02", "set04"}
VAL_SETS   = {"set05", "set06"}
TEST_SETS  = {"set03"}
POS_WEIGHT = 1.682
HIDDEN, LR = 128, 1e-3
EPOCHS, BATCH, WD, PATIENCE, THR = 100, 32, 1e-5, 15, 0.5
LAYERS = [1, 2, 3]
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
def metrics(model, X, y, device):
    model.eval()
    p = torch.sigmoid(model(torch.from_numpy(X).to(device)).squeeze(-1)).cpu().numpy()
    pred = (p >= THR).astype(int)
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                f1=f1_score(y, pred, zero_division=0), acc=accuracy_score(y, pred),
                prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0))


def train_one(num_layers, seed, device, data):
    set_seed(seed)
    Xtr, ytr, Xva, yva, Xte, yte = data
    mean = Xtr.reshape(-1, 5).mean(0); std = Xtr.reshape(-1, 5).std(0) + 1e-6
    Xtr, Xva, Xte = (Xtr - mean) / std, (Xva - mean) / std, (Xte - mean) / std

    dropout = 0.0 if num_layers == 1 else 0.3      # inter-layer dropout, inert at 1 layer
    model = BiLSTM(input_dim=5, hidden_dim=HIDDEN,
                   num_layers=num_layers, dropout=dropout).to(device)
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
        vauc = metrics(model, Xva, yva, device)["auc"]
        sched.step(vauc)
        if vauc > best_auc:
            best_auc, best_ep, noimp = vauc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            noimp += 1
            if noimp >= PATIENCE:
                break
    model.load_state_dict(best_state)
    m = metrics(model, Xte, yte, device)
    m.update(num_layers=num_layers, dropout=dropout, seed=seed, n_params=int(n_params),
             best_epoch=best_ep, val_auc=round(best_auc, 4))
    return m


def agg(rows, metric):
    v = np.array([r[metric] for r in rows]); return v.mean(), v.std(ddof=1)


def main():
    import warnings; warnings.simplefilter("ignore")     # silence nl=1 LSTM-dropout note
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = load_split()
    print(f"device {device} | num_layers {LAYERS} | seeds {SEEDS} | "
          f"hidden {HIDDEN}, lr {LR} (baseline)\n")
    runs_root = HERE / "runs_layers"; runs_root.mkdir(exist_ok=True)

    results = {L: [] for L in LAYERS}
    print(f"{'layers':>6s} {'params':>9s} {'seed':>4s} {'AUC':>6s} {'PR':>6s} "
          f"{'F1':>6s} {'Acc':>6s} {'ep':>3s} {'s':>4s}")
    print("-" * 56)
    for L in LAYERS:
        rdir = runs_root / f"nl{L}"; rdir.mkdir(exist_ok=True)
        for seed in SEEDS:
            jp = rdir / f"seed{seed}.json"
            if jp.exists():
                m = json.loads(jp.read_text())
            else:
                t0 = time.time()
                m = train_one(L, seed, device, data)
                m["seconds"] = round(time.time() - t0, 1)
                jp.write_text(json.dumps(m, indent=2))
            results[L].append(m)
            print(f"{L:>6d} {m['n_params']:>9,d} {m['seed']:>4d} {m['auc']:6.3f} "
                  f"{m['pr_auc']:6.3f} {m['f1']:6.3f} {m['acc']:6.3f} "
                  f"{m['best_epoch']:>3d} {m.get('seconds',0):>4.0f}")

    write_outputs(results)


def write_outputs(results):
    with open(HERE / "07b_num_layers_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_layers", "dropout", "n_params", "seed", "best_epoch"] + METRICS)
        for L in LAYERS:
            for r in results[L]:
                w.writerow([L, r["dropout"], r["n_params"], r["seed"], r["best_epoch"]]
                           + [round(r[m], 4) for m in METRICS])

    aucs = {L: np.array([r["auc"] for r in results[L]]) for L in LAYERS}
    means = {L: aucs[L].mean() for L in LAYERS}
    seed_std = float(np.mean([aucs[L].std(ddof=1) for L in LAYERS]))
    spread = max(means.values()) - min(means.values())
    params = {L: results[L][0]["n_params"] for L in LAYERS}
    best_L = max(means, key=means.get)

    L = ["# Issue 7b — Network-depth ablation (num_layers ∈ {1,2,3}, multi-seed)", "",
         f"Companion to the hidden-size (width) ablation: varies **depth** at the "
         f"baseline width (hidden=128, lr=1e-3, dropout=0.3). Clean baseline data "
         f"(`sequences_clean/`), {len(SEEDS)} seeds {SEEDS}, MPS; everything else "
         f"locked. num_layers=2 is the baseline. **Note:** dropout is inter-layer LSTM "
         f"dropout, inert at num_layers=1, so the 1-layer model runs with no dropout "
         f"(intrinsic to the architecture).", "",
         "| num_layers | dropout | params | best ep | AUC | PR-AUC | F1 | Acc |",
         "|---|---|---|---|---|---|---|---|"]
    for d in LAYERS:
        rows = results[d]
        ep = f"{np.mean([r['best_epoch'] for r in rows]):.0f}"
        do = "0.0 (inert)" if d == 1 else "0.3"
        cells = " | ".join(f"{agg(rows, m)[0]:.3f} ± {agg(rows, m)[1]:.3f}"
                           for m in ["auc", "pr_auc", "f1", "acc"])
        L.append(f"| {d}{' (baseline)' if d==2 else ''} | {do} | {params[d]:,} | {ep} | {cells} |")
    L += ["",
          f"**Between-depth mean-AUC spread = {spread:.4f}**, vs average within-depth "
          f"seed std = ±{seed_std:.4f}.", "",
          "Pairwise vs num_layers=2 (paired t-test, matched seeds; Mann-Whitney U):", "",
          "| pair | ΔAUC | paired-t p | Mann-Whitney p |", "|---|---|---|---|"]
    pvals = {}
    for d in [1, 3]:
        a, b = aucs[d], aucs[2]
        tp = stats.ttest_rel(a, b).pvalue
        up = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        pvals[d] = tp
        L.append(f"| nl{d} vs nl2 | {a.mean()-b.mean():+.4f} | {tp:.3f} | {up:.3f} |")
    kw = stats.kruskal(*[aucs[d] for d in LAYERS]).pvalue
    L.append(f"\nKruskal–Wallis across the three depths: p = {kw:.3f}.\n")

    no_sig = pvals[1] > 0.05 and pvals[3] > 0.05 and kw > 0.05
    L.append("## Verdict\n")
    if no_sig:
        L.append(f"**num_layers=2 is justified — depth past 1 layer gives no "
                 f"significant gain, and a 3rd layer adds none.** No depth differs "
                 f"significantly from 2 (nl1 p={pvals[1]:.3f}, nl3 p={pvals[3]:.3f}, "
                 f"Kruskal p={kw:.3f}); the spread ({spread:.4f}) is "
                 f"{'within' if spread <= seed_std else 'near'} seed noise "
                 f"(±{seed_std:.4f}). Depth-3 costs {params[3]/params[2]:.1f}× the "
                 f"parameters of depth-2 ({params[3]:,} vs {params[2]:,}) for no "
                 f"measurable benefit, and depth-1 (no inter-layer dropout) is "
                 f"{'no better' if means[1] <= means[2] else 'not significantly better'}. "
                 f"**2 layers is the right depth** — enough to model the sequence, not "
                 f"so deep it overfits the small training set (N=2178). num_layers=2 "
                 f"reproduces the baseline (this run {means[2]:.3f} ± "
                 f"{aucs[2].std(ddof=1):.3f} vs 0.932 ± 0.011).\n")
    else:
        L.append(f"**num_layers={best_L} has the highest mean AUC ({means[best_L]:.3f}).** "
                 f"nl1 p={pvals[1]:.3f}, nl3 p={pvals[3]:.3f}, Kruskal {kw:.3f}; spread "
                 f"{spread:.4f} vs seed std ±{seed_std:.4f}. "
                 + ("A depth reaches significance — report it and the cost trade-off."
                    if (pvals[1] < 0.05 or pvals[3] < 0.05) else
                    "No pairwise test is significant despite the spread — treat as a "
                    "mild, non-significant trend.")
                 + f" Params: 1L={params[1]:,}, 2L={params[2]:,}, 3L={params[3]:,}.\n")
    (HERE / "07b_num_layers_results.md").write_text("\n".join(L))

    make_figure(results, means, params)
    print(f"\nspread {spread:.4f} | seed std ±{seed_std:.4f} | best nl={best_L} "
          f"| nl1 p={pvals[1]:.3f} | nl3 p={pvals[3]:.3f} | Kruskal {kw:.3f}")
    print("wrote 07b_num_layers_results.csv/.md and figure")


def make_figure(results, means, params):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = LAYERS
    mu = [means[d] for d in LAYERS]
    sd = [agg(results[d], "auc")[1] for d in LAYERS]
    fig, ax = plt.subplots(figsize=(7, 5.2), facecolor="white")
    ax.errorbar(xs, mu, yerr=sd, color="#0891b2", lw=2.5, marker="o", ms=9,
                capsize=5, zorder=3, label="mean ± std (5 seeds)")
    for d in LAYERS:
        ys = [r["auc"] for r in results[d]]
        ax.scatter([d] * len(ys), ys, color="#0891b2", alpha=0.35, s=30, zorder=2)
    for d, m in zip(xs, mu):
        ax.annotate(f"{m:.3f}", (d, m), textcoords="offset points",
                    xytext=(0, 12), ha="center", color="#0891b2", fontweight="bold")
        ax.annotate(f"{params[d]/1000:.0f}k", (d, m), textcoords="offset points",
                    xytext=(0, -20), ha="center", color="#6b7280", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d} layer{'s' if d>1 else ''}{' (baseline)' if d==2 else ''}"
                        for d in xs])
    ax.set_ylabel("test AUC (set03)")
    ax.set_title("Network-depth ablation — num_layers ∈ {1,2,3}\n(baseline width, 5 seeds)")
    ax.set_ylim(0.90, 0.96); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(HERE / "07b_num_layers_figure.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

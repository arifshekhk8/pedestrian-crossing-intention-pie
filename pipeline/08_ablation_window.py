"""
08_ablation_window.py — Day 8: observation window ablation.

Comparison axis 3 from THESIS_PLAN.md:
  obs_len = 8 frames (0.27s) | 16 frames (0.53s) | 30 frames (1.0s)

For each obs_len:
  1. Builds sequences from pie_annotations.pkl via 02_build_sequences.py
  2. Trains baseline BiLSTM (identical arch + hyperparams to 04_train_bilstm.py)
  3. Saves results to paper_and_artifacts/runs/ablation_window_{N}/

obs_len is the ONLY variable. Everything else is locked:
  - same model architecture (BiLSTMIntentPredictor from 03_bilstm_model.py)
  - same seed, lr, patience, pos_weight, TTE=45, input_dim=5
"""

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle
import random
import subprocess
import time
from pathlib import Path
from importlib import import_module

import matplotlib
matplotlib.use("Agg")

BiLSTM = import_module("03_bilstm_model").BiLSTMIntentPredictor

TRAIN_SETS = {"set01", "set02", "set04"}
VAL_SETS = {"set05", "set06"}
TEST_SETS = {"set03"}
POS_WEIGHT = 1.44
OBS_LENS = [8, 16, 30]
TTE = 45


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── sequence building ─────────────────────────────────────────────────────────

def build_sequences(pkl_path: Path, obs_len: int, out_dir: Path):
    """
    Calls 02_build_sequences.py via subprocess.
    Args match the Day 3 CLI: --pkl --obs-len --tte --out-dir
    Skips if out_dir already contains X.npy (cached run).
    """
    if (out_dir / "X.npy").exists():
        print(
            f"  [cache hit] sequences already exist at {out_dir}, skipping build")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "02_build_sequences.py",
        "--pkl",     str(pkl_path),
        "--obs-len", str(obs_len),
        "--tte",     str(TTE),
        "--out-dir", str(out_dir),
    ]
    print(f"  building sequences: obs_len={obs_len} → {out_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  STDOUT:", result.stdout[-500:])
        print("  STDERR:", result.stderr[-500:])
        raise RuntimeError(
            f"02_build_sequences.py failed for obs_len={obs_len}")
    print(f"  done — {result.stdout.strip().splitlines()[-1]}")


# ── data loading ──────────────────────────────────────────────────────────────

def load_splits(seq_dir: Path):
    X = np.load(seq_dir / "X.npy").astype(np.float32)
    y = np.load(seq_dir / "y.npy").astype(np.float32)
    with open(seq_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    set_ids = np.array([m["set_id"] for m in meta])
    tr = np.isin(set_ids, list(TRAIN_SETS))
    va = np.isin(set_ids, list(VAL_SETS))
    te = np.isin(set_ids, list(TEST_SETS))
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def normalize(X, mean, std):
    return (X - mean) / std


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss, n = 0.0, 0
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb).squeeze(-1)
        total_loss += crit(logits, yb).item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(yb.cpu().numpy())
        n += yb.size(0)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)
    return {
        "loss": total_loss / n,
        "acc":  accuracy_score(labels, preds),
        "f1":   f1_score(labels, preds, zero_division=0),
        "auc":  roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan"),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec":  recall_score(labels, preds, zero_division=0),
        "probs": probs, "labels": labels, "preds": preds,
    }


# ── single training run ───────────────────────────────────────────────────────

def train_one(seq_dir: Path, out_dir: Path, obs_len: int,
              device, epochs=100, batch_size=32,
              lr=1e-3, weight_decay=1e-5, patience=15, seed=42):

    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xva, yva, Xte, yte = load_splits(seq_dir)
    print(f"  shapes → train {Xtr.shape} | val {Xva.shape} | test {Xte.shape}")

    # train-only normalization
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-6
    Xtr = normalize(Xtr, mean, std)
    Xva = normalize(Xva, mean, std)
    Xte = normalize(Xte, mean, std)
    np.save(out_dir / "norm_mean.npy", mean)
    np.save(out_dir / "norm_std.npy",  std)

    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=(device.type == "cuda"))

    train_loader = make_loader(Xtr, ytr, True)
    val_loader = make_loader(Xva, yva, False)
    test_loader = make_loader(Xte, yte, False)

    model = BiLSTM().to(device)
    pos_wt = torch.tensor([POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_wt)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    best_auc, no_improve = -1.0, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * yb.size(0)
            n += yb.size(0)

        val = evaluate(model, val_loader, device)
        scheduler.step(val["auc"])
        history.append({"epoch": epoch,
                        "train_loss": running / n, **{f"val_{k}": v
                                                      for k, v in val.items()
                                                      if k not in ("probs", "labels", "preds")},
                        "lr": optimizer.param_groups[0]["lr"]})

        print(f"  ep {epoch:3d} | trL {running/n:.4f} | "
              f"vAUC {val['auc']:.3f} vF1 {val['f1']:.3f} "
              f"vAcc {val['acc']:.3f} | lr {optimizer.param_groups[0]['lr']:.1e} "
              f"| {time.time()-t0:.1f}s")

        if val["auc"] > best_auc:
            best_auc = val["auc"]
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "obs_len": obs_len,
                        "val_metrics": {k: v for k, v in val.items()
                                        if k not in ("probs", "labels", "preds")}},
                       out_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stop at epoch {epoch}")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # test eval on best checkpoint
    ckpt = torch.load(out_dir / "best.pt",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device)
    cm = confusion_matrix(test["labels"], test["preds"]).tolist()

    final = {
        "model":      f"BiLSTM obs_len={obs_len}",
        "obs_len":    obs_len,
        "best_epoch": ckpt["epoch"],
        "val":        ckpt["val_metrics"],
        "test":       {k: v for k, v in test.items()
                       if k not in ("probs", "labels", "preds")},
        "test_confusion_matrix": cm,
    }
    with open(out_dir / "final.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"  TEST → AUC {test['auc']:.3f} | F1 {test['f1']:.3f} | "
          f"Acc {test['acc']:.3f} | P {test['prec']:.3f} | R {test['rec']:.3f}")
    return final


# ── comparison plot ───────────────────────────────────────────────────────────

def save_comparison_plot(results: list, plot_path: Path):
    metric_names = ["Acc", "F1", "AUC", "Prec", "Rec"]
    # light → dark blue for 8→16→30
    colors = ["#60a5fa", "#378ADD", "#0C447C"]
    labels = [
        f"obs={r['obs_len']}f\n({r['obs_len']/30:.2f}s)" for r in results]

    data = [[r["test"]["acc"], r["test"]["f1"], r["test"]["auc"],
             r["test"]["prec"], r["test"]["rec"]] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#0f1117")
    fig.suptitle("Observation Window Ablation  (axis 3) — test set: set03",
                 color="white", fontsize=13, fontweight="bold", y=0.97)

    # ── left: grouped bar chart ──
    ax = axes[0]
    ax.set_facecolor("#0f1117")
    x = np.arange(len(metric_names))
    w = 0.22
    offsets = [-w, 0, w]
    for i, (vals, color, lbl) in enumerate(zip(data, colors, labels)):
        bars = ax.bar(x + offsets[i], vals, w - 0.02,
                      label=lbl.replace("\n", " "), color=color, alpha=0.85, zorder=2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color=color)
    ax.set_ylim(0.65, 0.98)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, color="white", fontsize=11)
    ax.yaxis.set_tick_params(labelcolor="#9ca3af")
    ax.tick_params(colors="#4b5563", length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(axis="y", color="#374151", lw=0.5, zorder=1)
    for s in ax.spines.values():
        s.set_edgecolor("#374151")
    ax.legend(facecolor="#1f2937", edgecolor="#374151",
              labelcolor="white", fontsize=10)
    ax.set_title("All metrics by obs window",
                 color="white", fontsize=11, pad=10)

    # ── right: AUC line trend ──
    ax2 = axes[1]
    ax2.set_facecolor("#0f1117")
    obs_vals = [r["obs_len"] for r in results]
    auc_vals = [r["test"]["auc"] for r in results]
    ax2.plot(obs_vals, auc_vals, color="#378ADD", lw=2, marker="o",
             markersize=8, markerfacecolor="white", markeredgecolor="#378ADD",
             markeredgewidth=2, zorder=3)
    for x_v, y_v in zip(obs_vals, auc_vals):
        ax2.text(x_v, y_v + 0.002, f"{y_v:.3f}",
                 ha="center", va="bottom", color="#378ADD",
                 fontsize=11, fontweight="bold")
    ax2.set_xticks(obs_vals)
    ax2.set_xticklabels([f"{v}f\n({v/30:.2f}s)" for v in obs_vals],
                        color="white", fontsize=10)
    ax2.yaxis.set_tick_params(labelcolor="#9ca3af")
    ax2.tick_params(colors="#4b5563", length=0)
    ax2.grid(color="#374151", lw=0.5, zorder=1)
    for s in ax2.spines.values():
        s.set_edgecolor("#374151")
    ax2.set_xlabel("observation window (frames)", color="#9ca3af", fontsize=10)
    ax2.set_ylabel("AUC-ROC", color="#9ca3af", fontsize=10)
    ax2.set_title("AUC vs observation window",
                  color="white", fontsize=11, pad=10)
    ax2.set_ylim(min(auc_vals) - 0.03, max(auc_vals) + 0.03)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"Plot saved → {plot_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pkl = Path("sequences/pie_annotations.pkl")   # symlinked from dataset
    work = Path("/kaggle/working")
    results = []

    print(f"device : {device}")
    print(f"running obs_len ablation: {OBS_LENS}\n")

    for obs_len in OBS_LENS:
        print(f"\n{'='*60}")
        print(f"  obs_len = {obs_len} frames  ({obs_len/30:.2f}s @ 30fps)")
        print(f"{'='*60}")

        seq_dir = work / f"seqs_obs{obs_len}"
        out_dir = work / f"ablation_window_{obs_len}"

        # step 1: build sequences
        build_sequences(pkl, obs_len, seq_dir)

        # step 2: train + eval
        final = train_one(seq_dir, out_dir, obs_len, device)
        results.append(final)

    # ── summary table ──
    print(f"\n{'='*72}")
    print(f"  Axis 3 — Observation Window Ablation  (test set: set03)")
    print(f"{'='*72}")
    print(f"{'obs_len':>10} {'frames':>8} {'secs':>6} "
          f"{'Epoch':>6} {'Acc':>7} {'F1':>7} {'AUC':>7} {'P':>7} {'R':>7}")
    print(f"{'-'*72}")
    for r in results:
        t = r["test"]
        print(f"{r['obs_len']:>10} {r['obs_len']:>8} "
              f"{r['obs_len']/30:>6.2f} "
              f"{r['best_epoch']:>6} "
              f"{t['acc']:>7.3f} {t['f1']:>7.3f} {t['auc']:>7.3f} "
              f"{t['prec']:>7.3f} {t['rec']:>7.3f}")
    print(f"{'='*72}")

    # delta from 16-frame reference
    ref = next(r for r in results if r["obs_len"] == 16)
    print("\nDelta vs obs_len=16 reference:")
    for r in results:
        if r["obs_len"] == 16:
            continue
        d_auc = r["test"]["auc"] - ref["test"]["auc"]
        d_f1 = r["test"]["f1"] - ref["test"]["f1"]
        print(f"  obs_len={r['obs_len']:>2} vs 16 → "
              f"AUC {d_auc:+.3f} | F1 {d_f1:+.3f}")

    # save consolidated results
    all_results_path = work / "ablation_window_results.json"
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results → {all_results_path}")

    # comparison plot
    plot_path = work / "day8_window_ablation.png"
    save_comparison_plot(results, plot_path)


if __name__ == "__main__":
    main()

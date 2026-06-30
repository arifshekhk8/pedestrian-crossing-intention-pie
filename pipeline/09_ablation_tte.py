"""
09_ablation_tte.py — Day 9: prediction horizon (TTE) ablation.

Comparison axis 4 from THESIS_PLAN.md:
  TTE = 30 frames (1.00s) | 45 frames (1.50s) | 60 frames (2.00s)

For each TTE:
  1. Builds sequences from pie_annotations.pkl via 02_build_sequences.py
  2. Trains baseline BiLSTM (identical arch + hyperparams to 04_train_bilstm.py)
  3. Saves results to paper_and_artifacts/runs/ablation_tte_{N}/

TTE is the ONLY variable. Everything else is locked:
  - same model architecture (BiLSTMIntentPredictor from 03_bilstm_model.py)
  - same seed, lr, patience, obs_len=16, input_dim=5
  - pos_weight FIXED at 1.44 across ALL experiments (Days 5-9).
    We log each run's actual class balance for the record, but do NOT
    feed it into the loss — that keeps TTE the only moving variable, so
    the precision/recall trend is attributable to horizon alone, and
    TTE=45 reproduces the Day 5 baseline (AUC 0.931, F1 0.844) exactly.
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
POS_WEIGHT = 1.44          # fixed across all experiments (Days 5-9)
TTE_VALUES = [30, 45, 60]
OBS_LEN = 16


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── sequence building ─────────────────────────────────────────────────────────

def build_sequences(pkl_path: Path, tte: int, out_dir: Path):
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
        "--obs-len", str(OBS_LEN),
        "--tte",     str(tte),
        "--out-dir", str(out_dir),
    ]
    print(f"  building sequences: TTE={tte} → {out_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  STDOUT:", result.stdout[-500:])
        print("  STDERR:", result.stderr[-500:])
        raise RuntimeError(f"02_build_sequences.py failed for TTE={tte}")
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

def train_one(seq_dir: Path, out_dir: Path, tte: int,
              device, epochs=100, batch_size=32,
              lr=1e-3, weight_decay=1e-5, patience=15, seed=42):

    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xva, yva, Xte, yte = load_splits(seq_dir)
    print(f"  shapes → train {Xtr.shape} | val {Xva.shape} | test {Xte.shape}")

    # Log this run's class balance for the record, but DO NOT use it as the
    # loss weight — pos_weight is fixed at 1.44 so TTE stays the only variable.
    n_pos = int((ytr == 1).sum())
    n_neg = int((ytr == 0).sum())
    pos_weight_val = POS_WEIGHT
    print(f"  class balance: neg={n_neg} pos={n_pos} "
          f"(natural ratio {n_neg / max(n_pos, 1):.3f}) → "
          f"using FIXED pos_weight={pos_weight_val:.3f}")

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
    pos_wt = torch.tensor([pos_weight_val], device=device)
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
                        "tte": tte,
                        "pos_weight": pos_weight_val,
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
        "model":       f"BiLSTM TTE={tte}",
        "tte":         tte,
        "tte_seconds": tte / 30.0,
        "pos_weight":  pos_weight_val,
        "n_train_pos": n_pos,
        "n_train_neg": n_neg,
        "best_epoch":  ckpt["epoch"],
        "n_train":     int(len(ytr)),
        "n_val":       int(len(yva)),
        "n_test":      int(len(yte)),
        "val":         ckpt["val_metrics"],
        "test":        {k: v for k, v in test.items()
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
    # teal palette (distinct from Day 8 blues), light → dark for 30→45→60
    colors = ["#5eead4", "#14b8a6", "#0f766e"]
    labels = [f"TTE={r['tte']}f\n({r['tte']/30:.2f}s)" for r in results]

    data = [[r["test"]["acc"], r["test"]["f1"], r["test"]["auc"],
             r["test"]["prec"], r["test"]["rec"]] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#0f1117")
    fig.suptitle("Prediction Horizon Ablation  (axis 4) — test set: set03",
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
    ax.set_ylim(0.55, 0.98)
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
    ax.set_title("All metrics by TTE", color="white", fontsize=11, pad=10)

    # ── right: AUC line trend ──
    ax2 = axes[1]
    ax2.set_facecolor("#0f1117")
    tte_vals = [r["tte"] for r in results]
    auc_vals = [r["test"]["auc"] for r in results]
    ax2.plot(tte_vals, auc_vals, color="#14b8a6", lw=2, marker="o",
             markersize=8, markerfacecolor="white", markeredgecolor="#14b8a6",
             markeredgewidth=2, zorder=3)
    for x_v, y_v in zip(tte_vals, auc_vals):
        ax2.text(x_v, y_v + 0.003, f"{y_v:.3f}",
                 ha="center", va="bottom", color="#14b8a6",
                 fontsize=11, fontweight="bold")
    ax2.set_xticks(tte_vals)
    ax2.set_xticklabels([f"{v}f\n({v/30:.2f}s)" for v in tte_vals],
                        color="white", fontsize=10)
    ax2.yaxis.set_tick_params(labelcolor="#9ca3af")
    ax2.tick_params(colors="#4b5563", length=0)
    ax2.grid(color="#374151", lw=0.5, zorder=1)
    for s in ax2.spines.values():
        s.set_edgecolor("#374151")
    ax2.set_xlabel("prediction horizon (frames)", color="#9ca3af", fontsize=10)
    ax2.set_ylabel("AUC-ROC", color="#9ca3af", fontsize=10)
    ax2.set_title("AUC vs prediction horizon",
                  color="white", fontsize=11, pad=10)
    ax2.set_ylim(min(auc_vals) - 0.04, max(auc_vals) + 0.03)

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
    print(f"pos_weight (fixed, all runs): {POS_WEIGHT}")
    print(f"running TTE ablation: {TTE_VALUES}\n")

    for tte in TTE_VALUES:
        print(f"\n{'='*60}")
        print(f"  TTE = {tte} frames  ({tte/30:.2f}s @ 30fps)")
        print(f"{'='*60}")

        seq_dir = work / f"seqs_tte{tte}"
        out_dir = work / f"ablation_tte_{tte}"

        # step 1: build sequences
        build_sequences(pkl, tte, seq_dir)

        # step 2: train + eval
        final = train_one(seq_dir, out_dir, tte, device)
        results.append(final)

    # ── summary table ──
    print(f"\n{'='*78}")
    print(f"  Axis 4 — Prediction Horizon Ablation  (test set: set03)")
    print(f"{'='*78}")
    print(f"{'TTE':>6} {'secs':>6} {'pos_w':>7} {'N_test':>7} "
          f"{'Epoch':>6} {'Acc':>7} {'F1':>7} {'AUC':>7} {'P':>7} {'R':>7}")
    print(f"{'-'*78}")
    for r in results:
        t = r["test"]
        print(f"{r['tte']:>6} {r['tte']/30:>6.2f} {r['pos_weight']:>7.3f} "
              f"{r['n_test']:>7} {r['best_epoch']:>6} "
              f"{t['acc']:>7.3f} {t['f1']:>7.3f} {t['auc']:>7.3f} "
              f"{t['prec']:>7.3f} {t['rec']:>7.3f}")
    print(f"{'='*78}")

    # delta from TTE=45 reference (matches Day 5 baseline)
    ref = next(r for r in results if r["tte"] == 45)
    print("\nDelta vs TTE=45 reference (Day 5 baseline):")
    for r in results:
        if r["tte"] == 45:
            continue
        d_auc = r["test"]["auc"] - ref["test"]["auc"]
        d_f1 = r["test"]["f1"] - ref["test"]["f1"]
        print(f"  TTE={r['tte']:>2} vs 45 → "
              f"AUC {d_auc:+.3f} | F1 {d_f1:+.3f}")

    # reproducibility check: TTE=45 with fixed pos_weight=1.44 should now
    # reproduce Day 5 / Day 8(obs=16) exactly (AUC 0.931, F1 0.844)
    ref_auc, ref_f1 = ref["test"]["auc"], ref["test"]["f1"]
    drift_auc = abs(ref_auc - 0.931)
    drift_f1 = abs(ref_f1 - 0.844)
    print(f"\nReproducibility check (TTE=45 vs Day 5 baseline):")
    print(f"  AUC: {ref_auc:.3f}  (Day 5: 0.931, drift {drift_auc:+.3f})")
    print(f"  F1 : {ref_f1:.3f}  (Day 5: 0.844, drift {drift_f1:+.3f})")
    if drift_auc > 0.01 or drift_f1 > 0.01:
        print(f"  ⚠  drift > 0.01 — check pipeline before trusting deltas")
    else:
        print(f"  ✓  reproduced within noise")

    # save consolidated results
    all_results_path = work / "ablation_tte_results.json"
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results → {all_results_path}")

    # comparison plot
    plot_path = work / "day9_tte_ablation.png"
    save_comparison_plot(results, plot_path)


if __name__ == "__main__":
    main()

"""
04_train_bilstm.py — Day 5: full training run, baseline BiLSTM.

HISTORICAL ARTIFACT (2026-07-13 note): the DEFAULTS here are the RETRACTED
leaky-era protocol (seq_dir="sequences", POS_WEIGHT=1.44). Running this as-is does
NOT reproduce the published clean 0.932 — that needs
--seq_dir journal_prep/issue2_clean_protocol/sequences_clean --pos_weight 1.682
(exactly what issue2's 04_multiseed_baseline.py passes). For any NEW training use
the unified engine: journal_prep/issue12_unified_pipeline/12_unified_engine.py.
This file's behaviour is deliberately unchanged — it is the record of what produced
the legacy runs and defines the demo's inference contract.

Loads sequences/, splits by PIE set_id, normalizes with train-only stats,
trains with early stopping + best-checkpoint on val AUC, final test eval
with all 4 metrics + confusion matrix.
"""

import argparse
import json
import pickle
import random
import time
from pathlib import Path
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)

# Python modules can't start with a digit, so importlib instead of `from ... import`
BiLSTM = import_module("03_bilstm_model").BiLSTMIntentPredictor
TRAIN_SETS = {"set01", "set02", "set04"}
VAL_SETS = {"set05", "set06"}
TEST_SETS = {"set03"}
POS_WEIGHT = 1.44  # from Day 3: 819 negatives / 570 positives


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Tradeoff: deterministic=True is reproducible but slightly slower.
    # For a 20-day thesis we want reproducible numbers in the paper.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_splits(seq_dir: Path):
    X = np.load(seq_dir / "X.npy").astype(np.float32)
    y = np.load(seq_dir / "y.npy").astype(np.float32)
    with open(seq_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    # Assumes meta is a list of dicts with "set_id" per sequence.
    # If yours is a DataFrame, swap to: set_ids = meta["set_id"].to_numpy()
    set_ids = np.array([m["set_id"] for m in meta])

    tr = np.isin(set_ids, list(TRAIN_SETS))
    va = np.isin(set_ids, list(VAL_SETS))
    te = np.isin(set_ids, list(TEST_SETS))
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def compute_norm_stats(X_train: np.ndarray):
    # Flatten over (N, T) so we get per-feature stats across all timesteps.
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-6
    return mean, std


def normalize(X, mean, std):
    return (X - mean) / std


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    """Run model over loader; return metrics dict + raw probs/preds/labels."""
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
    preds = (probs >= threshold).astype(int)
    return {
        "loss": total_loss / n,
        "acc":  accuracy_score(labels, preds),
        "f1":   f1_score(labels, preds, zero_division=0),
        # AUC needs both classes present; val/test should have both, but guard anyway.
        "auc":  roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan"),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec":  recall_score(labels, preds, zero_division=0),
        "probs": probs, "labels": labels, "preds": preds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", type=Path, default=Path("sequences"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("paper_and_artifacts/runs/bilstm_baseline"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos_weight", type=float, default=POS_WEIGHT,
                    help="overrides the fixed Day-3 POS_WEIGHT (819 neg/570 pos); "
                         "pass the new train split's neg/pos ratio for a different dataset")
    args = ap.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    Xtr, ytr, Xva, yva, Xte, yte = load_splits(args.seq_dir)
    print(f"train: {Xtr.shape} | val: {Xva.shape} | test: {Xte.shape}")
    print(
        f"train pos rate: {ytr.mean():.3f} | val: {yva.mean():.3f} | test: {yte.mean():.3f}")

    # Train-only normalization — saved so inference (Day 15) reuses identical stats.
    mean, std = compute_norm_stats(Xtr)
    Xtr = normalize(Xtr, mean, std)
    Xva = normalize(Xva, mean, std)
    Xte = normalize(Xte, mean, std)
    np.save(args.out_dir / "norm_mean.npy", mean)
    np.save(args.out_dir / "norm_std.npy", std)

    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=(device.type == "cuda"))

    train_loader = make_loader(Xtr, ytr, shuffle=True)
    val_loader = make_loader(Xva, yva, shuffle=False)
    test_loader = make_loader(Xte, yte, shuffle=False)

    model = BiLSTM().to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    # mode='max' because we monitor AUC (higher = better)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_auc, epochs_no_improve = -1.0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running_loss, n = time.time(), 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            n += yb.size(0)
        train_loss = running_loss / n

        val = evaluate(model, val_loader, device)
        scheduler.step(val["auc"])

        row = {"epoch": epoch, "train_loss": train_loss,
               "val_loss": val["loss"], "val_acc": val["acc"], "val_f1": val["f1"],
               "val_auc": val["auc"], "val_prec": val["prec"], "val_rec": val["rec"],
               "lr": optimizer.param_groups[0]["lr"], "time": time.time() - t0}
        history.append(row)
        print(f"ep {epoch:3d} | trL {train_loss:.4f} | vL {val['loss']:.4f} | "
              f"vAcc {val['acc']:.3f} F1 {val['f1']:.3f} AUC {val['auc']:.3f} "
              f"P {val['prec']:.3f} R {val['rec']:.3f} | lr {row['lr']:.1e} "
              f"| {row['time']:.1f}s")

        if val["auc"] > best_auc:
            best_auc = val["auc"]
            epochs_no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_metrics": {k: v for k, v in val.items()
                                if k not in ("probs", "labels", "preds")},
            }, args.out_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stop at epoch {epoch} "
                      f"(no val AUC improvement for {args.patience} epochs)")
                break

    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---- Final test eval on the BEST val checkpoint (touched once) ----
    ckpt = torch.load(args.out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device)
    cm = confusion_matrix(test["labels"], test["preds"]
                          ).tolist()  # [[TN,FP],[FN,TP]]
    final = {
        "best_epoch": ckpt["epoch"],
        "val":  ckpt["val_metrics"],
        "test": {k: v for k, v in test.items()
                 if k not in ("probs", "labels", "preds")},
        "test_confusion_matrix": cm,
    }
    with open(args.out_dir / "final.json", "w") as f:
        json.dump(final, f, indent=2)

    print("\n===== TEST (best val-AUC checkpoint) =====")
    print(f"best epoch: {final['best_epoch']}")
    print(f"acc {test['acc']:.3f} | F1 {test['f1']:.3f} | AUC {test['auc']:.3f} | "
          f"P {test['prec']:.3f} | R {test['rec']:.3f}")
    print(f"confusion [[TN,FP],[FN,TP]]: {cm}")


if __name__ == "__main__":
    main()

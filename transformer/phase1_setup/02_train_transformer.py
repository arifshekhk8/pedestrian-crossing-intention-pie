"""02_train_transformer.py — the transformer training engine (transformer/PLAN.md #2, #4).

train_run() is the single training loop reused everywhere: Phase-T1 dev runs here,
the Kaggle search notebook's Stages A/B/C, and the Kaggle final notebook's Stage D. The
notebooks embed this file verbatim under a "KEEP IN SYNC WITH 02_train_transformer.py"
banner — edit here first, then copy down.

Frozen protocol (identical to the LSTM — pipeline/04_train_bilstm.py /
journal_prep/issue2_clean_protocol/06b_local_verify_seed42.py): sequences_clean/,
splits by recording set, train-only z-score, pos_weight=1.682, threshold 0.5, batch 32,
max 100 epochs, early-stop patience 15 on val AUC, best-on-val-AUC checkpoint,
seeds [42,0,1,2,3]. Only the model architecture and training RECIPE (lr / schedule /
dropout / weight decay / optimizer) are searchable — see PLAN.md #4 for the grids.

CLI (single dev run — NOT a search; --eval-test only belongs in Phase T4 / Stage D):
    python transformer/phase1_setup/02_train_transformer.py --preset default --seed 42 \
        --device cpu --out_dir transformer/phase1_setup/runs_dev/default_seed42
"""
import argparse
import importlib.util
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
SEQ_DIR = HERE.parent / "sequences_clean"  # shared data lives at transformer/ root

_spec = importlib.util.spec_from_file_location("transformer_model", HERE / "00_transformer_model.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
TransformerIntentPredictor = _m.TransformerIntentPredictor
count_params = _m.count_params

TRAIN_SETS = {"set01", "set02", "set04"}
VAL_SETS = {"set05", "set06"}
TEST_SETS = {"set03"}
POS_WEIGHT = 1.682
EPOCHS = 100
BATCH_SIZE = 32
PLATEAU_PATIENCE = 5          # ReduceLROnPlateau patience (LSTM parity)
EARLY_STOP_PATIENCE = 15
THRESHOLD = 0.5
WARMUP_EPOCHS = 5
COSINE_MIN_FRAC = 0.01

# The pre-registered "transformer_default" (PLAN.md #3): the transformer's own
# architecture, trained with the LSTM's exact recipe (Adam 1e-3 / wd 1e-5 / plateau).
# Zero search — fixed before any search run.
DEFAULT_CFG = dict(
    d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1, pool="cls", pos="learned",
    lr=1e-3, schedule="plateau", weight_decay=1e-5, optimizer="adam",
)

PRESETS = {"default": DEFAULT_CFG}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_raw(seq_dir=SEQ_DIR):
    X = np.load(Path(seq_dir) / "X.npy").astype(np.float32)
    y = np.load(Path(seq_dir) / "y.npy").astype(np.float32)
    with open(Path(seq_dir) / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    set_ids = np.array([m["set_id"] for m in meta])
    assert X.shape == (4906, 16, 5), f"unexpected X shape {X.shape} — wrong sequences dir?"
    return X, y, set_ids


def split_data(X, y, set_ids):
    tr = np.isin(set_ids, list(TRAIN_SETS))
    va = np.isin(set_ids, list(VAL_SETS))
    te = np.isin(set_ids, list(TEST_SETS))
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def build_model(cfg):
    return TransformerIntentPredictor(
        input_dim=5, d_model=cfg["d_model"], nhead=cfg.get("nhead", 4),
        num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"], dropout=cfg["dropout"],
        pool=cfg["pool"], pos=cfg["pos"],
    )


def build_optimizer(model, cfg):
    lr, wd = cfg["lr"], cfg["weight_decay"]
    if cfg.get("optimizer", "adam") == "adamw":
        no_decay_keys = ("bias", "norm", "pos_embed", "cls_token")
        decay_params, no_decay_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            target = no_decay_params if any(k in name.lower() for k in no_decay_keys) else decay_params
            target.append(p)
        groups = [{"params": decay_params, "weight_decay": wd},
                  {"params": no_decay_params, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def lr_for_epoch(epoch, base_lr):
    """warmup_cosine schedule only; 'plateau' is stepped via ReduceLROnPlateau instead."""
    if epoch <= WARMUP_EPOCHS:
        return base_lr * epoch / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (COSINE_MIN_FRAC + (1 - COSINE_MIN_FRAC) * cosine)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    probs_all, labels_all, total, n = [], [], 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb).squeeze(-1)
        total += crit(logits, yb).item()
        probs_all.append(torch.sigmoid(logits).cpu().numpy())
        labels_all.append(yb.cpu().numpy())
        n += yb.size(0)
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    preds = (probs >= THRESHOLD).astype(int)
    metrics = {
        "loss": total / n,
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan"),
        "pr_auc": average_precision_score(labels, probs),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec": recall_score(labels, preds, zero_division=0),
    }
    return metrics, probs, labels


def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=False)


def train_run(cfg, seed, device, data, eval_test=False, out_dir=None, verbose=False,
              pos_weight=None):
    """The training loop shared by every stage. `data` = (Xtr, ytr, Xva, yva, Xte, yte),
    RAW (unnormalized) arrays for the fixed set-based split (identical across seeds).
    eval_test=True touches the test set — only Stage D (Phase T4) may pass this.
    pos_weight overrides the frozen POS_WEIGHT=1.682 — only Phase T4's LOSO folds pass
    this (Issue-5 protocol: per-fold n_neg/n_pos, since each fold's train pool differs)."""
    set_seed(seed)
    Xtr, ytr, Xva, yva, Xte, yte = data
    pw = POS_WEIGHT if pos_weight is None else pos_weight

    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0) + 1e-6
    Xtr_n = (Xtr - mean) / std
    Xva_n = (Xva - mean) / std
    Xte_n = (Xte - mean) / std

    train_loader = make_loader(Xtr_n, ytr, shuffle=True)
    val_loader = make_loader(Xva_n, yva, shuffle=False)
    test_loader = make_loader(Xte_n, yte, shuffle=False) if eval_test else None

    model = build_model(cfg).to(device)
    n_params = count_params(model)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = build_optimizer(model, cfg)
    schedule = cfg.get("schedule", "plateau")
    plateau = None
    if schedule == "plateau":
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=PLATEAU_PATIENCE)

    best_auc, best_epoch, best_state, no_improve = -1.0, 0, None, 0
    history = []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        if schedule == "warmup_cosine":
            new_lr = lr_for_epoch(epoch, cfg["lr"])
            for g in opt.param_groups:
                g["lr"] = new_lr

        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()

        val_metrics, _, _ = evaluate(model, val_loader, device)
        if schedule == "plateau":
            plateau.step(val_metrics["auc"])
        history.append({"epoch": epoch, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if verbose:
            print(f"  ep {epoch:3d} vAUC {val_metrics['auc']:.4f}")

        if val_metrics["auc"] > best_auc:
            best_auc, best_epoch = val_metrics["auc"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    final_val_metrics, _, _ = evaluate(model, val_loader, device)
    seconds = round(time.time() - t0, 1)

    result = dict(**cfg, seed=seed, n_params=n_params, best_epoch=best_epoch,
                  val=final_val_metrics, seconds=seconds, pos_weight=pw)

    if eval_test:
        test_metrics, test_probs, test_labels = evaluate(model, test_loader, device)
        preds = (test_probs >= THRESHOLD).astype(int)
        tn = int(((test_labels == 0) & (preds == 0)).sum())
        fp = int(((test_labels == 0) & (preds == 1)).sum())
        fn = int(((test_labels == 1) & (preds == 0)).sum())
        tp = int(((test_labels == 1) & (preds == 1)).sum())
        result["test"] = test_metrics
        result["test_confusion_matrix"] = [[tn, fp], [fn, tp]]

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state, "epoch": best_epoch, "val_metrics": final_val_metrics},
                   out_dir / "best.pt")
        np.save(out_dir / "norm_mean.npy", mean)
        np.save(out_dir / "norm_std.npy", std)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        final_json = {"best_epoch": best_epoch, "val": final_val_metrics}
        if eval_test:
            final_json["test"] = result["test"]
            final_json["test_confusion_matrix"] = result["test_confusion_matrix"]
        (out_dir / "final.json").write_text(json.dumps(final_json, indent=2))

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="default", choices=list(PRESETS.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--seq_dir", default=str(SEQ_DIR))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--eval-test", action="store_true",
                    help="Touch the test set. Only pass this in Phase T4 (Stage D).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = PRESETS[args.preset]
    X, y, set_ids = load_raw(Path(args.seq_dir))
    data = split_data(X, y, set_ids)

    if args.eval_test:
        print("*** --eval-test set: this run touches set03. Only do this once, in Phase T4. ***")

    result = train_run(cfg, args.seed, device, data, eval_test=args.eval_test,
                       out_dir=args.out_dir, verbose=args.verbose)
    printable = {k: v for k, v in result.items() if k != "val"}
    print(json.dumps(printable, indent=2, default=str))
    print("val:", result["val"])
    if "test" in result:
        print("test:", result["test"])


if __name__ == "__main__":
    main()

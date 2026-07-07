"""Local re-verification of the multi-seed variant runs (seed 42).

Faithful port of 06_multiseed_variants_kaggle.ipynb (Cells 3-5), run locally to
confirm the original local bbox-only single-seed = 0.746 was a *stuck* run, not
the true behaviour. Baseline (~0.913) and attention (~0.933) are positive
controls: if they reproduce their known numbers but bbox-only lands ~0.88
(matching Kaggle seed-42 0.879), the 0.746 is a degenerate run and the 5-seed
0.883 is the trustworthy number.

Run on CPU for determinism. sklearn is present locally (1.9.0), so metrics match
the notebook exactly.
"""
import json, pickle, random, time, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)

HERE = Path(__file__).resolve().parent
SEQ_DIR = HERE / "sequences_clean"

# --- locked contract (identical to the notebook) ---
POS_WEIGHT   = 1.682
TRAIN_SETS   = {"set01", "set02", "set04"}
VAL_SETS     = {"set05", "set06"}
TEST_SETS    = {"set03"}
EPOCHS       = 100
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE     = 15
THRESHOLD    = 0.5


# ===== models (verbatim from notebook Cell 3) =====
class BiLSTMIntentPredictor(nn.Module):
    def __init__(self, input_dim=5, proj_dim=64, hidden_dim=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.ReLU())
        self.bilstm = nn.LSTM(proj_dim, hidden_dim, num_layers,
                              dropout=dropout, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden_dim * 2, 1)
    def forward(self, x):
        out, _ = self.bilstm(self.input_proj(x))
        return self.head(out[:, -1, :])


class BiLSTMIntentPredictorFlex(nn.Module):
    def __init__(self, input_dim=5, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 64)
        self.bilstm = nn.LSTM(64, hidden_size, num_layers,
                              dropout=dropout if num_layers > 1 else 0.0,
                              bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden_size * 2, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.drop(torch.relu(self.input_proj(x)))
        out, _ = self.bilstm(x)
        return self.head(self.drop(out[:, -1, :]))


class BiLSTMAttentionIntentPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_size=128, num_layers=2,
                 dropout=0.3, attn_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 64)
        self.bilstm = nn.LSTM(64, hidden_size, num_layers,
                              dropout=dropout if num_layers > 1 else 0.0,
                              bidirectional=True, batch_first=True)
        self.attn_W = nn.Linear(hidden_size * 2, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)
        self.head = nn.Linear(hidden_size * 2, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.drop(torch.relu(self.input_proj(x)))
        H, _ = self.bilstm(x)
        scores = self.attn_v(torch.tanh(self.attn_W(H)))
        weights = torch.softmax(scores, dim=1)
        context = (weights * H).sum(dim=1)
        return self.head(self.drop(context))


BBOX_ARCH = "flex"  # "flex" = notebook's BiLSTMIntentPredictorFlex (extra dropout);
                     # "baseline" = clean ablation: baseline arch with input_dim=4


def build_model(name):
    if name == "bilstm_baseline":
        return BiLSTMIntentPredictor(input_dim=5)
    if name == "bilstm_bbox_only":
        if BBOX_ARCH == "baseline":
            return BiLSTMIntentPredictor(input_dim=4)   # clean ablation, no extra dropout
        return BiLSTMIntentPredictorFlex(input_dim=4)
    if name == "bilstm_attention":
        return BiLSTMAttentionIntentPredictor(input_dim=5)
    raise ValueError(name)


USE_SPEED = {"bilstm_baseline": True, "bilstm_bbox_only": False, "bilstm_attention": True}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_raw():
    X = np.load(SEQ_DIR / "X.npy").astype(np.float32)
    y = np.load(SEQ_DIR / "y.npy").astype(np.float32)
    with open(SEQ_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    set_ids = np.array([m["set_id"] for m in meta])
    return X, y, set_ids


def split(X, y, set_ids):
    tr = np.isin(set_ids, list(TRAIN_SETS))
    va = np.isin(set_ids, list(VAL_SETS))
    te = np.isin(set_ids, list(TEST_SETS))
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


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
    probs = np.concatenate(probs_all); labels = np.concatenate(labels_all)
    preds = (probs >= THRESHOLD).astype(int)
    return {
        "loss": total / n,
        "acc":  accuracy_score(labels, preds),
        "f1":   f1_score(labels, preds, zero_division=0),
        "auc":  roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan"),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec":  recall_score(labels, preds, zero_division=0),
    }


def make_loader(X, y, shuffle, device):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def train_one(model_name, seed, device, X_ALL, Y_ALL, SETIDS_ALL, verbose=False):
    set_seed(seed)
    X = X_ALL if USE_SPEED[model_name] else X_ALL[:, :, :4]
    Xtr, ytr, Xva, yva, Xte, yte = split(X, Y_ALL, SETIDS_ALL)

    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0) + 1e-6
    Xtr, Xva, Xte = (Xtr - mean) / std, (Xva - mean) / std, (Xte - mean) / std

    train_loader = make_loader(Xtr, ytr, True, device)
    val_loader   = make_loader(Xva, yva, False, device)
    test_loader  = make_loader(Xte, yte, False, device)

    model = build_model(model_name).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=device))
    opt  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    best_auc, best_epoch, best_state, no_improve = -1.0, 0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward(); opt.step()
        val = evaluate(model, val_loader, device)
        sched.step(val["auc"])
        if verbose:
            print(f"  ep {epoch:3d} vAUC {val['auc']:.3f} vF1 {val['f1']:.3f}")
        if val["auc"] > best_auc:
            best_auc, best_epoch = val["auc"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    return {"model": model_name, "seed": seed, "best_epoch": best_epoch,
            "val_auc_at_best": best_auc, "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+",
                    default=["bilstm_baseline", "bilstm_attention", "bilstm_bbox_only"])
    ap.add_argument("--bbox-arch", default="flex", choices=["flex", "baseline"])
    args = ap.parse_args()
    global BBOX_ARCH
    BBOX_ARCH = args.bbox_arch
    device = torch.device(args.device)

    X_ALL, Y_ALL, SETIDS_ALL = load_raw()
    print(f"X {X_ALL.shape} pos_rate {float(Y_ALL.mean()):.3f} | device {device} | seed {args.seed}\n")
    print(f"{'model':18s} {'testAUC':>8s} {'F1':>6s} {'Acc':>6s} {'P':>6s} {'R':>6s} "
          f"{'ep':>3s} {'valAUC':>7s} {'sec':>5s}")
    print("-" * 70)
    expect = {"bilstm_baseline": "~0.913", "bilstm_attention": "~0.933 (Kaggle 0.933)",
              "bilstm_bbox_only": "Kaggle seed42=0.879 ; old local=0.746"}
    for name in args.models:
        t0 = time.time()
        r = train_one(name, args.seed, device, X_ALL, Y_ALL, SETIDS_ALL)
        t = r["test"]
        print(f"{name:18s} {t['auc']:8.3f} {t['f1']:6.3f} {t['acc']:6.3f} "
              f"{t['prec']:6.3f} {t['rec']:6.3f} {r['best_epoch']:3d} "
              f"{r['val_auc_at_best']:7.3f} {time.time()-t0:5.0f}   expect {expect.get(name,'')}")


if __name__ == "__main__":
    main()

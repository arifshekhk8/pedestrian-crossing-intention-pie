"""01_sanity_checks.py — Phase T1 gates 0-3 (transformer/PLAN.md #7, Phase T1).

Local-only, seconds-scale probes. No experiment-grade training happens here — that is
Kaggle's job (PLAN.md #5). Run this before uploading anything to Kaggle; all gates must
pass before Phase T2 (the search notebook) is created.

    python transformer/phase1_setup/01_sanity_checks.py [--device cpu]

Writes 01_sanity_report.md next to this script.
"""
import argparse
import importlib.util
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
BATCH_SIZE = 32


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def normalize(Xtr, Xva, Xte):
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0) + 1e-6
    return (Xtr - mean) / std, (Xva - mean) / std, (Xte - mean) / std, mean, std


@torch.no_grad()
def eval_auc(model, X, y, device):
    model.eval()
    p = torch.sigmoid(model(torch.from_numpy(X).to(device)).squeeze(-1)).cpu().numpy()
    return roc_auc_score(y, p)


def quick_train(build_model_fn, Xtr, ytr, Xva, yva, device, epochs, lr=1e-3, wd=1e-5, seed=42,
                pos_weight=POS_WEIGHT):
    """Takes a zero-arg model *factory*, not a pre-built model -- seeding must happen
    before construction, since layer init (Linear/LayerNorm/cls_token/pos_embed) draws
    from the global RNG. Passing an already-built model would make two "same seed" runs
    diverge from different initial weights while looking identically seeded."""
    set_seed(seed)
    model = build_model_fn().to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
    return eval_auc(model, Xva, yva, device)


def gate0_protocol_asserts(X, y, set_ids):
    lines = ["## Gate 0 — protocol asserts", ""]
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        mark = "PASS" if cond else "FAIL"
        ok = ok and cond
        lines.append(f"- [{mark}] {name}: {detail}")

    check("X shape", X.shape == (4906, 16, 5), f"got {X.shape}")
    Xtr, ytr, Xva, yva, Xte, yte = split(X, y, set_ids)
    check("train N", len(ytr) == 2178, f"got {len(ytr)}")
    check("val N", len(yva) == 634, f"got {len(yva)}")
    check("test N", len(yte) == 2094, f"got {len(yte)}")
    check("test positives", int(yte.sum()) == 681, f"got {int(yte.sum())}")
    n_pos, n_neg = int(ytr.sum()), int((1 - ytr).sum())
    pos_weight = n_neg / n_pos
    check("pos_weight", abs(pos_weight - POS_WEIGHT) < 1e-3,
          f"{n_neg}/{n_pos} = {pos_weight:.3f} (expect {POS_WEIGHT})")
    _, _, _, mean, std = normalize(Xtr, Xva, Xte)
    check("norm_mean shape", mean.shape == (5,), f"got {mean.shape}")
    check("norm_std shape", std.shape == (5,), f"got {std.shape}")

    return ok, lines, (Xtr, ytr, Xva, yva, Xte, yte)


def gate1_linear_probe_floor(Xtr, ytr, Xva, yva, device):
    lines = ["## Gate 1 — linear-probe floor", ""]

    floor_auc = quick_train(
        lambda: TransformerIntentPredictor(num_layers=0, d_model=64, dropout=0.0, pool="mean"),
        Xtr, ytr, Xva, yva, device, epochs=30)

    lr_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr_clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
    lr_probs = lr_clf.predict_proba(Xva.reshape(len(Xva), -1))[:, 1]
    lr_auc = roc_auc_score(yva, lr_probs)

    real_auc = quick_train(
        lambda: TransformerIntentPredictor(d_model=64, num_layers=2, dim_ff=128, dropout=0.1,
                                           pool="cls", pos="learned"),
        Xtr, ytr, Xva, yva, device, epochs=30)

    lines.append(f"- L=0 transformer wrapper (mean-pool, no encoder), 30 epochs: val AUC {floor_auc:.4f}")
    lines.append(f"- sklearn LogisticRegression (flat 80-D input, balanced): val AUC {lr_auc:.4f}")
    lines.append(f"- L=2 transformer (d64/ff128/cls/learned), 30 epochs: val AUC {real_auc:.4f}")
    floor = max(floor_auc, lr_auc)
    ok = real_auc > floor - 0.02
    mark = "PASS" if ok else "FAIL"
    lines.append(f"- [{mark}] L>=1 transformer clearly beats the linear floor "
                 f"({real_auc:.4f} vs floor max {floor:.4f})")
    return ok, lines


def gate2_overfit_tiny_batch(Xtr, ytr, device):
    lines = ["## Gate 2 — overfit a tiny batch", ""]
    set_seed(42)
    idx = np.arange(64)
    Xt, yt = Xtr[idx], ytr[idx]
    model = TransformerIntentPredictor(d_model=64, num_layers=2, dim_ff=128, dropout=0.0,
                                       pool="cls", pos="learned").to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xb = torch.from_numpy(Xt).to(device)
    yb = torch.from_numpy(yt).to(device)
    for _ in range(200):
        model.train()
        opt.zero_grad()
        loss = crit(model(xb).squeeze(-1), yb)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = (torch.sigmoid(model(xb).squeeze(-1)) >= 0.5).float()
        acc = (preds == yb).float().mean().item()
    ok = acc >= 0.99
    mark = "PASS" if ok else "FAIL"
    lines.append(f"- [{mark}] 64-window overfit, 200 epochs, dropout=0: train acc {acc:.4f} (expect 1.0)")
    return ok, lines


def gate3_determinism_and_params(Xtr, ytr, Xva, yva):
    lines = ["## Gate 3 — determinism probe (CPU) + parameter table", ""]
    device = torch.device("cpu")

    def build_default():
        return TransformerIntentPredictor(d_model=128, num_layers=2, dim_ff=256, dropout=0.1,
                                          pool="cls", pos="learned")

    auc1 = quick_train(build_default, Xtr, ytr, Xva, yva, device, epochs=5, seed=42)
    auc2 = quick_train(build_default, Xtr, ytr, Xva, yva, device, epochs=5, seed=42)
    ok = abs(auc1 - auc2) < 1e-6
    mark = "PASS" if ok else "FAIL"
    lines.append(f"- [{mark}] default config, seed 42 twice on CPU, 5 epochs: "
                 f"val AUC {auc1:.6f} vs {auc2:.6f} (|delta|={abs(auc1 - auc2):.2e})")

    lines.append("")
    lines.append("Parameter ladder (Stage-A sizes; brackets the 594,561-param BiLSTM):")
    lines.append("")
    lines.append("| (d_model, ff) | L | params |")
    lines.append("|---|---|---|")
    for d_model, ff in [(64, 128), (128, 256), (128, 512)]:
        for L in (2, 4):
            n = count_params(TransformerIntentPredictor(d_model=d_model, dim_ff=ff, num_layers=L))
            lines.append(f"| ({d_model}, {ff}) | {L} | {n:,} |")
    default_params = count_params(build_default())
    lines.append(f"\n`transformer_default` (d128/L2/ff256/cls/learned): **{default_params:,} params** "
                 f"(BiLSTM baseline: 594,561).")
    return ok, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = ap.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.backends.mps.is_available() else "cpu")

    print(f"device: {device}")
    X, y, set_ids = load_raw()
    ok0, lines0, splits = gate0_protocol_asserts(X, y, set_ids)
    Xtr, ytr, Xva, yva, Xte, yte = splits
    Xtr_n, Xva_n, Xte_n, mean, std = normalize(Xtr, Xva, Xte)
    print("Gate 0:", "PASS" if ok0 else "FAIL")

    t0 = time.time()
    ok1, lines1 = gate1_linear_probe_floor(Xtr_n, ytr, Xva_n, yva, device)
    print(f"Gate 1: {'PASS' if ok1 else 'FAIL'} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    ok2, lines2 = gate2_overfit_tiny_batch(Xtr_n, ytr, device)
    print(f"Gate 2: {'PASS' if ok2 else 'FAIL'} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    ok3, lines3 = gate3_determinism_and_params(Xtr_n, ytr, Xva_n, yva)
    print(f"Gate 3: {'PASS' if ok3 else 'FAIL'} ({time.time() - t0:.1f}s)")

    all_ok = ok0 and ok1 and ok2 and ok3
    report = ["# Phase T1 — Sanity report", "",
              f"Overall: {'**ALL GATES PASS**' if all_ok else '**SOME GATES FAILED — do not proceed to Kaggle**'}",
              ""]
    report += lines0 + [""] + lines1 + [""] + lines2 + [""] + lines3
    (HERE / "01_sanity_report.md").write_text("\n".join(report) + "\n")
    print(f"\n{'ALL GATES PASS' if all_ok else 'SOME GATES FAILED'} -- wrote 01_sanity_report.md")


if __name__ == "__main__":
    main()

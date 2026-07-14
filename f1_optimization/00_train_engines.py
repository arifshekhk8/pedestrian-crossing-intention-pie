"""00_train_engines.py — forked training loops for the F1-first program (PLAN.md §3.3).

Why forked and not imported/edited in place:
- `journal_prep/issue8_grid_search/08_grid_search.py` (the LSTM loop these runs must
  match) is import-broken post-reorg (stale `ROOT/"03_bilstm_model.py"` path), and
- `transformer/phase1_setup/02_train_transformer.py` is a frozen phase artifact whose
  code the Kaggle notebooks embed verbatim ("KEEP IN SYNC" banner) — editing it would
  break byte-parity with the runs it produced.
Both loops are forked here line-faithfully; the ONLY behavioural change is the
pre-registered hybrid rule: early stopping and the plateau scheduler still step on
**val AUC** (training dynamics identical to the frozen protocol), while the saved
checkpoint is the **best-val-F1 epoch** (tie-break val acc, then val AUC). Each run also
records `val_at_auc_best` (the same trajectory's AUC-best-epoch val metrics) so gate G1
is a pure within-run criterion comparison. `select="auc"` reproduces the frozen
behaviour exactly (used by 03's fork-fidelity gate and shortlist completion runs).

Every run records: cfg, seed, pos_weight, device, select, n_params, best_epoch,
val (selected checkpoint), val_at_auc_best, auc_best_epoch, seconds. Runs with an
out_dir save best.pt / final.json / history.json / norm_mean.npy / norm_std.npy in the
standard schema. VAL-ONLY: there is no test-evaluation code path in this file at all
(PLAN.md §8).
"""
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("f1_common", HERE / "00_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

EPOCHS = 100
BATCH = 32
WD_LSTM = 1e-5                 # frozen LSTM weight decay
PATIENCE = 15                  # early stop (on val AUC, frozen)
PLATEAU_PATIENCE = 5
THRESHOLD = 0.5                # training-time metric threshold (frozen)
WARMUP_EPOCHS = 5              # transformer warmup_cosine (phase1 parity)
COSINE_MIN_FRAC = 0.01


def _f1_key(m):
    """Checkpoint-selection key, PLAN.md §2 hierarchy: F1, then acc, then AUC."""
    return (round(m["f1"], 10), round(m["acc"], 10), m["auc"])


@torch.no_grad()
def _val_metrics(model, Xva_n, yva, device):
    model.eval()
    p = torch.sigmoid(model(torch.from_numpy(Xva_n).to(device)).squeeze(-1)).cpu().numpy()
    return C.metrics_at(yva, p, THRESHOLD)


def _finish(model, best, auc_best, cfg, seed, pw, device, select, n_params, t0,
            history, mean, std, out_dir):
    model.load_state_dict(best["state"])
    result = dict(cfg=cfg, seed=seed, pos_weight=pw, device=str(device), select=select,
                  n_params=int(n_params), best_epoch=best["epoch"], val=best["metrics"],
                  auc_best_epoch=auc_best["epoch"], val_at_auc_best=auc_best["metrics"],
                  seconds=round(time.time() - t0, 1))
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best["state"], "epoch": best["epoch"], "val_metrics": best["metrics"]},
                   out_dir / "best.pt")
        np.save(out_dir / "norm_mean.npy", mean)
        np.save(out_dir / "norm_std.npy", std)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        (out_dir / "final.json").write_text(json.dumps(result, indent=2))
    return result


def train_lstm(cfg, seed, device, data, pos_weight=None, select="f1", out_dir=None):
    """Forked line-faithfully from issue8's train() (protocol-identical: same op order,
    so a select='auc' run reproduces a cached grid JSON on the same device)."""
    C.set_seed(seed)
    Xtr, ytr, Xva, yva, _, _ = data
    pw = C.POS_WEIGHT if pos_weight is None else pos_weight
    mean = Xtr.reshape(-1, 5).mean(0)
    std = Xtr.reshape(-1, 5).std(0) + 1e-6
    Xtr_n, Xva_n = (Xtr - mean) / std, (Xva - mean) / std

    model = C.build_lstm(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=WD_LSTM)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=PLATEAU_PATIENCE)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr_n), torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True)

    best = {"key": None, "state": None, "epoch": 0, "metrics": None}
    auc_best = {"auc": -1.0, "epoch": 0, "metrics": None}
    noimp, history, t0 = 0, [], time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb).squeeze(-1), yb).backward()
            opt.step()
        m = _val_metrics(model, Xva_n, yva, device)
        sched.step(m["auc"])
        history.append({"epoch": ep, **{f"val_{k}": v for k, v in m.items()}})

        track_f1 = (select == "f1")
        key = _f1_key(m)
        if track_f1 and (best["key"] is None or key > best["key"]):
            best.update(key=key, epoch=ep, metrics=m,
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
        # AUC tracker drives early stop + (for select="auc") the checkpoint — frozen rule
        if m["auc"] > auc_best["auc"]:
            auc_best.update(auc=m["auc"], epoch=ep, metrics=m)
            if not track_f1:
                best.update(key=None, epoch=ep, metrics=m,
                            state={k: v.detach().clone() for k, v in model.state_dict().items()})
            noimp = 0
        else:
            noimp += 1
            if noimp >= PATIENCE:
                break
    return _finish(model, best, auc_best, cfg, seed, pw, device, select, n_params, t0,
                   history, mean, std, out_dir)


def _lr_for_epoch(epoch, base_lr):
    if epoch <= WARMUP_EPOCHS:
        return base_lr * epoch / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (COSINE_MIN_FRAC + (1 - COSINE_MIN_FRAC) * cosine)


def _build_tf_optimizer(model, cfg):
    lr, wd = cfg["lr"], cfg["weight_decay"]
    if cfg.get("optimizer", "adam") == "adamw":
        no_decay_keys = ("bias", "norm", "pos_embed", "cls_token")
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if any(k in name.lower() for k in no_decay_keys) else decay).append(p)
        return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def train_transformer(cfg, seed, device, data, pos_weight=None, select="f1", out_dir=None):
    """Forked line-faithfully from phase1's train_run() with the hybrid checkpoint rule
    swapped in and the test path removed. One deliberate difference from phase1: val is
    evaluated FULL-BATCH via _val_metrics (like the LSTM engines), not via phase1's
    batch-32 loader — mathematically equivalent per-sample probabilities, and both
    families in this program share the identical eval style (fairness-symmetric)."""
    C.set_seed(seed)
    Xtr, ytr, Xva, yva, _, _ = data
    pw = C.POS_WEIGHT if pos_weight is None else pos_weight
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0) + 1e-6
    Xtr_n, Xva_n = (Xtr - mean) / std, (Xva - mean) / std

    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr_n), torch.from_numpy(ytr)),
                              batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False)

    model = C.build_transformer(cfg).to(device)
    n_params = C.count_params(model)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = _build_tf_optimizer(model, cfg)
    schedule = cfg.get("schedule", "plateau")
    plateau = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                          patience=PLATEAU_PATIENCE)
               if schedule == "plateau" else None)

    best = {"key": None, "state": None, "epoch": 0, "metrics": None}
    auc_best = {"auc": -1.0, "epoch": 0, "metrics": None}
    noimp, history, t0 = 0, [], time.time()
    for ep in range(1, EPOCHS + 1):
        if schedule == "warmup_cosine":
            new_lr = _lr_for_epoch(ep, cfg["lr"])
            for g in opt.param_groups:
                g["lr"] = new_lr
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb).squeeze(-1), yb).backward()
            opt.step()

        m = _val_metrics(model, Xva_n, yva, device)
        if plateau is not None:
            plateau.step(m["auc"])
        history.append({"epoch": ep, **{f"val_{k}": v for k, v in m.items()}})

        track_f1 = (select == "f1")
        key = _f1_key(m)
        if track_f1 and (best["key"] is None or key > best["key"]):
            best.update(key=key, epoch=ep, metrics=m,
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
        if m["auc"] > auc_best["auc"]:
            auc_best.update(auc=m["auc"], epoch=ep, metrics=m)
            if not track_f1:
                best.update(key=None, epoch=ep, metrics=m,
                            state={k: v.detach().clone() for k, v in model.state_dict().items()})
            noimp = 0
        else:
            noimp += 1
            if noimp >= PATIENCE:
                break
    return _finish(model, best, auc_best, cfg, seed, pw, device, select, n_params, t0,
                   history, mean, std, out_dir)


def cached_run(train_fn, run_dir, cfg, seed, device, data, pos_weight=None, select="f1"):
    """Skip-if-final.json-exists caching (issue8 pattern) — interruption-safe sweeps."""
    run_dir = Path(run_dir)
    fj = run_dir / "final.json"
    if fj.exists():
        return json.loads(fj.read_text())
    return train_fn(cfg, seed, device, data, pos_weight=pos_weight, select=select,
                    out_dir=run_dir)

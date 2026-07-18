"""06_rnn_loso.py — RNN study Phase R4: 6-fold Leave-One-Set-Out CV (generalization check).

Mirrors Issue-5's protocol exactly (journal_prep/issue5_loso_cv/05_loso_cv.py): for each PIE
set held out as the test fold, the other 5 sets are split 85/15 by PEDESTRIAN (grouped), with
per-fold train-only normalization and per-fold pos_weight = n_neg/n_pos. Selection on val AUC
(so the held-out AUC is directly comparable to the BiLSTM's LOSO 0.928, the GRU's 0.946, and
the transformer's 0.939 — all AUC-selected).

TRAINING is done by the unified engine's train_run (family birnn, the F1-winner config) — NO
loop is duplicated. The engine ignores test, so the held-out set is scored separately from the
saved checkpoint. Seed 42. CPU.

Assert the 6 fold test_n = {set01 258, set02 310, set03 2094, set04 1610, set05 47, set06 587}
(Issue-5 fingerprint of genuine, untampered sequences_clean data).

Run from the repo root:  python rnn/phase4_final/06_rnn_loso.py
"""
import csv
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SEQ_DIR = ROOT / "journal_prep" / "issue2_clean_protocol" / "sequences_clean"
SEARCH = ROOT / "rnn" / "phase2_search"
RUNS_LOSO = HERE / "runs_loso"
CPU = torch.device("cpu")

SETS = ["set01", "set02", "set03", "set04", "set05", "set06"]
EXPECT_TEST_N = {"set01": 258, "set02": 310, "set03": 2094, "set04": 1610,
                 "set05": 47, "set06": 587}
VAL_FRAC = 0.15


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


engine = _load("rnn_engine", ROOT / "journal_prep" / "issue12_unified_pipeline" / "12_unified_engine.py")
F1_WINNER_ID = json.loads((SEARCH / "_stage_summary.json").read_text())["f1_winner"]
F1_WINNER_CFG = json.loads(
    (SEARCH / "runs_search" / F1_WINNER_ID / "seed42.json").read_text())["cfg"]


def load_all():
    X = np.load(SEQ_DIR / "X.npy").astype(np.float32)
    y = np.load(SEQ_DIR / "y.npy").astype(np.float32)
    meta = pickle.load(open(SEQ_DIR / "meta.pkl", "rb"))
    sid = np.array([r["set_id"] for r in meta])
    pid = np.array([f"{r['set_id']}/{r['ped_id']}" for r in meta])
    return X, y, sid, pid


def evaluate(run_dir, X, y):
    """Score a held-out set from a saved engine checkpoint (CPU, train-only norm applied)."""
    mean = np.load(run_dir / "norm_mean.npy")
    std = np.load(run_dir / "norm_std.npy")
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = engine.MODEL_REGISTRY["birnn"](F1_WINNER_CFG)
    model.load_state_dict(ck["model"]); model.eval()
    with torch.no_grad():
        Xn = ((X - mean) / std).astype(np.float32)
        p = torch.sigmoid(model(torch.from_numpy(Xn)).squeeze(-1)).numpy()
    pred = (p >= 0.5).astype(int)
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                f1=f1_score(y, pred, zero_division=0), acc=accuracy_score(y, pred),
                prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0))


def fold(test_set, X, y, sid, pid, seed=42):
    te = sid == test_set
    pool = ~te
    pool_peds = np.unique(pid[pool])
    rng = np.random.default_rng(seed); rng.shuffle(pool_peds)
    n_val = max(1, int(round(len(pool_peds) * VAL_FRAC)))
    val_peds = set(pool_peds[:n_val].tolist())
    is_val = np.array([p in val_peds for p in pid])
    va, tr = pool & is_val, pool & ~is_val

    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    Xte, yte = X[te], y[te]
    pw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))

    run_dir = RUNS_LOSO / test_set
    if not ((run_dir / "best.pt").exists()):
        # engine trains + selects on val (AUC checkpoint), saves best.pt + norm; ignores test
        engine.train_run("birnn", F1_WINNER_CFG, seed, CPU,
                         (Xtr, ytr, Xva, yva, Xte, yte),
                         pos_weight=pw, select="auc", out_dir=run_dir)
    fj = json.loads((run_dir / "final.json").read_text())
    m = evaluate(run_dir, Xte, yte)
    m.update(test_set=test_set, test_n=int(te.sum()), test_pos=float(yte.mean()),
             train_n=int(tr.sum()), val_n=int(va.sum()), pos_weight=round(pw, 3),
             best_epoch=fj["best_epoch"], val_auc=round(fj["val"]["auc"], 4))
    assert m["test_n"] == EXPECT_TEST_N[test_set], \
        f"{test_set}: test_n {m['test_n']} != {EXPECT_TEST_N[test_set]} (data fingerprint fail)"
    (run_dir / "loso.json").write_text(json.dumps(m, indent=2))
    return m


def main():
    X, y, sid, pid = load_all()
    print(f"LOSO on rnn_f1_winner {F1_WINNER_CFG} | seed 42 | CPU\n")
    print(f"{'fold':8s} {'testN':>6s} {'pos%':>5s} {'AUC':>6s} {'PR':>6s} {'F1':>6s} "
          f"{'Acc':>6s} {'pw':>5s} {'ep':>3s}")
    print("-" * 60)
    rows = []
    for s in SETS:
        m = fold(s, X, y, sid, pid)
        rows.append(m)
        print(f"{s:8s} {m['test_n']:6d} {m['test_pos']*100:5.1f} {m['auc']:6.3f} "
              f"{m['pr_auc']:6.3f} {m['f1']:6.3f} {m['acc']:6.3f} {m['pos_weight']:5.2f} "
              f"{m['best_epoch']:3d}")

    aucs = np.array([m["auc"] for m in rows])
    f1s = np.array([m["f1"] for m in rows])
    big = [m for m in rows if m["test_n"] >= 100]
    abig = np.array([m["auc"] for m in big])
    print("-" * 60)
    print(f"{'MEAN':8s} {'':>6s} {'':>5s} {aucs.mean():6.3f} {'':>6s} {f1s.mean():6.3f}   "
          f"(AUC std ddof=1 {aucs.std(ddof=1):.3f}; excl set05 {abig.mean():.3f})")

    cols = ["test_set", "test_n", "test_pos", "train_n", "val_n", "pos_weight",
            "best_epoch", "val_auc", "auc", "pr_auc", "f1", "acc", "prec", "rec"]
    with open(HERE / "06_loso_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for m in rows:
            w.writerow({c: m[c] for c in cols})
    (HERE / "06_loso_summary.json").write_text(json.dumps(dict(
        cfg=F1_WINNER_CFG, folds=rows,
        auc_mean=float(aucs.mean()), auc_std=float(aucs.std(ddof=1)),
        auc_mean_excl_set05=float(abig.mean()), auc_std_excl_set05=float(abig.std(ddof=1)),
        f1_mean=float(f1s.mean()), f1_std=float(f1s.std(ddof=1))), indent=2))
    print("\nwrote 06_loso_results.csv, 06_loso_summary.json, runs_loso/<set>/")


if __name__ == "__main__":
    main()

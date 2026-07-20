"""02_train_pie_full.py — PIE-side of the PSI cross-test: retrain RNN-F1 on the FULL PIE dataset.

Since PSI 2.0 is the external test set, we no longer hold out PIE set03. Split all six PIE sets:
  train = set01/02/03/04   (set03 folded in — maximize training data)
  val   = set05/06         (unchanged; used for early-stop, F1 checkpoint, and tau*)
No PIE test split. Trains the F1-optimised vanilla RNN (birnn) at OW 16/32/64, 5 seeds, CPU, via the
frozen unified engine. Saves each model + its PIE train-set norm_mean/std + per-seed tau* — the
artifacts 03_eval_on_psi.py will load to score PSI zero-shot (PSI standardized with PIE stats).

Run: python journal_prep/obs_window_extension/psi_crosstest/02_train_pie_full.py
"""
import csv
import importlib.util
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OWX = HERE.parent                       # obs_window_extension/
ROOT = OWX.parent.parent                # repo root


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load("engine12", ROOT / "journal_prep" / "issue12_unified_pipeline" / "12_unified_engine.py")
C = _load("f1_common", ROOT / "f1_optimization" / "00_common.py")

CPU = torch.device("cpu")
SEEDS = [42, 0, 1, 2, 3]
TRAIN_SETS = {"set01", "set02", "set03", "set04"}   # set03 folded in (PSI is the test)
VAL_SETS = {"set05", "set06"}

# The F1-optimised vanilla RNN recipe (rnn_f1_winner); pos_weight held at its recipe value.
RNN_F1 = dict(key="birnn", label="Vanilla-RNN-F1",
              cfg=dict(lr=1e-4, dropout=0.2, hidden=256, num_layers=2), pw=1.682)

# OW -> clean-sequence dir (all six sets present in each)
SEQ_DIRS = {
    16: ROOT / "journal_prep" / "issue2_clean_protocol" / "sequences_clean",
    32: OWX / "seq_ow32",
    64: OWX / "seq_ow64",
}


def load_full_pie(W):
    d = SEQ_DIRS[W]
    X = np.load(d / "X.npy").astype(np.float32)
    y = np.load(d / "y.npy").astype(np.float32)
    meta = pickle.load(open(d / "meta.pkl", "rb"))
    sid = np.array([m["set_id"] for m in meta])
    assert X.shape[1] == W and X.shape[2] == 5, f"bad shape {X.shape} for OW{W}"
    tr = np.isin(sid, sorted(TRAIN_SETS))
    va = np.isin(sid, sorted(VAL_SETS))
    return X[tr], y[tr], X[va], y[va]


def main():
    out_root = HERE / "models_pie_full" / "rnn_f1"
    results = {}
    t0 = time.time()

    for W in (16, 32, 64):
        Xtr, ytr, Xva, yva = load_full_pie(W)
        # train_run expects a 6-tuple (test slot ignored by the val-only loop)
        data = (Xtr, ytr, Xva, yva, Xva[:0], yva[:0])
        tr_pos = int(ytr.sum())
        print(f"\n{'='*72}\nOW {W}: full-PIE train {len(ytr)} (pos {tr_pos}, {ytr.mean()*100:.1f}%) "
              f"/ val {len(yva)} (pos {int(yva.sum())})\n{'='*72}")

        per_seed = []
        for s in SEEDS:
            run_dir = out_root / f"ow{W}" / f"seed{s}"
            r = E.train_run(RNN_F1["key"], RNN_F1["cfg"], s, CPU, data,
                            pos_weight=RNN_F1["pw"], select="f1", out_dir=run_dir)
            # tau* on val (carried over to PSI, never re-tuned there)
            pf = C.prob_fn_from_run_dir(run_dir, E.MODEL_REGISTRY[RNN_F1["key"]](RNN_F1["cfg"]))
            pv = pf(Xva)
            tau = C.best_threshold(yva, pv)
            mv = C.metrics_at(yva, pv, tau)
            (run_dir / "tau_star.json").write_text(json.dumps({"tau": float(tau)}))
            per_seed.append(dict(seed=s, tau=float(tau), n_params=r["n_params"],
                                 best_epoch=r["best_epoch"], seconds=r["seconds"],
                                 val_f1=mv["f1"], val_acc=mv["acc"], val_auc=mv["auc"]))
            print(f"  RNN-F1 OW{W} seed{s:<3d} tau*={tau:.3f} "
                  f"valF1={mv['f1']:.4f} valAUC={mv['auc']:.4f} ({r['seconds']}s ep{r['best_epoch']})")

        f1s = np.array([p["val_f1"] for p in per_seed])
        aucs = np.array([p["val_auc"] for p in per_seed])
        results[f"ow{W}"] = dict(
            cfg=RNN_F1["cfg"], pos_weight=RNN_F1["pw"], n_params=per_seed[0]["n_params"],
            n_train=len(ytr), n_val=len(yva), train_pos=tr_pos,
            per_seed=per_seed, val_f1_mean=float(f1s.mean()), val_f1_std=float(f1s.std(ddof=1)),
            val_auc_mean=float(aucs.mean()), model_dir=str((out_root / f"ow{W}").relative_to(ROOT)))
        print(f"  -> OW{W}: val F1 {f1s.mean():.4f}±{f1s.std(ddof=1):.4f}  val AUC {aucs.mean():.4f}")

    (HERE / "02_pie_full_results.json").write_text(json.dumps(results, indent=2))
    with open(HERE / "02_pie_full_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "n_train", "n_val", "params", "pos_weight",
                    "val_f1_mean", "val_f1_std", "val_auc_mean"])
        for W in (16, 32, 64):
            r = results[f"ow{W}"]
            w.writerow([W, r["n_train"], r["n_val"], r["n_params"], r["pos_weight"],
                        round(r["val_f1_mean"], 4), round(r["val_f1_std"], 4),
                        round(r["val_auc_mean"], 4)])
    print(f"\nTOTAL {time.time()-t0:.0f}s. models in models_pie_full/rnn_f1/, "
          f"summary in 02_pie_full_results.json/.csv — ready for PSI eval.")


if __name__ == "__main__":
    main()

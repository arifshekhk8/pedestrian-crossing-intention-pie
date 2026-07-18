"""04_gru_final.py — GRU study Phase G4: final training (val-only, produces checkpoints).

Trains the G3-confirmed arms x 5 seeds through the ONE unified engine, saving full run dirs
(best.pt + norm stats + final.json) needed for the paired comparison. VAL-ONLY: the engine
has no test code path — test set03 is touched only by 05_gru_test_eval.py, and only after this.

Arms (user-confirmed G3 gate, 2026-07-14 — "F1-winner + default only"):
  gru_f1_winner   lr5e-04_do0.3_h256_nl2  --select f1   (headline GRU)
  gru_default_f1  lr1e-03_do0.3_h128_nl2  --select f1   (un-searched-GRU control)
  gru_default_auc lr1e-03_do0.3_h128_nl2  --select auc  (AUC twin of the frozen BiLSTM)
pos_weight 1.682, seeds [42,0,1,2,3], CPU.

The F1-winner cfg is read from its raw search JSON (not hand-parsed from the cfg_id) —
transformer/phase4 discipline.

Run from the repo root:  python gru/phase4_final/04_gru_final.py
"""
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = HERE / "runs_final"
CPU = torch.device("cpu")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


engine = _load("gru_engine", ROOT / "journal_prep" / "issue12_unified_pipeline" / "12_unified_engine.py")

SEEDS = engine.SEEDS
POS_WEIGHT = engine.POS_WEIGHT   # 1.682

# F1-winner cfg from its raw search JSON (no hand-parsing)
F1_WINNER_ID = "lr5e-04_do0.3_h256_nl2"
_wjson = ROOT / "gru" / "phase2_search" / "runs_search" / F1_WINNER_ID / "seed42.json"
F1_WINNER_CFG = json.loads(_wjson.read_text())["cfg"]
DEFAULT_CFG = dict(engine.GRU_DEFAULT_CFG)

ARMS = [
    ("gru_f1_winner", F1_WINNER_CFG, "f1"),
    ("gru_default_f1", DEFAULT_CFG, "f1"),
    ("gru_default_auc", DEFAULT_CFG, "auc"),
]


def main():
    assert F1_WINNER_CFG == dict(lr=5e-4, dropout=0.3, hidden=256, num_layers=2), \
        f"F1-winner cfg drifted: {F1_WINNER_CFG}"
    data = engine.load_splits()
    print(f"device cpu | pos_weight {POS_WEIGHT} | seeds {SEEDS}")
    print(f"F1-winner cfg: {F1_WINNER_CFG}\n")

    for arm, cfg, select in ARMS:
        print(f"=== {arm}  (select={select}, cfg={cfg}) ===")
        for seed in SEEDS:
            out_dir = RUNS / arm / f"seed{seed}"
            if (out_dir / "best.pt").exists() and (out_dir / "final.json").exists():
                r = json.loads((out_dir / "final.json").read_text())
                print(f"  seed {seed:2d} [cached] val F1 {r['val']['f1']:.4f} "
                      f"AUC {r['val']['auc']:.4f}")
                continue
            r = engine.train_run("gru", cfg, seed, CPU, data, pos_weight=POS_WEIGHT,
                                 select=select, out_dir=out_dir)
            assert "test" not in r, "engine must not evaluate test"
            print(f"  seed {seed:2d} val F1 {r['val']['f1']:.4f}  acc {r['val']['acc']:.4f}  "
                  f"AUC {r['val']['auc']:.4f}  (ep {r['best_epoch']}, {r['seconds']:.0f}s)")
        print()
    print("done — checkpoints in runs_final/. Test still UNTOUCHED (05_gru_test_eval.py next).")


if __name__ == "__main__":
    main()

"""04_rnn_final.py — RNN study Phase R4: final training (val-only, produces checkpoints).

Trains the R3-confirmed arms x 5 seeds through the ONE unified engine, saving full run dirs
(best.pt + norm stats + final.json) needed for the paired comparison. VAL-ONLY: the engine
has no test code path — test set03 is touched only by 05_rnn_test_eval.py, and only after this.

Arms (pre-registered; final set confirmed at the R3 human checkpoint, 2026-07-14 — "add the
AUC-selected winner": since the search's F1-winner and AUC-winner are the SAME config, an
AUC-selected large-RNN arm is essentially free and closes the "no AUC-tuned large model" gap
the GRU study had to flag):
  rnn_f1_winner   <search F1-winner>       --select f1   (headline RNN, primary F1 comparison)
  rnn_winner_auc  <search F1-winner cfg>   --select auc  (dedicated AUC-optimized h256)
  rnn_default_f1  lr1e-03_do0.3_h128_nl2   --select f1   (un-searched-RNN control)
  rnn_default_auc lr1e-03_do0.3_h128_nl2   --select auc  (matched-size AUC twin of frozen BiLSTM)
seeds [42,0,1,2,3], CPU. The winner arms use the search's chosen pos_weight; the default arms
use the frozen 1.682 (the BiLSTM-recipe anchor).

The F1-winner cfg and chosen pos_weight are read from the search's `_stage_summary.json`
(the id) + the winner's raw seed42.json (the cfg dict) — no hand-parsing (transformer/gru
phase4 discipline).

Run from the repo root:  python rnn/phase4_final/04_rnn_final.py
"""
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = HERE / "runs_final"
SEARCH = ROOT / "rnn" / "phase2_search"
CPU = torch.device("cpu")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


engine = _load("rnn_engine", ROOT / "journal_prep" / "issue12_unified_pipeline" / "12_unified_engine.py")

SEEDS = engine.SEEDS
ANCHOR_PW = engine.POS_WEIGHT   # 1.682

_summary = json.loads((SEARCH / "_stage_summary.json").read_text())
F1_WINNER_ID = _summary["f1_winner"]
CHOSEN_PW = float(_summary["chosen_pw"])
F1_WINNER_CFG = json.loads(
    (SEARCH / "runs_search" / F1_WINNER_ID / "seed42.json").read_text())["cfg"]
DEFAULT_CFG = dict(engine.PRESETS["birnn"])   # lr1e-3/do0.3/h128/nl2

# arm -> (cfg, select, pos_weight)
ARMS = [
    ("rnn_f1_winner", F1_WINNER_CFG, "f1", CHOSEN_PW),
    ("rnn_winner_auc", F1_WINNER_CFG, "auc", CHOSEN_PW),
    ("rnn_default_f1", DEFAULT_CFG, "f1", ANCHOR_PW),
    ("rnn_default_auc", DEFAULT_CFG, "auc", ANCHOR_PW),
]


def main():
    data = engine.load_splits()
    print(f"device cpu | seeds {SEEDS} | family birnn")
    print(f"F1-winner id: {F1_WINNER_ID}  cfg: {F1_WINNER_CFG}  (pos_weight {CHOSEN_PW:g})")
    print(f"default cfg: {DEFAULT_CFG}  (pos_weight {ANCHOR_PW:g})\n")

    for arm, cfg, select, pw in ARMS:
        print(f"=== {arm}  (select={select}, pos_weight={pw:g}, cfg={cfg}) ===")
        for seed in SEEDS:
            out_dir = RUNS / arm / f"seed{seed}"
            if (out_dir / "best.pt").exists() and (out_dir / "final.json").exists():
                r = json.loads((out_dir / "final.json").read_text())
                print(f"  seed {seed:2d} [cached] val F1 {r['val']['f1']:.4f} "
                      f"AUC {r['val']['auc']:.4f}")
                continue
            r = engine.train_run("birnn", cfg, seed, CPU, data, pos_weight=pw,
                                 select=select, out_dir=out_dir)
            assert "test" not in r, "engine must not evaluate test"
            print(f"  seed {seed:2d} val F1 {r['val']['f1']:.4f}  acc {r['val']['acc']:.4f}  "
                  f"AUC {r['val']['auc']:.4f}  (ep {r['best_epoch']}, {r['seconds']:.0f}s)")
        print()
    print("done — checkpoints in runs_final/. Test still UNTOUCHED (05_rnn_test_eval.py next).")


if __name__ == "__main__":
    main()

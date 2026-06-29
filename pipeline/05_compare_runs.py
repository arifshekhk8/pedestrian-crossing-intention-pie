"""
05_compare_runs.py — Day 6: comparison table for supervisor.

Reads final.json from each run and prints a clean side-by-side table.
Run this after both training scripts have finished.
"""

import json
from pathlib import Path

RUNS = {
    "BiLSTM  bbox+speed (5D)": "paper_and_artifacts/runs/bilstm_baseline/final.json",
    "BiLSTM  bbox-only  (4D)": "paper_and_artifacts/runs/bilstm_bbox_only/final.json",
}

metrics = ["acc", "f1", "auc", "prec", "rec"]
header  = f"{'Model':<30} {'Epoch':>6} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7}"

print("\n" + "=" * 75)
print("  Day 6 Comparison — Bbox+Speed vs Bbox-Only (test set: set03)")
print("=" * 75)
print(header)
print("-" * 75)

for name, path in RUNS.items():
    p = Path(path)
    if not p.exists():
        print(f"{name:<30}   [not found — run training script first]")
        continue
    with open(p) as f:
        d = json.load(f)
    t = d["test"]
    print(f"{name:<30} {d['best_epoch']:>6} "
          f"{t['acc']:>7.3f} {t['f1']:>7.3f} {t['auc']:>7.3f} "
          f"{t['prec']:>7.3f} {t['rec']:>7.3f}")

print("=" * 75)
print("\nConclusion hint:")
print("  If AUC(5D) - AUC(4D) >= 0.01 → ego-speed adds meaningful signal.")
print("  If delta < 0.01              → bbox motion alone is highly predictive.\n")
"""02_rerank_searches.py — Phase F2: re-rank both cached searches by val F1 (PLAN.md §5).

Reads ONLY the cached val-only JSONs of the two existing searches (no training, no test):
- LSTM Issue-8 grid: journal_prep/issue8_grid_search/runs_grid/<cfgid>/seed<k>.json
- Transformer staged search: transformer/phase2_kaggle_search/runs_search/<cfgid>/seed<k>.json

Ranking key = the supervisor hierarchy: val F1, tie-break val acc, then val AUC.
Names the LSTM shortlist (seed-42 F1 top-5 UNION the configs that already have 5-seed
cache) and which of those need completion runs — written machine-readably to
02_shortlist.json for 03_lstm_shortlist.py.

Outputs: 02_rerank_lstm.csv, 02_rerank_transformer.csv, 02_rerank_report.md, 02_shortlist.json
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("f1_common", HERE / "00_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

TOPK = 5
KEY = lambda v: (round(v["f1"], 10), round(v["acc"], 10), v["auc"])


def load_pool(root):
    """{cfg_id: {seed: json}} for every cached run under root."""
    pool = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for jp in sorted(d.glob("seed*.json")):
            pool.setdefault(d.name, {})[int(jp.stem[4:])] = json.loads(jp.read_text())
    return pool


def rank_seed42(pool):
    rows = [(cid, js[42]) for cid, js in pool.items() if 42 in js]
    rows.sort(key=lambda r: KEY(r[1]["val"]), reverse=True)
    return rows


def five_seed_stats(pool):
    out = {}
    for cid, js in pool.items():
        if len(js) == len(C.SEEDS):
            f1s = np.array([js[s]["val"]["f1"] for s in C.SEEDS])
            accs = np.array([js[s]["val"]["acc"] for s in C.SEEDS])
            aucs = np.array([js[s]["val"]["auc"] for s in C.SEEDS])
            out[cid] = dict(f1_mean=f1s.mean(), f1_std=f1s.std(ddof=1),
                            acc_mean=accs.mean(), auc_mean=aucs.mean())
    return out


def write_csv(path, ranked, extra_cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["f1_rank", "cfg_id", "val_f1", "val_acc", "val_auc"] + extra_cols)
        for i, (cid, r) in enumerate(ranked, 1):
            w.writerow([i, cid, round(r["val"]["f1"], 4), round(r["val"]["acc"], 4),
                        round(r["val"]["auc"], 4)] +
                       [r.get(c, "") for c in extra_cols])


def main():
    lstm_pool = load_pool(C.LSTM_GRID)
    tf_pool = load_pool(C.TF_SEARCH)

    lstm_ranked = rank_seed42(lstm_pool)
    tf_ranked = rank_seed42(tf_pool)
    lstm_5s = five_seed_stats(lstm_pool)
    tf_5s = five_seed_stats(tf_pool)

    write_csv(HERE / "02_rerank_lstm.csv", lstm_ranked, ["lr", "dropout", "hidden", "num_layers"])
    write_csv(HERE / "02_rerank_transformer.csv", tf_ranked, [])

    # ---- LSTM shortlist: seed-42 F1 top-5 UNION configs with existing 5-seed cache ----
    top5 = [cid for cid, _ in lstm_ranked[:TOPK]]
    shortlist = sorted(set(top5) | set(lstm_5s), key=lambda c: -dict(lstm_ranked)[c]["val"]["f1"]
                       if c in dict(lstm_ranked) else 0)
    need_completion = [c for c in shortlist if c not in lstm_5s]
    shortlist_cfgs = {cid: {k: lstm_pool[cid][42][k] for k in ("lr", "dropout", "hidden", "num_layers")}
                      for cid in shortlist}
    (HERE / "02_shortlist.json").write_text(json.dumps(dict(
        shortlist=shortlist, need_completion=need_completion, cfgs=shortlist_cfgs,
        seeds=C.SEEDS), indent=2))

    # ---- transformer: verify the frozen winner is also the F1 pick (PLAN.md §3.2) ----
    tf_winner_id = "d128_ff512_L4_last_spe__adam_lr1e-03_plateau_do0.1_wd1e-05"
    tf_f1_rank1 = tf_ranked[0][0]
    tf_5s_leader = max(tf_5s, key=lambda c: tf_5s[c]["f1_mean"])
    tf_unchanged = (tf_f1_rank1 == tf_winner_id) and (tf_5s_leader == tf_winner_id)

    # ---- report ----
    L = ["# 02 — F1 re-ranking of both cached searches (val-only JSONs; no training)", "",
         "Ranking key: **val F1 -> val acc -> val AUC** (supervisor hierarchy).", "",
         "## Transformer (78 configs, seed-42) — top 10", "",
         "| F1 rank | cfg | val F1 | val acc | val AUC |", "|---|---|---|---|---|"]
    for i, (cid, r) in enumerate(tf_ranked[:10], 1):
        L.append(f"| {i} | `{cid}` | {r['val']['f1']:.4f} | {r['val']['acc']:.4f} | {r['val']['auc']:.4f} |")
    L += ["", "5-seed candidates by mean val F1:", "",
          "| cfg | val F1 (5-seed) | val AUC mean |", "|---|---|---|"]
    for cid in sorted(tf_5s, key=lambda c: -tf_5s[c]["f1_mean"]):
        s = tf_5s[cid]
        L.append(f"| `{cid}` | {s['f1_mean']:.4f} ± {s['f1_std']:.4f} | {s['auc_mean']:.4f} |")
    L += ["", f"**Transformer config decision: {'UNCHANGED' if tf_unchanged else 'CHANGED'}.** "
          f"The frozen AUC winner `{tf_winner_id}` is seed-42 F1 rank "
          f"{[c for c, _ in tf_ranked].index(tf_winner_id) + 1} and "
          f"{'ALSO the' if tf_5s_leader == tf_winner_id else 'NOT the'} 5-seed val-F1 leader "
          f"(`{tf_5s_leader}`). "
          + ("The searched architecture+recipe carries over to the F1-first program as-is; "
             "only pos_weight and the checkpoint rule are re-optimized (PLAN.md §3.3-3.4)."
             if tf_unchanged else
             "AMENDMENT REQUIRED: the pre-registered plan assumed no change — document before proceeding."),
          "",
          "## LSTM (36 configs, seed-42) — top 8", "",
          "| F1 rank | cfg | val F1 | val acc | val AUC |", "|---|---|---|---|---|"]
    for i, (cid, r) in enumerate(lstm_ranked[:8], 1):
        L.append(f"| {i} | `{cid}` | {r['val']['f1']:.4f} | {r['val']['acc']:.4f} | {r['val']['auc']:.4f} |")
    base_rank = [c for c, _ in lstm_ranked].index(C.lstm_cfg_id(C.BASELINE_LSTM_CFG)) + 1
    L += ["", f"(baseline `{C.lstm_cfg_id(C.BASELINE_LSTM_CFG)}` sits at F1 rank {base_rank}.)", "",
          "Existing 5-seed candidates by mean val F1:", "",
          "| cfg | val F1 (5-seed) | val AUC mean |", "|---|---|---|"]
    for cid in sorted(lstm_5s, key=lambda c: -lstm_5s[c]["f1_mean"]):
        s = lstm_5s[cid]
        L.append(f"| `{cid}` | {s['f1_mean']:.4f} ± {s['f1_std']:.4f} | {s['auc_mean']:.4f} |")
    L += ["", "## LSTM shortlist (PLAN.md §5 step 2)", "",
          f"Shortlist = seed-42 F1 top-{TOPK} UNION existing 5-seed configs "
          f"({len(shortlist)} configs). Missing seeds to complete (AUC protocol, val-only): ", ""]
    for cid in shortlist:
        tag = "**needs completion runs**" if cid in need_completion else "5-seed cache present"
        L.append(f"- `{cid}` — {tag}")
    L += ["", "Machine-readable: `02_shortlist.json` (consumed by `03_lstm_shortlist.py`).", ""]
    (HERE / "02_rerank_report.md").write_text("\n".join(L))

    print(f"transformer unchanged: {tf_unchanged}")
    print(f"LSTM shortlist ({len(shortlist)}): {shortlist}")
    print(f"need completion: {need_completion}")
    print("wrote 02_rerank_lstm.csv, 02_rerank_transformer.csv, 02_rerank_report.md, 02_shortlist.json")


if __name__ == "__main__":
    main()

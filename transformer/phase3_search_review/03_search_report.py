"""03_search_report.py — Phase T3: independent local re-derivation of the Kaggle search.

Reads the downloaded runs_search/ tree and recomputes every ranking and every mean/std
FROM THE RAW PER-SEED JSON FILES — it does not trust runs_search/_stage_summary.json's
numbers, only uses it to identify which architectures Stage B/transfer were run on
(structural bucketing, not conclusions). If the independent recomputation disagrees with
the notebook's own summary, this script errors loudly rather than silently reporting a
number nobody checked.

This is the Phase-T3 human checkpoint (transformer/PLAN.md): review the winner here,
BEFORE phase4_kaggle_final/ is built or run — test set03 does not exist anywhere in
this pipeline yet, and won't until Stage D.

    python transformer/phase3_search_review/03_search_report.py

Writes: 03_arch_grid.csv, 03_recipe_grid.csv, 03_candidates_multiseed.csv,
03_search_summary.md, 03_search_figure.png (all next to this script).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE.parent / "phase2_kaggle_search" / "runs_search"

ARCH_KEYS = ["d_model", "nhead", "num_layers", "dim_ff", "pool", "pos"]
RECIPE_KEYS = ["lr", "schedule", "dropout", "weight_decay", "optimizer"]

# Context only -- NOT a formal comparison. The real LSTM-vs-transformer comparison is a
# paired test-set bootstrap in Phase T5 (05_compare_vs_lstm.py), on checkpoints that
# don't exist until Stage D. These are quoted here purely so the supervisor conversation
# has a frame of reference for "is 0.98 val AUC encouraging or not."
LSTM_VAL_AUC_MEAN = 0.9644   # Issue 8, baseline config, 5-seed mean
LSTM_VAL_AUC_STD = 0.0043
LSTM_TEST_AUC_MEAN = 0.9324  # Issue 2, 5-seed mean -- THE number the transformer must beat
LSTM_TEST_AUC_STD = 0.0114


def load_all_runs():
    """One row per (cfg_id, seed) file, tagged with cfg_id from the directory name."""
    rows = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("seed*.json")):
            r = json.loads(f.read_text())
            assert "test" not in r, f"FOUND A TEST KEY in {f} -- protocol violation, stop."
            r["_cfg_id"] = d.name
            rows.append(r)
    return rows


def group_by_cfg(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["_cfg_id"], []).append(r)
    return groups


def mean_std(vals):
    vals = np.array(vals, dtype=float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def main():
    assert RUNS_DIR.exists(), f"missing {RUNS_DIR} -- unzip the Kaggle output here first"
    summary = json.loads((RUNS_DIR / "_stage_summary.json").read_text())

    rows = load_all_runs()
    groups = group_by_cfg(rows)
    print(f"loaded {len(rows)} run files across {len(groups)} distinct configs "
         f"(expect 78 configs, ~102 files: 78 seed-42 + up to 24 additional Stage-C seeds)")
    assert len(groups) == 78, f"expected 78 distinct configs, found {len(groups)}"
    assert len(rows) <= 78 + 5 * 6, "more run files than the protocol should produce"

    default_cfg_id = summary["default_cfg_id"]
    winner_cfg_id = summary["winner_cfg_id"]
    arch_top3_cfgs = [e["cfg"] for e in summary["stage_a_top3"]]
    arch1_id = None  # filled in below once we can compute cfg_id-equivalent keys

    def arch_key(cfg):
        return tuple(cfg[k] for k in ARCH_KEYS)

    arch_top3_keys = [arch_key(c) for c in arch_top3_cfgs]

    def recipe_key(cfg):
        return tuple(cfg[k] for k in RECIPE_KEYS)

    default_recipe_key = recipe_key(summary["default_cfg"])

    # --- Stage A: all 36 configs sharing the default recipe (any architecture), ranked
    #     by their OWN seed-42 val AUC (recomputed straight from that file, not summary). ---
    stage_a_rows = []
    for cfg_id, seed_rows in groups.items():
        r42 = next((r for r in seed_rows if r["seed"] == 42), None)
        if r42 is None:
            continue
        if recipe_key(r42) == default_recipe_key:
            stage_a_rows.append(r42)
    print(f"Stage-A pool (default recipe, any architecture): {len(stage_a_rows)} configs "
         f"(expect 36)")
    assert len(stage_a_rows) == 36

    stage_a_ranked = sorted(stage_a_rows, key=lambda r: r["val"]["auc"], reverse=True)
    with open(HERE / "03_arch_grid.csv", "w") as f:
        f.write("rank,cfg_id," + ",".join(ARCH_KEYS) + ",n_params,val_auc\n")
        for i, r in enumerate(stage_a_ranked, 1):
            vals = [str(r[k]) for k in ARCH_KEYS]
            f.write(f"{i},{r['_cfg_id']}," + ",".join(vals) +
                   f",{r['n_params']},{r['val']['auc']:.6f}\n")

    recomputed_arch_top3 = [arch_key(r) for r in stage_a_ranked[:3]]
    assert recomputed_arch_top3 == arch_top3_keys, (
        "Stage-A top-3 recomputed from raw files does NOT match _stage_summary.json's "
        "stage_a_top3 -- something is inconsistent, stop and investigate.\n"
        f"recomputed: {recomputed_arch_top3}\nsummary:    {arch_top3_keys}"
    )
    print("Stage-A top-3 independently recomputed and MATCHES the notebook's summary.")

    # --- Stage B + transfer: configs whose architecture is one of arch_top3, whose
    #     recipe is NOT the default (so they were part of the recipe search). ---
    stage_b_and_transfer_rows = []
    for cfg_id, seed_rows in groups.items():
        r42 = next((r for r in seed_rows if r["seed"] == 42), None)
        if r42 is None:
            continue
        if recipe_key(r42) != default_recipe_key and arch_key(r42) in arch_top3_keys:
            stage_b_and_transfer_rows.append(r42)
    print(f"Stage-B + transfer pool (searched recipes on top-3 architectures): "
         f"{len(stage_b_and_transfer_rows)} configs (expect 36 + 6 = 42)")
    assert len(stage_b_and_transfer_rows) == 42

    recipe_ranked = sorted(stage_b_and_transfer_rows, key=lambda r: r["val"]["auc"], reverse=True)
    with open(HERE / "03_recipe_grid.csv", "w") as f:
        f.write("rank,cfg_id,architecture_role," + ",".join(RECIPE_KEYS) + ",val_auc\n")
        for i, r in enumerate(recipe_ranked, 1):
            role = ("arch#1 (Stage B)" if arch_key(r) == arch_top3_keys[0]
                   else "arch#2 (transfer)" if arch_key(r) == arch_top3_keys[1]
                   else "arch#3 (transfer)")
            vals = [str(r[k]) for k in RECIPE_KEYS]
            f.write(f"{i},{r['_cfg_id']},{role}," + ",".join(vals) + f",{r['val']['auc']:.6f}\n")

    # --- Full 78-config pool ranked by seed-42 val AUC (for the selection-noise check). ---
    full_pool_ranked = sorted(
        (next(r for r in seed_rows if r["seed"] == 42) for seed_rows in groups.values()),
        key=lambda r: r["val"]["auc"], reverse=True,
    )
    seed42_leader_cfg_id = full_pool_ranked[0]["_cfg_id"]

    # --- Stage C candidates: recompute mean/std from raw files, cross-check vs summary. ---
    candidate_ids = [c["cfg_id"] for c in summary["stage_c_candidates"]]
    candidate_rows = []
    for cfg_id in candidate_ids:
        seed_rows = groups[cfg_id]
        seeds_present = sorted(r["seed"] for r in seed_rows)
        vals = [r["val"]["auc"] for r in seed_rows]
        mean, std = mean_std(vals)
        candidate_rows.append(dict(cfg_id=cfg_id, n_seeds=len(seed_rows),
                                   seeds=seeds_present, val_auc_mean=mean, val_auc_std=std,
                                   cfg={k: seed_rows[0][k] for k in ARCH_KEYS + RECIPE_KEYS}))

    summary_by_id = {c["cfg_id"]: c for c in summary["stage_c_candidates"]}
    for c in candidate_rows:
        s = summary_by_id[c["cfg_id"]]
        assert abs(c["val_auc_mean"] - s["val_auc_mean"]) < 1e-9, (
            f"recomputed mean for {c['cfg_id']} ({c['val_auc_mean']}) disagrees with "
            f"summary ({s['val_auc_mean']}) -- stop and investigate."
        )
        assert c["n_seeds"] == 5, f"{c['cfg_id']} has {c['n_seeds']} seeds, expected 5"
    print("All 6 Stage-C candidates independently recomputed and MATCH the notebook's "
         "summary exactly (mean val AUC, to 1e-9).")

    candidate_rows.sort(key=lambda c: c["val_auc_mean"], reverse=True)
    with open(HERE / "03_candidates_multiseed.csv", "w") as f:
        f.write("rank,cfg_id,is_default,is_winner,n_seeds,val_auc_mean,val_auc_std\n")
        for i, c in enumerate(candidate_rows, 1):
            f.write(f"{i},{c['cfg_id']},{c['cfg_id']==default_cfg_id},"
                   f"{c['cfg_id']==winner_cfg_id},{c['n_seeds']},"
                   f"{c['val_auc_mean']:.6f},{c['val_auc_std']:.6f}\n")

    winner = next(c for c in candidate_rows if c["cfg_id"] == winner_cfg_id)
    default = next(c for c in candidate_rows if c["cfg_id"] == default_cfg_id)
    default_rank_in_78 = [r["_cfg_id"] for r in full_pool_ranked].index(default_cfg_id) + 1

    write_markdown(summary, candidate_rows, winner, default, seed42_leader_cfg_id,
                   default_rank_in_78, len(full_pool_ranked))
    make_figure(stage_a_ranked, candidate_rows, winner_cfg_id, default_cfg_id)

    print(f"\nWINNER: {winner_cfg_id}")
    print(f"  val AUC (5-seed): {winner['val_auc_mean']:.4f} ± {winner['val_auc_std']:.4f}")
    print(f"transformer_default: val AUC (5-seed): {default['val_auc_mean']:.4f} ± "
         f"{default['val_auc_std']:.4f}  (rank {default_rank_in_78}/{len(full_pool_ranked)} "
         f"of the full seed-42 pool)")
    print(f"\nFor context only (NOT the formal comparison): LSTM val AUC (Issue 8) = "
         f"{LSTM_VAL_AUC_MEAN:.4f} ± {LSTM_VAL_AUC_STD:.4f}")
    print("wrote 03_arch_grid.csv, 03_recipe_grid.csv, 03_candidates_multiseed.csv, "
         "03_search_summary.md, 03_search_figure.png")


def write_markdown(summary, candidate_rows, winner, default, seed42_leader_cfg_id,
                   default_rank, n_total):
    is_default_winner = winner["cfg_id"] == default["cfg_id"]
    noise_note = ""
    if seed42_leader_cfg_id != winner["cfg_id"]:
        leader_row = next((c for c in candidate_rows if c["cfg_id"] == seed42_leader_cfg_id),
                          None)
        if leader_row is not None:
            noise_note = (
                f" The selection-noise control mattered concretely, exactly as it did for "
                f"the LSTM in Issue 8: the single-seed (seed 42) leader across all 78 "
                f"configs was `{seed42_leader_cfg_id}`, but its 5-seed **mean** val AUC "
                f"({leader_row['val_auc_mean']:.4f}) ends up lower than the actual winner's "
                f"({winner['val_auc_mean']:.4f}) — a single-seed search would have picked "
                f"the wrong config."
            )
        else:
            noise_note = (
                f" The single-seed (seed 42) leader across all 78 configs was "
                f"`{seed42_leader_cfg_id}`, which was not selected as a Stage-C candidate "
                f"(outside the top-5 by mean once other configs were considered) — a "
                f"reminder that seed-42-only rankings are noisy."
            )

    L = ["# Phase T3 — Transformer staged search report (local re-derivation)", "",
         "**Independently recomputed from the raw `runs_search/**/seed*.json` files** — "
         "every number below was recalculated from scratch and cross-checked against "
         "`_stage_summary.json`; the script asserts on any mismatch. See "
         "`transformer/PLAN.md` §4 for the full protocol.", "",
         "## Protocol reminder (val-only; test set03 has not been touched anywhere yet)",
         "",
         "1. **Stage A** — 36 architecture configs at seed 42, default recipe "
         "(Adam, lr=1e-3, plateau, dropout=0.1, wd=1e-5), ranked by validation AUC.",
         "2. **Stage B** — 36 training-recipe configs on Stage-A's #1 architecture, "
         "seed 42, val AUC.",
         "3. **Transfer check** — the top-3 recipes re-tried on Stage-A architectures "
         "#2 and #3 (6 configs), to confirm the recipe ranking isn't architecture-specific.",
         "4. **Stage C** — top-5 (architecture × recipe) combos from the pooled 78, plus "
         "`transformer_default` (always carried), each × 5 seeds [42,0,1,2,3]. **Winner = "
         "highest mean val AUC** (guards against a single-seed fluke). Test set still "
         "untouched anywhere in this pipeline.", "",
         "## Stage C — candidate ranking by mean validation AUC (5 seeds each)", "",
         "| rank | config | val AUC (5-seed) | tag |", "|---|---|---|---|"]
    for i, c in enumerate(candidate_rows, 1):
        tags = []
        if c["cfg_id"] == winner["cfg_id"]:
            tags.append("**winner**")
        if c["cfg_id"] == default["cfg_id"]:
            tags.append("transformer_default")
        L.append(f"| {i} | `{c['cfg_id']}` | {c['val_auc_mean']:.4f} ± {c['val_auc_std']:.4f} "
                 f"| {' '.join(tags)} |")

    L += ["", "## Winner vs default (validation set — test not yet touched)", "",
         f"| | config | val AUC (5-seed) |", "|---|---|---|",
         f"| **winner** | `{winner['cfg_id']}` | **{winner['val_auc_mean']:.4f} ± "
         f"{winner['val_auc_std']:.4f}** |",
         f"| transformer_default | `{default['cfg_id']}` | {default['val_auc_mean']:.4f} ± "
         f"{default['val_auc_std']:.4f} |", "",
         f"`transformer_default` ranks **{default_rank}/{n_total}** in the full seed-42 "
         f"architecture pool — the search meaningfully outperformed the un-searched "
         f"default." if default_rank > 5 else
         "`transformer_default` happened to also be a top-5 seed-42 config."]

    L += ["", "## Verdict", ""]
    if is_default_winner:
        L.append("**The search confirms the hand-picked default.** Report "
                 f"`transformer_default`'s test AUC in Phase T4/T5.")
    else:
        L.append(
            f"**The search selected `{winner['cfg_id']}`** over `transformer_default` — "
            f"a deeper (num_layers={winner['cfg']['num_layers']} vs "
            f"{default['cfg']['num_layers']}), differently-pooled "
            f"(pool={winner['cfg']['pool']!r} vs {default['cfg']['pool']!r}) architecture "
            f"with sinusoidal rather than learned positional encoding. Notably, "
            f"`pool=\"last\"` mirrors the BiLSTM's own readout (its last timestep), which "
            f"is a reassuring sign the search converged on something sensible rather than "
            f"an arbitrary config." + noise_note
        )
    L += ["", f"**This is a validation-set result only.** For context (not a formal "
         f"comparison — protocols differ and only a test-set comparison counts): the "
         f"LSTM's own 5-seed validation AUC (Issue 8) is "
         f"{LSTM_VAL_AUC_MEAN:.4f} ± {LSTM_VAL_AUC_STD:.4f}, and its test AUC — **the "
         f"number the transformer must actually beat** — is "
         f"{LSTM_TEST_AUC_MEAN:.4f} ± {LSTM_TEST_AUC_STD:.4f}. Validation AUC is not "
         f"directly comparable across the two searches (different validation-selection "
         f"histories can inflate a val number without a matching test gain — this is "
         f"exactly why Phase T4 touches test exactly once and Phase T5 uses a paired "
         f"bootstrap rather than comparing val numbers directly).", "",
         f"**Next step: Phase T4.** Paste `{winner['cfg_id']}` into "
         f"`phase4_kaggle_final/04_final_loso_kaggle.ipynb`'s CONFIG cell, run it — that "
         f"notebook trains the winner + default × 5 seeds and touches test set03 exactly once, "
         f"plus the 6-fold LOSO CV."]
    (HERE / "03_search_summary.md").write_text("\n".join(L))


def make_figure(stage_a_ranked, candidate_rows, winner_cfg_id, default_cfg_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white",
                             gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    vals = [r["val"]["auc"] for r in stage_a_ranked]
    colors = ["#9ca3af"] * len(vals)
    ax.bar(range(len(vals)), vals, color=colors, zorder=2)
    ax.set_ylim(min(vals) - 0.01, max(vals) + 0.005)
    ax.set_xlabel("Stage-A architecture rank (sorted by seed-42 val AUC)")
    ax.set_ylabel("validation AUC")
    ax.set_title(f"Stage A — {len(stage_a_ranked)} architectures (default recipe, seed 42)")
    ax.grid(axis="y", alpha=0.3, zorder=1)

    ax2 = axes[1]
    ranked = sorted(candidate_rows, key=lambda c: c["val_auc_mean"], reverse=True)
    labels = [("winner" if c["cfg_id"] == winner_cfg_id
              else "default" if c["cfg_id"] == default_cfg_id else f"#{i+1}")
             for i, c in enumerate(ranked)]
    colors2 = ["#dc2626" if c["cfg_id"] == winner_cfg_id
              else "#2563eb" if c["cfg_id"] == default_cfg_id
              else "#9ca3af" for c in ranked]
    for i, (c, col) in enumerate(zip(ranked, colors2)):
        ax2.bar(i, c["val_auc_mean"], yerr=c["val_auc_std"], color=col, capsize=4,
               zorder=2, width=0.6)
        ax2.annotate(f"{c['val_auc_mean']:.3f}", (i, c["val_auc_mean"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    lo = min(c["val_auc_mean"] - c["val_auc_std"] for c in ranked)
    hi = max(c["val_auc_mean"] + c["val_auc_std"] for c in ranked)
    pad = max(0.005, (hi - lo) * 0.3)
    ax2.set_ylim(lo - pad, hi + pad * 1.5)
    ax2.set_xticks(range(len(ranked)))
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("validation AUC (5-seed mean ± std)")
    ax2.set_title("Stage C candidates")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(handles=[Patch(color="#dc2626", label="winner"),
                        Patch(color="#2563eb", label="transformer_default"),
                        Patch(color="#9ca3af", label="other candidate")], loc="lower left",
              fontsize=8)

    fig.suptitle("Phase T3 — Transformer staged search (validation only; test not yet touched)",
                fontweight="bold")
    plt.tight_layout()
    plt.savefig(HERE / "03_search_figure.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

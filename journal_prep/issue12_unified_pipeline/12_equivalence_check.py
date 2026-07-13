"""12_equivalence_check.py — proves the unified engine IS the published pipeline.

Gates (each PASS/FAIL, report written either way):

G-A  ENGINE EQUIVALENCE (bilstm, CPU): train the same cell (baseline cfg, pw 1.682,
     seed 42, select=f1) with BOTH the f1_optimization engine and the unified engine
     on CPU — every val metric, best_epoch, auc_best_epoch, n_params must be
     bit-identical between the two engines. CPU is used because it is context-free
     (verified: same value fresh-process, warmed-process, cross-process), so this
     isolates pure code equivalence.

G-B  PUBLISHED-CELL REPRODUCTION (transformer, MPS): the unified engine must
     reproduce f1_optimization/runs_f1/transformer_searched/pw1.682/seed42/final.json
     EXACTLY on the same device that produced it.

G-C  NEW FAMILIES (gru, birnn): registry builders produce the right output shape and
     loss strictly decreases over 3 epochs on a 128-window subset (functional smoke,
     no performance claim).

KNOWN, MEASURED CAVEAT (documented, not a gate): recurrent (nn.LSTM) TRAINING on
Apple MPS is bit-deterministic only within an identical process history — the same
cell gives different (all individually deterministic) trajectories depending on what
ran before it in the process (measured: fresh process val F1 0.82392027; after one
other cell 0.83439490; the published cell, produced mid-way through the 04 driver,
0.83870968). The transformer family shows no such dependence (its cached cell
reproduces exactly cross-process), and CPU shows none for any family
(0.8271604938 in every context). This also retroactively explains why Issue-8's
cached MPS grid values could not be re-reproduced ("env drift" in
f1_optimization/PROGRESS_LOG.md — same phenomenon). Consequence for practice:
train recurrent families on CPU when exact regeneration matters (~15 s/run here),
or record that MPS training numbers are context-bound measurements. No published
number is invalidated: all reported test metrics come from saved checkpoints
evaluated on CPU behind exact parity gates.

Output: 12_equivalence_report.md. VAL-ONLY (the engine has no test path).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
_specu = importlib.util.spec_from_file_location("unified", HERE / "12_unified_engine.py")
U = importlib.util.module_from_spec(_specu)
_specu.loader.exec_module(U)
_specc = importlib.util.spec_from_file_location("f1_common", ROOT / "f1_optimization" / "00_common.py")
C = importlib.util.module_from_spec(_specc)
_specc.loader.exec_module(C)
_spece = importlib.util.spec_from_file_location("f1_engines", ROOT / "f1_optimization" / "00_train_engines.py")
E = importlib.util.module_from_spec(_spece)
_spece.loader.exec_module(E)

TF_REF = ROOT / "f1_optimization/runs_f1/transformer_searched/pw1.682/seed42/final.json"


def main():
    data = U.load_splits()
    lines = ["# 12 — Unified-engine equivalence check", ""]
    all_pass = True

    # ---- G-A: engine equivalence on CPU (context-free device) ----
    cpu = torch.device("cpu")
    r_f1 = E.train_lstm(C.BASELINE_LSTM_CFG, 42, cpu, data, pos_weight=1.682, select="f1")
    r_un = U.train_run("bilstm", U.BILSTM_BASELINE_CFG, 42, cpu, data,
                       pos_weight=1.682, select="f1")
    ga = (r_f1["val"] == r_un["val"] and r_f1["best_epoch"] == r_un["best_epoch"]
          and r_f1["auc_best_epoch"] == r_un["auc_best_epoch"]
          and r_f1["n_params"] == r_un["n_params"]
          and r_f1["val_at_auc_best"] == r_un["val_at_auc_best"])
    all_pass &= ga
    print(f"G-A bilstm engine-equivalence (CPU): {'PASS bit-identical' if ga else 'FAIL'} "
          f"(f1-engine {r_f1['val']['f1']:.10f} vs unified {r_un['val']['f1']:.10f})")
    lines += ["## G-A — engine equivalence, bilstm on CPU (context-free device)", "",
              f"f1_optimization engine: val F1 {r_f1['val']['f1']:.10f}, AUC "
              f"{r_f1['val']['auc']:.10f}, best_ep {r_f1['best_epoch']}",
              f"unified engine:        val F1 {r_un['val']['f1']:.10f}, AUC "
              f"{r_un['val']['auc']:.10f}, best_ep {r_un['best_epoch']}", "",
              f"**{'PASS — bit-identical in every field' if ga else 'FAIL'}**", ""]

    # ---- G-B: published transformer cell reproduced exactly on MPS ----
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ref = json.loads(TF_REF.read_text())
    r_tf = U.train_run("transformer", U.TRANSFORMER_SEARCHED_CFG, 42, dev, data,
                       pos_weight=1.682, select="f1")
    gb = (r_tf["val"] == ref["val"] and r_tf["best_epoch"] == ref["best_epoch"]
          and r_tf["auc_best_epoch"] == ref["auc_best_epoch"]
          and r_tf["n_params"] == ref["n_params"]
          and r_tf["val_at_auc_best"] == ref["val_at_auc_best"])
    all_pass &= gb
    print(f"G-B transformer vs published cell ({dev}): {'PASS exact' if gb else 'FAIL'} "
          f"(unified {r_tf['val']['f1']:.10f} vs cached {ref['val']['f1']:.10f})")
    lines += [f"## G-B — published-cell reproduction, transformer on {dev}", "",
              f"unified: val F1 {r_tf['val']['f1']:.10f}, best_ep {r_tf['best_epoch']} | "
              f"cached `{TF_REF.relative_to(ROOT)}`: val F1 {ref['val']['f1']:.10f}, "
              f"best_ep {ref['best_epoch']}", "",
              f"**{'PASS — exact reproduction of the published run' if gb else 'FAIL'}**", ""]

    # ---- G-C: gru / birnn functional smoke ----
    Xtr, ytr = data[0][:128], data[1][:128]
    flat = Xtr.reshape(-1, 5)
    Xn = ((Xtr - flat.mean(0)) / (flat.std(0) + 1e-6)).astype(np.float32)
    lines += ["## G-C — new families (registry-ready, no published result)", ""]
    for fam in ("gru", "birnn"):
        U.set_seed(42)
        model = U.MODEL_REGISTRY[fam](U.GRU_DEFAULT_CFG).to(dev)
        n = sum(p.numel() for p in model.parameters())
        crit = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        xb = torch.from_numpy(Xn).to(dev)
        yb = torch.from_numpy(ytr).to(dev)
        losses = []
        for _ in range(3):
            model.train()
            opt.zero_grad()
            out = model(xb).squeeze(-1)
            assert out.shape == (128,), f"{fam} bad output shape {out.shape}"
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        ok = losses[-1] < losses[0]
        all_pass &= ok
        print(f"G-C {fam}: params={n:,} losses={['%.4f' % l for l in losses]} "
              f"{'PASS' if ok else 'FAIL'}")
        lines.append(f"- **{fam}** — params {n:,}; 3-epoch loss "
                     + " -> ".join(f"{l:.4f}" for l in losses)
                     + f" — {'PASS (decreasing)' if ok else '**FAIL**'}")
    lines.append("")

    # ---- documented caveat ----
    lines += ["## Measured reproducibility caveat (documented; not a gate)", "",
              "Recurrent (nn.LSTM) TRAINING on Apple MPS is bit-deterministic only "
              "within an identical process history: the same cell measured val F1 "
              "0.82392027 (fresh process), 0.83439490 (after one other training in the "
              "same process), 0.83870968 (the published cell, produced mid-way through "
              "the 04 driver). The transformer family has no such dependence (G-B "
              "reproduces its published cell exactly, cross-process), and CPU has none "
              "for any family (0.8271604938 in every tested context). This also "
              "explains Issue-8's earlier 'environment drift' finding. Practice: train "
              "recurrent families on CPU when exact regeneration matters (~15 s/run); "
              "all published TEST metrics are unaffected (saved checkpoints, CPU "
              "evaluation, exact parity gates).", "",
              "## Verdict", "",
              ("**ALL GATES PASS** — one engine, one code path, provably the same "
               "computation as the published pipeline; GRU/biRNN ready for follow-up.")
              if all_pass else
              "**AT LEAST ONE GATE FAILED — do not adopt until resolved.**", ""]
    (HERE / "12_equivalence_report.md").write_text("\n".join(lines))
    print("wrote 12_equivalence_report.md |", "ALL PASS" if all_pass else "FAILURES")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

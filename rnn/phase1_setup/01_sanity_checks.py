"""01_sanity_checks.py — RNN study Phase R1 sanity gates (local CPU).

Mirrors gru/phase1_setup/01_sanity_checks.py and transformer/phase1_setup/01_sanity_checks.py:
every gate must pass before any search is run. NO training-engine code is duplicated —
everything routes through journal_prep/issue12_unified_pipeline/12_unified_engine.py (the one
unified engine), which is the entire point of this study.

The family here is `birnn` = a BIDIRECTIONAL VANILLA RNN (tanh cell), the exact twin of our
BiLSTM with only the recurrent cell swapped (nn.LSTM -> nn.RNN), so the comparison isolates
gating. Unlike the GRU (gated twin), the vanilla RNN removes gating entirely — it is the
family most likely to break the thesis's "the cell doesn't matter" tie.

Gates:
  G0  protocol asserts        engine.load_splits() clean; X=(4906,16,5); splits 2178/634/2094;
                              test positives 681; pos_weight = n_neg/n_pos = 1366/812 = 1.682;
                              train-only norm shape (5,).
  G1  param count             birnn default (h128/nl2/do0.3) = 149,121 EXACTLY; print the search
                              param ladder (hidden {64,128,256} x num_layers {1,2}) — the whole
                              vanilla-RNN family sits BELOW the BiLSTM's 594,561 (un-gated cell
                              has ~1/4 the recurrent weights).
  G2  forward/backward CPU    output (B,1), no NaN, one BCE+Adam step runs; loss finite.
  G3  determinism (the load-bearing gate)  birnn default seed 42 trained TWICE on CPU ->
                              bit-identical val F1/acc/AUC and val_at_auc_best.auc (|delta|=0).
  G4  engine parity           the SAME engine builds the published BiLSTM cell: bilstm baseline
                              n_params = 594,561 (fingerprint), and is itself bit-reproducible
                              on CPU across two runs.

Writes phase1_setup/01_sanity_report.md.
Run from the repo root:  python rnn/phase1_setup/01_sanity_checks.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # repo root
ENGINE_PATH = ROOT / "journal_prep" / "issue12_unified_pipeline" / "12_unified_engine.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


engine = _load("rnn_engine", ENGINE_PATH)

CPU = torch.device("cpu")
BIRNN_DEFAULT = dict(engine.PRESETS["birnn"])       # lr1e-3/do0.3/h128/nl2 (BiLSTM-recipe twin)
BILSTM_BASE = dict(engine.BILSTM_BASELINE_CFG)      # lr1e-3/do0.3/h128/nl2
EXPECTED_BIRNN_PARAMS = 149_121
EXPECTED_BILSTM_PARAMS = 594_561


def n_params(family, cfg):
    return sum(p.numel() for p in engine.MODEL_REGISTRY[family](cfg).parameters())


def gate0(lines):
    """Protocol asserts — the engine's own load_splits() already asserts most of this."""
    Xtr, ytr, Xva, yva, Xte, yte = engine.load_splits()
    n_pos_tr, n_neg_tr = int(ytr.sum()), int(len(ytr) - ytr.sum())
    pw = n_neg_tr / n_pos_tr
    test_pos = int(yte.sum())
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0) + 1e-6

    ok = (Xtr.shape[1:] == (16, 5) and len(ytr) == 2178 and len(yva) == 634
          and len(yte) == 2094 and test_pos == 681 and abs(pw - 1.682) < 1e-3
          and mean.shape == (5,) and std.shape == (5,))
    lines += [
        "## Gate 0 — protocol asserts",
        "",
        f"- X shape (per window): {Xtr.shape[1:]} (expect (16, 5))",
        f"- split sizes: train {len(ytr)} / val {len(yva)} / test {len(yte)} "
        f"(expect 2178 / 634 / 2094)",
        f"- train class balance: {n_pos_tr} pos / {n_neg_tr} neg -> pos_weight "
        f"{pw:.4f} (expect 1366/812 = 1.6823)",
        f"- test positives: {test_pos} (expect 681)",
        f"- train-only norm shapes: mean {mean.shape}, std {std.shape} (expect (5,))",
        f"- **Gate 0: {'PASS' if ok else 'FAIL'}**",
        "",
    ]
    return ok, (Xtr, ytr, Xva, yva, Xte, yte)


def gate1(lines):
    p_default = n_params("birnn", BIRNN_DEFAULT)
    ok = (p_default == EXPECTED_BIRNN_PARAMS)
    lines += [
        "## Gate 1 — parameter count & search ladder",
        "",
        f"- vanilla-RNN default (h128 / nl2 / do0.3) = **{p_default:,}** params "
        f"(expect {EXPECTED_BIRNN_PARAMS:,})",
        "",
        "Search-space param ladder (bidirectional tanh RNN, proj_dim 64) — note the whole "
        f"family sits **below** the BiLSTM's {EXPECTED_BILSTM_PARAMS:,} (un-gated cell has "
        "~1/4 the recurrent weights of the 4-gate LSTM):",
        "",
        "| hidden | num_layers | params | ×BiLSTM |",
        "|---|---|---|---|",
    ]
    for hidden in (64, 128, 256):
        for nl in (1, 2):
            do = 0.0 if nl == 1 else 0.3   # dropout inert at nl=1 (inter-layer)
            cfg = dict(lr=1e-3, dropout=do, hidden=hidden, num_layers=nl)
            p = n_params("birnn", cfg)
            lines.append(f"| {hidden} | {nl} | {p:,} | {p / EXPECTED_BILSTM_PARAMS:.2f}× |")
    lines += ["", f"- **Gate 1: {'PASS' if ok else 'FAIL'}**", ""]
    return ok


def gate2(lines, data):
    engine.set_seed(42)
    model = engine.MODEL_REGISTRY["birnn"](BIRNN_DEFAULT).to(CPU)
    Xtr, ytr = data[0], data[1]
    xb = torch.from_numpy(Xtr[:32]).to(CPU)
    yb = torch.from_numpy(ytr[:32]).to(CPU)
    out = model(xb)
    shape_ok = tuple(out.shape) == (32, 1)
    nan_ok = torch.isfinite(out).all().item()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.682]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss0 = crit(model(xb).squeeze(-1), yb)
    opt.zero_grad(); loss0.backward(); opt.step()
    step_ok = torch.isfinite(loss0).item()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters())
    ok = shape_ok and nan_ok and step_ok and grad_ok
    lines += [
        "## Gate 2 — forward / backward on CPU",
        "",
        f"- forward output shape: {tuple(out.shape)} (expect (32, 1)) — "
        f"{'ok' if shape_ok else 'FAIL'}",
        f"- output finite (no NaN/Inf): {nan_ok}",
        f"- one BCE+Adam step: loss {loss0.item():.4f} finite {step_ok}, grads finite {grad_ok}",
        f"- **Gate 2: {'PASS' if ok else 'FAIL'}**",
        "",
    ]
    return ok


def _train_twice(family, cfg, data):
    r1 = engine.train_run(family, cfg, 42, CPU, data, select="f1", out_dir=None)
    r2 = engine.train_run(family, cfg, 42, CPU, data, select="f1", out_dir=None)
    keys = [("val", "f1"), ("val", "acc"), ("val", "auc"), ("val_at_auc_best", "auc")]
    deltas = {f"{a}.{b}": abs(r1[a][b] - r2[a][b]) for a, b in keys}
    identical = all(d == 0.0 for d in deltas.values())
    return r1, r2, deltas, identical


def gate3(lines, data):
    r1, r2, deltas, identical = _train_twice("birnn", BIRNN_DEFAULT, data)
    lines += [
        "## Gate 3 — determinism (bit-identical same-seed CPU runs)",
        "",
        "`birnn` default, seed 42, trained twice on CPU (full protocol, early stop on val AUC):",
        "",
        f"- run 1: val F1 {r1['val']['f1']:.6f}, acc {r1['val']['acc']:.6f}, "
        f"AUC {r1['val']['auc']:.6f}, val_at_auc_best AUC {r1['val_at_auc_best']['auc']:.6f} "
        f"(best epoch {r1['best_epoch']}, {r1['seconds']:.0f}s)",
        f"- run 2: val F1 {r2['val']['f1']:.6f}, acc {r2['val']['acc']:.6f}, "
        f"AUC {r2['val']['auc']:.6f}, val_at_auc_best AUC {r2['val_at_auc_best']['auc']:.6f} "
        f"(best epoch {r2['best_epoch']}, {r2['seconds']:.0f}s)",
        f"- |deltas|: " + ", ".join(f"{k} {v:.1e}" for k, v in deltas.items()),
        f"- **Gate 3: {'PASS' if identical else 'FAIL'}** "
        f"({'bit-identical' if identical else 'NON-DETERMINISTIC — investigate before search'})",
        "",
    ]
    return identical, r1


def gate4(lines, data):
    p = n_params("bilstm", BILSTM_BASE)
    fingerprint_ok = (p == EXPECTED_BILSTM_PARAMS)
    r1, r2, deltas, identical = _train_twice("bilstm", BILSTM_BASE, data)
    sane = r1["val"]["auc"] > 0.90   # BiLSTM baseline val AUC is ~0.96
    ok = fingerprint_ok and identical and sane
    lines += [
        "## Gate 4 — engine parity (same engine builds the published BiLSTM)",
        "",
        f"- bilstm baseline (h128/nl2/do0.3) n_params = **{p:,}** "
        f"(expect published {EXPECTED_BILSTM_PARAMS:,}) — {'ok' if fingerprint_ok else 'FAIL'}",
        f"- bilstm seed 42 twice on CPU: val AUC {r1['val']['auc']:.6f} / "
        f"{r2['val']['auc']:.6f}; |delta AUC| {deltas['val.auc']:.1e} "
        f"({'bit-identical' if identical else 'NON-DETERMINISTIC'})",
        f"- val AUC in sane band (>0.90): {sane}",
        f"- **Gate 4: {'PASS' if ok else 'FAIL'}**",
        "",
        "> Note: issue12's `12_equivalence_check.py` already proved this engine is "
        "bit-equivalent to the published pipeline; this is a lightweight re-confirmation that "
        "the RNN study rides the same rails as the BiLSTM/GRU/Transformer.",
        "",
    ]
    return ok


def main():
    print("RNN R1 sanity gates (CPU) — this runs 4 short trainings, ~1-2 min...")
    lines = ["# RNN study — Phase R1 sanity report", "",
             "All gates run **local CPU** through the unified engine "
             "(`journal_prep/issue12_unified_pipeline/12_unified_engine.py`). No trainer code "
             "is duplicated. Family = `birnn` (bidirectional vanilla tanh RNN, the un-gated "
             "twin of the BiLSTM). Generated by `rnn/phase1_setup/01_sanity_checks.py`.", ""]

    ok0, data = gate0(lines); print(f"  Gate 0 (protocol asserts): {'PASS' if ok0 else 'FAIL'}")
    ok1 = gate1(lines);       print(f"  Gate 1 (param count 149,121): {'PASS' if ok1 else 'FAIL'}")
    ok2 = gate2(lines, data); print(f"  Gate 2 (fwd/bwd CPU): {'PASS' if ok2 else 'FAIL'}")
    ok3, _ = gate3(lines, data); print(f"  Gate 3 (determinism): {'PASS' if ok3 else 'FAIL'}")
    ok4 = gate4(lines, data); print(f"  Gate 4 (engine parity): {'PASS' if ok4 else 'FAIL'}")

    all_ok = ok0 and ok1 and ok2 and ok3 and ok4
    lines += ["---", "",
              f"## Summary: {'ALL GATES PASS ✅' if all_ok else 'GATE FAILURE ❌'}",
              "",
              "| gate | check | result |",
              "|---|---|---|",
              f"| G0 | protocol asserts (splits, pos_weight, norm) | {'PASS' if ok0 else 'FAIL'} |",
              f"| G1 | birnn param count = 149,121 | {'PASS' if ok1 else 'FAIL'} |",
              f"| G2 | forward/backward on CPU | {'PASS' if ok2 else 'FAIL'} |",
              f"| G3 | same-seed CPU determinism (|Δ|=0) | {'PASS' if ok3 else 'FAIL'} |",
              f"| G4 | engine builds published BiLSTM (594,561) | {'PASS' if ok4 else 'FAIL'} |",
              ""]
    (HERE / "01_sanity_report.md").write_text("\n".join(lines))
    print(f"\n{'ALL GATES PASS' if all_ok else 'GATE FAILURE'} — wrote 01_sanity_report.md")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

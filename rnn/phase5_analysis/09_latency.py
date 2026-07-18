"""09_latency.py — RNN study Phase R5: isolated inference latency (Issue-9 protocol, M4).

Measures the RNN F1-winner's forward-pass latency in isolation, exactly as Issue 9 measured the
BiLSTM: 50 warmup + 1000 timed forwards per cell, CPU and MPS × batch {1, 8, 32}, with
torch.mps.synchronize() inside every timed MPS call (so we time compute, not async dispatch).
Latency is weight-independent, but we load the real F1-winner checkpoint for fidelity.

Reference (M4, CPU batch-1, ms/window): BiLSTM 0.575 (Issue 9), Transformer 0.459
(transformer/phase5), GRU 0.721 (gru/phase5). Reports the RNN next to all three. No training.
The vanilla RNN is the smallest family, so it should be the fastest recurrent model.

Outputs: 09_latency_results.json, 09_latency_report.md
Run from the repo root:  python rnn/phase5_analysis/09_latency.py
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SEARCH = ROOT / "rnn" / "phase2_search"
WARMUP, TIMED = 50, 1000
BATCHES = [1, 8, 32]
REF = {"BiLSTM": 0.575, "Transformer": 0.459, "GRU": 0.721}  # M4 CPU batch-1 ms/window
BUDGET = 33.3                                                 # ms/frame at 30 fps


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


engine = _load("rnn_engine", ROOT / "journal_prep" / "issue12_unified_pipeline" / "12_unified_engine.py")
F1_WINNER_ID = json.loads((SEARCH / "_stage_summary.json").read_text())["f1_winner"]
F1_WINNER_CFG = json.loads(
    (SEARCH / "runs_search" / F1_WINNER_ID / "seed42.json").read_text())["cfg"]
CKPT = ROOT / "rnn" / "phase4_final" / "runs_final" / "rnn_f1_winner" / "seed42" / "best.pt"


def devices():
    d = ["cpu"]
    if torch.backends.mps.is_available():
        d.append("mps")
    return d


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def load_model(device):
    model = engine.MODEL_REGISTRY["birnn"](F1_WINNER_CFG).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@torch.no_grad()
def time_forward(model, x, dev):
    for _ in range(WARMUP):
        model(x)
    sync(dev)
    ts = np.empty(TIMED)
    for i in range(TIMED):
        t0 = time.perf_counter()
        model(x)
        sync(dev)
        ts[i] = (time.perf_counter() - t0) * 1e3
    return ts


def main():
    n_params = sum(p.numel() for p in engine.MODEL_REGISTRY["birnn"](F1_WINNER_CFG).parameters())
    print(f"RNN F1-winner {F1_WINNER_CFG} ({n_params:,} params) | devices {devices()} | "
          f"{WARMUP} warmup + {TIMED} timed\n")
    out = {"cfg": F1_WINNER_CFG, "n_params": int(n_params), "model": {}}
    for dev in devices():
        model = load_model(dev)
        out["model"][dev] = {"batch": {}}
        for b in BATCHES:
            x = torch.randn(b, 16, 5, device=dev)
            ts = time_forward(model, x, dev)
            out["model"][dev]["batch"][str(b)] = dict(
                ms_forward=float(ts.mean()), ms_std=float(ts.std()),
                ms_p99=float(np.percentile(ts, 99)),
                ms_per_window=float(ts.mean() / b),
                windows_per_s=float(b / (ts.mean() / 1e3)))
            print(f"  [{dev}] batch={b:2d}: {ts.mean():.3f} ms/forward "
                  f"({ts.mean()/b:.4f} ms/window, {b/(ts.mean()/1e3):,.0f} win/s)")

    cpu1 = out["model"]["cpu"]["batch"]["1"]["ms_per_window"]
    (HERE / "09_latency_results.json").write_text(json.dumps(out, indent=2))

    L = ["# RNN study — Phase R5 isolated inference latency (Apple M4)", "",
         f"RNN F1-winner ({n_params:,} params), Issue-9 protocol: {WARMUP} warmup + "
         f"{TIMED} timed forwards per cell, MPS synced inside each timed call. Latency is "
         "weight-independent; the real F1-winner checkpoint is loaded for fidelity.", "",
         "| device | batch | ms/forward | ms/**window** | windows/s | p99 ms |",
         "|---|---|---|---|---|---|"]
    for dev in out["model"]:
        for b in BATCHES:
            c = out["model"][dev]["batch"][str(b)]
            L.append(f"| {dev.upper()} | {b} | {c['ms_forward']:.3f} | "
                     f"**{c['ms_per_window']:.4f}** | {c['windows_per_s']:,.0f} | {c['ms_p99']:.3f} |")
    L += ["",
          f"**RNN single-window latency = {cpu1:.3f} ms** (CPU, batch 1) = "
          f"~{BUDGET/cpu1:.0f}× inside a 30 fps frame budget ({BUDGET:.1f} ms). "
          f"vs BiLSTM {REF['BiLSTM']:.3f} ms, GRU {REF['GRU']:.3f} ms, and Transformer "
          f"{REF['Transformer']:.3f} ms (all M4 CPU batch-1). The vanilla RNN is the smallest "
          f"family ({n_params:,} params), so a low latency is expected; either way it is ~2 "
          f"orders of magnitude inside the frame budget and the live pipeline stays "
          f"detection-bound (Issue 9). Latency is not a deployment discriminator among the four "
          f"model families.", ""]
    (HERE / "09_latency_report.md").write_text("\n".join(L))
    print(f"\nRNN CPU batch-1: {cpu1:.3f} ms/window  (BiLSTM {REF['BiLSTM']}, GRU {REF['GRU']}, "
          f"Transformer {REF['Transformer']})")
    print("wrote 09_latency_results.json, 09_latency_report.md")


if __name__ == "__main__":
    main()

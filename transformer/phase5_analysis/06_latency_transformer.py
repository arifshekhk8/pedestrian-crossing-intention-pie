"""06_latency_transformer.py — Phase T5: isolated inference-latency benchmark (M4).

Issue-9 core applied to the Phase-T3 winner (`transformer_searched`, 794,241 params,
num_layers=4): 50 warmup + 1000 timed forwards, CPU + MPS x batch {1, 8, 32}, MPS
timings synchronised inside each timed call (torch.mps.synchronize) since MPS is
asynchronous -- otherwise we'd time kernel dispatch, not compute.

NO TRAINING. Loads the real `transformer_searched/seed42/best.pt` for fidelity (latency
is weight-independent, but this matches Issue 9's own convention of loading real
weights rather than a randomly-initialized model).

Deployment hardware = this M4 (same machine the live demo runs on). Compared against
the BiLSTM's own Issue-9 numbers (0.575 ms/window CPU batch-1).

    python transformer/phase5_analysis/06_latency_transformer.py
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PHASE1 = HERE.parent / "phase1_setup"
PHASE4 = HERE.parent / "phase4_kaggle_final"
CKPT = PHASE4 / "runs_final" / "transformer_searched" / "seed42" / "best.pt"

_spec_engine = importlib.util.spec_from_file_location("train_engine", PHASE1 / "02_train_transformer.py")
engine = importlib.util.module_from_spec(_spec_engine)
_spec_engine.loader.exec_module(engine)

WINNER_CFG = dict(
    d_model=128, nhead=4, num_layers=4, dim_ff=512, dropout=0.1,
    pool="last", pos="sin",
    lr=1e-3, schedule="plateau", weight_decay=1e-5, optimizer="adam",
)

WARMUP, TIMED = 50, 1000
BATCHES = [1, 8, 32]

# BiLSTM's own Issue-9 numbers (journal_prep/issue9_latency/09_latency_results.json) --
# frozen, for side-by-side context only.
LSTM_LATENCY = json.loads(
    (HERE.parent.parent / "journal_prep" / "issue9_latency" / "09_latency_results.json").read_text()
)["bilstm"]


def devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def load_model(device):
    model = engine.build_model(WINNER_CFG).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def time_forward(model, x, device, warmup=WARMUP, timed=TIMED):
    for _ in range(warmup):
        model(x)
    sync(device)
    ts = np.empty(timed)
    for i in range(timed):
        t0 = time.perf_counter()
        model(x)
        sync(device)
        ts[i] = (time.perf_counter() - t0) * 1e3  # ms
    return ts


def bench_transformer():
    out = {}
    for dev in devices():
        model = load_model(dev)
        out[dev] = {"batch": {}}
        for b in BATCHES:
            x = torch.randn(b, 16, 5, device=dev)
            ts = time_forward(model, x, dev)
            out[dev]["batch"][b] = dict(
                ms_forward=float(ts.mean()), ms_std=float(ts.std()),
                ms_p50=float(np.percentile(ts, 50)), ms_p99=float(np.percentile(ts, 99)),
                ms_per_window=float(ts.mean() / b),
                windows_per_s=float(b / (ts.mean() / 1e3)))
            print(f"  [transformer/{dev}] batch={b:2d}: {ts.mean():.3f} ms/forward "
                 f"({ts.mean()/b:.4f} ms/window, {b/(ts.mean()/1e3):,.0f} win/s)")
    return out


def write_outputs(transformer):
    json.dump(transformer, open(HERE / "06_latency_results.json", "w"), indent=2)

    has_mps = "mps" in transformer
    BUDGET = 33.3  # ms per frame at 30 fps
    single = {d: transformer[d]["batch"][1]["ms_per_window"] for d in transformer}
    best_dev = min(single, key=single.get)
    best_single = single[best_dev]
    cpu1 = transformer["cpu"]["batch"][1]["ms_per_window"]
    mps1 = transformer["mps"]["batch"][1]["ms_per_window"] if has_mps else None
    lstm_cpu1 = LSTM_LATENCY["cpu"]["batch"]["1"]["ms_per_window"]
    lstm_mps1 = LSTM_LATENCY["mps"]["batch"]["1"]["ms_per_window"] if "mps" in LSTM_LATENCY else None

    L = ["# Phase T5 — Isolated inference-latency benchmark (Apple M4)", "",
        "Issue-9 protocol applied to the Phase-T3 winner (`transformer_searched`, "
        f"794,241 params, num_layers=4, d_model=128) on this M4 — the same hardware "
        "the live demo runs on. Inference only, no training. MPS timings synchronise "
        f"inside each timed call (`torch.mps.synchronize`), {WARMUP} warmup + {TIMED} "
        "timed forwards per cell. Loads the real `transformer_searched/seed42/best.pt` "
        "(latency is weight-independent, but this matches Issue 9's own convention).",
        "",
        "## Transformer latency vs BiLSTM baseline", "",
        "| device | batch | ms / forward | ms / **window** | windows / s | p99 ms | "
        "BiLSTM ms/window (ref) |",
        "|---|---|---|---|---|---|---|"]
    for dev in transformer:
        lstm_dev = LSTM_LATENCY.get(dev, {}).get("batch", {})
        for b in BATCHES:
            c = transformer[dev]["batch"][b]
            lstm_ref = lstm_dev.get(str(b), {}).get("ms_per_window")
            lstm_str = f"{lstm_ref:.4f}" if lstm_ref is not None else "n/a"
            L.append(f"| {dev.upper()} | {b} | {c['ms_forward']:.3f} | "
                     f"**{c['ms_per_window']:.4f}** | {c['windows_per_s']:,.0f} | "
                     f"{c['ms_p99']:.3f} | {lstm_str} |")

    ratio_cpu1 = cpu1 / lstm_cpu1  # >1 => transformer slower; <1 => transformer faster
    transformer_faster = ratio_cpu1 < 1.0
    factor = (1 / ratio_cpu1) if transformer_faster else ratio_cpu1
    comparison_word = "faster" if transformer_faster else "slower"

    insight = ""
    if has_mps and mps1 > cpu1:
        insight = (f" As with the BiLSTM, **CPU beats MPS at batch 1** "
                  f"({cpu1:.3f} vs {mps1:.3f} ms) — GPU kernel-dispatch overhead "
                  f"dominates a model this small at batch 1. At batch 32 MPS reaches "
                  f"{transformer['mps']['batch'][32]['ms_per_window']:.4f} ms/window vs "
                  f"CPU {transformer['cpu']['batch'][32]['ms_per_window']:.4f}.")

    L += ["",
         f"**The transformer is {factor:.2f}x {comparison_word} than the BiLSTM per "
         f"window** at batch 1 CPU ({cpu1:.4f} ms vs {lstm_cpu1:.4f} ms) — "
         + (f"despite having ~1.3x the parameters (794,241 vs 594,561), the fully "
            f"parallel self-attention forward pass over T=16 tokens apparently "
            f"outruns the BiLSTM's inherently sequential recurrence (which must step "
            f"through the sequence one timestep at a time and can't parallelize "
            f"across time), on this hardware and batch size."
            if transformer_faster else
            f"the extra parameters (794,241 vs 594,561) and 4 self-attention+FFN "
            f"layers cost more per window than the BiLSTM's 2 recurrent layers.")
         + f" **In absolute terms both are effectively free**: fastest single-window "
         f"latency is **{best_single:.3f} ms** ({best_dev.upper()}, batch 1) = "
         f"{1e3/best_single:,.0f} windows/s — about **{BUDGET/best_single:.0f}x "
         f"inside** a 30 fps frame budget ({BUDGET:.1f} ms).{insight} At T=16, "
         f"self-attention's O(T²) cost (16²=256 pairs) is trivial next to the linear "
         f"projections and FFN either way — sequence length is not the deciding "
         f"factor here.", "",
         "## Verdict", "",
         f"The transformer is {factor:.2f}x {comparison_word} than the BiLSTM per "
         f"window ({cpu1:.4f} vs {lstm_cpu1:.4f} ms CPU batch-1), but **neither model "
         f"is a latency concern on this hardware**: both are roughly 2 orders of "
         f"magnitude inside the 30 fps budget, and the live pipeline remains "
         f"detection-bound (Issue 9: YOLO26-M is the actual bottleneck; the intent "
         f"model — either architecture — is a rounding error next to it). If "
         f"deploying the transformer in place of the BiLSTM, expect no perceptible "
         f"change in end-to-end frame rate."]
    (HERE / "06_latency_report.md").write_text("\n".join(L) + "\n")
    print("\nwrote 06_latency_results.json, 06_latency_report.md")


def main():
    print(f"M4 latency benchmark (transformer_searched) | devices {devices()} | "
         f"{WARMUP} warmup + {TIMED} timed\n")
    transformer = bench_transformer()
    write_outputs(transformer)


if __name__ == "__main__":
    main()

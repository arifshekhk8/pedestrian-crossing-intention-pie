# Issue 3 — Master comparison table (INTERNAL reference, for understanding)

**What this is.** A *superset* reference table of PIE crossing-intention methods —
every column a reviewer or you might want (Acc / AUC / F1 / **Precision / Recall /
Params / Latency / hyperparameters / modalities / split**), and **new 2024–2026
papers** added on top of the finalized baseline set. This is the "for-my-understanding"
working table.

> **This file is NOT the publication table.** The lean, publication-finalized table is
> [`03_baseline_comparison.md`](03_baseline_comparison.md) — it deliberately carries
> only clean, standard-protocol, own-paper-verified rows. This master table is broader
> and more permissive: it keeps off-protocol and secondary-sourced rows so you can see
> the whole landscape, each clearly flagged. **Do not paste ◻/⚠ rows into the
> manuscript without first-hand verification.**

## How to read the flags (read this first — it is the whole point)

| Flag | Meaning | Trust for the paper? |
|---|---|---|
| ✅ | Number verified **first-hand** against the method's **own paper** (or GTransPDM Table I / PedFormer Table I, which we read directly) | Yes |
| ◻ | **Secondary** — transcribed from *another* paper's comparison table (mostly MFT Table 1, Nov-2025 arXiv). Cross-paper tables in this field disagree (see caveats), so treat as **indicative only** | Verify first |
| ⚠ | **Off-protocol / not directly comparable** — different split, different task, or different horizon. Kept for context; **candidate to remove** | No — context only |
| ~ | Approximate / our own value not yet recomputed exactly | — |
| N/R | Not reported by the source | — |

**⚠ The single most important caveat.** The recent comparison tables (MFT, GTransPDM,
ACIT) **do not agree with each other**, because each re-implements or re-transcribes
the others. Example: GTransPDM's *own* paper reports its full model at 0.90/0.87/0.82
(and 0.92/0.90/0.86 without the pose stream), but MFT's Table 1 lists "GTransPDM" as
0.90/0.87/**0.87**/0.86. So a ◻ number is *someone's transcription*, not ground truth.
Anywhere it matters, go to the primary PDF.

---

## Table 1 — Accuracy / ranking metrics (PIE, standard protocol unless flagged)

Standard protocol = train set01/02/04 · val set05/06 · **test set03**; obs 16 frames
(0.5 s); TTE 30–60 frames (1–2 s); 32.5 % positive. Rows sorted newest-ish / by family.

| Method | Year | Venue | Acc | AUC | **F1** | Prec | Rec | Modalities (#streams) | Src / flag |
|---|---|---|---|---|---|---|---|---|---|
| **— Our models (this thesis) —** | | | | | | | | | |
| **BiLSTM (clean)** | 2026 | this work | 0.883 | 0.932 | 0.828 | 0.794 | 0.865 | bbox + ego-speed (**2**) | ✅ ours |
| **Transformer (searched)** | 2026 | this work | 0.894 | **0.950** | 0.845 | N/R | N/R | bbox + ego-speed (**2**) | ✅ ours |
| **BiLSTM-F1** (F1-first) | 2026 | this work | 0.897 | 0.940 | **0.844** | N/R | N/R | bbox + ego-speed (**2**) | ✅ ours |
| **Transformer-F1** (F1-first) | 2026 | this work | 0.896 | 0.947 | **0.847** | N/R | N/R | bbox + ego-speed (**2**) | ✅ ours |
| **GRU-F1** (recurrent-cell twin) | 2026 | this work | 0.901 | 0.941 | **0.849** | N/R | N/R | bbox + ego-speed (**2**) | ✅ ours |
| **RNN-F1** (un-gated recurrent twin) | 2026 | this work | 0.902 | 0.948 | **0.852** | N/R | N/R | bbox + ego-speed (**2**) | ✅ ours |
| **— Verified baselines (own paper / GTransPDM Tbl I / PedFormer Tbl I) —** | | | | | | | | | |
| PCPA | 2021 | WACV | 0.87 | 0.86 | 0.77 | 0.76 | 0.81 | bbox+pose+context+speed (4) | ✅ |
| SingleRNN | 2020 | IV | 0.81 | 0.75 | 0.64 | 0.67 | N/R | multimodal RNN | ✅ (GTransPDM Tbl I) |
| SF-RNN | 2020 | BMVC | 0.82 | 0.79 | 0.69 | 0.67 | N/R | multimodal RNN | ✅ (GTransPDM Tbl I) |
| MultiRNN | 2019 | — | 0.83 | 0.80 | 0.71 | 0.69 | N/R | multimodal RNN | ✅ (GTransPDM Tbl I) |
| IntFormer | 2021 | arXiv | 0.89 | 0.92 | 0.81 | N/R | N/R | multimodal | ✅ (GTransPDM Tbl I) |
| TrouSPI-Net | 2021 | FG | 0.88 | 0.88 | 0.80 | 0.73 | 0.89 | pose + bbox (atrous+U-GRU) | ✅ (GTransPDM Tbl I) |
| Pedestrian Graph+ | 2022 | T-ITS | 0.89 ‡ | 0.90 ‡ | 0.81 ‡ | 0.83 | N/R | pose graph + ego (2–3) | ✅ ‡ |
| FF-STP | 2021 | — | 0.89 | 0.86 | 0.80 | 0.79 | N/R | multimodal | ✅ (GTransPDM Tbl I) |
| PIT | 2023 | T-ITS | 0.91 | 0.92 | 0.82 | 0.84 | N/R | multimodal transformer | ✅ (GTransPDM Tbl I) |
| BiPed | 2023 | ICCV | 0.91 ‡ | 0.90 ‡ | 0.85 ‡ | 0.82 | N/R | multimodal (multitask) | ✅ ‡ (PedFormer Tbl I) |
| **PedFormer** | 2023 | ICRA | **0.93** | 0.90 | **0.87** | N/R | N/R | multimodal (traj+action multitask) | ✅ (own Tbl I) — **F1/Acc ceiling** |
| **— New 2024–2026 (verify ◻ before quoting) —** | | | | | | | | | |
| **GTransPDM** (full) | 2024 | arXiv Sep | 0.90 | 0.87 | 0.82 | 0.86 | N/R | bbox + pose + ego (3) | ✅ (own paper) |
| **GTransPDM** (w/o pose) | 2024 | arXiv Sep | 0.92 | 0.90 | 0.86 | N/R | N/R | bbox + ego (**2**) — closest cousin | ✅ (own "w/o Xke") |
| **PedCMT** | 2024 | **T-ITS** | ~0.92 | ~0.81 | ~0.876 | ~0.79 | N/R | **bbox + ego-speed (2)** + multitask | ◻ garbled 2ary — **verify from paper/code** |
| **Faster-PCPNet** | 2024 | IEEE (ITSC/T-ITS) | **0.94** | 0.92 | **0.89** | 0.89 | 0.88 | pose+ego+bbox+polar-coord | Acc ✅ (2 sources); rest ◻ |
| **RAIDN** | ~2024 | (MFT ref 36) | 0.92 | 0.89 | 0.85 | 0.82 | 0.89 | multimodal | ◻ (MFT Tbl 1) |
| **LSOP-Net** | ~2024 | (MFT ref 24) | 0.89 | 0.87 | 0.81 | 0.80 | 0.82 | raw-modality, implicit cues | ◻ (MFT Tbl 1) |
| **PFRN** | ~2024 | (MFT ref 19) | 0.90 | 0.85 | 0.77 | 0.81 | 0.74 | multimodal | ◻ (MFT Tbl 1) |
| **STFF-MANet** | ~2024 | (MFT ref 21) | 0.89 | 0.88 | 0.82 | 0.79 | 0.85 | multimodal | ◻ (MFT Tbl 1) |
| **MMH-PAP** | ~2022 | IV (MFT ref 16) | 0.89 | 0.88 | 0.81 | 0.77 | N/R | multimodal hybrid | ◻ (MFT Tbl 1) |
| **V-PedCross** | ~2024 | (MFT ref 10) | 0.89 | 0.88 | 0.67 | 0.74 | 0.84 | visual-only | ◻ (MFT Tbl 1) |
| **RU-LSTM** | ~2022 | (MFT ref 20) | 0.87 | 0.84 | 0.77 | N/R | N/R | multimodal RNN | ◻ (MFT Tbl 1) |
| **MTMGN** | ~2023 | (MFT ref 22) | 0.90 | 0.87 | 0.92 (?) | 0.95 (?) | 0.90 | multimodal graph | ◻ **F1>AUC looks wrong — likely transcription error** |
| **Dual-STGAT** | ~2024 | (MFT ref 23) | 0.86 | 0.87 | 0.91 (?) | 0.92 (?) | 0.90 | spatio-temporal graph-attn | ◻ **F1>AUC looks wrong — likely transcription error** |
| **MFT** (Multi-Context Fusion Transf.) | 2025 | arXiv Nov | 0.90 | **0.94** | 0.83 | 0.83 | 0.82 | 4 context streams (P/L/V/E) | ✅ (own Tbl 1); **split text differs †2** |
| **— ⚠ Off-protocol — keep as context, CAN REMOVE —** | | | | | | | | | |
| **PIP-Net** ⚠ | 2025 | T-ITS | ~0.91 | ~0.90 | ~0.84 | N/R | 0.84 | **7 modalities** (RGB+flow+semseg+depth+…) | ⚠ **custom random split ~50/40/10** — not standard protocol |
| **PIEPredict** ⚠ | 2019 | ICCV | — | — | — | — | — | bbox + ego + context | ⚠ **trajectory predictor**, not a binary classifier — no native Acc/AUC/F1 |
| **Occlusion-Aware Diffusion** ⚠ | 2025 | T-ITS | 0.90 | 0.95 | 0.90 | N/R | N/R | bbox + ego (2) | ⚠ **occluded-only, ~1-frame horizon** — modality precedent, not a row |
| **ACIT** ⚠ | 2025 | arXiv Nov | — | — | — | — | — | 5 streams (RGB+flow+pose+ctx+speed) | ⚠ **JAAD-only paper — no PIE number** (kept for its cost table below) |

‡ GTransPDM flags Ped-Graph+ and BiPed as configured differently from the standard
protocol ("Except BiPed and Pedestrian Graph+…") — verify vs their originals.
†2 MFT's text states train set01/02/**06** · val set04 · test set03, which differs from
the canonical set01/02/04 · 05/06 · 03; likely a paraphrase slip, but it is another
reason its transcribed rows are ◻.

---

## Table 2 — Efficiency & complexity (params / size / latency / key hyperparameters)

**⚠ Latency is NOT cross-comparable across rows.** The baseline latencies come from the
**ACIT** and **MFT** cost tables, measured on **their desktop GPUs** (RTX 3090 Ti class)
for the **full model**. Ours are **M4** (CPU or Apple-Metal GPU), **classifier-forward
only**. Different hardware *and* different scope. The honest efficiency story is
**parameter count + stream count** (0.6–0.8 M params / 2 streams vs 5–61 M / 3–7
streams) and **no auxiliary extractor** — *not* a hardware-matched ms benchmark. That
said, we beat every baseline by **15–40×** on **any** of our measurements (our slowest,
GPU batch-1 = 1.65 ms, is still ~14× under MFT's 23 ms), so nothing needs cherry-picking.

**Which of our latency numbers to quote → CPU batch-1** (see the breakdown table below,
Table 2b). At batch-1 (single-window real-time latency) our **CPU is 2.4–2.9× lower than
our GPU** — a tiny 0.6–0.8 M model is dominated by GPU kernel-launch/transfer overhead,
so the GPU only wins on *batched throughput*, which is a different metric. CPU batch-1 is
the honest single-inference latency, it is also our *lower* batch-1 number, and it is
deterministic (±0.008 ms vs the GPU's ±0.77 ms).

| Method | Params | Model size | Latency (their HW) | Key hyperparameters | Src |
|---|---|---|---|---|---|
| **BiLSTM (ours)** | **594,561** (0.59 M) | ~2.3 MB | **0.575 ms** (M4 CPU, b1; see 2b) | h128, 2-layer BiLSTM, dropout 0.3, Adam lr 1e-3, pos_weight 1.682, obs 16 | ours |
| **Transformer (ours, searched)** | **794,241** (0.79 M) | ~3.1 MB | **0.459 ms** (M4 CPU, b1; see 2b) | d128, ff512, 4 layers, 8 heads, last-token pool, sinusoidal PE, dropout 0.1, wd 1e-5, lr 1e-3 | ours |
| **BiLSTM-F1 (ours)** | ≈2–3 M (h256)~ | ~9 MB | (h256; Issue-9 latency measured at h128) | **h256**, 2-layer, F1-checkpointing, val-τ*≈0.5, pos_weight 1.682 | ours |
| **Transformer-F1 (ours)** | 794,241 (same arch) | ~3.1 MB | 0.459 ms (M4 CPU, b1) | searched cfg, F1-checkpointing, val-τ*≈0.5, pos_weight 2.5(val)→1.682(test) | ours |
| **GRU-F1 (ours)** | 1,678,209 (1.68 M) | ~6.7 MB | 0.721 ms (M4 CPU, b1) | h256, 2-layer bidir GRU, dropout 0.3, Adam lr 5e-4, F1-ckpt, val-τ*≈0.53, pos_weight 1.682 | ours |
| **RNN-F1 (ours)** | 560,001 (0.56 M) | ~2.2 MB | **0.316 ms** (M4 CPU, b1) — fastest | h256, 2-layer bidir vanilla RNN (tanh), dropout 0.2, Adam lr 1e-4, F1-ckpt, val-τ*≈0.5, pos_weight 1.682 | ours |
| PCPA | 31.17 M | 118.80 MB | 38.60 ms | 3D-CNN + RNN, 4 streams | ACIT/MFT cost tbl |
| Global PCPA | 60.92 M | 374.20 MB | 70.83 ms | larger PCPA variant | ACIT/MFT cost tbl |
| FUSSI-Net | 1.00 M | 8.40 MB | 34.92 ms | skeleton fusion RNN | ACIT/MFT cost tbl |
| MTC | 8.25 M | 99.70 MB | 36.23 ms | multi-task | MFT cost tbl |
| VMI | N/R | 19.07 MB | 11.03 ms | — | MFT cost tbl |
| MTMGN | N/R | N/R | 56.00 ms | multimodal graph | MFT cost tbl |
| ACIT (JAAD model) | 5.15 M | 62.50 MB | 43.93 ms | Swin-V2 + dual-path attn, dropout 0.3, L2 1e-3, Adam lr 2e-5 | ACIT/MFT cost tbl |
| **MFT** | **0.95 M** | 9.40 MB | 23.20 ms | 4 heads, d128, dropout 0.2, Adam lr 2e-5 (PIE), 60 ep, bs 2 | MFT cost tbl |
| GTransPDM | N/R (4 GCN + 4 Transf, d64) | N/R | "0.05 ms" (own claim) | 4 GCN blocks, 4 Transf layers, 64-d | ✅ own paper |
| PedCMT | N/R | N/R | N/R | cross-modal transformer + uncertainty multitask | github.com/xbchen82/PedCMT |

Takeaways from Table 2: (1) **FUSSI (1.0 M) and MFT (0.95 M) are the only published
sub-million-param models** — ours (0.59 M / 0.79 M) sit right there, and below them,
while using **fewer input streams**. (2) The heavyweight RGB models (PCPA 31 M, Global
PCPA 61 M) are 30–60× our size. (3) GTransPDM's "0.05 ms" is its own headline and not
independently reproduced; treat with the same skepticism as our sub-ms number.

### Table 2b — our latency, full CPU-vs-GPU breakdown (M4, so no cherry-picking)

Measured on the M4, classifier-forward pass only, 100 timed runs.
Sources: [`../issue9_latency/09_latency_results.json`](../issue9_latency/09_latency_results.json)
(BiLSTM) and [`../../transformer/phase5_analysis/06_latency_results.json`](../../transformer/phase5_analysis/06_latency_results.json)
(Transformer).

| Model | Device | **Batch-1 latency** (ms/win) | Batched throughput (ms/win, b32) |
|---|---|---|---|
| BiLSTM | CPU (M4) | **0.575** ±0.008 | 0.135 |
| BiLSTM | GPU (M4 MPS) | 1.647 ±0.773 | **0.083** |
| Transformer | CPU (M4) | **0.459** ±0.018 | **0.084** |
| Transformer | GPU (M4 MPS) | 1.388 ±0.135 | 0.116 |

Reading it: at **batch-1** (the real-time, one-pedestrian latency) **CPU beats GPU** for
both models — GPU launch/transfer overhead swamps the compute of a sub-million-param
model. GPU only wins on **batched throughput** for the BiLSTM (0.083 vs 0.135); for the
Transformer, CPU wins even batched. So "GPU latency was lower" is really the *batch-32
throughput* number (0.083) — a different metric from single-window latency, and not the
one to put in a latency column. **Quote CPU batch-1** (0.459–0.575 ms): it is the honest
single-inference latency, our lower batch-1 figure, and deterministic. Every one of these
still lands 14–70× under the multimodal baselines' 23–71 ms — the parsimony claim does
not depend on which we pick.

---

## Table 3 — Our models, full reproducible config (the row we can defend in depth)

| | BiLSTM (clean) | Transformer (searched) | BiLSTM-F1 | Transformer-F1 |
|---|---|---|---|---|
| Params | 594,561 | 794,241 | ≈2–3 M (h256) | 794,241 |
| Inputs | bbox + ego-speed | bbox + ego-speed | bbox + ego-speed | bbox + ego-speed |
| obs_len / TTE | 16 / 30–60 | 16 / 30–60 | 16 / 30–60 | 16 / 30–60 |
| Hidden / depth | h128 / 2 | d128 ff512 / 4 layers, 8 heads | **h256** / 2 | d128 ff512 / 4 layers |
| Pool / PE | (BiLSTM final) | last-token / sinusoidal | (BiLSTM final) | last-token / sinusoidal |
| Dropout / wd | 0.3 / 0 | 0.1 / 1e-5 | 0.3 / 0 | 0.1 / 1e-5 |
| Optimizer / lr | Adam / 1e-3 | Adam / 1e-3 | Adam / 1e-3 | Adam / 1e-3 |
| pos_weight | 1.682 | 1.682 | 1.682 | 2.5(val)→1.682(test) |
| Selection / stop | val AUC | val AUC | **best val F1** (hybrid) | **best val F1** (hybrid) |
| Threshold | 0.5 | 0.5 | val-τ* ≈ 0.5 | val-τ* ≈ 0.5 |
| **Acc / AUC / F1** | 0.883 / 0.932 / 0.828 | 0.894 / 0.950 / 0.845 | 0.897 / 0.940 / **0.844** | 0.896 / 0.947 / **0.847** |
| Precision / Recall | 0.794 / 0.865 | N/R | N/R | N/R |
| Seeds | [42,0,1,2,3] | [42,0,1,2,3] | [42,0,1,2,3] | [42,0,1,2,3] |

(Deployable 5-seed **probability-ensemble** F1: BiLSTM-F1 **0.856**, Transformer-F1
**0.857** — a *different statistic* from the per-seed means above; never mixed into the
comparison tables.)

---

## Where our result sits (the reading, F1-first)

- **On F1** (supervisor's primary metric): our F1-optimized models are **0.844–0.847**.
  The verified standard-protocol F1 band is **0.77 → 0.87** (ceiling PedFormer 0.87;
  next GTransPDM-w/o-pose 0.86). If Faster-PCPNet's 0.89 and RAIDN's 0.85 survive
  first-hand verification, the ceiling rises — **but both use more streams than us**
  (Faster-PCPNet adds pose + polar coords). We are within **0.02–0.03 of the multimodal
  ceiling with 2 raw streams**.
- **On AUC**: our 0.940–0.950 is **at/above the top of the table** (next: MFT 0.94,
  PIT/IntFormer 0.92). MFT (Nov-2025) now **ties our AUC at 0.94** — worth citing as the
  current strongest published AUC, and it does it with 0.95 M params / 4 context streams
  vs our 0.79 M / 2 streams.
- **On Accuracy**: mid/upper-mid (0.896–0.897) vs a band topping at PedFormer 0.93 and
  Faster-PCPNet 0.94. We do **not** lead on Acc — stating that is what keeps the AUC
  claim credible.
- **Parsimony is the through-line**: 2 raw streams, 0.6–0.8 M params, no
  pose/flow/semseg/depth extractor — within a couple of F1 points of models using 3–7
  streams and 5–61 M params.

## Sources

- PCPA / benchmark: Kotseruba et al., WACV 2021.
- GTransPDM: arXiv 2409.20223 (own Table I — read first-hand).
- PedFormer: arXiv 2210.07886 (own Table I).
- **MFT** (Multi-Context Fusion Transformer): arXiv 2511.20011 (Nov 2025) — Table 1
  (many secondary rows) + Table 4 (cost).
- **ACIT**: arXiv 2511.20020 (Nov 2025) — JAAD-only; Table III (cost).
- **Faster-PCPNet**: IEEE Xplore 10418196 (2024).
- **PedCMT**: IEEE T-ITS 2024, 10.1109/TITS.2024.3386689; code github.com/xbchen82/PedCMT.
- PIP-Net: arXiv 2402.12810 / IEEE T-ITS 2025 (⚠ custom split).
- PIEPredict: Rasouli et al., ICCV 2019 (⚠ trajectory model).
- Occlusion-Aware Diffusion: arXiv 2511.00858 (⚠ occluded-only).

> **To promote any ◻ row to the manuscript:** open the primary PDF, confirm the split
> is standard set01/02/04·05/06·03, confirm the number, then move it into
> `03_baseline_comparison.md` with a ✅. Until then it stays here.

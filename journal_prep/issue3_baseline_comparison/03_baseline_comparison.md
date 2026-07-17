# Issue 3 — Published-Baseline Comparison (PIE crossing prediction)

**Purpose.** Place our leak-free result against published PIE crossing-intention
numbers, with a *modality* column so the input-parsimony story is visible. Valid
only now that Issues 1–2 (leakage removed, canonical protocol) are done — the
number below is the clean `0.932`, **not** the old leaky `0.931`.

> **Verification status (2026-06):** baseline figures below are now confirmed
> against primary sources (mostly GTransPDM's Table I, which tabulates the field on
> PIE, plus each method's own paper). Remaining `[verify]` items are flagged.
> Citations are abbreviated pending a BibTeX pass.

---

## Our result (clean protocol, 5-seed)

| | Acc | AUC | PR-AUC | F1 | Precision | Recall | Inputs |
|---|---|---|---|---|---|---|---|
| **BiLSTM (ours, clean)** | 0.883 ± 0.009 | 0.932 ± 0.011 | 0.876 ± 0.016 | 0.828 ± 0.012 | 0.794 ± 0.022 | 0.865 ± 0.024 | bbox + ego-speed (**2 streams**) |
| **Transformer (ours, searched)** | 0.894 ± 0.009 | **0.950 ± 0.003** | 0.901 ± 0.010 | 0.845 ± 0.013 | — | — | bbox + ego-speed (**2 streams**) |
| **BiLSTM-F1 (ours, F1-first program)** | 0.897 ± 0.006 | 0.940 ± 0.004 | — | **0.844 ± 0.008** | — | — | bbox + ego-speed (**2 streams**) |
| **Transformer-F1 (ours, F1-first program)** | 0.896 ± 0.011 | 0.947 ± 0.003 | — | **0.847 ± 0.017** | — | — | bbox + ego-speed (**2 streams**) |
| **GRU-F1 (ours, `gru/` study)** | 0.901 ± 0.010 | 0.941 ± 0.007 | — | **0.849 ± 0.011** | — | — | bbox + ego-speed (**2 streams**) |
| **Vanilla RNN-F1 (ours, `rnn/` study)** | 0.902 ± 0.008 | 0.948 ± 0.002 | — | **0.852 ± 0.012** | — | — | bbox + ego-speed (**2 streams**) |

The two `-F1` rows are the supervisor-directed F1-first optimization (`f1_optimization/`,
2026-07-12: val-tuned operating point + best-val-F1 checkpointing + F1-re-selected
config for the LSTM — `h256` — all selection on val only). Numbers above are at
**threshold 0.5** (their val-tuned τ\* landed ≈0.5, so the 0.5-threshold numbers are
the cross-paper-comparable ones); the deployable 5-seed probability ensembles reach
F1 **0.856** (BiLSTM-F1) / **0.857** (Transformer-F1) — a different statistic, always
labeled. Verdicts: the LSTM's F1 gain is significant (ΔF1 +0.0187, 95% CI
[+0.0073, +0.0300]); the transformer's is not; and **under F1 the two families are a
statistical TIE** (ΔF1 +0.0008, CI [−0.0124, +0.0142]) — the transformer's AUC win
does not carry to F1.

The **GRU-F1 row** is the recurrent-cell follow-up (`gru/`, 2026-07-14): the GRU is the
BiLSTM's gated twin (only `nn.LSTM`→`nn.GRU`), given the identical Issue-8 search + F1-first
optimization on the same unified CPU engine. It **ties the BiLSTM-F1 on F1** (10k paired
bootstrap ΔF1 +0.0071, CI [−0.0043, +0.0187]) and **ties the frozen BiLSTM on AUC at matched
capacity/selection** (ΔAUC −0.0008, CI [−0.0039, +0.0021]), both robust to the
pedestrian-cluster bootstrap; it does **not** reach the searched transformer's AUC (ΔAUC
−0.0070, CI [−0.0101, −0.0038]). So the recurrent cell type is not what moves the metrics —
the input signal is — while the transformer's AUC edge stays attributable to its search, not
to attention-vs-recurrence. Caveat: the GRU F1-winner is a larger h256 model (1.68 M params);
the matched-size 446k GRU is the one that ties the 595k BiLSTM on AUC. Full detail:
`gru/phase5_analysis/07_comparison_report.md`.

The **Vanilla RNN-F1 row** is the un-gated recurrent follow-up (`rnn/`, 2026-07-14): a
bidirectional plain tanh RNN, the BiLSTM's twin with its **gating removed** (only
`nn.LSTM`→`nn.RNN`), given the identical Issue-8 search + F1-first optimization on the same
unified CPU engine. It **ties BiLSTM-F1 and GRU-F1 on F1** (10k paired bootstrap ΔF1 +0.0033,
CI [−0.0083, +0.0187] vs BiLSTM-F1; ΔF1 −0.0038, CI [−0.0117, +0.0039] vs GRU-F1) and is
level-to-marginally-better than the frozen BiLSTM on AUC at matched h128 capacity/selection
(ΔAUC +0.0059, CI [+0.0032, +0.0088]); **no cell-isolation endpoint is a loss**, all robust to
the pedestrian-cluster bootstrap. And — unlike the GRU, which *lost* to the searched transformer
on AUC — the AUC-optimized vanilla RNN **ties it** (ΔAUC −0.0013, CI [−0.0041, +0.0015]): once an
un-gated recurrent net gets the same search, it reaches the same ~0.95 AUC, confirming that edge
was the *search*, not attention. So **not even the LSTM's gating is what moves the metrics over
this 16-step window — the input signal is** (the strongest form of the claim: LSTM ≈ GRU ≈ vanilla
RNN). The vanilla RNN is also the smallest (h256 winner 560k; h128 149k) and fastest (0.316
ms/window CPU) of the four families. Caveat: this holds *at a 16-step horizon* — an un-gated RNN
would likely fall behind over long sequences. Full detail: `rnn/phase5_analysis/07_comparison_report.md`.

Test = PIE set03, 2,094 windows (**32.5% positive**); mean ± std over seeds
[42,0,1,2,3]; obs_len 16, TTE∈[30,60], 0.5 overlap, threshold 0.5. The full
evidence chain is now complete: **bootstrap 95% CI ≈ [0.92, 0.95]** (Issue 4, 10k
resamples), **LOSO 6-fold AUC 0.928 ± 0.041** with set03 = 0.931 ≈ fixed-split
(Issue 5, so set03 isn't an easy fold), and eval-parity verified (Issue 2:
per-pedestrian 0.914 ≈ per-window 0.913, benchmark-min-track subset 0.919) — so the
0.932 is **not** an easier-evaluation artifact.

**Update (`transformer/`, 2026-07-12): a small pre-LN Transformer encoder over the
identical 2-stream input, with a 78-config staged architecture+recipe search
(val-only selection, test touched exactly once), **measurably beats the BiLSTM on
AUC** — and on AUC specifically; under the F1-first hierarchy the families TIE, see
the metric-choice section below — on the same frozen protocol and the same 2,094 test
windows: a 10k paired bootstrap of ΔAUC gives **+0.0135, 95% CI [+0.0097, +0.0174]**
(excludes 0), paired t-test p=0.025. Critically, the *un-searched* transformer (same architecture family, LSTM's
own recipe, zero tuning) is a statistical **tie** with the BiLSTM
(Δ=+0.0005, CI [-0.0034, +0.0043], p=0.83) — so the win is attributable to the search
finding a better architecture (deeper: 4 layers vs 2; last-token pooling; sinusoidal
positional encoding), not to attention over recurrence per se. The BiLSTM row above
remains the primary reported result for this table's existing evidence chain
(bootstrap CI, LOSO, eval-parity, detector robustness); the Transformer row is
reported alongside it as the strongest model found, with the full comparison in
`transformer/phase5_analysis/05_comparison_report.md` and `transformer/PLAN.md`.
Latency (M4, isolated forward pass): the transformer is *faster* per window than the
BiLSTM despite ~1.3× the parameters (0.459 vs 0.575 ms, CPU batch-1; artifact of
record `transformer/phase5_analysis/06_latency_results.json`) — fully-parallel
self-attention over T=16 tokens apparently outruns the BiLSTM's sequential
recurrence on this hardware, so neither model is a latency concern for the live
pipeline (still detection-bound, Issue 9).

### On metric choice (updated 2026-07-13: F1-first, per the supervisor's directive)

**Reporting hierarchy: F1 first, then accuracy, then AUC** (supervisor directive,
implemented end-to-end in `f1_optimization/`). This is consistent with the
literature's practice — every PIE benchmark paper reports Acc/AUC/F1 jointly (PCPA,
GTransPDM, PIP-Net, PedFormer all do) — though we do not claim the field treats F1 as
primary; F1 is the imbalance-appropriate operating-point metric and the one our
supervisor prioritizes. AUC (+ PR-AUC) remains reported as the threshold-free
corroboration, where both our models still lead the standard-protocol table.

Context that stays true and defensible:

- **Cross-paper tables compare at each paper's own single operating point** (no PIE
  paper we verified states a threshold-selection procedure — implicitly 0.5). Our
  table rows are therefore **at threshold 0.5**. The F1-first program *does* select an
  operating point on validation (never test), but its F1-optimized models' val-optimal
  τ\* landed ≈0.5, so the @0.5 and @τ\* numbers coincide to ±0.001 — no asterisk
  needed. Val-tuned and ensemble numbers appear only outside the comparison table,
  explicitly labeled.
- **The operating point is deliberately recall-favoring** (`pos_weight=1.682` →
  R 0.865 > P 0.794 for the frozen baseline): in an AV-safety setting a missed crosser
  costs more than a false alarm. The F1-first program swept pos_weight symmetrically
  for both families ({1.0…2.5}, val-selected) — the anchor 1.682 won for the LSTM;
  the transformer's marginal val preference for 2.5 did not transfer to test
  (reported plainly in `f1_optimization/06_comparison_report.md`).
- **Modest Acc with the highest AUC is the *opposite* of an "easier-eval" artifact** —
  our 50%-overlap protocol admits 3.5× more, and Issue-2-confirmed *harder*,
  windows (short-track AUC 0.863 vs 0.919), which depresses thresholded metrics more
  than ranking AUC.

## Comparison table — standard PIE protocol

All rows use the standard PIE benchmark (train set01/02/04 · val set05/06 · test
set03; obs T=16 frames / 0.5 s; TTE 30–60 frames / 1–2 s; metrics Acc/AUC/F1).
✅ = figure confirmed from the cited source.

| Method | Venue / Year | Acc | AUC | F1 | Modalities (streams) | Src |
|---|---|---|---|---|---|---|
| **PCPA** (Kotseruba et al.) | WACV 2021 | 0.87 ✅ | 0.86 ✅ | 0.77 ✅ | bbox + pose + context + speed (4) | benchmark; GTransPDM Tbl I + PIP-Net Tbl |
| **Pedestrian Graph+** | 2022 | 0.89 ✅ ‡ | 0.90 ✅ ‡ | 0.81 ✅ ‡ | pose graph + ego (2–3) | GTransPDM Tbl I |
| **IntFormer** | 2021 | 0.89 ✅ | 0.92 ✅ | 0.81 ✅ | multimodal | GTransPDM Tbl I |
| **PIT** | 2023 | 0.91 ✅ | 0.92 ✅ | 0.82 ✅ | multimodal | GTransPDM Tbl I |
| **BiPed** | 2023 | 0.91 ✅ ‡ | 0.90 ✅ ‡ | 0.85 ✅ ‡ | multimodal | PedFormer Tbl I; GTransPDM Tbl I |
| **PedFormer** (Rasouli & Kotseruba) | 2023 | **0.93** ✅ | 0.90 ✅ | **0.87** ✅ | multimodal (traj+action multitask) | PedFormer Tbl I (arXiv 2210.07886) |
| **GTransPDM** | arXiv Sept 2024 | 0.90 ✅ † | 0.87 ✅ | 0.82 ✅ | bbox + pose + ego motion (3) | GTransPDM Tbl I |
| **GTransPDM (w/o pose)** | arXiv Sept 2024 | 0.92 ✅ | 0.90 ✅ | 0.86 ✅ | bbox + ego motion (2) | GTransPDM Tbl I ("w/o Xke") |
| **BiLSTM (ours, clean)** | 2026 | 0.883 | 0.932 | 0.828 | **bbox + ego-speed (2)** | this work |
| **Transformer (ours, searched)** | 2026 | 0.894 | **0.950** | 0.845 | **bbox + ego-speed (2)** | this work (`transformer/`) |
| **BiLSTM-F1 (ours)** | 2026 | 0.897 | 0.940 | 0.844 | **bbox + ego-speed (2)** | this work (`f1_optimization/`) |
| **Transformer-F1 (ours)** | 2026 | 0.896 | 0.947 | 0.847 | **bbox + ego-speed (2)** | this work (`f1_optimization/`) |
| **GRU-F1 (ours)** | 2026 | 0.901 | 0.941 | 0.849 | **bbox + ego-speed (2)** | this work (`gru/`) |
| **Vanilla RNN-F1 (ours)** | 2026 | 0.902 | 0.948 | 0.852 | **bbox + ego-speed (2)** | this work (`rnn/`) |

† GTransPDM's own Table I lists **0.90** Acc for the full model; its abstract's
**92%** headline is the **ablation without the skeleton-pose encoder** ("w/o Xke") —
given its own row above since, at 2 streams (bbox + ego motion), it is the closest
published cousin of our minimal-modality design (verified against arXiv 2409.20223,
2026-07-13).
‡ GTransPDM states: "Except BiPed and Pedestrian Graph+, other solutions show the
same data splits and configurations as ours" — these two rows are benchmark numbers
whose configuration GTransPDM itself flags as differing; verify against their
original papers in the BibTeX pass.

**PIP-Net (IEEE T-ITS 2025) was REMOVED from this table (2026-07-13):** its own paper
states a **custom random split** ("~50% (880 samples) training, 40% (719) testing,
10% (243) validation"), not the standard set01/02/04·05/06·03 protocol, and its
reported triple varies by paper version (arXiv v-fetched Table II: 0.91/0.90/0.84).
Like Occlusion-Aware Diffusion below, it is cited as context, not as a
protocol-comparable row. The previous revision of this table carried it at
0.915/0.897/0.846 marked "confirmed" — that was wrong on both counts.

**PIEPredict** (Rasouli et al., ICCV 2019; bbox + ego + context) is the
dataset-origin model but is a *trajectory* predictor, not a binary crossing
classifier — no directly comparable Acc/AUC/F1 row. It is the candidate for an
optional "run-it-on-our-split" experiment (below), not a transcribed row.

## ⚠ Occlusion-Aware Diffusion is a *modality precedent*, not a comparison row

The PLAN earmarked **Occlusion-Aware Diffusion** (Liu et al., arXiv 2511.00858,
accepted IEEE T-ITS Nov 2025) as the "apples-to-apples" row. On inspection it is
**not protocol-comparable**:

- It reports **only occluded scenarios** (EO1–EO5 element / PO1–PO5 partial
  occlusion) — there is **no standard fully-observed PIE number** in the paper.
- Its prediction horizon is **~1 frame ahead** (15-frame obs, 16th-frame target),
  far shorter/easier than our **TTE 30–60 frames (1–2 s)**.
- Its best PIE figure (EO5: Acc 0.90 / AUC 0.95 / F1 0.90, vs TrEP 0.85/0.91/0.85)
  is an *occlusion-robustness* result, not a benchmark crossing-prediction score.

What it **does** give us: confirmation that **bbox + ego-velocity only** is a
deliberate, published minimal modality ("the proposed model and TrEP only leverage
bounding boxes and ego-vehicle speed"). So cite it as a **precedent for the
two-stream design choice**, with the protocol caveat stated — do **not** put its
0.95 in the same table as our 0.932.

## How to read this (the honest framing)

- **On AUC, both of our models lead the standard-protocol table.** The BiLSTM
  (0.932) already topped the band (next: PIT / IntFormer 0.92); the searched
  Transformer (0.950) extends that lead further, using the same **2 input streams**
  vs the 3–7 of every method above it. This is the parsimony headline, now with two
  architectures confirming it independently.
- **On F1 (the supervisor's primary metric) our F1-optimized models are upper-mid
  band**: BiLSTM-F1 0.844 / Transformer-F1 0.847 against a verified standard-protocol
  band of 0.77–**0.87** (ceiling: PedFormer 0.87, a trajectory+action multitask
  multimodal model; next GTransPDM-w/o-pose 0.86). We get within 0.02–0.03 of the
  multimodal F1 ceiling with **2 input streams**; the deployable 5-seed ensembles
  reach 0.856/0.857 (labeled as ensembles, never mixed into this table).
- **On Accuracy we are mid-band** (F1-optimized: 0.897/0.896 vs band 0.87–**0.93**;
  PedFormer's 0.93 tops it) — the F1-first program lifted Acc too (BiLSTM
  0.883→0.897). We do **not** dominate on Acc; reporting this honestly is what makes
  the AUC claim credible rather than "suspiciously easy." (Test base rate: 32.5%
  positives.)
- **The claim to make (F1-first):** *"On the standard PIE protocol, a 2-stream
  (bbox + ego-speed) model reaches F1 within 0.02–0.03 of the multimodal
  state-of-the-art (0.844–0.847 vs PedFormer's 0.87), with the table's highest AUC
  (0.94–0.95), at a fraction of the feature-extraction cost and latency; this holds
  for both a BiLSTM and a staged-search Transformer over the identical input — and
  once both families receive identical F1-first optimization they are statistically
  indistinguishable on F1, so the parsimony finding is about the input signal, not an
  architecture artifact."* Use Occlusion-Aware Diffusion as the precedent that
  minimal modality is a legitimate choice (with its protocol caveat).
- **The latency half of the claim is now measured (Issue 9):** the BiLSTM runs in
  **0.575 ms/window** (~58× inside a 30 fps budget; the transformer 0.459 ms) and is
  **4.5% of the live pipeline** (detection-bound). So "fewest inputs" is backed by
  "lowest compute" — the 3–7-stream methods above need pose/optical-flow/semseg/depth
  extractors we do not, which is the parsimony+efficiency story, not just an accuracy
  claim.
- **Pre-empt the "top AUC + fewest inputs ⇒ easier eval" reflex** with the now-
  complete evidence, cited inline next to the table: leakage removed + **0%
  verified** (Issue 1–2); eval-parity per-ped ≈ per-window (Issue 2); bootstrap
  95% CI **[0.92, 0.95]** window-level, **[0.92, 0.96] pedestrian-cluster** (Issue 4;
  `f1_optimization/07_cluster_bootstrap.md` — quote the cluster interval, windows are
  ped-correlated); **LOSO 0.928 ± 0.041** across all 6 sets, set03 representative
  (Issue 5); multi-seed ablations (Issue 6) and a documented grid search (Issue 8 —
  AUC-conditional, see the metric-conditional notes there); and detector-in-the-loop
  robustness (Issue 10: AUC drop only +0.009 GT→**oracle-matched** YOLO boxes).

## Split & protocol alignment

| Baseline | Standard split? | Notes |
|---|---|---|
| PCPA | ✅ (defines it) | reference protocol |
| PIT / IntFormer | ✅ (as tabulated by GTransPDM) | GTransPDM: "same data splits and configurations as ours" |
| Ped-Graph+ / BiPed | ✅ ‡ | GTransPDM flags these two as configured differently ("Except BiPed and Pedestrian Graph+…") — verify vs originals in BibTeX pass |
| PedFormer | ✅ (verified 2026-07-13) | its own paper: "default data split", 50% overlap, TTE 1–2 s (arXiv 2210.07886 Tbl I) |
| GTransPDM (+ w/o-pose variant) | ✅ | 4770/1332/3816 (0.5:0.1:0.4) = set01/02/04 · 05/06 · 03; obs 16, TTE 30–60 |
| PIP-Net | ✗ **verified custom split (2026-07-13)** | its own paper: random ~50%/40%/10% (880/719/243 samples) — removed from the table |
| Occlusion-Aware Diffusion | ✗ different task | occluded scenarios, ~1-frame-ahead TTE — not comparable |

## Optional but strong: PIEPredict on **our** split

`PIEPredict/` is vendored locally. Running it on our exact set03 split → one
*directly comparable* row ("original PIE model, our split, our metrics") that
sidesteps cross-paper caveats. Indicative, not mandatory.

## Remaining to-do (Issue 3 checklist)

Internal finalization is **DONE** — every downstream issue's number is now folded in.
Only external-source verification and the BibTeX pass remain (need paper access).

- [x] Lock Acc/AUC/F1 for PCPA, GTransPDM (+ w/o-pose), PedFormer, BiPed + landscape
      (PIT/IntFormer/Ped-Graph+) — confirmed from sources.
- [x] Resolve Occlusion-Diffusion — **reclassified** as modality precedent (occluded
      protocol, not a comparison row).
- [x] **(2026-07-13 audit)** Split the misattributed BiPed/PedFormer row (PedFormer
      is 0.93/0.90/0.87); removed PIP-Net (verified custom split); fixed the
      GTransPDM footnote (0.92 = w/o-pose ablation, given its own row); folded in the
      F1-first program rows + reframed the metric-choice section F1-first.
- [x] **(2026-07-13 audit)** GTransPDM 0.90-vs-0.92 resolved against the source
      (w/o-Xke ablation).
- [ ] **External (pre-submission):** verify PIP-Net's published IEEE T-ITS version
      numbers (arXiv version fetched 2026-07-13 shows 0.91/0.90/0.84; a later version
      reportedly differs) — context citation only either way. Verify IntFormer vs its
      own paper (single-source row), and Ped-Graph+/BiPed vs originals (config flag ‡).
- [ ] **External (pre-submission):** full BibTeX + venue/DOI pass (arXiv ids:
      GTransPDM 2409.20223, PIP-Net 2402.12810, ODM 2511.00858, IntFormer 2105.08647,
      PedFormer 2210.07886).
- [ ] (Optional, strong) Run vendored `PIEPredict/` on our split for one directly
      comparable "original PIE model, our split" row.

## Sources

- PCPA / benchmark: Kotseruba, Rasouli, Tsotsos, "Benchmark for Evaluating
  Pedestrian Action Prediction," WACV 2021 — openaccess.thecvf.com; repo
  github.com/ykotseruba/PedestrianActionBenchmark
- GTransPDM: arXiv:2409.20223 (Table I tabulates PCPA/PIT/IntFormer/Ped-Graph+/BiPed)
- PIP-Net: Azarmi et al., IEEE T-ITS 2025; arXiv:2402.12810; eprints.whiterose.ac.uk
- Occlusion-Aware Diffusion: arXiv:2511.00858 (accepted IEEE T-ITS, Nov 2025)

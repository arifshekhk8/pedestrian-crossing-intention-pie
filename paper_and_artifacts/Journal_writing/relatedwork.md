# relatedwork.md — recent related work + honest defensibility assessment

> **Purpose.** Two things in one file, as requested:
> 1. **A verified catalogue of recent related work** on PIE pedestrian crossing-intention
>    prediction — organized by how it relates to *our* contributions — with proper
>    links, each paper's modality/rigor profile, and the limitation we address.
> 2. **An honest answer** (§7) to "is my journal defensible against all of this?"
>
> **How the links/numbers were sourced (2026-07-21).** I web-searched the landscape and
> then fetched each paper's arXiv/DOI page to confirm title, authors, venue, and claims.
> Items I confirmed first-hand are marked ✅; items known only from a secondary source
> (another paper's table, an aggregator, or a search snippet) are marked ◻ **[verify
> against the primary PDF before citing]**. **Do not paste a ◻ number into the manuscript
> without opening the source.** This mirrors the discipline in
> `../../journal_prep/issue3_baseline_comparison/`.
>
> **Metric hierarchy for reading the tables: F1 → accuracy → AUC** (our supervisor's
> directive). Our own numbers: see `ProjectDescription.md` §13–14.

---

## 1. The one thing to internalize first

The single most important paper for our defense is **PedCMT (IEEE T-ITS, 2024)**. It
already published our headline parsimony claim — *"only bounding box + ego-vehicle speed
is enough to match multimodal SOTA on PIE"* — two years before us, at a **stronger venue
than our target (MTI)**. Our parsimony *result* is therefore **not novel on its own.**
What is still novel and defensible is **the leakage-free re-evaluation, the statistical
rigor, and the four-family "the architecture doesn't matter, the input does" isolation**.
The paper's framing must pivot accordingly (see §7). Read §2 first, then §7.

---

## 2. Category A — direct minimal-modality competitors (these bear on our *novelty*)

These are the works that already argue "few/cheap inputs are enough." They constrain what
we can claim as new.

### A1. PedCMT — *the key competitor* ✅ (claim) / ◻ (exact PIE numbers)
- **Title:** Pedestrian Crossing Intention Prediction Based on Cross-Modal Transformer and
  Uncertainty-Aware Multi-Task Learning for Autonomous Driving
- **Venue/Year:** IEEE Transactions on Intelligent Transportation Systems, 2024
- **DOI:** 10.1109/TITS.2024.3386689 · **IEEE Xplore:** <https://ieeexplore.ieee.org/document/10507743/>
  · **Code:** <https://github.com/xbchen82/PedCMT>
- **Inputs:** **bounding box + ego-vehicle speed ONLY** (2 streams) — identical to ours.
- **Method:** cross-modal + self-attention transformer with bottleneck feature fusion;
  **uncertainty-aware multi-task** (jointly predicts future bbox *and* crossing action).
- **Reported:** "on par with or superior to SOTA methods that rely on more inputs";
  PIE Acc **~0.92** ◻ (exact AUC/F1 need primary verification — our internal table had
  ~0.92 / ~0.81 / ~0.876 flagged garbled).
- **What it does NOT do (our openings):** no temporal-leakage audit; no bootstrap/cluster
  CIs; no LOSO; no multi-seed variance; no detector-in-the-loop; single dataset-pair
  numbers at one operating point. It also *adds* modeling complexity (multi-task +
  uncertainty) rather than isolating *why* 2 streams suffice.
- **Bottom line:** **overlaps our parsimony headline directly.** We must cite it up front,
  concede the parsimony result to it, and pivot our novelty to leakage + rigor + the
  architecture-isolation finding.

### A2. GTransPDM (w/o pose ablation) — closest 2-stream cousin ✅
- **Title:** GTransPDM: A Graph-embedded Transformer with Positional Decoupling for
  Pedestrian Crossing Intention Prediction
- **Authors:** Chen Xie, Ciyun Lin, Xiaoyu Zheng, Bowen Gong, Antonio M. López
- **Venue/Year:** IEEE Signal Processing Letters, 2025 · **arXiv:** 2409.20223 —
  <https://arxiv.org/abs/2409.20223>
- **Inputs:** full model = bbox + **pose/skeleton** + ego-motion (3); its **"w/o Xke"
  ablation drops pose → bbox + ego-motion (2)** and *improves* to **0.92 / 0.90 / 0.86**
  (Acc/AUC/F1) — the closest published 2-stream number to ours.
- **Limitation we address:** the full model needs a **skeleton-pose estimator** (extra
  model, degrades at distance/occlusion); no CIs/LOSO/multi-seed/detector-in-the-loop.
- **Our response:** we reach comparable AUC/F1 with **no pose pipeline at all**, and we
  show (four-family study) that even the graph/attention machinery is unnecessary.

### A3. "Is attention to bounding boxes all you need for pedestrian action prediction?" ◻
- **Authors:** Achaji, Lorenzo, et al. · **Venue:** IEEE IV 2022 · **arXiv:** 2107.08031
  **[verify arXiv id + authors]**
- **Relevance:** an **explicit bbox-centric precedent** — the title itself pre-empts our
  "input is what matters" story. Cite it as prior support for minimal-modality, and note
  it did *not* do a leakage audit or cross-architecture isolation.

### A4. Occlusion-Aware Diffusion Model — modality precedent, **not** a comparison row ✅
- **Authors:** Yu Liu, Zhijie Liu, Zedong Yang, You-Fu Li, He Kong
- **Venue/Year:** IEEE T-ITS 2025 (accepted) · **arXiv:** 2511.00858 —
  <https://arxiv.org/abs/2511.00858>
- **Inputs:** bbox + ego-velocity only (2). **But protocol differs:** occlusion-specific
  scenarios, ~1-frame-ahead horizon — not standard fully-observed crossing prediction.
- **Use:** cite as a **precedent that bbox + ego-velocity is a legitimate minimal
  modality**, with the protocol caveat — do NOT tabulate its ~0.95 against our 0.94.

---

## 3. Category B — recent multimodal SOTA (we address heavy-modality / weak-rigor)

These use 3–7 streams and heavier pipelines. They set the AUC/F1 ceiling and are what our
"parsimony + rigor" story is measured against.

### B1. PIP-Net ✅ — but **custom split**, so context-only
- **Authors:** Mohsen Azarmi, Mahdi Rezaei, He Wang · **Venue:** IEEE T-ITS, Vol. 26,
  No. 7, July 2025 · **arXiv:** 2402.12810 — <https://arxiv.org/abs/2402.12810>
- **Inputs:** kinematics + spatial scene features + **categorical depth map + local motion
  flow + up to 3 cameras** (heavy multimodal); introduces a custom **"Urban-PIP"** dataset.
- **Critical:** its own paper uses a **custom random split (~880/719/243)**, *not* the
  standard set01/02/04·05/06·03 — so it was **removed** from our comparison table and is
  cited as prose context only.
- **Limitation we address:** prohibitive feature pipeline (flow/semseg/depth extractors),
  many failure points, not real-time; non-standard split.

### B2. MFT (Multi-Context Fusion Transformer) ✅ — ties our AUC
- **Authors:** Yuanzhe Li, Hang Zhong, Steffen Müller · **arXiv:** 2511.20011 (Nov 2025,
  rev. Mar 2026) — <https://arxiv.org/abs/2511.20011>
- **Inputs:** 4 numerical context dimensions (behavior / environment / localization /
  vehicle-motion). **PIE Acc 0.90** ◻ (AUC ~0.94 from its cost table, ties our AUC; F1
  0.83). ~0.95 M params.
- **Limitation we address:** no CIs/multi-seed/LOSO/detector-in-the-loop; more streams.
  It's the current strongest published AUC — worth citing as the AUC peer we match at 2
  streams / 0.6–0.8 M params.

### B3. ACIT ✅ — JAAD-only, no PIE number
- **Title:** ACIT: Attention-Guided Cross-Modal Interaction Transformer · **arXiv:**
  2511.20020 — <https://arxiv.org/abs/2511.20020>
- **Inputs:** 5 streams (RGB + flow + pose + context + speed). **JAAD-only** — kept for its
  cost table (43.9 ms, 5.15 M params), not a PIE comparison row.

### B4. TrajFusionNet ✅
- **Authors:** François G. Landry, Moulay A. Akhloufi · **Venue:** IEEE T-IV 2025 ·
  **arXiv:** 2508.19866 — <https://arxiv.org/abs/2508.19866>
- **Inputs:** two branches — trajectory+speed sequences (SAM) + **visual** (predicted bbox
  overlaid on scene images, VAM). Claims SOTA on 3 datasets + **lowest inference time**.
- **Limitation we address:** needs the visual branch (scene images); no CIs/LOSO/multi-seed
  reported. Its explicit efficiency claim makes it a good latency peer to cite.

### B5. IntentFormer (multimodal co-learning) ◻ [sciencedirect blocked — verify]
- **Title:** Predicting pedestrian intentions with multimodal IntentFormer: A Co-learning
  approach · **Venue:** Pattern Recognition, 2025 · <https://www.sciencedirect.com/science/article/abs/pii/S0031320324009567>
- **Inputs:** multimodal co-learning (verify exact streams). Distinct from the older 2021
  "IntFormer" (Lorenzo et al., arXiv 2105.08647) — do not conflate them.

### B6. TCL (Temporal-contextual Event Learning) ✅
- **Authors:** Hongbin Liang, Hezhe Qiao, Wei Huang, Qizhou Wang, Mingsheng Shang, Lin Chen
- **Venue:** ICONIP 2024 · **arXiv:** 2504.06292 — <https://arxiv.org/abs/2504.06292>
- **Inputs:** visual + non-visual, ego-view frames; temporal event clustering to fight
  frame redundancy. Claims SOTA on PIE/JAAD-beh/JAAD-all (numbers ◻).
- **Limitation we address:** no CIs/multi-seed/LOSO/detector-in-the-loop.

### B7. Multimodal Fusion Network ◻ [verify] · **arXiv:** 2511.20008 (2025) —
<https://arxiv.org/abs/2511.20008>. Recent fusion model; verify inputs/numbers before use.

---

## 4. Category C — foundational & landscape baselines (the standard-protocol table)

These are the verified rows already in
`../../journal_prep/issue3_baseline_comparison/03_baseline_comparison.md` (Acc/AUC/F1,
standard split). Cite for the landscape; do not re-verify unless a number is challenged.

| Method | Venue/Year | Acc/AUC/F1 | Inputs | Link |
|---|---|---|---|---|
| **PIE / PIEPredict** (dataset origin) | ICCV 2019 | trajectory model (no clf row) | bbox+ego+context | [CVF PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.pdf) |
| **PCPA** (benchmark anchor) | WACV 2021 | 0.87 / 0.86 / 0.77 | bbox+pose+context+speed (4) | [repo](https://github.com/ykotseruba/PedestrianActionBenchmark) |
| **Pedestrian Graph+** | T-ITS 2022 | 0.89 / 0.90 / 0.81 ‡ | pose graph + ego (2–3) | [rg 363071497] ◻ |
| **IntFormer** (Lorenzo) | 2021 | 0.89 / 0.92 / 0.81 | multimodal | arXiv 2105.08647 ◻ |
| **PIT** | T-ITS 2023 | 0.91 / 0.92 / 0.82 | multimodal transformer | [rg 373835703] ◻ |
| **BiPed** | ICCV 2023 | 0.91 / 0.90 / 0.85 ‡ | multimodal | arXiv 2210.07886 |
| **PedFormer** (F1/Acc ceiling) | ICRA 2023 | **0.93 / 0.90 / 0.87** | multimodal multitask | [arXiv 2210.07886](https://arxiv.org/abs/2210.07886) ✅ |
| **Faster-PCPNet** | IEEE 2024 | 0.94 / 0.92 / 0.89 ◻ | pose+ego+bbox+polar | IEEE 10418196 ◻ |

‡ GTransPDM flags Ped-Graph+ and BiPed as configured differently — verify vs originals.
**These are the "heavy modality, single split, no CI, GT-box" cluster our rigor addresses.**

---

## 5. Category D — surveys & critical reviews (the field-wide gaps we close)

These are the strongest support for our *methodological* contributions — cite them to
establish that the gaps we close are real and acknowledged.

### D1. Diving Deeper Into Pedestrian Behavior Understanding ✅ — the authoritative critique (full text pulled 2026-07-21)
- **Authors:** Amir Rasouli, Iuliia Kotseruba (the PIE/PCPA authors themselves)
- **Venue:** IEEE IV 2024 · **arXiv:** 2407.00446 — <https://arxiv.org/abs/2407.00446>
- **Why it matters:** the PIE/PCPA authors' *own* critique of how the field evaluates on
  PIE/JAAD. **This is the single best citation to motivate our leakage audit + rigor** — and
  several of its critiques map directly onto our contributions. Verbatim/close critiques and
  our response:

  | "Diving Deeper" critique (quote/close paraphrase) | Our response (contribution) |
  |---|---|
  | **Task confusion:** *"it has become difficult to discern models trained for intention estimation and action prediction as the terms are often used interchangeably"* — intention is an unobservable state of mind, crossing action is an observable event | Our leakage audit *operationalizes* this distinction: by removing windows where the pedestrian is already crossing (67.9%), we strip out the *action-detection* contamination and evaluate genuine *pre-onset prediction* (TTE ≥ 30). Issues 1–2. |
  | **Anchor misalignment (≈ our leakage point):** intention labels come from clips ending at a **`critical_point`**, while action samples use **1–3 s TTE windows** — *"not all samples have overlaps,"* creating misalignment in what is actually predicted | This is *precisely* the anchor problem we audit and fix. We re-anchor every window at PIE's **`crossing_point`** with TTE ∈ [30,60] → 0% verified leakage. **Cite this as independent acknowledgment that the anchor matters.** Issue 2. |
  | **Narrow evaluation:** *"the narrow focus of evaluation procedures that measure performance by averaging accuracy of models over all observations"* — fails to address consistency and different horizons/risk levels | We report **F1-first** (not averaged accuracy) + per-horizon behavior (TTE ablation, Issue 6: AUC declines 0.960→0.919 with horizon) + bootstrap/cluster CIs + LOSO. Issues 4–6. |
  | **Metric weakness:** hard/consistency metrics show *"significant performance drop on all models, suggesting their overall consistency is low"* | Motivates our multi-seed + CI reporting and the F1-first hierarchy over single averaged accuracy. Issues 4, 6; f1_optimization. |
  | **Modality finding (supports us):** dynamics-oriented models excel at *action* prediction, and **JAAD's poor ego-motion data causes severe degradation** — *"dynamics information is crucial"* | **Independent confirmation of our ego-speed (dynamics) dominance finding** AND the citable justification for why JAAD can't test ego-speed (→ our cross-dataset Track-A limitation is the survey's own point). Issue 2; `cross_dataset_validation/PLAN.md`. |
  | **Temporal consistency:** *"lack of temporal consistency in model predictions, even within the short span of 2 s … can lead to irrational behavior by the vehicle"* | **We do NOT fully address this** (we don't measure per-frame prediction stability within one track). Honest gap → name it as future work; our multi-seed variance + TTE ablation only partially touch it. |

### D2. Feature Importance in Pedestrian Intention Prediction: A Context-Aware Review ✅
- **arXiv:** 2409.07645 (2024) — <https://arxiv.org/abs/2409.07645>
- **Why it matters (double-edged):** its CAPFI analysis independently finds **bbox and
  ego-speed are the dominant features**, and explicitly warns that **"reliance on
  ego-vehicle speed may induce driver-side bias, especially in yielding scenarios."**
  - **Supports us:** corroborates ego-speed dominance (our Issue-2 finding) from an
    independent method.
  - **Constrains us:** it *also* pre-published the "ego-speed is dominant" observation
    *and* the "driver-side bias" caveat we list as our honest limitation. Our feature-
    ablation finding is therefore **confirmatory, not first**. Cite it, don't claim the
    ego-speed insight as ours alone.

### D3. Pedestrian Crossing Intention Prediction in the Wild: A Survey ◻ [verify]
- researchgate 387985435 (2024/2025). A recent survey — use for landscape framing +
  to cross-check that no one else reports a leakage audit.

---

## 6. Category E — emerging paradigms & other datasets (context, not competitors)

Different task setups or datasets — cite to show breadth/awareness, not head-to-head.

- **Psychological Features + Transformer Fusion** ✅ — Sima Ashayer, Hoang H. Nguyen, Yu
  Liang, Mina Sartipi, **IEEE IV 2026**, arXiv 2603.19533 —
  <https://arxiv.org/abs/2603.19533>. Modality-agnostic behavioral streams; evaluated on
  **PSI 1.0/2.0** (not PIE). Relevant to our (paused) PSI cross-test.
- **VRU-CIPI** ✅ — Abdelrahman, Abdel-Aty, Tran, arXiv 2505.09935 (2025) —
  <https://arxiv.org/abs/2505.09935>. **UCF-VRU** dataset (infrastructure cameras), not
  PIE — 96.45% acc. Different setting; cite for breadth only.
- **VLM / GPT-4 approaches** ◻ — e.g. "Pedestrian Intention Prediction via Vision-Language
  ..." arXiv 2507.04141; "Understanding Pedestrian Gesture Misrecognition" arXiv 2508.06801
  (2025). A zero-shot LLM paradigm (GPT-4 mini ~0.73 F1 on PIE). Cite as an emerging,
  higher-latency alternative that our lightweight model contrasts against.
- **ESIA** ◻ [verify] — Energy-Based Spatiotemporal Interaction-Aware, arXiv 2604.23728
  (2026). Very recent; verify before citing.
- **Perception stack (for our live pipeline):** **ByteTrack** (Zhang et al., ECCV 2022,
  arXiv 2110.06864) and **Ultralytics YOLO** — cite in Methods for the demo, not as
  crossing-prediction competitors.

---

## 7. HONEST DEFENSIBILITY ASSESSMENT (the frank answer you asked for)

**Short answer: Yes — the paper is defensible for MDPI MTI, but *only if the framing
pivots*. As currently headlined ("2 streams is enough / parsimony"), it is partially
undercut by prior work and would be vulnerable. Re-headlined around the leakage audit +
rigor + the four-family isolation, it is a solid, honest contribution.** Here is the
unvarnished breakdown.

### 7.1 What is genuinely novel and strongly defensible ✅
1. **The temporal-leakage audit + leakage-free re-extraction (Issues 1–2).** This is the
   real contribution. In everything I surveyed — PedCMT, GTransPDM, PIP-Net, MFT, TCL,
   TrajFusionNet, the surveys — **no one audits observation-window leakage against the
   per-frame `cross` label and re-anchors at `crossing_point` with a 0%-leakage
   verification.** The closest is the "Diving Deeper" benchmark's general complaint about
   inconsistent protocols, but they don't isolate this specific leak. **This is
   publishable on its own and is your defensible core.**
2. **Statistical rigor almost no one else reports.** Across every paper I checked, **none**
   report bootstrap CIs, pedestrian-cluster CIs, LOSO across all 6 sets, *and* multi-seed
   variance. Point estimates at one seed/one split are the field norm. Our full rigor stack
   is a legitimate, checkable differentiator.
3. **The four-family architecture-isolation study (Transformer / GRU / vanilla RNN vs
   BiLSTM, matched search budgets).** "Attention beats recurrence only via its search; the
   gated GRU and even the un-gated RNN tie; therefore the input signal, not the
   architecture, carries the task" is a **genuinely new, well-controlled finding.** No
   surveyed paper does this cross-architecture isolation with matched search + F1-first +
   cluster-bootstrap. This is arguably your *second* real contribution and is what makes the
   parsimony story causal rather than incidental.
4. **Detector-in-the-loop realism (Issue 10).** Measuring GT-box vs YOLO-box degradation +
   tracker fragmentation is rare — most papers assume ground-truth boxes at inference.

### 7.2 Where the paper is weak / exposed ⚠ (state these before a reviewer does)
1. **The parsimony *result* is not new.** **PedCMT (T-ITS 2024)** and **GTransPDM-w/o-pose
   (SPL 2025)** already showed bbox + ego-speed matches multimodal SOTA on PIE, and
   **Achaji et al. (IV 2022)** pre-framed "is bbox all you need?". If the paper leads with
   "we show 2 streams is enough," a reviewer who knows PedCMT will reject the novelty
   claim. **Fix: cite PedCMT prominently, concede the result, and reframe our parsimony as
   *corroboration under a leakage-free protocol + a causal explanation via the four-family
   study*.** ("PedCMT showed 2 streams suffice; we show *why* — it's the input, not the
   architecture — and that it holds once leakage is removed.")
2. **The ego-speed dominance finding is also not first.** The **CAPFI review (2409.07645)**
   independently reported ego-speed dominance *and* the driver-side-bias caveat. Present
   ours as confirmatory + cite them; do not claim discovery.
3. **Single dataset (PIE only).** This is now the **biggest reviewer risk**, because the
   PSI cross-test is paused. Nearly every competitor reports **PIE + JAAD**. A single-dataset
   paper reads as thin for a journal. **Highest-value mitigation:** run the leakage audit +
   clean protocol + the four families on **JAAD** (you already have JAAD tooling via the
   PIE/PIEPredict repos — no external access needed, unlike PSI). Even a compact JAAD
   replication would materially raise defensibility and directly answer "does this
   generalize beyond PIE?".
4. **Non-standard (leakage-free) protocol cuts both ways.** It's our contribution, but it
   means our numbers are **not strictly comparable** to the standard-protocol table rows.
   We handle this honestly (we say so, and we show the leak fix barely moved AUC), but a
   reviewer may still push on it. Keep the "our clean number vs their leaky number"
   framing explicit and never tabulate them as if identical protocols.
5. **We lead on AUC, not F1 or accuracy.** Under the supervisor's own F1-first hierarchy we
   are *mid-band* on F1 (0.844–0.852 vs PedFormer 0.87) and accuracy. The headline must not
   overclaim "SOTA" — "competitive, within 0.02–0.03 of the multimodal ceiling at a
   fraction of the cost" is the honest and still-strong claim.
6. **Venue realism.** This work is **right-sized for MTI** (mid-tier, open-access,
   application/interaction-oriented). It would be a harder sell at IEEE T-ITS where
   PedCMT/PIP-Net/GTransPDM/MFT sit — those have richer models, two datasets, and stronger
   raw SOTA. Target MTI and the framing works.

### 7.3 The defensible thesis (the sentence the paper should actually argue)
> *"Prior work (PedCMT, GTransPDM-w/o-pose) already hinted that minimal inputs suffice for
> PIE crossing prediction, but under protocols we show contain up-to-68% temporal leakage
> and with single-point, single-seat evaluation. We (i) audit and remove that leakage, (ii)
> re-establish the minimal-modality result under a 0%-leakage protocol with full statistical
> rigor (bootstrap + cluster CIs, LOSO, multi-seed), and (iii) demonstrate — across four
> matched-budget architecture families — that the result is driven by the input signal, not
> the temporal model or its gating, with real-time detector-in-the-loop validation."*

That thesis is **not** refuted by any paper in this file. Leading with it, and conceding
the raw-parsimony precedent up front, is what makes the paper defensible and honest.

### 7.4 Verdict
- **Defensible for MTI: Yes**, with the pivot above. The leakage audit + rigor + four-family
  isolation are enough for a mid-tier journal even though the parsimony result itself has
  precedent.
- **Would it survive IEEE T-ITS as-is: Probably not** — single dataset + non-SOTA F1 +
  precedented parsimony would draw a major-revision-or-reject.
- **Single highest-leverage move to strengthen it:** add a **JAAD** replication of the
  leakage audit + clean protocol (available now, no PSI dependency). Second: get the full
  text of "Diving Deeper" (2407.00446) and cite its specific protocol critiques as the
  motivation for our audit.

---

## 8. Field-wide limitation → our response matrix (paper-ready)

| Common gap in PIE crossing-prediction work | Which papers exhibit it | Our response | Evidence |
|---|---|---|---|
| **Temporal leakage in the observation window never checked** | ~all (PedCMT, GTransPDM, MFT, TCL, PIP-Net, …) | audit vs per-frame `cross` + re-anchor at `crossing_point`, **0% verified** | Issues 1–2 — *our most distinctive point* |
| **Heavy multimodal pipelines** (pose/flow/semseg/depth/multi-cam) | PIP-Net (7), ACIT (5), MFT (4), GTransPDM (3) | **2 cheap streams**, no auxiliary extractor, 0.32–0.72 ms/window | Issues 3, 9; four-family study |
| **No confidence intervals** | ~all | 10k bootstrap + **pedestrian-cluster** CIs | Issue 4 |
| **Single fixed split** | ~all | LOSO across all 6 sets (0.928 ± 0.041) | Issue 5 |
| **Single-seed ablations as conclusions** | ~all | 5-seed mean ± std + significance | Issue 6 |
| **Ground-truth boxes assumed at inference** | ~all | detector-in-the-loop (YOLO→ByteTrack), robust (+0.009 drop) | Issue 10 |
| **Unjustified hyperparameters** | many | documented 36-config grid (val-only) + 4 matched-budget searches | Issues 7–8, transformer/gru/rnn |
| **Architecture asserted, not tested** | all | four-family isolation: input ≫ architecture/gating | transformer/, gru/, rnn/ |
| **Parsimony shown but not *explained*** | **PedCMT, GTransPDM-w/o-pose** | we show *why* 2 streams suffice (ablation + cell isolation) | Issue 2 + four families |

---

## 9. Citation to-do before drafting Related Work

- [x] **Fetch the full text of "Diving Deeper" (2407.00446)** and extract its specific
      protocol/metric critiques — **DONE 2026-07-21**, folded into §5 D1 as a quote→response
      table (task-confusion, `critical_point`/TTE anchor misalignment ≈ our leakage point,
      narrow-averaged-accuracy, dynamics-crucial ego-motion support, temporal-consistency gap).
- [ ] **Verify PedCMT's exact PIE Acc/AUC/F1** from the IEEE Xplore PDF (10507743) or the
      GitHub `pie.py` results — critical, since it's the paper we concede parsimony to.
- [ ] Verify the ◻ rows in §4 (IntFormer 2105.08647, PIT, Ped-Graph+, Faster-PCPNet) and
      §3 (Achaji arXiv id) against primary PDFs; add BibTeX entries.
- [ ] Confirm whether **any** surveyed paper reports a leakage check (I found none) — if
      confirmed, it strengthens the Issue-1 novelty claim; state it carefully ("to our
      knowledge, no prior PIE crossing-prediction work audits observation-window leakage").
- [ ] Add all §2–§6 entries to `references.bib` with verified DOIs; clear the `% VERIFY`
      flags. arXiv ids on hand: PedCMT (T-ITS 10.1109/TITS.2024.3386689), GTransPDM
      2409.20223, PIP-Net 2402.12810, PedFormer/BiPed 2210.07886, MFT 2511.20011, ACIT
      2511.20020, TrajFusionNet 2508.19866, TCL 2504.06292, ODM 2511.00858, Diving-Deeper
      2407.00446, CAPFI 2409.07645, Psych-Transformer 2603.19533, VRU-CIPI 2505.09935.

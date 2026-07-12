# Journal Paper — Writing Plan (MDPI MTI)

Target journal: **MTI — Multimodal Technologies and Interaction** (MDPI, open access).
Tool: **Overleaf** + the official **MDPI Article LaTeX template** (`mdpi.cls` + `mdpi.bst`).
All numbers are final (see `../journal_prep/` Issues 1–10). This plan maps our work
onto the MDPI structure, section by section, with the exact results and figures that
go in each.

> **Golden rule:** every claim in the paper must trace to a number in `journal_prep/`.
> Do not invent results. When in doubt, cite the issue folder.

---

## 0. Working title (pick one, refine later)

- *"Leakage-Free Pedestrian Crossing-Intention Prediction from Bounding Boxes and
  Ego-Speed: A Parsimonious BiLSTM Baseline on PIE"*
- *"Less Is Enough: A Two-Stream BiLSTM for Real-Time Pedestrian Crossing Prediction"*

## 1. The paper's one-sentence thesis (the spine)

> On a **leakage-free, canonical PIE protocol**, a BiLSTM using only **bounding-box
> motion + ego-vehicle speed** reaches **AUC 0.932 [0.92–0.95]** — the top of the
> standard-protocol band — at **0.575 ms/window**, i.e. competitive with 3–7-stream
> multimodal SOTA at a fraction of the cost, shown with full statistical rigor and
> detector-in-the-loop realism.

Three contributions to hammer throughout:
1. **A temporal-leakage audit + fix** (most papers never check this) — Issues 1–2.
2. **Parsimony**: 2 streams ≈ top AUC, with measured real-time latency — Issues 3, 9.
3. **Rigor**: bootstrap CIs, LOSO, multi-seed ablations, documented HP search,
   detector-in-the-loop — Issues 4, 5, 6, 7, 8, 10.

---

## 2. MDPI MTI section structure (the order to follow)

MDPI's required order (verify against the MTI "Instructions for Authors" page before
submitting — section names/positions can vary slightly by journal):

| # | Section | Our content | Source issues |
|---|---|---|---|
| — | **Title + Authors + Affiliations** | — | — |
| — | **Abstract** (~200 words, unstructured) | problem → leakage finding → method → headline 0.932 + latency → contribution | all |
| — | **Keywords** (5–8) | pedestrian crossing intention; intention prediction; PIE dataset; BiLSTM; temporal leakage; ADAS; ego-vehicle speed; real-time | — |
| 1 | **Introduction** | problem & safety motivation; why intention≠trajectory; the gap (leakage, heavy modality, weak rigor); our 3 contributions; paper roadmap | 1, 2, 3 |
| 2 | **Related Work** | PIE crossing-prediction landscape (PCPA, Ped-Graph+, PIT, IntFormer, BiPed, GTransPDM, PIP-Net); occlusion-diffusion as a *minimal-modality precedent*; the field-wide gaps we close | 3 (`03_baseline_comparison.md`, `04_positioning_vs_prior_work.md`) |
| 3 | **Materials and Methods** | PIE dataset; **the leakage problem + the clean crossing_point-anchored protocol**; feature set (bbox + ego-speed, raw px, train-only norm); BiLSTM architecture (hidden 128, 2 layers — justified); training contract (split, pos_weight 1.682, early stop); **documented hyperparameter search**; evaluation metrics; the YOLO26-M + ByteTrack live pipeline | 1, 2, 7, 8, (10 pipeline) |
| 4 | **Results** | headline AUC 0.932 ± 0.011 + **bootstrap CI**; **baseline comparison table**; **LOSO**; ablations: **ego-speed dominance (+0.18)**, window/TTE (incl. matched cohort), hidden-size/depth; **latency + pipeline breakdown**; **GT-box vs YOLO-box** degradation | 2, 3, 4, 5, 6, 7, 9, 10 |
| 5 | **Discussion** | interpret the parsimony result; high-AUC/mid-Acc honest framing; positioning vs each prior work; deployment realism (detection-bound, tracker fragmentation); **limitations** (ego-speed encodes ego-driver anticipation; tracker; 2 clips for Issue 10) | 3, 9, 10 + limitation note |
| 6 | **Conclusions** | restate contributions + the headline; future work (re-ID, speed-perturbation robustness, more clips) | all |
| — | **Back matter** | Author Contributions; Funding; Data Availability (PIE is public + our code); Conflicts of Interest; Abbreviations; References | — |

---

## 3. What goes in each Results subsection (exact numbers + figures)

Pull figures straight from the issue folders (don't regenerate). Suggested order:

1. **Main result + CI** — AUC 0.932 ± 0.011 (5-seed), 95% CI [0.92, 0.95], PR-AUC
   0.876, Acc 0.883. → `issue4_bootstrap_ci/`.
2. **Baseline comparison (Table)** — our row vs PCPA/Ped-Graph+/PIT/IntFormer/BiPed/
   GTransPDM/PIP-Net, with a **modalities** column. → `issue3_baseline_comparison/03_baseline_comparison.md`.
3. **Feature ablation (the headline finding)** — ego-speed dominant: baseline 0.932 vs
   bbox-only 0.753 (−0.18); attention no benefit (0.925). → `issue2_clean_protocol/05_variant_comparison.md`.
4. **Generalization — LOSO** — 6-fold 0.928 ± 0.041; set03 representative. → `issue5_loso_cv/` (table).
5. **Window + TTE ablation** — window insensitive; TTE declines 0.960→0.919, matched-
   cohort confirmed. → `issue6_window_tte_ablation/06_ablation_figure.png` + `06b_matched_tte_figure.png`.
6. **Capacity** — hidden 64/128/256 ≈ 0.93; depth 1/2/3 ≈ 0.93. → `issue7_hidden_size/` (2 figures).
7. **Hyperparameter search** — 36-config grid confirms the hand-set config. → `issue8_grid_search/08_grid_search_figure.png`.
8. **Latency + pipeline** — 0.575 ms/window; YOLO 93% / BiLSTM 4.5%. → `issue9_latency/09_latency_figure.png`.
9. **Detector-in-the-loop** — GT vs YOLO AUC drop +0.009; tracker fragmentation. → `issue10_gt_vs_detector/10_gt_vs_detector_figure.png`.

**Figure budget:** MDPI is fine with ~8–12 figures/tables. Prioritise: baseline table,
feature-ablation bar, window/TTE, latency breakdown, GT-vs-YOLO scatter. Convert the
positioning matrix (`04_positioning_vs_prior_work.md`) into one Discussion table.

---

## 4. Drafting order (write in THIS order — not top to bottom)

Writing methods/results first (concrete) before intro/discussion (framing) is faster
and avoids rework:

1. **Materials and Methods** — most concrete; we have every detail nailed.
2. **Results** — paste numbers + tables + figures; prose comes easily from the issue READMEs.
3. **Related Work** — from `03_baseline_comparison.md` + `04_positioning_vs_prior_work.md`.
4. **Introduction** — now that contributions are crisp.
5. **Discussion** — interpret results + limitations.
6. **Conclusions** — short, mirror the intro contributions.
7. **Abstract** — write LAST (summarise the finished paper).
8. **Back matter + references** — fill `references.bib`, run the BibTeX pass.

Target length: MDPI articles typically 6,000–10,000 words; aim ~7,000 + ~9 floats.

---

## 5. Status checklist

- [ ] Set up Overleaf project from the official MDPI template (journal = `mti`) — see `README.md`
- [ ] Fill `references.bib` (starter provided) + verify the 2 external baseline splits
- [ ] §3 Materials and Methods
- [ ] §4 Results (+ tables/figures from issue folders)
- [ ] §2 Related Work
- [ ] §1 Introduction
- [ ] §5 Discussion (+ limitations)
- [ ] §6 Conclusions
- [ ] Abstract + Keywords
- [ ] Back matter (Author Contributions, Data Availability, etc.)
- [ ] Full read-through + de-AI pass (see README "how to use Claude")
- [ ] Cross-check every number against `journal_prep/`

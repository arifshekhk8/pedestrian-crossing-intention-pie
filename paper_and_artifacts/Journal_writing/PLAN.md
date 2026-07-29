# Journal Paper — Writing Plan (MDPI MTI)

> **✅ STATUS 2026-07-26 — the manuscript is complete and submission-ready apart from
> front matter. Read this block first.**
>
> **Where it lives:** `MDPI_Article_Template/` — `main.tex`, `references.bib`
> (59 cited refs, every one verified first-hand, **zero `% VERIFY` flags left**),
> `figures/` (9 figures + their generator scripts), and the compiled `main.pdf`
> (25 pages). Build with **`tectonic main.tex`** from that folder. Overleaf/pdfLaTeX
> works too and needs no local fixes; the EPS→PDF logo conversion in `Definitions/`
> is only there so tectonic can render them.
>
> **Written this session (2026-07-26):** Materials and Methods, Results, Discussion,
> Conclusions, and the Abstract, all to full depth, plus every figure. Earlier
> sessions produced the Introduction (with the verified road-safety statistics, see
> `../STATISTICS_SOURCES.md`) and the Related Work.
>
> **Figures — all reproducible, none hand-drawn.** Generators live in
> `MDPI_Article_Template/figures/`; run `python make_figN_*.py` from anywhere.
> `figstyle.py` holds the shared palette (validated for colour-vision deficiency and
> ≥3:1 contrast) so the eight figures read as one document.
>
> | Fig | Script | What it shows | Data source |
> |---|---|---|---|
> | 1 | `make_intro_figure.py` | pedestrian share of road deaths (WHO) + US trend (NHTSA) | `../STATISTICS_SOURCES.md` |
> | 2 | `make_fig2_system.py` | the two-stream model + the live pipeline | schematic |
> | 3 | `make_fig3_protocol.py` | track-end vs crossing-point anchoring | schematic; real rates |
> | 4 | `make_fig4_leakage.py` | the leakage audit, before/after | **computed live** from both `leakage_per_sequence.csv` |
> | 5 | `make_fig5_curves.py` | ROC + PR, four families + bbox-only | **computed live** from `_probs.npz` (`prep_probs.py`) |
> | 6 | `make_fig6_forest.py` | ΔF1 / ΔAUC forest with cluster CIs | the three `*_cluster_bootstrap.json` |
> | 7 | `make_fig7_ablations.py` | ego-speed, horizon, window length | **computed live** from `06b_matched_tte_results.csv` + OW tables |
> | 8 | `make_fig8_latency.py` | per-family latency + pipeline budget | `latency_comparison.csv`, Issue 9 |
>
> Run `prep_probs.py` once before Figure 5 to build the cached probability vectors.
>
> **Every number in the paper was re-checked against its source file** on 2026-07-26:
> split counts (2,178/634/2,094 and 616/186/587) recomputed from the `.npy` files,
> leakage rates and rank-biserial effects recomputed from the audit CSVs, all
> bootstrap intervals read from the stored JSON, and the metric table cross-checked
> against `journal_prep/Analysis/model_comparison.csv`. No discrepancies remain.
>
> **Decisions still standing from 2026-07-21** (reasoning in `relatedwork.md` §7):
> - **Venue: MDPI MTI, locked.** Fits both pillars: minimal *multimodal* fusion
>   (pedestrian visual bbox + ego-vehicle OBD telemetry) and *interaction*
>   (pedestrian↔AV anticipation).
> - **Metric hierarchy: F1 → accuracy → AUC** (locked). Every table and claim leads with F1.
> - **Reframed novelty (mandatory pivot):** the headline is NOT "2 streams is enough"
>   (PedCMT/GTransPDM-w/o-pose/Achaji got there first — we **concede that up front**).
>   Our novelty = **(1) the temporal-leakage audit + leak-free protocol, (2) full
>   statistical rigor, (3) the four-family "input, not architecture, decides"
>   isolation.** Parsimony is corroboration + explanation, not the claim.
> - **Title (option B, chosen):** *"Two Streams, Four Architectures: A Leakage-Free,
>   Statistically Rigorous Benchmark for Real-Time Pedestrian Crossing-Intention
>   Prediction on PIE."*
> - **Scope:** full draft on the PIE results; **cross-dataset generalization is
>   ongoing/future work**, stated as such in §5.5 (running in a separate session —
>   JAAD primary, nuScenes stretch; PePScenes dead, PSI paused; see
>   `../../journal_prep/cross_dataset_validation/PLAN.md`).
> - **Framing:** BiLSTM = headline model; Transformer/GRU/vanilla-RNN = a controlled
>   "does the architecture matter?" isolation study.
>
> **✅ RESOLVED 2026-07-28 — the qualitative figure now exists (Figure 9).** Section 4.11
> shows the full detection-to-prediction pipeline on two PIE test scenes: one pedestrian
> correctly flagged 2.0 s before stepping into the road, one worker at a kerb correctly
> left alone. Faces blurred. Video S1 (26 s, 3.3 MB) is cut and declared. Everything is
> driven by the clean-protocol BiLSTM-F1 ensemble, gated by a parity check that
> reproduces Table 3 exactly. Full record in **`QUALITATIVE_FIGURE_PLAN.md`**.
>
> **Remaining before submission — one blocking item and three optional:**
> 1. **BLOCKING: front matter.** Authors, affiliations, ORCID, corresponding e-mail,
>    Author Contributions initials, and the data-availability repository URL are all
>    still `PLACEHOLDER`. Nothing else blocks submission.
> 2. Optional: a supervisor read for scientific framing.
> 3. Optional: MTI's cover-letter and graphical-abstract requirements.
> 4. Optional: fold in the cross-dataset results if that session finishes in time
>    (§5.5 and the Conclusions are already written to accommodate them).
>
> The roadmap below (from 2026-07-13) is still a valid section-by-section reference.

> **⚠ UPDATE 2026-07-13 — read this before using the roadmap below.** Two things
> changed after this plan was first written and the roadmap must be read through them:
> 1. **Metric hierarchy is now F1 → accuracy → AUC** (supervisor directive). Every
>    "headline AUC 0.932" framing below should be re-read **F1-first**: the headline
>    is **F1 0.844 (BiLSTM) / 0.847 (Transformer)**, with AUC (0.94–0.95, top of the
>    table) as threshold-free corroboration. The `paper_skeleton.tex` is already
>    F1-first; this PLAN's prose is the straggler.
> 2. **Two new programs must appear in the paper:** the `../../transformer/` extension
>    (Transformer beats the BiLSTM **on AUC**, ties on F1) and the `../../f1_optimization/`
>    program (symmetric F1-first optimization of both families), replicated under the
>    unified engine `../../journal_prep/issue12_unified_pipeline/`. So it is a
>    **12-issue** program, two model families, F1-first. Also: **PIP-Net was removed**
>    from the baseline table (custom split); see `issue3_baseline_comparison/`.
> The most current numbers/framing live in `f1_optimization/README.md`,
> `issue3_baseline_comparison/03_baseline_comparison.md` (+ `05_master_comparison_table.md`),
> and `transformer/SUPERVISOR_SUMMARY.md`.

Target journal: **MTI — Multimodal Technologies and Interaction** (MDPI, open access).
Tool: **Overleaf** + the official **MDPI Article LaTeX template** (`mdpi.cls` + `mdpi.bst`).
All numbers are final (see `../journal_prep/` Issues 1–12 + `../../f1_optimization/`).
This plan maps our work onto the MDPI structure, section by section, with the exact
results and figures that go in each.

> **Golden rule:** every claim in the paper must trace to a number in `journal_prep/`.
> Do not invent results. When in doubt, cite the issue folder.

---

## 0. Working title (pick one, refine later)

- *"Leakage-Free Pedestrian Crossing-Intention Prediction from Bounding Boxes and
  Ego-Speed: A Parsimonious BiLSTM Baseline on PIE"*
- *"Less Is Enough: A Two-Stream BiLSTM for Real-Time Pedestrian Crossing Prediction"*

## 1. The paper's one-sentence thesis (the spine, F1-first)

> On a **leakage-free, canonical PIE protocol**, two-stream (**bounding-box motion +
> ego-vehicle speed**) models reach **F1 0.844–0.847** — within 0.02–0.03 of the
> multimodal F1 ceiling (PedFormer 0.87) — while holding the **highest AUC in the
> standard-protocol table** (0.94–0.95) at **0.46–0.58 ms/window**, i.e. competitive
> with 3–7-stream multimodal SOTA at a fraction of the cost, shown with full
> statistical rigor and detector-in-the-loop realism. A BiLSTM and a staged-search
> Transformer over the identical input **tie on F1** (the Transformer wins on AUC only),
> so the parsimony result is about the input signal, not the architecture.

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

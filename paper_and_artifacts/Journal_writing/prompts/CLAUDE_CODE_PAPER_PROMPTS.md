# Claude Code prompt pack — writing the MDPI MTI paper from scratch

**How to use this file.** Open a **fresh** Claude Code session at the repo root
(`/Users/arif/Developer/pedestrian-thesis`). Paste **Prompt 0 (the Master Prompt)** first —
it is the constitution for the whole session and everything after it depends on it. Then run
the **Stage-1 research prompts** (R1–R4) one at a time. Then run the **Stage-2 section
prompts** (S1–S9) in the order given. Finish with the **Stage-3 assembly prompts** (A1–A3).

Do not skip Prompt 0. Do not run the section prompts before the research prompts — the whole
point of the ordering is that no sentence gets written before its evidence exists.

---
---

# PROMPT 0 — THE MASTER PROMPT (paste this first, once)

```
You are helping me write a journal manuscript for MDPI MTI (Multimodal Technologies and
Interaction). This session has exactly one job: produce a complete, submission-ready
LaTeX manuscript, written from scratch, section by section, with every factual claim
traceable either to a run output in this repository or to a verified citation.

Read this brief completely before doing anything. Then read the source files listed in
section D. Do not write a single line of the manuscript until I give you a section prompt.

================================================================
A. WHAT THE PAPER IS
================================================================

Working title (may be refined, must keep the same three ideas):
"Two Streams, Four Architectures: A Leakage-Free, Statistically Rigorous Benchmark for
Pedestrian Crossing-Intention Prediction on PIE"

One-sentence thesis:
On a leakage-free re-extraction of the PIE benchmark, models consuming only a pedestrian
bounding box and the ego-vehicle's speed reach F1 0.844-0.852 and the highest ROC-AUC in
the standard-protocol table, and four very different temporal encoders trained through one
identical engine are statistically indistinguishable on the primary metric -- so the input
signal, not the architecture, carries this task.

Three contributions, in this order of emphasis:
  1. A quantified temporal-leakage audit of the common PIE windowing protocol, and a
     leakage-free re-extraction anchored at PIE's own crossing_point event (0% verified
     leakage, 3.5x the data).
  2. A rigorous re-evaluation of a deliberately parsimonious two-stream model: window and
     pedestrian-cluster bootstrap intervals, leave-one-set-out cross-validation, multi-seed
     ablations, a documented hyperparameter search, and a detector-in-the-loop test.
  3. A four-family architecture isolation (LSTM, Transformer, GRU, un-gated RNN) under one
     engine at matched and at tuned budgets, showing the families tie on the primary metric.

Metric hierarchy, locked by my supervisor and never to be reordered:
F1 -> accuracy -> ROC-AUC. Lead every table, every claim and every sentence with F1.
AUC and PR-AUC are threshold-free corroboration, never the headline.

MANDATORY FRAMING RULE. We do NOT claim to be first to show that minimal inputs suffice on
PIE. PedCMT, the pose-free ablation of GTransPDM, and Achaji et al. got there first, and the
paper must concede this explicitly and early, in the Introduction and again in the
Discussion. Our novelty is the leakage audit, the statistical rigor, and the causal
four-family explanation of WHY two streams suffice. A draft that headlines "two streams are
enough" as if it were new is wrong and will be rejected; rewrite it if you catch yourself
drifting there.

================================================================
B. HARD CONSTRAINTS
================================================================

B1. THE MDPI TEMPLATE IS UNTOUCHABLE.
    - The manuscript is built on the official MDPI article template already in this repo at
      paper_and_artifacts/Journal_writing/MDPI_Article_Template/ (class Definitions/mdpi.cls,
      style Definitions/mdpi.bst, reference template template.tex).
    - You may NEVER edit, patch, monkey-patch, or "fix" any file under Definitions/.
      Not one character. If something does not render, change our content, not the class.
    - The document class line is exactly:
      \documentclass[mti,article,submit,moreauthors]{Definitions/mdpi}
    - Every template macro must be present and used as the template uses it: \Title,
      \Author, \AuthorNames, \address, \corres, \abstract, \keyword, \begin{document},
      the numbered sections, \supplementary, \authorcontributions, \funding,
      \institutionalreview, \informedconsent, \dataavailability, \acknowledgments,
      \conflictsofinterest, \abbreviations, \reftitle{References}, the bibliography.
    - Do not invent your own preamble, do not add packages the template does not already
      load unless a section prompt explicitly tells you to, do not restyle headings,
      captions, or tables. MDPI section order is fixed: Introduction, Materials and Methods,
      Results, Discussion, Conclusions (plus the extra numbered sections I authorise below).
    - Tables use the template's tabularx/booktabs idiom exactly as template.tex demonstrates.
      Figures use \includegraphics with the template's figure environment.

B2. LENGTH: 15 to 17 pages in the submit layout, hard ceiling 17. This is roughly
    7,000-7,500 words of body text plus 6 figures and 5 tables. If a draft overruns, cut
    prose, never cut a result, a limitation, or a citation. Report the compiled page count
    after every assembly.

B3. NUMBERS. Every number in the paper must trace to a file in this repository. Never
    invent, never round differently in two places, never carry a number from memory. If you
    cannot find a number's source file, say so and stop; do not approximate it. Two
    statistics exist and must never be mixed in one sentence: the per-seed mean over five
    seeds (these are the paper's numbers, comparable to other papers) and the five-seed
    probability ensemble (deployable, slightly higher, always labelled as an ensemble).

B4. CITATIONS. Never invent a citation, a DOI, an author list, a venue, or a page range.
    Every reference must come from the verified bibliography produced in Stage 1. If a claim
    needs a source we do not have, flag it in your response as MISSING CITATION and leave a
    visible \todo-style comment in the .tex rather than writing an unsourced sentence.

B5. EVERY DESIGN DECISION NEEDS A WARRANT. This is a journal paper, so no methodological
    choice may appear as a bare assertion. Each of the following must be justified either by
    a citation to the literature or, where it is our own choice, by an explicit stated reason
    plus the experiment that supports it: the 16-frame observation window; the 30-60 frame
    prediction horizon; the crossing_point anchoring rule; the fixed split by recording set;
    train-only z-score normalization; raw pixel coordinates rather than image-normalized
    ones; the positive class weight of 1.682; the F1-first hierarchy; validation-only
    threshold tuning; the pedestrian-cluster bootstrap; paired bootstraps for model
    contrasts; the matched-versus-tuned two-design comparison; the choice of five seeds;
    hidden size and depth. If a warrant is missing, say so before writing the sentence.

B6. TONE. Professional academic register for an engineering journal. Declarative,
    measured, specific. Forbidden: promotional adjectives (novel, powerful, remarkable,
    seamless, robustly demonstrates), hedge-stacking, rule-of-three lists as a stylistic
    tic, "moreover/furthermore/additionally" chains, em-dash overuse, paragraphs that open
    with a participial phrase ("Building on this, ..."), and any sentence that could be
    deleted without losing information. Prefer the active voice where the actor matters
    ("we re-anchored the windows"), the passive where the object matters. Every paragraph
    must carry a claim; no throat-clearing openers. After each section is drafted, run the
    /humanizer skill on it and apply the result.

B7. HONESTY. State weaknesses before a reviewer finds them. Specifically these must appear
    in the manuscript and must not be softened away:
    - Ego-speed partly encodes the ego-driver's own anticipation (the instrumented car slows
      for a pedestrian the driver expects to cross); the field has independently flagged this
      driver-side bias.
    - Our accuracy is mid-band (0.897-0.902) against PedFormer's 0.93; we lead on AUC and are
      within about 0.02 of the F1 ceiling. Never write "state of the art" unqualified.
    - Our leakage-free protocol is not identical to the protocol the tabulated baselines
      used, so the comparison table is indicative, not a like-for-like leaderboard. Say this
      in the table caption and in the text.
    - The family equivalence is horizon-bounded: at a 64-frame window the un-gated RNN alone
      falls behind (directional, per-seed intervals overlap).
    - The detector-in-the-loop study rests on two clips with ground-truth-guided association,
      so its +0.009 AUC drop is a lower bound.
    - Five seeds is low statistical power; the paired and cluster bootstraps carry the
      argument, not the n=5 t-tests.
    - Single dataset (PIE). Cross-dataset validation is stated as ongoing/future work.

B8. SCOPE EXCLUSION. The JAAD cross-dataset work in
    journal_prep/cross_dataset_validation/ is DELIBERATELY OUT OF SCOPE for this paper.
    Do not put its numbers in any table, figure, or result sentence. Cross-dataset
    generalization is written only as future work in the Discussion and Conclusions.

B9. Front matter (authors, affiliations, ORCID, corresponding e-mail, author-contribution
    initials, repository URL in Data Availability) stays as clearly marked PLACEHOLDER text.
    I will fill it in myself. Do not invent names or a URL.

B10. MY EXISTING WORK IS STRICTLY READ-ONLY. This is not a preference, it is the single
    constraint I care most about. Everything already in this repository is evidence and
    history; the new manuscript is written ALONGSIDE it, never on top of it. Concretely:

    Read freely, modify NEVER, for any reason:
      paper_and_artifacts/Journal_writing/MDPI_Article_Template/   (incl. main.tex,
                                                                   main_short.tex,
                                                                   main_drawio.tex,
                                                                   template.tex,
                                                                   references.bib,
                                                                   references2.bib,
                                                                   figures/, figures_drawio/,
                                                                   Definitions/, all PDFs)
      paper_and_artifacts/Journal_writing/Overleaf_package/        (incl. main.tex, its
                                                                   references.bib, figures/,
                                                                   Definitions/, main.pdf)
      paper_and_artifacts/Journal_writing/Overleaf_package*.zip
      paper_and_artifacts/Journal_writing/*.md                     (PLAN, README,
                                                                   ProjectDescription,
                                                                   relatedwork,
                                                                   STATISTICS_SOURCES,
                                                                   QUALITATIVE_FIGURE_PLAN)
      paper_and_artifacts/Journal_writing/drawio/  and  supplementary/
      journal_prep/  in its entirety, including every run output, CSV, JSON and figure
      transformer/, f1_optimization/, gru/, rnn/, pipeline/, paper_and_artifacts/runs/
      every figure generator script anywhere in the repository

    The ONLY paths you may create or write to in this whole session:
      paper_and_artifacts/Journal_writing/submission/**            (the new manuscript)
      paper_and_artifacts/Journal_writing/submission/evidence/**   (the Stage-1 research)

    Rules that follow from this, and that you must apply without being reminded:
      - Never run a figure generator, never regenerate a figure, never re-run a training
        or analysis script. Every figure you need already exists as a PDF; COPY it into
        submission/figures/ and use the copy. If a figure seems wrong, tell me, do not fix it.
      - Never edit an existing .bib file. submission/references.bib is a NEW file that you
        create; you may seed it by copying entries out of the existing bibliographies, but
        the originals stay byte-identical.
      - Never "tidy", reformat, rename, move, or delete anything outside submission/.
      - Never git add, git commit, git checkout, git restore, git stash, or git clean.
        Leave the working tree exactly as you found it apart from the new submission/ folder.
      - If a task seems to require touching a protected path, stop and ask me instead.

    Before you finish any prompt in this session, verify you have written nothing outside
    submission/. If you have, say so immediately and restore the file.

================================================================
C. THE FACTS OF THE STUDY (verify each against its source file before use)
================================================================

Dataset. PIE (Rasouli et al., ICCV 2019). ~6 h on-board video, 1920x1080 at 30 fps, Toronto,
six recording sets. Per-frame boxes, per-frame `cross` state, per-pedestrian crossing label,
per-pedestrian `crossing_point` event frame, synchronized OBD ego-speed. 582,376 frame rows,
1,374 pedestrians after dropping crossing_label == -1.

Split by recording set: train set01/02/04, validation set05/06, test set03. No random split.

The leak (old track-end anchor). The audit's unit is the SEQUENCE, not strictly the pedestrian
(1,389 sequences under one window per contiguous segment); follow the wording of
journal_prep/issue1_leakage_audit/01_leakage_report.md and do not silently upgrade "sequences"
to "pedestrians". Of 570 crossing sequences, 387 (67.9%) have at least one
frame inside the observation window in which they are already crossing; 369 (64.7%) are
crossing in all sixteen frames; only 183 (32.1%) have a clean window. Median gap between
window end and crossing onset is +182 frames. Static shortcut: rank-biserial +0.65 box area,
+0.63 height, +0.49 bottom edge. Convergence at epoch 3.

The fix: anchor at PIE's crossing_point, truncate the containing contiguous segment at that
frame inclusive, slide 16-frame windows at 50% overlap (stride 8), keep only windows whose
last observed frame is 30-60 frames before the crossing point. 107 of 1,374 pedestrians
(7.8%) excluded as too short. Validity: crossing_point equals the first `cross == crossing`
frame for 516 of 519 crossers (99.4%) and never precedes it.

Protocol comparison:
  track-end anchor : 1,389 windows (616/186/587), 41.0% positive, leakage 387/570 (67.9%),
                     class weight 1.44
  crossing-point   : 4,906 windows (2,178/634/2,094), 33.6% positive overall
                     (train 37.3 / val 24.4 / test 32.5), leakage 0/4,906 (0.000%),
                     class weight 1.682
Post-fix static effects: area +0.25, height +0.21, bottom edge +0.09. Convergence epoch 17.
Eval parity: per-window AUC 0.9131, per-pedestrian 0.9143, benchmark-filter subset 0.9194,
short tracks 46-75 frames 0.8634 (the extra tracks we admit are harder, not easier).

Features. 16-frame window (0.53 s) of [x1, y1, x2, y2, vehicle_speed]; raw PIE pixel
coordinates, NOT normalized by image size; per-feature z-score with mean and std from the
training split only; decision threshold 0.5 unless a validation-tuned tau is stated.

Training, frozen and identical for every family (the one engine is
journal_prep/issue12_unified_pipeline/12_unified_engine.py, families bilstm | transformer |
gru | birnn, --select f1 | auc): BCEWithLogitsLoss with pos_weight 1.682 applied only in the
training gradient, WITH ONE STATED EXCEPTION -- the tuned Transformer row was trained at
pos_weight 2.5, selected on validation in the symmetric class-weight sweep (see
journal_prep/Analysis/hyperparameters.md and f1_optimization/README.md arm F4). This exception
must be stated in Experimental Settings; omitting it makes the frozen-protocol claim false; Adam, weight decay 1e-5; batch 32; at most 100 epochs; early stopping
patience 15; ReduceLROnPlateau factor 0.5 patience 5; five seeds [42, 0, 1, 2, 3]; test set03
touched exactly once per experiment on the validation-selected checkpoint.

Main results table (test set03, 2,094 windows, 32.5% positive, per-seed means over 5 seeds):

  Matched configuration (width 128, 2 layers, dropout 0.3, LR 1e-3)
    LSTM          594,561 params   F1 0.828   Acc 0.883   AUC 0.932
    GRU           446,081          F1 0.840   Acc 0.898   AUC 0.933
    RNN           149,121          F1 0.836   Acc 0.889   AUC 0.942
    Transformer   268,417          F1 0.821   Acc 0.878   AUC 0.942
  CAREFUL: the phrase "only the cell changes" applies to the THREE RECURRENT ROWS ONLY.
  Attention has no hidden-size equivalent, so the matched Transformer row is the un-searched
  default (d128/ff256, 2 layers, dropout 0.1) and is the one row in this block selected on
  validation F1 rather than validation AUC. The same architecture under the AUC rule reads
  0.934 +/- 0.006 (transformer/SUPERVISOR_SUMMARY.md section 4), not 0.942. Carry this as a
  table footnote, as the prior draft did, and never write "only the cell changes" about a
  block that includes the Transformer.
  Tuned per family at an equal search budget
    LSTM        2,237,313          F1 0.844   Acc 0.897   AUC 0.940
    Transformer   794,241          F1 0.847   Acc 0.896   AUC 0.947
    GRU         1,678,209          F1 0.849   Acc 0.901   AUC 0.941
    RNN           560,001          F1 0.852   Acc 0.902   AUC 0.948
  Ablations (LSTM, matched configuration)
    bounding box only (4-D)  594,497  F1 0.551  Acc 0.744  AUC 0.753
    + temporal attention     611,265  F1 0.821  Acc 0.879  AUC 0.925

Selected configurations (what each search chose): LSTM width 256 / 2 layers / dropout 0.3 /
LR 1e-3 / tau 0.52; Transformer 4 heads, 128/512, 4 layers, dropout 0.1, LR 1e-3, tau 0.65,
last-token pooling, sinusoidal positional encoding; GRU 256 / 2 / 0.3 / 5e-4 / tau 0.53;
RNN (Elman tanh) 256 / 2 / 0.2 / 1e-4 / tau 0.53.

Search budgets: LSTM, GRU and RNN each got the identical 36-configuration a-priori grid over
LR {1e-3, 5e-4, 1e-4} x dropout {0.2, 0.3, 0.5} x hidden {64, 128, 256} x depth {1, 2}
(54 nominal cells collapse to 36 because inter-layer dropout is inert at depth 1), selected in
three stages, test touched once. The Transformer got a 78-configuration staged search over
width {(64,128),(128,256),(128,512)} x depth {2,4} x pooling x positional encoding, then a
recipe stage. No stage evaluated the test set.

Statistics. Bootstrap AUC 95% CI [0.92, 0.95] at the window level and [0.92, 0.96] over
pedestrian clusters, 10,000 resamples on the 2,094 test windows, 541 test pedestrians
resampled as whole units for the cluster bootstrap; PR-AUC 0.876. Model contrasts are paired
bootstraps on identical resamples. Key contrasts, with 95% pedestrian-cluster intervals:
  GRU vs LSTM on F1            +0.0071  [-0.0043, +0.0187]   ties
  RNN vs LSTM on F1            +0.0033  [-0.0083, +0.0150]   ties
  RNN vs GRU on F1             -0.0038  [-0.0117, +0.0039]   ties
  Transformer vs LSTM on F1    +0.0008  [-0.0124, +0.0142]   ties, p = 0.762
  BiLSTM F1-optimized vs its own AUC-selected baseline: +0.0187 [+0.0073, +0.0300] real
  AUC-selected Transformer vs AUC-selected LSTM: +0.0135 [+0.0097, +0.0174] real
  Un-searched Transformer vs LSTM on AUC: +0.0005 [-0.0034, +0.0043] ties (the win is the
    search, not attention)
  RNN vs searched Transformer on AUC: -0.0013 [-0.0041, +0.0015] ties
  GRU vs searched Transformer on AUC: -0.0070 [-0.0101, -0.0038] loses

Other results.
  Ego-speed ablation: dropping v takes AUC 0.932 -> 0.753, F1 0.828 -> 0.551, Acc 0.883 -> 0.744.
  Horizon: two runs exist and must never be mixed in one sentence. The MATCHED-COHORT run over
    the same 493 test pedestrians gives AUC 0.961 / 0.946 / 0.919 at 1.0 / 1.5 / 2.0 s with
    every pairwise paired t-test p <= 0.004 (journal_prep/issue6_window_tte_ablation/
    06b_matched_tte_report.md); the ALL-ELIGIBLE sweep gives 0.960 / 0.948 / 0.919 at
    p <= 0.008. Kruskal-Wallis p = 0.002. Sample effect between the two <= 0.002 AUC.
  Observation length: 8 / 16 / 30 frames give AUC 0.931 / 0.933 / 0.937, all pairwise
    p > 0.21; extending to 32 and 64 frames makes every family slightly worse; at 64 frames
    the un-gated RNN alone loses 0.050 F1 versus its own 16-frame result, about twice the
    gated cells' decline (directional only, per-seed intervals overlap). Note the window
    sweep is not a matched cohort: test N shrinks 2,094 -> 1,009 -> 458.
  LOSO across all six sets: AUC 0.928 +/- 0.041; the set03 fold scores 0.931, at the fold
    mean; excluding the 47-window set05 fold, the remaining five give 0.915 +/- 0.029.
  Capacity: hidden 64/128/256 give 0.927/0.933/0.938 (256 vs 128 not significant, p = 0.34,
    at 3.8x the parameters); depth 1/2/3 give 0.930/0.932/0.931. The 36-config grid winner
    beat the hand-set configuration on test by +0.0006, p = 0.91.
  Latency, Apple M4 CPU at batch 1: RNN 0.316 ms/window, Transformer 0.459, LSTM 0.575,
    GRU 0.721; 46x to 105x inside a 30 fps frame budget. Full pipeline: detector about 93%
    of per-frame cost, intention model 4.5%, about 27.5 fps end to end.
  Detector-in-the-loop: two test clips, 98 pedestrians, 311 windows, mean IoU 0.75; AUC
    0.962 -> 0.953 per window and 0.958 -> 0.948 per pedestrian; decisions flip in 10 of 311
    windows (3%). Detector recall 88% of pedestrians; ByteTrack dominant-identity purity 39%.
  Qualitative: five-seed LSTM ensemble at validation-tuned tau 0.516; a pedestrian flagged at
    p = 0.71 about 1.5 s before stepping off the kerb, and a kerbside worker correctly left at
    p = 0.31; across the two clips the ensemble reaches accuracy 0.920 and F1 0.884 over 439
    windows. Faces blurred. Video S1 is 26 s of continuous pipeline output.

Environment: training, statistics, the recurrent searches and all timing on an Apple M4
(10 CPU cores, 16 GB unified memory, macOS 26.5); the larger Transformer search and the
observation-length sweeps on a Kaggle NVIDIA T4; every tabled configuration retrained locally
through the same engine, inference parity verified at about 1e-6. Python 3.13.5, PyTorch
2.12.0, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0, Ultralytics 8.4.68 with a YOLO26-M
detector and the bundled ByteTrack. Reproducibility caveat: training nn.LSTM on Apple MPS is
process-history-dependent, so recurrent runs needing exact reproduction were run on CPU,
where training is bit-reproducible; Transformer training does not show this.

Baseline table on the standard PIE protocol (Acc / AUC / F1 / streams), to be re-verified in
Stage 1 before use:
  PCPA (WACV 2021)                0.87 / 0.86 / 0.77  box+pose+context+speed (4)
  Pedestrian Graph+ (2022)        0.89 / 0.90 / 0.81  pose graph + ego (2-3)
  IntFormer (2021)                0.89 / 0.92 / 0.81  multimodal
  PIT (2023)                      0.91 / 0.92 / 0.82  multimodal
  BiPed (2023)                    0.91 / 0.90 / 0.85  multimodal
  PedFormer (2023)                0.93 / 0.90 / 0.87  multimodal multitask (the F1/Acc ceiling)
  GTransPDM (2024)                0.90 / 0.87 / 0.82  box+pose+ego (3)
  GTransPDM without pose          0.92 / 0.90 / 0.86  box+ego (2), our closest cousin
  PedCMT                          0.92 / -- / --      box+ego-speed (2)   UNVERIFIED
  MFT (2025)                      0.90 / 0.94 / 0.83  4 context streams   UNVERIFIED
Two rows are flagged UNVERIFIED and constraint B4 forbids printing them until R1 confirms them
against the primary PDF: PedCMT's exact PIE Accuracy/AUC/F1, which our whole novelty concession
rests on, and MFT's AUC, which relatedwork.md notes was inferred from a cost table rather than
read from a metrics table. Our headline claim of the highest ROC-AUC in the table depends on
MFT's 0.94 being real, so if R1 cannot verify it, the claim is reworded, not printed anyway.
BiPed also has no primary citation of its own: relatedwork.md gives it arXiv 2210.07886, which
is PedFormer's identifier, and the old bibliography points the BiPed row at the PedFormer entry.
R1 must find BiPed's own paper and its own numbers, or the row comes out.
Do not tabulate PIP-Net (custom random split, prose context only) and do not tabulate the
Occlusion-Aware Diffusion model (occluded-only, roughly one-frame-ahead protocol; cite it as
a minimal-modality precedent only). PedFormer and BiPed are separate rows with different
numbers; an earlier draft of ours conflated them, do not regress that.

Road-safety statistics for the Introduction, already verified first-hand and recorded in
paper_and_artifacts/Journal_writing/STATISTICS_SOURCES.md: WHO Global Status Report on Road
Safety 2023 gives 1.19 million road traffic deaths in 2021, road traffic injury the leading
cause of death for ages 5-29, and pedestrians at 23% of fatalities (four-wheel occupants 30%,
powered two- and three-wheeler users 21%, cyclists 6%, other 20%). NHTSA Traffic Safety Facts
2023 Data: Pedestrians (DOT HS 813 727) gives 7,314 US pedestrians killed in 2023 against
4,910 in 2014, the pedestrian share rising 15% -> 18%, with 84% of fatalities urban, 74% away
from intersections, 77% in the dark. Use no road-safety number that is not in that file.

================================================================
D. WHAT TO READ, IN THIS ORDER, BEFORE WRITING ANYTHING
================================================================
 1. paper_and_artifacts/Journal_writing/ProjectDescription.md   the whole project in one file
 2. journal_prep/Analysis/model_comparison.md                   the consolidated results
 3. journal_prep/Analysis/hyperparameters.md                    per-model configurations
 4. journal_prep/Analysis/latency_comparison.md                 timing
 5. journal_prep/Analysis/documentation.md                      the reviewer Q&A
 6. journal_prep/issue1_leakage_audit/  and  issue2_clean_protocol/   the leakage story
 7. journal_prep/issue3_baseline_comparison/03_baseline_comparison.md
    and 04_positioning_vs_prior_work.md                         the comparison + positioning
 8. journal_prep/issue4_bootstrap_ci/ 5_loso_cv/ 6_window_tte_ablation/ 7_hidden_size/
    8_grid_search/ 9_latency/ 10_gt_vs_detector/                one headline each
 9. transformer/SUPERVISOR_SUMMARY.md, f1_optimization/README.md, gru/README.md, rnn/README.md
10. journal_prep/obs_window_extension/PLAN.md and 01_ow_results.csv
11. paper_and_artifacts/Journal_writing/relatedwork.md          the literature landscape and
                                                               the honest defensibility analysis
12. paper_and_artifacts/Journal_writing/STATISTICS_SOURCES.md   every road-safety figure
13. paper_and_artifacts/Journal_writing/MDPI_Article_Template/template.tex   the template idiom

Prior drafts exist at MDPI_Article_Template/main.tex (25 pp) and Overleaf_package/main.tex
(17 pp). OPEN THEM READ-ONLY, ONCE, FOR FACTS AND FOR WHAT LENGTH FITS, THEN SET THEM ASIDE.
They are my existing papers and they are frozen: do not edit them, do not improve them, do
not "bring them up to date", do not copy a paragraph across into the new manuscript. We are
writing fresh prose in a separate file. See constraint B10.

================================================================
E. WORKING DIRECTORY AND OUTPUT
================================================================
Create a new folder paper_and_artifacts/Journal_writing/submission/ and do ALL of your work
there. Per constraint B10 it is the only writable location in the repository:
  - Definitions/ copied in from MDPI_Article_Template/, unchanged (copy the folder; the
    originals are read-only, and you may never edit either the copy or the original)
  - figures/ holding COPIES of the existing figure PDFs you use; never regenerate a figure
  - main.tex, the new manuscript you build up section by section
  - references.bib, a NEW bibliography file that Stage 1 produces
  - evidence/, the Stage-1 research outputs
Build with `tectonic main.tex` from that folder, and report the page count each time.
Everything outside submission/ must be byte-identical when the session ends.

================================================================
F. HOW I WANT YOU TO WORK
================================================================
- One section per prompt. Do not run ahead and draft sections I have not asked for.
- Before writing a section, list the specific numbers it will use and the file each comes
  from, and the specific citations it will use. Wait for nothing; just show me the list at the
  top of your response, then write the section.
- After writing a section: run /humanizer over it, apply the result, recompile, report the
  page count and word count, and list anything you had to flag as MISSING CITATION or
  UNSOURCED NUMBER.
- Never touch Definitions/. Never edit a figure generator to make a number match the prose;
  if prose and figure disagree, the run output wins and the prose changes.
- If you believe a claim in this brief is wrong, say so with the file that contradicts it
  before writing it.

Confirm you have read this brief and the files in section D by giving me: (a) the three
contributions in your own words, (b) the list of paths you are forbidden to write to and the
one path you may write to, (c) the things you are forbidden to do, and (d) any contradiction
you found between this brief and the repository. Then stop and wait.
```

---
---

# STAGE 1 — RESEARCH AND VERIFICATION (run R1–R4 before any writing)

These use the `/deep-research` skill. Each produces an evidence file under
`paper_and_artifacts/Journal_writing/submission/evidence/`. Nothing gets written into the
manuscript until all four exist.

## R1 — Verify every baseline number in the comparison table

```
/deep-research Verify, against primary sources only, the reported PIE crossing-prediction results for each of the following methods, and report Accuracy, ROC-AUC and F1 together with the exact evaluation protocol each used. Methods: PCPA (Kotseruba, Rasouli & Tsotsos, WACV 2021); Pedestrian Graph+ (Cadena et al., 2022); IntFormer (Lorenzo et al., arXiv 2105.08647); PIT (Zhou et al., 2023); BiPed and PedFormer (Rasouli & Kotseruba, arXiv 2210.07886 and the ICRA/journal versions); GTransPDM full model and its without-pose ablation (Xie et al., arXiv 2409.20223); PedCMT (Chen et al., IEEE T-ITS 2024, DOI 10.1109/TITS.2024.3386689); MFT (Li et al., arXiv 2511.20011); PIP-Net (Azarmi et al., arXiv 2402.12810). For each method I need: (1) the numbers as printed in the method's OWN paper, not as re-tabulated by a third party, and where a third-party table is the only source, say so explicitly and name it; (2) whether it used the standard PIE split train set01/02/04, validate set05/06, test set03; (3) the observation length and time-to-event horizon in frames or seconds; (4) how many input streams and which modalities; (5) whether the paper states any check on observation-window temporal leakage; (6) a complete BibTeX entry with a verified DOI. Three specific questions I need settled: whether GTransPDM's abstract figure of 92 percent refers to the without-pose ablation rather than the full model; PedCMT's exact PIE Accuracy, AUC and F1, since our paper concedes the minimal-modality precedent to it; and whether Pedestrian Graph+ and BiPed were evaluated under a different configuration, as GTransPDM's table footnote implies. Flag any number you cannot verify from a primary source rather than reporting it. Write the result to paper_and_artifacts/Journal_writing/submission/evidence/R1_baselines.md as a table plus a per-method notes block, and append every verified BibTeX entry to submission/references.bib.
```

## R2 — Find the citable warrant for every methodological decision

```
/deep-research Find published, citable justification for each of the following methodological choices in a pedestrian crossing-intention prediction study on the PIE dataset, and for each one tell me whether the literature supports it, contradicts it, or is silent. (1) Anchoring the observation window at a labelled crossing-onset event rather than at the end of an annotated track, and any prior work that audits temporal leakage or label leakage in pedestrian action or intention benchmarks. (2) Treating F1 as the primary metric ahead of accuracy and ROC-AUC for an imbalanced, safety-critical binary decision, including what the pedestrian-action-prediction benchmark literature says about metric choice under class imbalance. (3) Using a clustered or grouped bootstrap when evaluation units are overlapping windows drawn from the same tracked subject, and the standard statistical treatment of non-independent evaluation units in machine-learning evaluation. (4) Paired bootstrap resampling for comparing two models on an identical test set. (5) The argument that architecture comparisons are confounded by unequal hyperparameter search budgets, and the practice of tuning every candidate architecture separately under an equal budget: I am aware of Greff et al. LSTM: A Search Space Odyssey, Lucic et al. Are GANs Created Equal, and Melis et al. On the State of the Art of Evaluation in Neural Language Models, and I need their exact claims verified plus any more recent equivalent. (6) Matching width versus matching parameter count when swapping a recurrent cell, including how Chung et al. 2014 handled it. (7) Tuning a decision threshold on validation data only. (8) Computing normalization statistics on the training split only. (9) Class-weighted binary cross-entropy in place of resampling for moderate imbalance. (10) Reporting five random seeds and the known limits of small-seed-sample significance testing in deep learning. For each item give me the strongest one or two citations with complete verified BibTeX and a one-sentence statement of exactly what the source supports, phrased so it can be cited in a Materials and Methods section. Where the literature is silent, say so plainly, since I will then justify the choice from our own experiments instead. Write the result to paper_and_artifacts/Journal_writing/submission/evidence/R2_method_warrants.md and append the BibTeX to submission/references.bib.
```

## R3 — Sweep for work newer than mid-2026 that could scoop or contradict us

```
/deep-research Survey pedestrian crossing-intention and pedestrian action prediction work published or preprinted from mid-2025 onward, with emphasis on anything from 2026, and tell me specifically whether any of it undercuts, duplicates, or contradicts the following four claims. Claim 1: that the common PIE observation-window protocol admits the crossing event into the observation window for a large fraction of positive samples, and that re-anchoring at PIE's crossing_point event removes it; I need to know whether anyone has now published a temporal-leakage audit of PIE or JAAD windowing. Claim 2: that a two-stream input of pedestrian bounding box plus ego-vehicle speed is competitive with three-to-seven-stream multimodal models on PIE; I already concede precedence to PedCMT, the pose-free GTransPDM ablation, and Achaji et al., and I need to know whether anything newer strengthens or weakens that concession. Claim 3: that LSTM, Transformer, GRU and un-gated Elman RNN encoders trained on the same input under matched search budgets are statistically indistinguishable on F1 for this task; I need any published architecture-comparison study on this task or a close neighbour. Claim 4: that ego-vehicle speed dominates the bounding-box stream and that this partly reflects the ego-driver's own anticipation. Also report the current best published PIE numbers on the standard split as of now, so I can state honestly where we sit, and list any new dataset released for this task since 2025. For each relevant paper give the venue, date, exact claim, the PIE numbers if any, and a complete verified BibTeX entry, and rank the papers by how much they matter to the four claims above. Write the result to paper_and_artifacts/Journal_writing/submission/evidence/R3_recent_work.md and append the BibTeX to submission/references.bib.
```

## R4 — MDPI MTI author instructions

```
/deep-research Retrieve the current MDPI Multimodal Technologies and Interaction (MTI) Instructions for Authors and the MDPI general author guidelines, and produce a checklist of every requirement that constrains how I write and format a research article for that journal. I specifically need: the required manuscript structure and section order for a research article; whether the abstract must be structured or unstructured and its word limit; the keyword count range; the required back-matter statements and their exact expected wording, covering author contributions and the CRediT taxonomy, funding, institutional review board statement, informed consent, data availability, acknowledgments including MDPI's current generative-AI disclosure requirement, and conflicts of interest; the reference style, numbering convention and whether DOIs are required; figure and table requirements covering resolution, placement, caption style, and whether figure parts are labelled (a) and (b); supplementary material and video declaration rules; the rules on preprints and on data and code availability; MTI's aims and scope in enough detail that I can state in a cover letter why a pedestrian-crossing-intention paper combining bounding-box vision with ego-vehicle telemetry fits a multimodal-interaction journal; and any word or page guidance for research articles. Quote the exact wording of anything that must appear verbatim in the manuscript. Write the result to paper_and_artifacts/Journal_writing/submission/evidence/R4_mti_rules.md as a checklist I can tick off before submission.
```

## R5 — reconcile (run after R1–R4, no web search needed)

```
Read the four evidence files in paper_and_artifacts/Journal_writing/submission/evidence/ and reconcile them against section C of the master brief. Produce submission/evidence/R5_reconciliation.md containing: (1) every number in section C that R1 or R3 contradicts, with both values and the source of each; (2) every design decision from constraint B5 that R2 could NOT find a citation for, which I will then have to justify from our own experiments; (3) every MDPI MTI requirement from R4 that the plan in this brief does not yet satisfy; (4) any claim in section A that R3 shows is no longer defensible as stated, with a proposed replacement wording. Change nothing in the brief yourself. Present it as a decision list for me.
```

---
---

# STAGE 2 — WRITE THE MANUSCRIPT (S1–S9, in this order)

Concrete before framing: Methods, Settings, Results, then Related Work, Introduction,
Discussion, Conclusions, Abstract, back matter. Each prompt assumes Prompt 0 and Stage 1
are done.

## S1 — Set up the document shell

```
Set up submission/main.tex now, and nothing else. Create paper_and_artifacts/Journal_writing/submission/ and COPY Definitions/ and the figure PDFs you will need into it from MDPI_Article_Template/ (a copy, not a move; the originals stay untouched, per constraint B10 nothing outside submission/ may be written to at any point in this session). Build the file from the official template: the exact documentclass line from the master brief, the MDPI internal commands block, the front-matter macros with PLACEHOLDER author, affiliation, ORCID and correspondence fields, the title, empty \abstract{} and \keyword{} to be filled last, \begin{document}, the six numbered section headings we will fill (Introduction; Related Work; Materials and Methods; Experimental Settings; Results; Discussion; Conclusions), then the complete back-matter macro block in the template's order, then the bibliography. Take the float-placement relaxations from the previous draft's preamble only if you can show me they are content-neutral, and say so explicitly; otherwise leave the template's defaults. Do not write any prose. Compile it and confirm it builds clean, then show me the file.
```

## S2 — Materials and Methods

```
Write Section 3, Materials and Methods, in full. Target 1,500-1,700 words across these subsections: an Overview stating the fixed-versus-varied design; The PIE Dataset and Splits; Temporal Leakage and the Leakage-Free Protocol; Features and Preprocessing; Model Architectures; Metric Hierarchy, Evaluation and Statistical Procedure; Live Perception-to-Prediction Pipeline.

The leakage subsection is the methodological centrepiece and gets the most space: state the diagnosis, the fix, the algorithm precisely enough to reimplement, and the two properties that make it leak-free by construction rather than by inspection (crossing_point equals the first crossing-labelled frame for 516 of 519 crossers and never precedes it; the 30-frame minimum look-ahead). Include Table 1, the track-end versus crossing-point protocol comparison, and reference Figure 3 (fig3_protocol.pdf) and Figure 2 (fig2_system.pdf).

Every choice in constraint B5 that this section touches must carry its warrant from evidence/R2_method_warrants.md, cited. Where R2 found nothing, justify from our own experiment and say which one. In the architectures subsection make the ladder explicit: the three recurrent families are the same network with only the cell swapped, so the GRU isolates which gated cell and the un-gated RNN isolates gating itself. State that all models are trained from scratch and that the only pretrained component anywhere is the detector in the live pipeline. In the statistics subsection justify the pedestrian-cluster bootstrap on the 541 test pedestrians and the paired-bootstrap contrasts, with citations.

List your numbers and their source files first, then write. Then run /humanizer, apply it, compile, report page and word count.
```

## S3 — Experimental Settings

```
Write Section 4, Experimental Settings, in full. Target 700-850 words, no subsections. Cover: the hardware and where each class of run executed, and the statement that every tabled configuration was retrained locally through the same engine with inference parity verified at about 1e-6; the software stack with versions; the MPS reproducibility caveat for nn.LSTM and why recurrent runs went to CPU; the frozen training recipe shared by all four families; the search budgets, with the point that the three recurrent families received the identical 36-configuration grid and the Transformer a larger 78-configuration staged search because it has more architectural freedom, and that no stage evaluated the test set.

Then the passage that carries the most argumentative weight in this section: why the comparison is run twice, tuned and matched. Ground the tuned design in the architecture-audit literature verified in R2, and give the concrete reason imposing one family's recipe on another relocates the bias rather than removing it, using our own case that the RNN's search chose a learning rate an order of magnitude below the LSTM's. Then state the width-versus-parameter-count problem honestly: at width 128 an LSTM carries four gate matrices to the un-gated cell's one, so 594,561 parameters against 149,121, we match width because a cell swap requires it, Chung et al. matched parameter count instead, and we report parameters throughout so either reading is available.

Include Table 2, the selected configuration per family. List numbers and sources first, then write, then /humanizer, compile, report counts.
```

## S4 — Results

```
Write Section 5, Results, in full. Target 2,000-2,300 words, and this is the section most at risk of overrunning, so keep every paragraph load-bearing. Subsections in this order: The Temporal-Leakage Audit; Main Result; Does the Architecture Matter; Comparison with Published Baselines; Which Input Carries the Signal; Prediction Horizon and Observation Length; Cross-Set Generalization, Capacity and Latency; Detector-in-the-Loop Robustness; Qualitative Behaviour.

Floats: Figure 4 (fig4_leakage.pdf) in the audit subsection; Table 3, all four families under both comparison designs plus the two ablations, and Figure 6 (fig6_forest_compact.pdf) in the architecture subsection; Table 4, the baseline comparison with a modalities-and-stream-count column, using ONLY the numbers verified in evidence/R1_baselines.md and stating in the caption that baseline figures are as reported by their sources and that our protocol is leakage-free and therefore not strictly identical; Figure 9 (fig9_qualitative.pdf) in the qualitative subsection.

Rules specific to this section. Lead every comparison with F1. Report the architecture result as a tie with the interval that establishes it, never as a win for whichever family has the largest point estimate. Immediately after the ties, show the test is not underpowered by giving the contrasts where it does separate. Report the Transformer's AUC advantage plainly and then immediately give the un-searched-Transformer result that locates the advantage in the search rather than in attention. In the main-result subsection make the point that removing leakage barely moved AUC and that this is the point rather than a reassurance, and give the eval-parity checks that rule out the alternative explanation that the new protocol is simply easier. In the horizon subsection note that the decline overturns an earlier insensitive-to-horizon conclusion of our own that was itself a leakage artifact. In the window subsection state that the sweep is not a matched cohort and that N shrinks. In the detector subsection state the lower-bound caveat.

List numbers and sources first, then write, then /humanizer, compile, report counts.
```

## S5 — Related Work

```
Write Section 2, Related Work, in full. Target 900-1,100 words in three subsections: Datasets, Benchmarks and Models; Minimal-Modality Prediction and the Perception Front End; Pedestrian-Vehicle Interaction and Remaining Gaps.

This section has to do three jobs at once and every sentence should serve one of them: map the field accurately, concede the minimal-modality precedent explicitly and early, and set up the four gaps the paper closes, namely unaudited temporal leakage, heavy multimodal pipelines, thin statistical evaluation, and untested architecture assumptions. Use evidence/R3_recent_work.md to make sure nothing published since mid-2025 is missing and that no claim of ours is stated more strongly than the current literature allows.

The concession paragraph is mandatory and must name PedCMT, the pose-free GTransPDM ablation, and Achaji et al. with their verified numbers, then state precisely what we add: the result under a leakage-free protocol, with full statistical scrutiny, and with an explanation of why it holds. Note in passing that Achaji's numbers were obtained under the pre-audit protocol.

The interaction subsection must earn the MTI venue: the external-HMI literature concerns explicit vehicle-to-pedestrian communication, whereas our ego-speed result points the other way, making the vehicle's own motion an implicit channel a model can read, with the driver-anticipation caveat attached. Keep it to a compact paragraph.

Use only citations that exist in submission/references.bib with verified fields. List your citations first, then write, then /humanizer, compile, report counts.
```

## S6 — Introduction

```
Write Section 1, Introduction, in full. Target 850-1,000 words. Structure: the safety motivation with the verified WHO and NHTSA figures and Figure 1 (fig1_pedestrian_statistics.pdf); how the risk distributes, urban, away from intersections, in the dark, and why that means an ADAS must anticipate a crossing before it begins; the task definition and why intention prediction is not trajectory forecasting, with the PIE dataset introduced; the three recurring methodological weaknesses, leakage never checked, models heavy with three to seven streams, evaluation thin, each with a citation; the explicit concession that minimal-modality prediction is not itself new, with a forward reference to Related Work; the three contributions as a bulleted list with forward references to the sections that deliver them; and a closing statement of the metric hierarchy.

Use no road-safety number that is not in STATISTICS_SOURCES.md, and cite WHO for global figures and NHTSA for US figures with the correct data years, the report year and the data year being different for WHO. Do not oversell: the contributions list should read as a description of what was done, not as a claim of significance.

List numbers, sources and citations first, then write, then /humanizer, compile, report counts.
```

## S7 — Discussion

```
Write Section 6, Discussion, in full. Target 1,100-1,300 words, no subsections, four to five paragraphs, each with a single job.

Paragraph one, what the parsimony result means: two cheap inputs reach within about 0.02 of the multimodal F1 ceiling and the highest AUC in the table, we are not first to find this and name who was, and here is what we add, namely that it survives a leakage-free protocol and full statistical scrutiny and that we can explain it. Give the explanation: removing ego-speed collapses the model while replacing the encoder three times changes nothing measurable, so the predictive content sits in the ego-speed profile and coarse box dynamics, legible within half a second, leaving no long-range structure for a heavier encoder to exploit. Draw the practitioner's implication about where engineering effort belongs.

Paragraph two, the metric-dependence finding: on AUC the Transformer wins, on F1 the four families tie, neither result is wrong, and a paper reporting one of them would tell half the story. Then the sharper methodological point, that a single-metric single-seed unequal-budget architecture comparison can produce a headline in either direction, and that ours could have.

Paragraph three, the transferable leakage finding: it is not specific to our model, any study anchoring at track end inherits it, and give the three symptoms by which others can recognize it, implausibly fast convergence, static geometry separating the classes on its own, and insensitivity to the prediction horizon as the clearest tell. Note the audit is cheap because it needs only annotation the dataset already provides.

Paragraph four, limitations, every item in constraint B7, stated plainly and each with the mitigation or the experiment that would settle it. Single dataset first, since it is the largest exposure, and state cross-dataset replication as ongoing work without reporting any JAAD number.

Do not introduce any number that did not appear in Results. List your points first, then write, then /humanizer, compile, report counts.
```

## S8 — Conclusions

```
Write Section 7, Conclusions, in full. Target 350-450 words, three paragraphs. First, the leakage finding and the fix with the two numbers that matter, 67.9 percent and 3.5-fold. Second, what the leakage-free protocol then shows: the two-stream F1 range and the AUC standing with the honest stream-count comparison, ego-speed dominance, the architecture tie including the un-gated result, and the sentence that the input signal rather than the architecture decides, closing with the statement that no comparison rests on a point estimate. Third, three future directions: cross-dataset replication of the audit and the model, a speed-perturbation study to probe how much of the ego-speed result is driver anticipation, and the perception front end, since detector recall and track identity rather than the intention model are what would limit such a system in a vehicle. No new numbers, no new citations, no restating the whole Results section. Then /humanizer, compile, report counts.
```

## S9 — Abstract, keywords, and back matter

```
Write the abstract, the keywords, and the complete back matter, last, now that the paper exists.

Abstract: one unstructured paragraph, at most 200 words, following MDPI's background-methods-results-conclusions arc without headings, checked against the MTI requirement in evidence/R4_mti_rules.md. It must state the leakage finding, the re-anchored protocol, the two-stream input, the four-family comparison, and the actual headline numbers. It must contain no result that is not substantiated in the body and must not exaggerate. Count the words and tell me the count.

Keywords: eight to ten, MDPI style, semicolon-separated, drawn from terms a reader would search.

Back matter, in the template's order and using MDPI's expected wording verified in R4: the supplementary declaration for Video S1; author contributions with the CRediT terms and PLACEHOLDER initials; funding; institutional review board statement, which must state that the study used the public PIE dataset, involved no new human or animal experiments, and that every face in the reproduced frames was irreversibly blurred before publication; informed consent; data availability with PIE public and a PLACEHOLDER repository URL; acknowledgments including MDPI's current generative-AI disclosure wording if any AI assistance is to be declared, and tell me what that wording requires so I can decide; conflicts of interest; and the abbreviations block covering ADAS, AUC, BCE, CPU, GRU, IoU, LSTM, OBD, PIE, PR-AUC, RNN, ROC, TTE and any other abbreviation actually used in the text.

Then /humanizer on the abstract only, compile, report the final page count.
```

---
---

# STAGE 3 — VERIFY, FIT, AND FINISH

## A1 — Number and citation audit

```
Audit the finished submission/main.tex end to end and produce submission/evidence/A1_audit.md. For every numeric value in the manuscript, list the value, where it appears, and the repository file it comes from, and flag any value you cannot trace, any value that appears with two different roundings in different places, and any place a per-seed mean and an ensemble figure sit in the same sentence. For every \cite key, confirm the entry exists in references.bib with a complete author list, venue, year and DOI, and flag any entry that is incomplete or that was not verified in Stage 1. Confirm that every claim which constraint B5 says needs a warrant has one in the text. Confirm nothing from journal_prep/cross_dataset_validation/ leaked into a result. Then verify constraint B10 held: run `git status --porcelain` and confirm that the ONLY paths that differ from how the session started are under paper_and_artifacts/Journal_writing/submission/. If any other file was created, modified or deleted, list it, restore it, and tell me what happened before doing anything else. Fix what is unambiguously fixable, list what needs my decision, and do not paper over anything.
```

## A2 — Fit to 15–17 pages

```
Compile and report the exact page count. If it exceeds 17 pages, cut prose only, in this priority order: redundant restatement between Results and Discussion; background sentences in the Introduction and Related Work that the citation alone already carries; procedural detail in Experimental Settings that a reader could recover from the cited engine; over-explained figure callouts. Never cut a result, a limitation, a caveat, a citation, or a float. Tighten figure captions to two or three lines each if needed. Show me a before-and-after word count per section and the new page count. If it is below 15 pages, tell me rather than padding, and propose what could usefully be expanded.
```

## A3 — Final read as a reviewer

```
Read the finished manuscript once through as a hostile but fair MTI reviewer and write submission/evidence/A3_review.md. Judge it on: whether the novelty claim survives given the minimal-modality precedent we concede; whether the leakage audit is convincing enough to carry the paper; whether the statistical treatment is correct, particularly the cluster bootstrap and the paired contrasts; whether any limitation is buried, softened, or missing; whether the baseline table's protocol mismatch is disclosed clearly enough; whether the tone reads as human academic prose rather than generated text, naming any sentence that gives it away; whether it fits MTI's scope; and whether every MDPI requirement in evidence/R4_mti_rules.md is met. Give me a decision verdict, accept, minor revision, major revision or reject, with the three changes that would most improve the outcome. Be blunt. Do not fix anything in this pass.
```

---
---

# Notes for you (not for pasting)

- **What deep-research actually returns.** `/deep-research` is a web-search-and-verify
  harness: it produces a cited report, not LaTeX. That is why Stage 1 writes evidence files
  and Stage 2 consumes them. Do not expect R1–R4 to touch the manuscript.
- **The JAAD decision.** Per your call, `journal_prep/cross_dataset_validation/` stays out.
  Constraint B8 enforces it, and A1 checks it did not leak in. The finished JAAD audit
  (93% of crossers leaking under the naive anchor, 0% under the event anchor) remains
  available if a reviewer asks for cross-dataset evidence, and it would answer the single
  largest exposure this paper has.
- **The one item that could change the paper.** R1 is asked to settle PedCMT's exact PIE
  numbers. Our concession of the minimal-modality precedent is built on that paper, so if
  its numbers differ from the 0.92 accuracy we carry, the Related Work concession and the
  baseline table both need adjusting before S4 and S5 are written.
- **Front matter** stays PLACEHOLDER throughout by design (B9). You fill in authors,
  affiliations, ORCID, corresponding e-mail, contribution initials and the repository URL.
- **Your existing papers are protected by constraint B10**, which names the protected paths
  explicitly, restricts all writes to `submission/`, forbids regenerating any figure, editing
  any existing `.bib`, or running any git command, and makes the session declare the
  protected-path list back to you before it starts. A1 then checks `git status` at the end to
  prove nothing outside `submission/` moved. The new manuscript is built beside the old one,
  never on top of it.

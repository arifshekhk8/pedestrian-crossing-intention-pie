I'm continuing a master's-thesis → journal-paper project in this repo
(/Users/arif/Developer/pedestrian-thesis). Please read this whole briefing, then read
the key files I list, then continue from where we are.

## What the project is
Predicting **pedestrian crossing intention** on the **PIE dataset**: a 2-stream
**BiLSTM** that takes 16 frames (~0.5 s) of a pedestrian's bounding box + the
ego-vehicle's speed and outputs the probability they are about to cross. There's also
a live demo pipeline (YOLO26-M detection → ByteTrack tracking → BiLSTM). It's a
linear, numbered research pipeline (scripts 01_…→10_ at the repo root), not an app.

## Where we are RIGHT NOW (2026-06-28)
The experimental work is **DONE**. We ran an 11-issue "journal-readiness" program
(all in `journal_prep/`, one folder per issue) that fixed every weakness a reviewer
would attack. **All 11 issues are complete.** My supervisor has now told me to
**write the journal paper** for **MDPI MTI (Multimodal Technologies and Interaction)**
in **Overleaf/LaTeX**, strictly following the MDPI structure and BibTeX format. The
writing workspace is set up in `Journal_writing/`. **Your main job now is to help me
draft the paper, section by section.** No more experiments unless I ask.

## The headline result (the spine of the paper)
On a **leakage-free, canonical PIE protocol**, the 2-stream (bbox + ego-speed) BiLSTM
reaches **test AUC 0.932 ± 0.011** (5 seeds; 95% bootstrap CI **[0.92, 0.95]**;
PR-AUC 0.876; Acc 0.883) — the **top of the standard-protocol band** — at
**0.575 ms/window** latency. Competitive with 3–7-stream multimodal SOTA using only
2 cheap inputs, shown with full statistical rigor and detector-in-the-loop realism.

## What each issue established (key numbers + folder)
- **Issue 1** `issue1_leakage_audit/` — found TEMPORAL LEAKAGE: 67.9% of crossers were
  already mid-crossing inside the observation window (old AUC 0.931 inflated).
- **Issue 2** `issue2_clean_protocol/` — rebuilt leak-free (anchored at PIE's
  `crossing_point`, TTE∈[30,60], N 1,389→4,906, **0% leakage verified**). Clean
  headline **AUC 0.932 ± 0.011**. Feature ablation: **ego-speed is dominant** —
  bbox+ego 0.932 vs **bbox-only 0.753** (−0.18); **attention 0.925** (no benefit).
- **Issue 3** `issue3_baseline_comparison/` — baseline table (`03_baseline_comparison.md`)
  + positioning matrix (`04_positioning_vs_prior_work.md`). Ours = top AUC, 2 streams.
  Occlusion-Diffusion = minimal-modality *precedent*, not a comparison row. **FINALIZED
  internally; only external BibTeX/PIP-Net-split verification remains.**
- **Issue 4** `issue4_bootstrap_ci/` — 10k bootstrap: AUC 0.932, **95% CI [0.92, 0.95]**, PR-AUC 0.876.
- **Issue 5** `issue5_loso_cv/` — leave-one-set-out: **6-fold AUC 0.928 ± 0.041**;
  set03 fold 0.931 ≈ fixed split (set03 is representative, not easy).
- **Issue 6** `issue6_window_tte_ablation/` — multi-seed. **Window length insensitive**
  (obs 8/16/30 → 0.931/0.933/0.937, within seed noise). **Prediction horizon matters**:
  TTE 30/45/60 → 0.960/0.948/0.919 (every pairwise p≤0.008), **confirmed on a matched
  cohort** (`06b_`, sample effect ≤0.002) — overturns old "insensitive to TTE".
- **Issue 7** `issue7_hidden_size/` — hidden 64/128/256 → 0.927/0.933/0.938 (256 n.s.
  at 3.8× params). Depth companion `07b_`: layers 1/2/3 → 0.930/0.932/0.931. Model is
  small-data-limited; **hidden=128, 2 layers justified**.
- **Issue 8** `issue8_grid_search/` (supervisor-requested) — full **36-config grid**,
  **val-only selection + test touched once**; the search **CONFIRMS the hand-set
  baseline** (val-winner beats it on test by Δ+0.0006, p=0.91, n.s.).
- **Issue 9** `issue9_latency/` — isolated **BiLSTM 0.575 ms/window** (CPU; ~58× inside
  a 30 fps budget; CPU>MPS at batch 1 due to GPU dispatch overhead). Pipeline
  **detection-bound**: YOLO26-M 93%, BiLSTM 4.5% → 27.5 fps.
- **Issue 10** `issue10_gt_vs_detector/` — GT-box vs YOLO-box on 98 peds: **prediction
  robust to box noise** (AUC drop +0.009/+0.010, 3% decision flips). Weak links are
  perception: detector recall 88%, **ByteTrack fragmentation severe** (track purity 39%).
- **Issue 11** — root docs (THESIS_PLAN/CODE_STATE/PROGRESS_LOG) cleaned to match reality.

The ONE honest limitation to keep stating: **ego-speed carries much of the AUC and
partly encodes the ego-driver's anticipation** (instrumented car slows for expected
crossers) — legitimate but not pure-vision; candidate speed-perturbation robustness check.

## Repo layout you need
- `journal_prep/` — all experiments, one folder per issue, each with a README +
  results .md/.csv + figures (.png). Master plan: `journal_prep/PLAN.md`; index:
  `journal_prep/README.md`; plain-English summary: `journal_prep/PROJECT_SUMMARY.pdf`.
- `Journal_writing/` — the paper workspace: **`PLAN.md`** (section-by-section roadmap +
  issue→section map + drafting order), **`README.md`** (Overleaf setup, BibTeX
  workflow, how to use you), **`paper_skeleton.tex`** (MDPI section scaffold to paste
  into Overleaf), **`references.bib`** (starter bibliography; `% VERIFY` flags on
  unconfirmed fields).
- Root docs: `THESIS_PLAN.md`, `PROGRESS_LOG.md`, `CODE_STATE.md`, `CLAUDE.md`.
- Code/data: numbered scripts `01_…10_` at root; `runs/`, `journal_prep/.../runs*`;
  `pie_annotations.pkl`; `PIE/annotations*`; demo clips `PIE_clips/set03/*.mp4`;
  `yolo26m.pt`; venv at `.venv/` (torch 2.12 + MPS, sklearn 1.9, scipy, xhtml2pdf all present).

## Critical working rules (please honour)
1. **Don't fabricate** numbers, citations, or DOIs. Every paper claim must trace to a
   number in `journal_prep/`. In `references.bib`, fields you can't confirm are marked
   `% VERIFY` — keep them flagged, don't guess.
2. **Training runs on the M4 GPU (MPS)**, locally, via `.venv`. (Shell state doesn't
   persist between tool calls — use `.venv/bin/python` with absolute paths.)
3. **Before any heavy local run, tell me the time estimate; if it could exceed
   30 minutes, ask permission first.** (Most paper work is writing — no compute.)
4. **After finishing a task, update** `journal_prep/PLAN.md` + `README.md`, the
   positioning matrix, and the memory files, and keep the docs consistent.
5. When I ask for an explanation, give it in **simple plain English**.
6. **MDPI specifics:** official MDPI template (`mdpi.cls` + `mdpi.bst`), journal code
   `mti`, pdfLaTeX, **numbered BibTeX references in order of appearance**. Structure:
   Intro → Related Work → Materials & Methods → Results → Discussion → Conclusions +
   back matter. Write sections in the order given in `Journal_writing/PLAN.md`
   (Methods → Results → Related Work → Intro → Discussion → Conclusions → Abstract last).

## What to do next (immediate)
First READ, in this order: `Journal_writing/PLAN.md`, `Journal_writing/README.md`,
`Journal_writing/paper_skeleton.tex`, and the memory file
`~/.claude/projects/-Users-arif-Developer-pedestrian-thesis/memory/project-state.md`.
Then we draft the paper section by section in MDPI LaTeX, pulling numbers/figures from
the issue folders. Unless I say otherwise, **start by drafting §3 Materials and
Methods** (subsections: PIE dataset; the temporal-leakage problem + the leakage-free
crossing_point protocol; features + preprocessing; BiLSTM architecture; training +
the documented hyperparameter search; the YOLO+ByteTrack live pipeline), ~600–800
words, MDPI LaTeX with `\cite{}` keys matching `references.bib`. Confirm you've read
the files and give me the §3 draft.

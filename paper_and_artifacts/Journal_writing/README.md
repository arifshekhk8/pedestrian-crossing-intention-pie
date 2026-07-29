# Journal_writing/ — how to write the MTI paper

> **✅ UPDATE 2026-07-26 — the manuscript is finished apart from front matter.**
> All seven sections (Abstract, Introduction, Related Work, Materials and Methods,
> Results, Discussion, Conclusions), nine figures, and four tables are written into
> **`MDPI_Article_Template/main.tex`** using the official MDPI class
> (`Definitions/mdpi.cls`, journal `mti`, numeric BibTeX). The compiled `main.pdf`
> runs to 24 pages with 59 references and no unresolved citations.
>
> **Build it:** from inside `MDPI_Article_Template/`, run **`tectonic main.tex`**.
> (Two one-time local fixes let tectonic render the MDPI logos: ghostscript was
> reinstalled and `logo-mdpi.eps`/`logo-updates.eps` were pre-converted to PDF, with
> the class patched to point at the PDFs; a backup sits at `Definitions/mdpi.cls.orig`.
> On Overleaf or plain pdfLaTeX none of this is needed — use the pristine class.)
>
> **Regenerate the figures:** every figure is produced by a script in
> `MDPI_Article_Template/figures/`. Run `python prep_probs.py` once, then any
> `make_figN_*.py`. Four of the eight read their numbers straight out of the result
> files, so they cannot drift from what the paper claims. `figstyle.py` carries the
> shared palette and chrome.
>
> **What is left:** only the front matter — authors, affiliations, ORCID,
> corresponding e-mail, Author Contributions initials, and the repository URL in the
> Data Availability statement, all currently marked `PLACEHOLDER`. See `PLAN.md` for
> the full status block, `relatedwork.md` §7 for the framing and the honest
> defensibility analysis, `ProjectDescription.md` for the whole-project reference,
> and `STATISTICS_SOURCES.md` for every road-safety figure quoted in the Introduction.

This folder is the workspace for drafting the journal paper. The **actual LaTeX
compiling happens on Overleaf** (and now also locally via tectonic); this folder holds the
plan, the reframed related-work analysis, the whole-project reference, and---in
`MDPI_Article_Template/`---the paper itself.

```
Journal_writing/
├── PLAN.md            section-by-section roadmap (read this first)
├── README.md          this file: Overleaf setup + BibTeX + how to use Claude
├── paper_skeleton.tex MDPI-structured section scaffold (paste into the Overleaf template)
└── references.bib     starter bibliography (BibTeX; extend as you cite)
```

---

## Part A — Setting up Overleaf (one time, ~10 min)

**1. Start from the OFFICIAL MDPI template — do not hand-build the preamble.**
The MDPI class (`mdpi.cls`) and bibliography style (`mdpi.bst`) are fiddly; always
start from MDPI's template so the formatting is correct out of the box.

- Go to <https://www.overleaf.com> → **New Project → Templates** → search **"MDPI"**
  → open **"MDPI Article Template"** (the official one), OR
- Download the LaTeX template zip from <https://www.mdpi.com/authors/latex> and
  **Upload Project** to Overleaf.

**2. Set the journal to MTI.** In `main.tex`, the first line is the document class.
Change the journal code to `mti`:

```latex
\documentclass[mti,article,submit,pdftex,moreauthors]{Definitions/mdpi}
```

(`submit` = submission layout; switch to `accept` only after acceptance. Keep the
`Definitions/` folder — it contains `mdpi.cls`.)

**3. Set the compiler to pdfLaTeX.** Overleaf → **Menu → Compiler → pdfLaTeX**.
MDPI's class needs pdfLaTeX + BibTeX (not biber). Overleaf runs BibTeX automatically;
if references don't appear, hit **Recompile** twice (LaTeX → BibTeX → LaTeX → LaTeX).

**4. Add the bibliography.** Upload `references.bib` (this folder) into the project.
The MDPI template points at it near the end of `main.tex`:

```latex
\externalbibliography{yes}
\bibliography{references}     % <- your .bib filename, no extension
```

`mdpi.bst` (shipped with the template) makes **numbered references in order of
appearance** — MDPI's required style. You cite with `\cite{key}` and it renders `[1]`.

**5. Paste our content.** Open `paper_skeleton.tex` (this folder), copy its section
bodies into the template's body (between `\begin{document}`/abstract and the back
matter). Keep the template's front-matter and back-matter macros.

**6. Upload figures.** Our figures already exist as PNGs in `../journal_prep/`:
e.g. `issue9_latency/09_latency_figure.png`. Upload the ones you use into an Overleaf
`figures/` folder and reference them with `\includegraphics`.

---

## Part B — BibTeX workflow (the citation format your supervisor wants)

MDPI = **numbered references, BibTeX, `mdpi.bst`**. The mechanics:

1. Every source gets an entry in `references.bib`, e.g.:
   ```bibtex
   @inproceedings{rasouli2019pie,
     title     = {PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction},
     author    = {Rasouli, Amir and Kotseruba, Iuliia and Kunic, Toni and Tsotsos, John K.},
     booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
     pages     = {6262--6271},
     year      = {2019}
   }
   ```
2. In the text: `...the PIE dataset~\cite{rasouli2019pie}...` → renders `[1]`.
3. MDPI wants complete entries (authors, title, venue, year, **volume/pages/DOI** for
   journals). Add the **DOI** field where possible — MDPI prints it.
4. `references.bib` here is seeded with the key citations (PIE, PCPA, the baselines,
   YOLO, ByteTrack, LSTM). **Two still need their exact split/DOI verified** against
   the source PDFs (flagged in the file) — do that before submission.

> Tip: don't edit the numbered list by hand — just `\cite{}` and let `mdpi.bst`
> number everything. Reorder happens automatically.

---

## Part C — How to use Claude (me) to write this efficiently

I have **all the project results in context** (every issue, every number, the figures,
the honest framing). Use me as a drafting engine, not a black box. The high-leverage
moves:

**1. Draft one section at a time, give me the target.**
> "Write the Materials and Methods 'Dataset and Leakage-Free Protocol' subsection in
> MDPI LaTeX. Use Issue 1 + Issue 2. ~350 words, cite the PIE dataset and the
> crossing_point anchor."

I'll produce LaTeX prose with `\cite{}` keys, `\ref{}`s, and the right numbers.

**2. Turn results into tables/figures.** Point me at a CSV/MD in `journal_prep/`:
> "Make the baseline-comparison LaTeX table from `issue3_baseline_comparison/03_baseline_comparison.md`."
> "Generate the LaTeX `table` for LOSO from `issue5_loso_cv/05_loso_results.csv`."
I'll emit a `\begin{table}` with `booktabs`, caption, and label.

**3. Write figure captions** that state the finding (MDPI likes self-contained
captions): give me the figure, I write the caption + the in-text paragraph that
references it.

**4. Keep the prose human.** After a draft, run:
> "Apply the humanizer pass to this paragraph."
I'll strip AI-tells (em-dash overuse, "moreover/furthermore" stacks, rule-of-three,
inflated phrasing) so a reviewer doesn't smell a generator. (There's a dedicated
`/humanizer` skill for this.)

**5. Build `references.bib` entries.** Give me a paper name/arXiv id:
> "Add a BibTeX entry for GTransPDM (arXiv:2409.20223)."
I'll format it MDPI-style. (I can't browse paywalled PDFs — I'll flag any field that
needs your manual verification rather than guess a DOI.)

**6. Check consistency.** Paste a drafted section back:
> "Cross-check every number in this Results section against `journal_prep/`."
I'll flag any value that doesn't match an issue.

**7. Iterate paragraph-by-paragraph**, not whole-paper-at-once — you stay in control,
the prose stays coherent, and your supervisor's feedback is easy to fold in.

**What I should NOT do for you:** invent citations/DOIs, fabricate numbers, or claim
results we didn't run. If something needs an external source or a new experiment, I'll
say so (and estimate the time before running anything heavy — per our standing rule).

---

## Part D — Suggested first three prompts to me

1. *"Draft §3.1 Dataset + §3.2 Leakage-Free Protocol (Materials and Methods) in MDPI
   LaTeX from Issues 1–2, ~500 words, with the crossing_point anchor and the 0%-leakage
   verification."*
2. *"Make the §4 main-results paragraph + the baseline-comparison table (Issue 3) and
   the feature-ablation figure callout (ego-speed +0.18, Issue 2)."*
3. *"Write the Limitations subsection of the Discussion from the honest notes in
   `journal_prep/issue3_baseline_comparison/04_positioning_vs_prior_work.md` (ego-speed
   signal, tracker fragmentation, 2-clip indicative)."*

A plain-English summary of the whole project (for your own reference / co-authors) is
in **`../journal_prep/PROJECT_SUMMARY.pdf`**.

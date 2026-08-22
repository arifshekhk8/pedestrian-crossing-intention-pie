# Overleaf package — three-contribution version

Everything Overleaf needs to compile the manuscript, and nothing it doesn't.
Self-contained: no file here points outside this folder.

This is the **three-contribution** version. The JAAD cross-dataset replication
that used to be contribution 4 has been removed: the section, its table, the
contribution bullet, and every mention of it in the abstract, keywords,
discussion, conclusions, and back matter. JAAD survives in one place only, as
background in Related Work, where it is described as a dataset the field uses
and nothing more.

## Uploading

Zip this folder and use **New Project → Upload Project** on Overleaf. Keep the
folder structure: `mdpi.cls` is loaded as `Definitions/mdpi` and the figures are
loaded as `figures/...`, so flattening the tree breaks the build.

Then check two settings under the Overleaf **Menu**:

| Setting | Value |
|---|---|
| Compiler | **pdfLaTeX** |
| Main document | **main.tex** |

The bibliography is **already typeset into `main.tex`**, so Overleaf runs no
BibTeX and needs only two pdfLaTeX passes. Citations resolve on the first
compile.

### Why the bibliography is inlined

The first version of this package used `\bibliography{references}` and timed out
on Overleaf's free plan. That plan allows roughly 20 seconds of compile time on
shared hardware, and the `.bib` workflow makes Overleaf run four tools in
sequence: pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX. Inlining the bibliography cuts
that to two pdfLaTeX passes.

The inlined block is the exact BibTeX output, not hand-written, and the
resulting PDF is byte-for-byte identical to the `.bib` build. `references.bib`
is still in the folder: it is what you edit if you add or change a citation, and
MDPI may ask for it.

**If you change `references.bib`,** the inlined block will not update by itself.
Either re-run the `.bib` workflow once (see the comment block above the
bibliography in `main.tex`, which explains how to switch back), or regenerate
locally with `tectonic --keep-intermediates main.tex` and paste the new
`main.bbl` in.

### If it still times out

In order of what to try:

1. Hit **Recompile** again. Overleaf caches intermediate files, so the second
   compile is much cheaper than the first.
2. **Menu → Compile Mode → Fast [draft]**, which reuses those cached files.
3. Delete `main.pdf` and `README.md` from the Overleaf project. Neither is
   compiled; they are only there for your convenience.

## What is here

```
main.tex           the manuscript: 17 pages, 3 contributions, 6 figures, 4 tables
main.pdf           a local build, so you can see the result before uploading
references.bib     38 entries; every one is cited
figures/           the 6 figures used, as vector PDFs
Definitions/       the official MDPI class, .bst styles, and logos
```

Overleaf regenerates `main.pdf` on its first compile, so the copy here is only a
preview and can be deleted without consequence.

### On the page count

17 pages, against the 15–16 asked for. The reference list is what sets the
floor: 38 entries occupy roughly two and a half pages, and the body ends part
way down page 15. Measured directly by recompiling at different reference
counts, the paper reaches 16 pages only at **32 references**, which is below the
35 the supervisor asked for. Between 33 and 38 references it is 17 pages
regardless, so trimming prose alone cannot buy the page back.

`figures/fig6_forest_compact.pdf` is the forest plot at a shorter aspect ratio
than the version in the thesis repository. Same rows, same numbers, tighter
vertical spacing, produced by the same generator with `--compact`. It exists
because the full-height figure was leaving a quarter of its page empty.

## Before you submit

Five placeholders are deliberately left in `main.tex`, since only the authors
can fill them:

- author names, affiliations, ORCID, and corresponding e-mail (lines 47–57)
- `\authorcontributions` — replace `[author initials]` and `[supervisor initials]`
- `\dataavailability` — replace with the public repository URL

MDPI also requires a statement disclosing any generative-AI use, in Materials
and Methods or in the Acknowledgments. It is not in the file because it is a
statement about how you worked, not something that can be written for you.

## Rebuilding locally

```bash
tectonic main.tex          # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

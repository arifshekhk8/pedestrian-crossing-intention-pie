# R4 — MDPI MTI submission rules: pre-submission checklist

**Sources actually read.** MDPI blocks automated access to `mdpi.com/journal/mti/instructions`
and `/about` (HTTP 403, browser user-agent included), so this was assembled from three
primary sources that are reachable:

1. **MDPI Style Guide, first edition**, Martyn Rittman, MDPI —
   `https://mdpi-res.com/data/mdpi-author-layout-style-guide.pdf` (read in full).
2. **MTI journal flyer**, July 2026 — `https://mdpi-res.com/journals/229/flyer.pdf`
   (aims, scope, metrics, editorial office).
3. **The official MDPI LaTeX template in this repo**, `Definitions/mdpi.cls` + `template.tex`
   — the authority for macro names and for the exact template wording of every back-matter
   statement.

Anything I could not confirm from those three is marked **UNVERIFIED — check the MTI
instructions page manually before submission**. Do not treat this file as a substitute for
one read of that page in a browser.

---

## A. Manuscript structure

- [ ] **IMRAD order**, per the Style Guide §3.1: *"The majority of journals use a so-called
      IMRAD structure, meaning the sections are Introduction, Materials and Methods, Results,
      and Discussions. Some journals require a Conclusion section at the end... Authors may
      choose to have results and discussion as one or two sections."*
      Our locked order — Introduction, Materials and Methods, Results, Discussion,
      Conclusions — is compliant.
- [ ] Section numbering starts at 1 (the template ships with `\setcounter{section}{-1}` for
      its own how-to section; that line must be deleted).
- [ ] **Floats appear shortly after first citation**, same section where possible:
      *"Figures, tables, and schemes should appear in the text shortly after the first time
      they are cited. Where possible, they should be in the same section as the citation...
      they should not break paragraphs."* Production may move them.

## B. Front matter

- [ ] **Abstract: up to 200 words, single unstructured paragraph**, but written to a
      structured shape. Style Guide §2.4: *"The abstract contains a summary of the entire
      paper and can be up to 200 words long. It must not contain any images or tables...
      Authors should follow the style of a structured abstract, which is based on the IMRAD
      structure of a paper, but without using headings... Abstracts without headings should
      consist of a single paragraph."*
- [ ] **Abstract must be self-contained** — every abbreviation defined inside it, since it is
      read separately from the paper.
- [ ] **Keywords: three to ten.** Template: *"List three to ten pertinent keywords specific to
      the article; yet reasonably common within the subject discipline."*
- [ ] Title, `\Author`, `\AuthorNames`, `\address`, `\corres` all populated (PLACEHOLDER text
      per B9 until the author fills them in).

## C. Figures and tables

- [ ] **Minimum 600 dpi** recommended for figures; tif/jpg/png all acceptable. (Our figures
      are vector PDFs, which is better than the floor.)
- [ ] **Multi-part figures use lowercase letters**: *"For figures with more than one part, the
      panels should be labeled a, b, c, d, etc. and each part can be separately cited in the
      main text. Each part must be individually described in the caption."*
      The template renders these as `(\textbf{a})`, `(\textbf{b})` inside the caption.
- [ ] **Figure captions go BELOW the figure. Table captions go ABOVE the table.** Both are
      mandatory.
- [ ] **Captions must stand alone.** Style Guide §7.6 gives the explicit contrast: *"The four
      methods used."* is not helpful, whereas *"The four minimization methods used to find the
      optimum parameters of the Navier–Stokes equation for three microfluidic devices."* is
      better — *"figures and captions sometimes appear online separate to the rest of the
      article and so must make sense when not accompanied by the main text."*
      → Our baseline-table caption must therefore carry the protocol caveat inside the caption
      itself, which is what B7 already requires.
- [ ] **Tables must be editable text, never images**; colour discouraged; merged cells sparing.
- [ ] Tables use the template's `tabularx` + `booktabs` idiom; wide floats use
      `\begin{adjustwidth}{-\extralength}{0cm}` with `\fulllength`.

## D. Back matter — required statements and their exact template wording

Every one of these macros must be present. Wording below is quoted from the official template.

- [ ] `\authorcontributions{...}` — **CRediT taxonomy, standard wording**. Style Guide §8.4:
      *"MDPI uses the CREDiT taxonomy for authorship and a standard wording as given in the
      journal article template."* Template pattern:
      > "Conceptualization, X.X. and Y.Y.; methodology, X.X.; software, X.X.; validation, X.X.,
      > Y.Y. and Z.Z.; formal analysis, X.X.; investigation, X.X.; resources, X.X.; data
      > curation, X.X.; writing—original draft preparation, X.X.; writing—review and editing,
      > X.X.; visualization, X.X.; supervision, X.X.; project administration, X.X.; funding
      > acquisition, Y.Y. All authors have read and agreed to the published version of the
      > manuscript."
      The final sentence is not optional.
- [ ] `\funding{...}` — either *"This research received no external funding"* or *"This
      research was funded by NAME OF FUNDER grant number XXX"*. Funder names must match the
      Crossref funder registry spelling.
- [ ] `\institutionalreview{...}` — for us the relevant options are the waiver or *"Not
      applicable."* **Note:** our Figure 9 shows blurred pedestrian faces from PIE, and the
      qualitative-figure plan already commits to declaring the blurring here. Decide between
      a waiver justification (secondary analysis of a public dataset) and "Not applicable",
      and state the blurring either way.
- [ ] `\informedconsent{...}` — *"Not applicable"* is the expected form for studies not
      involving human subjects; PIE is a pre-existing public dataset.
- [ ] `\dataavailability{...}` — **mandatory for research articles.** Must say where the data
      supporting the results can be found, with links to publicly archived datasets. Ours:
      PIE is public; our code repository URL stays PLACEHOLDER per B9.
- [ ] `\acknowledgments{...}` — typically **up to 100 words**. Carries the **generative-AI
      disclosure** if applicable:
      > "During the preparation of this manuscript/study, the author(s) used [tool name,
      > version information] for the purposes of [description of use]. The authors have
      > reviewed and edited the output and take full responsibility for the content of this
      > publication."
- [ ] `\conflictsofinterest{...}` — *"The authors declare no conflicts of interest."*
- [ ] `\abbreviations{Abbreviations}{...}` — two-column tabular of abbreviations used.
- [ ] `\supplementary{...}` — required because we ship **Video S1**. Template form:
      *"The following supporting information can be downloaded at \linksupplementary{s1},
      Figure S1: title; Table S1: title; Video S1: title."*
- [ ] `\reftitle{References}` then the bibliography.

### GenAI disclosure — note the second location

The template puts a GenAI instruction in **Materials and Methods**, not only in
Acknowledgments: *"In this section, where applicable, authors are required to disclose details
of how generative artificial intelligence (GenAI) has been used in this paper (e.g., to
generate text, data, or graphics, or to assist in study design, data collection, analysis, or
interpretation). The use of GenAI for superficial text editing (e.g., grammar, spelling,
punctuation, and formatting) does not need to be declared."*
→ **Author decision required.** This manuscript is being drafted with AI assistance that goes
beyond superficial text editing, so a disclosure is owed. Flagging it rather than writing it,
since the wording is the author's call.

## E. References

- [ ] **MTI uses the numeric ACS-based style**, not APA and not Chicago. Derived from the
      template's own journal lists: `mti` appears in neither the APA list nor the Chicago
      list, so the default numeric branch applies. Cite with `\citep{key}`; `mdpi.bst`
      renders `[n]` in order of appearance.
- [ ] ACS pattern for a journal article, per Style Guide §8.8.1:
      > Fisher, J.A.; Krapf, C.B.E.; Lang, S.C.; Nichols, G.J.; Payenberg, T.H.D.
      > Sedimentology and architecture of the Douglas Creek terminal splay, Lake Eyre,
      > central Australia. *Sedimentology* **2008**, *55*, 1915–1930.
- [ ] Conference paper pattern: *"In Proceedings of the [conference], [city, country], [dates];
      pp. X–Y."*
- [ ] Preprint pattern: *"arXiv 2004, arXiv:physics/0402096. Available online: URL (accessed on
      DD Month YYYY)."* → our arXiv-only entries (IntFormer, MFT) need an access date.
- [ ] **Cite only static content.** Style Guide §8.8: *"The citation list should contain only
      references to static content... Content that does not fulfil these criteria may be listed
      directly in the main text and might include company websites, or websites to track
      project development (such as github)."*
      → **Consequence for us:** the Ultralytics YOLO and ByteTrack repository links, and our
      own code repository, belong in the text or Data Availability, not the reference list.
      ByteTrack has an ECCV paper, so cite that instead.
- [ ] **DOIs: UNVERIFIED.** The Style Guide's examples do not show DOIs, but MDPI journals
      generally request them. Check the MTI instructions page; including verified DOIs is
      safe either way.

## F. Supplementary material

- [ ] Video S1 (`supplementary/video_s1.mp4`, 26 s, 3.3 MB) is submitted **with** the
      manuscript rather than third-party hosted — permitted: *"Either it can be submitted to
      MDPI along with the manuscript, or it can be hosted on a third-party platform."*
- [ ] If any file were instead hosted externally, it would need a DataCite DOI and a
      preservation policy; a personal website is explicitly unsuitable.

## G. Length

- [ ] **No length limit imposed by the journal.** MTI flyer: *"No Space Constraints, No Extra
      Space or Color Charges — No restriction on the maximum length of the papers, number of
      figures or colors."*
      Our 15–17 page ceiling is therefore the author's own editorial constraint (brief B2),
      not a journal rule. Worth knowing if a section genuinely needs another half page.

## H. Aims and scope, for the cover letter

Verbatim from the MTI flyer (July 2026):

> "Multimodal Technologies and Interaction (ISSN 2414-4088) is an international,
> multi/interdisciplinary, open access, peer-reviewed journal which publishes original
> articles, critical reviews, research notes, and short communications on this subject. The
> journal is focused on presenting research that combines different types of input and output
> in ways that can enrich user experience. MTI covers research in a wide range of areas,
> including but not limited to data analysis, artificial intelligence, graphics, psychology,
> social sciences, communication, design, engineering, and the arts."

Scope list, verbatim: displays/sensors (visual, tactile/haptic, sonic, taste, smell);
multimodal interaction, interfaces, and communication; **human–computer, human–human, and
human–robot interaction**; human factors, cognition; multimodal perception; smart wearable
technology; psychology and neuroscience; digital and sensory marketing; enabling, disruptive
technologies; multimodal science, technology, and interfaces; theoretical, social, and
cultural issues; virtual reality, augmented reality, extended reality; ubiquitous computing;
design and evaluation; content creation, environments processes and methods; **application
domains**; usable and secure computing.

Journal facts for the letter: ISSN 2414-4088; Editor-in-Chief Prof. Mark Billinghurst;
Co-Editor-in-Chief Prof. Cristina Portales; 2025 Impact Factor 3.3 (JCR, Clarivate 2026);
CiteScore 6.6; JCR Q2 (Computer Science, Cybernetics), CiteScore Q1 (Neuroscience,
miscellaneous); indexed in Scopus, ESCI, Inspec, dblp; editorial office `mti@mdpi.com`,
Grosspeteranlage 5, 4052 Basel; median 22.8 days to first decision.

**The fit argument to make.** The paper is a multimodal-input study: it fuses a *visual*
stream (pedestrian bounding-box geometry from a camera) with a *vehicle-telemetry* stream
(OBD ego-speed), and shows experimentally that the combination, not the model, carries the
task — removing the telemetry stream collapses F1 from 0.828 to 0.551. That is a result about
modality contribution, which sits squarely in "multimodal interaction" and "application
domains". The second hook is human–vehicle interaction: the system infers a pedestrian's
intention to enter the roadway so an automated vehicle can respond, i.e. machine perception of
a human communicative act in traffic. The honest caveat we already carry — that ego-speed
partly encodes the *driver's* anticipation — is itself a human-interaction finding, since it
says the telemetry stream is contaminated by a second human's behaviour.

---

## I. Items still to confirm manually (MDPI blocks automated access)

1. MTI-specific keyword count, if it differs from the template's three-to-ten.
2. Whether DOIs are mandatory in the reference list.
3. Any MTI-specific section ordering or article-type requirement.
4. Current APC and the submission-system checklist.
5. Preprint policy specifics — the Style Guide covers patents and supplementary hosting but
   the preprint/prior-publication rules live on the instructions page. **UNVERIFIED.**
6. Whether MTI requires a graphical abstract (the Style Guide mentions one may "also be
   submitted", implying optional).

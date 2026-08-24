# A4 — Introduction statistics refreshed to the 2024/2025 releases

Requested 24 August 2026: the Introduction was carrying a 2014–2023 evidence window while
2024, 2025 and 2026 releases exist. This file records what was replaced, what was verified at
source, and the one number that deliberately stays where it was.

Every figure below was read from the primary document, not from a search summary. The two PDFs
are in the session scratchpad; the two web sources were fetched and their text extracted.

---

## 1. What was wrong with the old text

| Old sentence | Status |
|---|---|
| "Road traffic killed an estimated 1.19 million people in 2021" | Superseded. WHO's July 2026 release puts the figure at 1.16 million for 2025. |
| "The trend is not improving where it has been measured most closely" | **False under 2024 data.** US pedestrian deaths have fallen three years running. |
| "7,314 pedestrians were killed in 2023 against 4,910 in 2014" | Both superseded. 2023 was revised to 7,367 in the FARS final file; 2014 falls outside the current ten-year table. |
| "84% urban, 74% away from intersections, 77% in the dark" (2023) | Superseded by the 2024 values 84 / 73 / 76. |

The second row is the one that mattered. The claim was defensible on 2023 data and is not
defensible now, and a reviewer with the June 2026 fact sheet open would have caught it.

## 2. Sources verified at origin

**NHTSA, *Traffic Safety Facts 2024 Data: Pedestrians*, DOT HS 813 818, June 2026.**
Retrieved from `crashstats.nhtsa.dot.gov` (publication 813818) and read with `pdftotext`. The
server returns the PDF wrapped in a multipart body; the payload was extracted between the
`%PDF-` and `%%EOF` markers before conversion.

Table 1, pedestrian fatalities and share of all traffic fatalities:

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|
| 5,494 | 6,080 | 6,075 | 6,374 | 6,272 | 6,565 | 7,470 | 7,593 | 7,367 | 7,080 |
| 15% | 16% | 16% | 17% | 17% | 17% | 17% | 18% | 18% | 18% |

Key findings, verbatim: 7,080 killed in 2024, "a 3.9-percent decrease from the 7,367 pedestrian
fatalities in 2023"; 18% of all traffic fatalities; urban 84%; "Seventy-three percent of the
pedestrian fatalities occurred at locations that were not intersections"; "More pedestrian
fatalities occurred in the dark (76%)".

Derived values used in the text, and their arithmetic:
- 7,080 / 5,494 = 1.289, printed as **29% above 2015**.
- 7,080 / 7,593 = 0.932, printed as **down 7% from the 2022 peak** (figure subtitle only).

Note for the record: this release revises 2023 from the 7,314 printed in DOT HS 813 727 to
7,367. The manuscript no longer prints either 2023 value, so there is nothing to reconcile.

**WHO, *Road Traffic Injuries*, fact sheet, stamped 20 July 2026.** Added 24 August 2026 at the
author's request, and cited first. Verbatim key facts: "Approximately 1.16 million people die
each year as a result of road traffic crashes"; "Road traffic injuries are the leading cause of
death for children and young adults aged 5-29 years"; "More than half of all road traffic deaths
are among vulnerable road users, including pedestrians, cyclists and motorcyclists".

What it does **not** contain, checked by string search of the retrieved page: no "21%", no
"2011", and no pedestrian share. It also says "each year" rather than naming 2025. So it cannot
carry the trend sentence or the 2025 anchor on its own, and the news release below stays cited
alongside it. Note that WHO's statement that more than half of road deaths are vulnerable road
users appears in the fact sheet in WHO's own voice, where the news release has it only as a
quotation from an individual; the fact sheet is the better authority for that clause.

The URL as supplied carried a `?utm_source=chatgpt.com` tracking parameter, stripped before use.
The page is updated in place rather than versioned, so the access date is doing real work here
under MDPI's static-content rule.

**WHO, *Road Deaths Fall by 21% Globally but Stronger Action Is Needed to Save Lives*, news
release, 20 July 2026.** Fetched from `who.int` and read as text. Verbatim: "road traffic
crashes still claimed 1.16 million people's lives in 2025"; "the global rate of road traffic
deaths declined by 21% between 2011 and 2025"; "Road traffic injuries remain the leading cause
of death among children and youth aged 5–29 years"; and, quoting Dr Etienne Krug, "More than
half of all road deaths are people who aren't even in a car".

**GHSA, *Pedestrian Traffic Fatalities by State: 2025 Preliminary Data*, 14 July 2026.** PDF
downloaded and read. Verbatim: "GHSA projects 6,732 pedestrians were killed in 2025 in all 50
states and D.C. This represents a projected 7% decrease from the 7,237 pedestrian fatalities
reported in 2024, a collective 505 fewer lives lost and the third consecutive year of declines.
While this is encouraging, the predicted total is still 5% above the 6,412 pre-pandemic deaths
reported in 2019."

**Cross-source hazard, handled.** GHSA works from state-reported preliminary data and NHTSA from
FARS, so their counts for the same year differ (2024: GHSA 7,237, FARS 7,080; 2019: GHSA 6,412,
FARS 6,272). No sentence in the manuscript computes a change across the two. The GHSA sentence
reports only GHSA's own direction and magnitude, and does not print a count.

## 3. The number that deliberately did not change

Figure 1(a) still shows the road user breakdown for **2021** from the WHO 2023 status report:
pedestrians 23%, four-wheel occupants 30%, powered two- and three-wheelers 21%, other 20%,
cyclists 6%.

WHO's July 2026 release publishes a global total and regional rate changes. It does **not**
publish a split by road user type; the closest it comes is "more than half of all road deaths
are people who aren't even in a car" and "deaths among motorcyclists now account for nearly a
third". There is no 2026 successor to the status report: the WHO road-safety reporting page
still lists the 2023 edition as the current one. So 2021 remains the most recent year for which
WHO has published this breakdown, and the caption now says so rather than leaving a reader to
wonder why a 2026 paper plots 2021 data.

## 4. Changes made

**`main.tex`, Introduction.**
- New opening paragraph: a single concrete scene, with the braking arithmetic that makes early
  commitment worth something (50 km/h is 13.9 m/s, so half a second is 6.9 m, printed as
  "about 7 m").
- Statistics paragraph rewritten around the 2025 global figure, the 2015–2024 US series and the
  2025 projection. The "trend is not improving" claim is replaced by the accurate one:
  improvement measured against a raised baseline.
- Circumstances paragraph updated to the 2024 values and shortened, because the vignette now
  carries the mid-block-at-night detail it used to state twice.
- Figure 1 caption rewritten to stand alone, to name the 2022 peak, and to explain the 2021
  vintage of panel (a).

**`references.bib`.** Four entries added: `who2026factsheet`, `who2026roaddeaths`,
`nhtsa2026pedestrians`, `ghsa2026pedestrian`. The two WHO web entries carry no `year` field,
because `mdpi.bst` otherwise prints a redundant ", 2026." after the access date; without it the
output matches MDPI's web-reference format exactly, and BibTeX raises no warning. `nhtsa2025pedestrians` is now uncited and left in place; a numeric style
prints only cited entries, so it does not appear in the reference list.

**`figures/fig1_pedestrian_statistics.pdf`.** Regenerated from
`evidence/make_intro_figure.py`, a copy of the original generator with the series moved to
2015–2024, a direct label added at the 2022 peak, and the subtitle restated. Panel geometry is
unchanged, so the float occupies the same space as before. Nothing outside `submission/` was
touched: the original generator under `MDPI_Article_Template/figures/` is untouched and still
produces the old figure.

## 5. Effect on length

21 pages to **22**. The whole increase is this change: about 140 net words of prose and three
added reference entries. Roughly 150 words would have to come back out to return to 21, which
is A2's decision and not one to make inside this pass.

## 6. Not done, and worth deciding

The manuscript's newest research citations are from 2025. R3 surfaced two 2026 items, a
psychological-features paper (arXiv 2603.19533) and a vision-language approach (arXiv
2606.09142), neither of which has been read at source in this session. Adding one verified 2026
citation to Related Work would answer the currency question for the literature as this pass has
answered it for the statistics. It needs a source read first, per B4.

---

## 7. Introduction restructured to the author's outline, 24 August 2026

Requested: one paragraph for the story, one for the figure, and the whole section ordered as
real-world problem, why anticipation matters, the research task, what is already known, the
methodological gaps, why the gaps matter, what this study does, contributions.

| # | Beat | Words |
|---|---|---|
| P1 | Real-world problem and why anticipation matters. Opens from the automated vehicle's standpoint: detection is half the problem, anticipation is the other half, and it counts only if it is early. The scene and the braking arithmetic follow. | 139 |
| P2 | The figure. Global toll, pedestrian share, the US series, the 2025 projection, and the crash circumstances that make the scene typical. Was two paragraphs of 215 words. | 138 |
| P3 | The research task. Binary question at a one to two second horizon, distinguished from trajectory forecasting; PIE as the testbed. | 131 |
| P4 | What is already known. The Kotseruba benchmark protocol, the drift to three to seven streams, and the minimal-input precedent. **The concession moved here from after the gaps**, which is where "what is already known" belongs and keeps it early per the framing rule. | 132 |
| P5 | The gaps. Leakage, thin evaluation, untested encoder. | 130 |
| P6 | Why the gaps matter. What each one does to the meaning of a published number. **New.** | 83 |
| P7 | What this study does, then the contributions list. **New.** | 67 |
| P8 | Metric hierarchy. Unchanged. | 56 |

Two substantive moves, neither of which drops anything:

1. **Input cost moved from the gap list to what is already known.** It is a property of the
   field, not a flaw in its method, and putting it in P4 lets the third gap be the one the paper
   actually closes with contribution 3, the untested encoder. The Introduction's three gaps now
   line up one to one with the three contributions. Related Work still lists four gaps, counting
   modality cost separately; that is a longer treatment, not a contradiction.
2. **`kotseruba2021benchmark` now cited in the Introduction.** It was cited only in Related Work,
   which left P4's claim about a shared protocol unsourced. No new reference-list entry.

Introduction length: 743 words as first written, 905 after the statistics refresh, **876 now**,
carrying two more structural beats than either earlier version. Still 22 pages, 0 errors, 0
undefined references, 0 BibTeX warnings.

---

## 8. New opening figure, 25 August 2026

Requested: a figure for the opening scenario, built from real dataset imagery.

**Source.** `PIE_clips/set03/video_0016.mp4`, pedestrian `3_16_919`, `crossing_point`
frame 2567 from the clean-protocol metadata. Panels at t = −1.5 s, −0.5 s and +0.5 s
(frames 2522, 2552, 2582) at 30 fps. Boxes are PIE's own per-frame annotations, not
detections; ego speeds are PIE's recorded `vehicle_speed` channel: 28, 24 and 18 km/h.

**Why this pedestrian, recorded so the choice is auditable.** Only two set03 clips are held
locally, and within them only six annotated crossers have a non-zero ego speed at the crossing
point. The rest cross in front of a vehicle already stopped at a signal, which is not the
scenario the Introduction describes. Of the six, this one has the clearest view and the largest
subject. Nothing beyond that was chosen by eye.

**Distances are measured, not assumed.** The metre values come from integrating the recorded
per-frame speed: 0.0 m at (a), 7.2 m at (b), 10.4 m at the step, 13.1 m at (c). The half second
between (b) and the step accounts for 3.2 m. The caption states the 50 km/h equivalent (about
7 m) so the figure and the text's generic arithmetic cannot be read as contradicting each other.

**Privacy.** Head regions of every person the detector finds are blurred in the pixels before
cropping, using the same routine as the qualitative figure: 30, 28 and 24 regions across the
three frames. The blur is applied to the pixels, so it cannot be undone from the published file.

**Design.** Generated by `evidence/make_fig_scenario.py`, which imports the manuscript's shared
`figstyle` module. Accent, ink tokens, panel-title hierarchy, label plates and provenance line
all follow the qualitative figure, so the two read as the work of one hand.

**Numbering.** The figure is referenced in the first paragraph, so it becomes Figure 1 and the
statistics figure becomes Figure 2. Every cross-reference in the manuscript is by `\ref`, so the
renumbering is automatic; a grep confirmed no hard-coded figure number anywhere in `main.tex`.

**Cost: 22 pages to 23.** Against a 17-page target this is now six over. See §9.

## 9. Standing page-budget position

| Stage | Pages |
|---|---|
| After S9, before any of this | 21 |
| Statistics refresh and hook | 22 |
| Opening scenario figure | 23 |

Constraint B2 sets 15 to 17 pages with a hard ceiling of 17, and forbids cutting a result, a
limitation or a citation to get there. R4 established that MTI itself imposes no length limit.
The gap is now roughly 2,000 words of prose, which is more than A2's priority order can remove
without touching material B2 protects. This needs an author decision: relax the ceiling, or
authorise cuts deep enough to reach it.

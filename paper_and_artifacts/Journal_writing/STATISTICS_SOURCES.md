# STATISTICS_SOURCES.md — verified road-safety statistics used in the Introduction

> **Purpose.** Every pedestrian-casualty number quoted in the manuscript's Introduction and
> in Figure 1 traces to a primary source recorded here, with the exact wording and where it
> came from. **Rule: no number enters the paper unless it appears below.**
>
> **Verification method (2026-07-22).** Each figure was read **first-hand from the primary
> PDF or the issuing organisation's own page**, not from a secondary summary. This mattered:
> web summaries reported the 2023 US pedestrian-fatality count variously as "7,367" and
> "7,514"; the actual NHTSA fact sheet says **7,314**. Secondary summaries of official
> statistics are not reliable — always re-read the source PDF.

---

## 1. WHO — Global Status Report on Road Safety 2023

- **Citation:** World Health Organization. *Global Status Report on Road Safety 2023*.
  WHO: Geneva, Switzerland, 2023. ISBN 978-92-4-008651-7.
- **Landing page:** <https://www.who.int/publications/i/item/9789240086517>
- **Full PDF:** <https://iris.who.int/bitstream/handle/10665/375016/9789240086517-eng.pdf>
  (mirror used for extraction: `https://assets.bbhub.io/dotorg/sites/64/2023/12/WHO-Global-status-report-on-road-safety-2023.pdf`)
- **Published:** 13 December 2023. **Data year: 2021** (important — the report is 2023, the
  estimates are for 2021).
- **Verification:** text extracted from the PDF with `pdftotext`; quotes below are verbatim.

| Statistic | Value | Verbatim source text |
|---|---|---|
| Global road traffic deaths | **1.19 million** (2021) | "There were an estimated 1.19 million road traffic deaths in 2021" |
| Death rate | 15 per 100,000 population | "this corresponds to a rate of 15 road traffic deaths per 100 000 population" |
| Leading cause of death, ages 5–29 | yes (as of 2019) | "road traffic injury remains the leading cause of death for children and young people aged 5–29 years and is the 12th leading cause of death when all ages are considered" |
| Low-/middle-income share | 92% of deaths | "92% of deaths occur in low- and middle-income countries" |

**Global distribution of road traffic deaths by road user type (2021)** — Section 1,
"Fatalities by road user type" (used for Figure 1a):

| Road user type | Share | Verbatim source text |
|---|---|---|
| 4-wheel vehicle occupants | **30%** | "occupants of 4-wheel vehicles represent 30% of fatalities" |
| **Pedestrians** | **23%** | "followed by pedestrians who represent 23% of fatalities" |
| Powered 2- and 3-wheeler users | **21%** | "powered two- and three-wheeler users make up 21% of fatalities" |
| Cyclists | **6%** | "Cyclists account for 6% of fatalities" |
| Other / unknown | **20%** | "Occupants of vehicles carrying more than 10 people, heavy goods vehicles, 'other' users and 'unknown' user types comprise the remaining 20% of deaths" |

(Sums to 100%. Micro-mobility/e-scooter users are 3% of deaths and are *included within*
the "other" category — do not add them as a separate slice.)

---

## 2. NHTSA — Traffic Safety Facts, "Pedestrians: 2023 Data"

- **Citation:** National Highway Traffic Safety Administration. *Traffic Safety Facts 2023
  Data: Pedestrians*; Report No. DOT HS 813 727; NHTSA, U.S. Department of Transportation:
  Washington, DC, USA, June 2025.
- **PDF:** <https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/813727.pdf>
- **Verification:** the PDF was downloaded and **read page-by-page directly** (pages 1–2).
  Data sources within it: FARS 2014–2022 Final File, 2023 Annual Report File (ARF);
  NASS GES 2014–2015 and CRSS 2016–2023.

### Headline figures (2023, United States)

| Statistic | Value |
|---|---|
| Pedestrians killed | **7,314** (a 3.7% decrease from 7,593 in 2022) |
| Pedestrians injured (estimated) | **68,244** (1.3% increase from 67,341 in 2022) |
| Share of all traffic fatalities | **18%** |
| Frequency | a pedestrian killed **every 72 minutes**, injured **every 8 minutes** |
| Urban vs rural | **84% urban**, 16% rural |
| Location | **74% at non-intersection locations**, 17% at intersections, 9% other |
| Lighting | **77% in the dark**, 19% daylight, 2% dusk, 2% dawn |
| Single-vehicle crashes | 89% |
| Hit-and-run | 24% |
| Male victims | 70% |
| Alcohol involved (driver and/or pedestrian) | 46% of fatal pedestrian crashes |

### Ten-year series — Table 1 of the fact sheet (used for Figure 1b)

| Year | Total traffic fatalities | Pedestrian fatalities | Pedestrians as % of total |
|---|---|---|---|
| 2014 | 32,744 | 4,910 | 15% |
| 2015 | 35,484 | 5,494 | 15% |
| 2016 | 37,806 | 6,080 | 16% |
| 2017 | 37,473 | 6,075 | 16% |
| 2018 | 36,835 | 6,374 | 17% |
| 2019 | 36,355 | 6,272 | 17% |
| 2020 | 39,007 | 6,565 | 17% |
| 2021 | 43,230 | 7,470 | 17% |
| 2022 | 42,721 | 7,593 | 18% |
| 2023 | 40,901 | 7,314 | 18% |

**Derived figure used in the paper:** 7,314 / 4,910 = **1.49**, i.e. pedestrian fatalities
rose **≈49%** between 2014 and 2023, while their share of all traffic deaths rose from
**15% to 18%**. (This is our own arithmetic on the table above, stated as such.)

---

## 3. European Commission — 2024 preliminary road-safety figures

- **Citation:** European Commission. *EU Road Fatalities Drop by 3% in 2024, but Progress
  Remains Slow*; Directorate-General for Mobility and Transport: Brussels, Belgium,
  18 March 2025.
- **URL:** <https://transport.ec.europa.eu/news-events/news/eu-road-fatalities-drop-3-2024-progress-remains-slow-2025-03-18_en>

| Statistic | Value |
|---|---|
| EU road deaths, 2024 (preliminary) | **≈19,800** (−3% vs 2023, ≈600 fewer) |
| EU average fatality rate | 44 deaths per million inhabitants |
| By road user type (2023 data) | car occupants 44%, powered two-wheelers 20%, **pedestrians 18%**, cyclists 10% |
| By road type (2023 data) | rural 52%, **urban 38%**, motorways 9% |
| Vulnerable road users in urban areas | **≈70%** of urban road deaths |

> Note: the road-user and road-type splits in this release are labelled as 2023 data even
> though the headline count is preliminary 2024 — quote them as such.

---

## 4. How these are used in the manuscript

- **Introduction, paragraph 1** — global scale (WHO: 1.19 M deaths; pedestrians 23%;
  leading killer of ages 5–29).
- **Introduction, paragraph 1–2** — the trend is not improving for pedestrians (NHTSA:
  +49% over 2014–2023; share 15%→18%) and the risk is concentrated in exactly the setting
  our system targets (NHTSA: 84% urban, 74% away from intersections, 77% in the dark;
  EC: ≈70% of urban EU deaths are vulnerable road users).
- **Figure 1a** — WHO global distribution by road user type (2021).
- **Figure 1b** — NHTSA US pedestrian fatalities and share of total, 2014–2023.
- Figure generated by `figures/make_intro_figure.py`; regenerate with
  `python paper_and_artifacts/Journal_writing/MDPI_Article_Template/figures/make_intro_figure.py`.

## 5. Statistics still to add if the framing changes

- Country-specific data for the authors' own region, if the supervisor wants a local hook.
- WHO *Global Status Report on Road Safety 2025*, if/when published — check
  <https://www.who.int/teams/social-determinants-of-health/safety-and-mobility> before
  submission, and update the 1.19 M/23% figures if superseded.

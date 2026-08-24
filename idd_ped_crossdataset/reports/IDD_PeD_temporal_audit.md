# IDD-PeD temporal audit

Independent and frame-level. **The PIE contamination rate was NOT assumed** — every number below is computed from IDD-PeD's own per-frame `CrossingBehavior` annotations by `scripts/04_temporal_audit.py`. Per-window evidence: `results/IDD_PeD_temporal_audit.csv`.

## Method

Ground truth: a pedestrian is *crossing at frame f* iff IDD-PeD's own per-frame `CrossingBehavior` tag at f is one of **CU, CFU, CD, CFD** (crossing undesignated / fast-undesignated / designated / fast-designated). `CI` ("crossing the road but **not** in the ego-vehicle's path") is deliberately **excluded**, because the prediction target is crossing *in front of the ego-vehicle*.

A window is **contaminated** iff it contains ≥ 1 such frame — the model would see the pedestrian already crossing while being asked to predict whether they will.

Three anchors are compared **on identical tracks**:

| anchor | event frame | rationale |
|---|---|---|
| **naive** | 45 frames before the track's last annotated frame | the widely-used "track-end" convention; the anchor PIE's original (retracted) builder used |
| **cp_anchor** | `crossing_point` | the literal port of this project's PIE clean protocol |
| **strict** | `min(crossing_point, first crossing-tagged frame)` | **this study's protocol on IDD-PeD** — cannot be later than true onset, so post-onset contamination is impossible by construction |

The third anchor exists because of a dataset difference, not a preference: on PIE, `crossing_point` *equals* the first crossing frame for 99.4 % of crossers, so PIE's rule is already "anchor at onset". On IDD-PeD that equality holds for only 68.9 %, so reproducing PIE's *semantics* requires taking the earlier of the two markers. This is a documented adaptation, not a new temporal rule invented for convenience.

## 1. Counts required by the brief

| # | quantity | value |
|---|---|---|
| 1 | pedestrian tracks considered | **4,746** (of 4,916 parsed; the remainder lack a POI attribute record) |
| 2 | tracks with usable crossing annotations (label + event frame inside a contiguous run) | **4,659** (`crossing_point`) / **4,661** (strict) |
| 3 | tracks with valid ego-speed alignment | **4,916 / 4,916 (100 %)** — schema audit §4 |
| 4 | nominal observation windows — naive | **8,651** (from 2,561 pedestrians) |
| 4 | nominal observation windows — cp_anchor | **7,919** (from 2,496 pedestrians) |
| 4 | nominal observation windows — strict | **7,318** (from 2,336 pedestrians) |
| 5 | contaminated — naive | **2,433 / 8,651 = 28.1 %** of all windows; **1,992 / 2,451 = 81.3 %** of crossing windows |
| 5 | contaminated — cp_anchor | **515 / 7,919 = 6.5 %** of all windows; **159 / 537 = 29.6 %** of crossing windows |
| 5 | contaminated — strict | **0 / 7,318 = 0.0 %** of all windows; **0 / 352 = 0.0 %** of crossing windows |
| 6 | **strictly pre-crossing windows — strict protocol** | **7,318 / 7,318 = 100.0 %** |
| 7 | other temporal inconsistencies | §5 |

## 2. Window leakage, side by side

| anchor | group | N | windows with ≥1 crossing frame | % |
|---|---|---|---|---|
| naive | crossers (label=1) | 2,451 | 1,992 | **81.3 %** |
| naive | non-crossers (label=0) | 6,200 | 441 | 7.1 % |
| naive | **all** | 8,651 | 2,433 | **28.1 %** |
| cp_anchor | crossers (label=1) | 537 | 159 | **29.6 %** |
| cp_anchor | non-crossers (label=0) | 7,382 | 356 | 4.8 % |
| cp_anchor | **all** | 7,919 | 515 | **6.5 %** |
| strict | crossers (label=1) | 352 | 0 | **0.0 %** |
| strict | non-crossers (label=0) | 6,966 | 0 | 0.0 % |
| strict | **all** | 7,318 | 0 | **0.0 %** |

- **naive** — crossing windows whose *entire* 16 frames are already crossing: **1,874** (76.5 % of crossers); whose *anchor frame itself* is already crossing: **1,905** (77.7 %); with a labelled onset at or before the window end: **2,322** (94.7 %).
- **cp_anchor** — crossing windows whose *entire* 16 frames are already crossing: **146** (27.2 % of crossers); whose *anchor frame itself* is already crossing: **154** (28.7 %); with a labelled onset at or before the window end: **180** (33.5 %).
- **strict** — crossing windows whose *entire* 16 frames are already crossing: **0** (0.0 % of crossers); whose *anchor frame itself* is already crossing: **0** (0.0 %); with a labelled onset at or before the window end: **0** (0.0 %).

## 3. Interpretation

**The leakage class reproduces on IDD-PeD — more severely than on PIE.** Under the track-end convention **81.3 %** of crossing windows already contain the pedestrian crossing (PIE: 67.9 %). Anchoring at `crossing_point` cuts that to **29.6 %** — but **not to zero**, because IDD-PeD's `crossing_point` is late in 19.0 % of crossers. Only the strict anchor reaches **0.0 %**.

| dataset | naive anchor | project's clean protocol | strict |
|---|---|---|---|
| PIE (Issue 1 / 2) | 67.9 % of crossers leak | **0.0 %** | n/a (identical to clean) |
| JAAD (Track A) | not run | **0.0 %** of 972 sequences | n/a |
| **IDD-PeD (this work)** | **81.3 %** | **29.6 %** | **0.0 %** |

**This is a genuinely new result, not a re-run.** On PIE and JAAD the event annotation was reliable enough that anchoring on it sufficed. On IDD-PeD it is not, and a naive port of the "clean" protocol would still have trained on ~30 % contaminated positives. The rate was computed independently for each dataset, exactly as the brief required.

## 4. Static-shortcut test (anchor-frame box geometry, strict protocol)

Can the last observed box alone separate the classes? Mann-Whitney U with rank-biserial effect size; the PIE audit used |rb| < 0.3 as "no strong shortcut".

| feature | crosser median | non-crosser median | p | rank-biserial |
|---|---|---|---|---|
| bbox_bottom_y | 918.8 | 974.2 | 6.49e-27 | -0.339 ⚠️ |
| bbox_height | 140.3 | 146.0 | 5.00e-04 | -0.110 |
| bbox_xcenter | 1062.2 | 647.5 | 4.53e-81 | +0.602 ⚠️ |
| bbox_area | 6486.6 | 7686.2 | 6.22e-05 | -0.126 |

⚠️ **A static shortcut IS present** on: bbox_bottom_y, bbox_xcenter. Unlike PIE (all |rb| < 0.3), anchor-frame geometry alone partially separates the classes on IDD-PeD. The strongest is `bbox_xcenter`, which is expected and somewhat tautological: IDD-PeD's positive label is *"crosses **in front of the ego-vehicle**"*, so positives are by definition pedestrians near the image centre. Any IDD-PeD result — ours and the dataset authors' alike — is therefore partly a *position* classifier rather than a pure *intention* classifier. **This is disclosed in the final report and must be disclosed in the paper.**

## 5. Other temporal inconsistencies found

| finding | detail |
|---|---|
| **`crossing_point` is a weaker onset marker than PIE's** | it equals the first crossing-tagged frame in **68.9 %** of crossers (PIE: 99.4 %) and is at or before it in **81.0 %**. In the remaining **19.0 %** it is *late*, by a median of 30 frames (max 291) — those tracks are the entire residual contamination of the `cp_anchor` variant. |
| **crossers are annotated from onset, not before it** | `crossing_point == first_frame` for **65.4 %** of crossing tracks; median pre-event track length is **1 frame** for crossers vs **52** for non-crossers. IDD-PeD simply does not contain a long pre-crossing observation for most crossers. |
| corrupt `crossing_point` values | a handful lie far outside the annotated range (e.g. −8,506 and 65,963); all are caught by the "must lie in a contiguous run" rule and excluded (85–87 tracks). |
| tracks with gaps | 29 tracks have > 1 contiguous segment; handled exactly as PIE (keep the segment containing the event frame). |
| duplicate frames / non-monotonic OBD ids | **none**. |
| label vs per-frame tags | 210 tracks labelled `crossing=0` still contain CU/CFU/CD/CFD frames — consistent with the label's definition (*crossing **in front of the ego-vehicle***, not crossing the road anywhere). |

## 6. Consequence for the experiment

Requiring a genuine pre-crossing observation is expensive on IDD-PeD:

| rule | crossing tracks | non-crossing tracks | % positive |
|---|---|---|---|
| authors' IDD-PeD protocol (L ≥ 16, TTE = 0) | 197 | 3,728 | 5.0 % |
| obs 16 + TTE ≥ 15 (0.5 s lead) | 175 | 3,149 | 5.3 % |
| `cp_anchor`: obs 16 + TTE ≥ 30 (≥1.0 s lead) | 149 | 2,347 | 6.0 % |
| **strict: obs 16 + TTE ≥ 30, anchored at min(cp, onset)** | **102** | **2,234** | **4.4 %** |
| obs 16 + TTE ≥ 60 (2.0 s lead) | 113 | 1,239 | 8.4 % |

The two window sets actually used:

| protocol | windows | train | val | test | test % positive | pos_weight |
|---|---|---|---|---|---|---|
| **strict** (main) | 7,318 | 3,944 (138 pos) | 1,017 (46 pos) | 2,357 (168 pos) | 7.1 % | 27.58 |
| `cp_anchor` (sensitivity) | 7,919 | 4,189 (224 pos) | 1,147 (102 pos) | 2,583 (211 pos) | 8.2 % | 17.70 |

Enforcing a strictly pre-onset ≥1 s observation costs about half the crossing tracks relative to the authors' own zero-lead protocol. The resulting positive rate is **3.5–7.1 %**, against PIE's 32.5 %. **That imbalance is intrinsic to IDD-PeD's "crosses in front of the ego-vehicle" label, not a by-product of our protocol** (the authors' own protocol yields 5.0 %), and it is the dominant caveat on every number reported from this dataset.

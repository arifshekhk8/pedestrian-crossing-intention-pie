# IDD-PeD cross-dataset validation — final scientific report

**Date:** 2026-08-25 · **Folder:** `idd_ped_crossdataset/` · **Status:** complete

> **Headline.** IDD-PeD is genuinely compatible with the PIE input contract — per-frame
> boxes plus per-frame OBD ego-speed **on the same km/h scale** — so this is the first real
> out-of-domain test of the project's actual 5-D model. Three things came out of it:
> **(1)** the temporal-leakage finding **reproduces and worsens** (81.3 % of crossing windows
> contaminated under the track-end convention, vs PIE's 67.9 %) — and, new here, the
> project's own clean protocol is **not sufficient** on IDD-PeD (29.6 % residual), because
> IDD-PeD's `crossing_point` is a much weaker onset marker than PIE's; a strict
> `min(crossing_point, onset)` anchor is needed to reach 0.0 %.
> **(2)** **Ego-speed is the domain-invariant signal *for transfer*, but the ego-speed
> dominance result does NOT replicate under independent training.** Neutralising ego-speed in
> the frozen PIE models costs **−0.165 AUC** on IDD-PeD (boxes alone: +0.022, i.e. box
> geometry actively *hurts* transfer). Yet a model trained *on* IDD-PeD gains only
> **+0.012 AUC** from ego-speed on average — against **+0.179** on PIE — and one family
> reverses. The likely reason is a static shortcut IDD-PeD has and PIE does not (§14).
> **(3)** **Neither the classifier nor the architecture-equivalence finding transfers.**
> Zero-shot F1 is 0.129–0.131, **at or below the trivial "predict-all-positive" baseline of
> 0.133** (only ranking survives, AUC 0.68–0.75); and under independent training **5 of 6**
> pairwise family comparisons differ significantly, with the PIE ranking roughly reversed.

---

## 1. Was IDD-PeD actually compatible with the PIE input representation?

**Yes — more so than any other candidate dataset examined by this project.**

| PIE contract | IDD-PeD | verdict |
|---|---|---|
| per-frame `[x1, y1, x2, y2]`, raw pixels | per-frame CVAT `[xtl, ytl, xbr, ybr]`, raw pixels | ✅ identical |
| per-frame ego-vehicle OBD speed | per-frame `OBD_speed` in `*_obd.xml` | ✅ identical schema |
| 30 fps | 30 fps (GoPro; the 25 fps DDPAI directories are **empty** in the release) | ✅ no conversion |
| binary `crossing` label | binary `crossing` (*in front of the ego-vehicle*) | ✅ same task |
| `crossing_point` event frame | `crossing_point` per-pedestrian attribute | ⚠️ present but **less reliable** (§5) |
| 1920×1080 | **1920×1440 in 29 of 33 videos** | ⚠️ the one real confound (§14) |

This is what makes IDD-PeD worth the effort: the project's earlier JAAD track
(`journal_prep/cross_dataset_validation/`) could only run **bbox-only**, because JAAD has no
ego-speed, and its four families landed at **AUC ≈ 0.50 — chance**. IDD-PeD is the first
second dataset on which the real two-stream model can be tested.

Licence is **CC BY 4.0** with direct, unauthenticated downloads — the project's earlier note
that IDD-PeD was "access-gated" is outdated.

## 2. Was ego speed successfully synchronised with video frames?

**Yes, exactly, with zero missing values.**

- Every one of the 33 videos has **exactly one OBD record per video frame** (OBD row count
  equals `meta/task/size` in all 33 cases).
- The OBD `id` **is** the video frame index — alignment is index-to-index by construction. No
  interpolation, resampling, or timestamp matching was needed or performed.
- All 34 OBD files have contiguous ids starting at 0 (0 gaps, 0 non-monotonic).
- **4,916 / 4,916 tracks (100 %) have full speed coverage.** 0 missing, 0 negative, 0 impossible.

**Scale compatibility with PIE (this is what made zero-shot transfer legitimate):**

| statistic | PIE `vehicle_speed` | IDD-PeD `OBD_speed` |
|---|---|---|
| min / median / p99 / max | 0.00 / 16.00 / 44.02 / 56.01 | 0.00 / 20.00 / 43.30 / 63.00 |
| % exactly zero | 22.7 % | 8.7 % |

Both are km/h on the same scale. Standardised with PIE's *own* training statistics, IDD-PeD's
speed channel has mean **z = −0.002** — essentially the same distribution. No unit conversion
was applied. Had the scales disagreed, Experiment A would have been invalid and this work
would have stopped.

**Disclosed limitation:** the OBD sensor logs at **10 Hz**; the released 30 fps signal is that
series linearly upsampled in thirds (25.8 % non-integer values; constant-run lengths cluster
at 3k+1 frames). Effective ego-speed resolution is therefore ~5.3 independent measurements
per 16-frame window, not 16. The signal is consumed exactly as published.

## 3. How many valid pedestrian sequences were obtained?

Parsed **4,916 pedestrian tracks** — an exact match to the authors' stated 3,284 train +
1,632 test, which independently validates the parser.

Under the main (**strict**) protocol:

| split | windows | positive | negative | positive rate | pedestrians |
|---|---|---|---|---|---|
| train | 3,944 | 138 | 3,806 | 3.5 % | 1,250 |
| val | 1,017 | 46 | 971 | 4.5 % | 329 |
| **test** | **2,357** | **168** | **2,189** | **7.1 %** | **757** |
| **total** | **7,318** | **352** | **6,966** | 4.8 % | **2,336** |

`pos_weight` = 27.58. For comparison, PIE's clean protocol yields 4,906 windows at a
**32.5 %** test positive rate with `pos_weight` 1.682.

## 4. How many were excluded and why?

| reason | tracks | note |
|---|---|---|
| pre-event segment shorter than `obs_len + TTE_MIN` = 46 frames | **2,325** | the dominant class — see §5 |
| no POI attribute record (annotator id mismatch in the release) | **170** | no label and no event frame; the authors' own interface skips these too |
| event frame outside any contiguous run of the track | **85** | catches track gaps **and** the handful of corrupt `crossing_point` values (e.g. −8,506, 65,963) |
| missing ego-speed on an observed frame | **0** | — |
| degenerate box (x2 ≤ x1 or y2 ≤ y1) | **0** | — |
| **total excluded** | **2,580** | **2,336 tracks retained** |

Nothing was silently repaired. Every rule is in `reports/temporal_protocol_IDD_PeD.md` §8 and
every excluded track is listed in `results/IDD_PeD_exclusions.csv`.

## 5. What was the IDD-PeD temporal contamination rate?

**Computed independently — the PIE rate was not assumed.** Ground truth is IDD-PeD's own
per-frame `CrossingBehavior` tag (CU/CFU/CD/CFD).

| anchor | windows | crossing windows contaminated | all windows contaminated |
|---|---|---|---|
| **naive** (track-end − 45) | 8,651 | **1,992 / 2,451 = 81.3 %** | 28.1 % |
| **`crossing_point`** (literal PIE port) | 7,919 | **159 / 537 = 29.6 %** | 6.5 % |
| **strict** — `min(cp, onset)` | 7,318 | **0 / 352 = 0.0 %** | **0.0 %** |

Reference: PIE naive **67.9 %** → clean **0.0 %**; JAAD clean **0.0 %** of 972 sequences.

**Two findings here, and the second is new.** First, the leakage class reproduces on a third
dataset, *more severely* than on PIE. Second — and this is the part that is not a re-run —
**the project's own clean protocol is not sufficient on IDD-PeD.** Anchoring at
`crossing_point` still leaves 29.6 % of crossing windows contaminated, because IDD-PeD's
`crossing_point`:

- equals the first crossing-tagged frame in only **68.9 %** of crossers (PIE: **99.4 %**);
- is at or before it in **81.0 %**;
- is **late** in **19.0 %**, by a median of 30 frames (max 291).

`figures/fig0_window_examples.png` shows a real instance: `gp_set_0002_vid_0002 / gp_5022`,
onset 8528 but `crossing_point` 8600 — the `crossing_point`-anchored window [8555, 8570] is
**16/16 frames already crossing**, while the strict window [8483, 8498] is **0/16**.

## 6. Could the same strict pre-crossing protocol be implemented?

**Yes — but only with an explicit, documented adaptation, and it is expensive.**

The adaptation is the strict anchor `event = min(crossing_point, first crossing-tagged frame)`.
This is not a new temporal rule: on PIE the two markers coincide for 99.4 % of crossers, so
taking the minimum **restores the semantics PIE's `crossing_point` already had** on a dataset
whose annotation of the same quantity is noisier. It guarantees 0 % contamination by
construction rather than by filtering.

The cost is severe, and it is a property of the dataset:

| rule | crossing tracks | non-crossing | % positive |
|---|---|---|---|
| authors' own IDD-PeD protocol (L ≥ 16, TTE = 0) | 197 | 3,728 | 5.0 % |
| obs 16 + TTE ≥ 15 (0.5 s lead) | 175 | 3,149 | 5.3 % |
| `crossing_point` anchor, TTE ≥ 30 | 149 | 2,347 | 6.0 % |
| **strict anchor, TTE ≥ 30 (this study)** | **102** | **2,234** | **4.4 %** |

**`crossing_point == first_frame` for 65.4 % of crossing tracks**, and the median pre-event
track length is **1 frame** for crossers versus **52** for non-crossers. IDD-PeD's annotators
generally began crosser tracks *at* onset rather than before it, so for most crossers a
pre-crossing observation window **does not exist in the data at all**. No protocol choice can
recover it. Broadening the label to "crosses the road anywhere" (including the `CI` class)
moves the positive rate only from 4.4 % to 5.3 %, confirming the constraint is structural
rather than a labelling artefact.

## 7. Could the PIE-trained model be evaluated zero-shot on IDD-PeD?

**Yes, cleanly, with no adaptation that touches IDD-PeD labels.** All 20 frozen checkpoints
(4 families × 5 seeds) were evaluated directly, using **PIE's** training normalization
statistics and an operating point τ\* fitted on **PIE's** validation split and carried over
unchanged. No fine-tuning, no IDD-PeD hyperparameter search, no IDD-PeD threshold tuning, no
IDD-PeD-derived normalization.

A **parity gate** ran first: the frozen BiLSTM's per-seed PIE test AUC was regenerated from
its checkpoints and matched the stored values to **|Δ| = 0.00e+00** (exact), confirming the
existing PIE results were not perturbed.

The one unavoidable adaptation is the coordinate frame (29 of 33 videos are 1920×1440 vs
PIE's 1920×1080). Two mappings are reported: `rescale` (pre-registered primary) and `raw`
(sensitivity). See §14.

---

## 8. What were the zero-shot results?

Experiment A, strict protocol, pre-registered `rescale` coordinate mapping, per-seed-mean ± std
over 5 seeds. τ\* fitted on **PIE validation**, never re-tuned.

| model | AUC | PR-AUC | F1 @ 0.5 | F1 @ PIE-τ\* | Prec | Rec | PIE in-domain AUC | Δ AUC |
|---|---|---|---|---|---|---|---|---|
| BiLSTM-F1 | 0.675 ± 0.075 | 0.139 | 0.131 | 0.131 ± 0.001 | 0.070 | 0.975 | 0.940 | **−0.265** |
| Transformer-F1 | 0.698 ± 0.075 | 0.176 | 0.130 | 0.130 ± 0.001 | 0.069 | 0.962 | 0.947 | **−0.249** |
| GRU-F1 | 0.713 ± 0.074 | 0.173 | 0.129 | 0.129 ± 0.001 | 0.069 | 0.960 | 0.941 | **−0.228** |
| Vanilla RNN-F1 | 0.720 ± 0.069 | 0.172 | 0.130 | 0.130 ± 0.002 | 0.070 | 0.958 | 0.948 | **−0.228** |
| *trivial baseline* | *0.500* | *0.071* | — | ***0.133*** | *0.071* | *1.000* | — | — |

**Read the F1 column against the baseline.** A classifier that labels every window positive
scores F1 = 0.133 on this test set. All four zero-shot models score **0.129–0.131** — i.e.
*at or marginally below* the do-nothing classifier, with precision 0.070 and recall ~0.96.
**As a decision-making classifier, zero-shot transfer fails outright.** The models inherit
PIE's 32.5 % base rate and flood a 7.1 % test set with positives.

What *does* survive is the **ranking**: AUC 0.675–0.720 with pedestrian-cluster 95 % CIs
(e.g. Transformer [0.672, 0.822]) that exclude 0.5, and PR-AUC 0.139–0.176 against a 0.071
chance line — roughly a 2–2.5× lift. So the PIE models retain a genuine but weak ordering
ability across a continent-scale domain shift, and no usable operating point.

**Which channel carries that surviving signal** (inference-only ablation on the frozen
checkpoints, `results/channel_ablation.md`, `figures/fig5_channel_ablation.png`):

| model | full | − ego-speed | − all boxes | − y only |
|---|---|---|---|---|
| BiLSTM | 0.723 | **0.568 (−0.156)** | 0.741 (+0.018) | 0.764 (+0.040) |
| Transformer | 0.752 | **0.565 (−0.187)** | 0.754 (+0.001) | 0.774 (+0.022) |
| GRU | 0.722 | **0.572 (−0.150)** | 0.751 (+0.029) | 0.764 (+0.042) |
| Vanilla RNN | 0.713 | **0.548 (−0.165)** | 0.753 (+0.040) | 0.771 (+0.058) |
| **mean Δ** | — | **−0.165** | **+0.022** | **+0.041** |

Removing ego-speed collapses transfer to near chance; removing the entire box stream
*improves* it. **Ego-speed is the domain-invariant channel; bounding-box pixel geometry is
domain-specific and actively harmful across datasets.** This is consistent to the decimal
with the measured distribution shift: under PIE's normalization IDD-PeD's speed sits at
z = −0.002 while its y-channels sit at z = +1.23 / +1.77 (raw) or −2.78 / −2.98 (rescaled).

> The "− all boxes" row is a **diagnostic, not a proposed model**. It was computed on test
> data and is reported to explain *why* transfer behaves as it does. We do not promote a
> speed-only model as a result, because selecting it would be test-set selection.

## 9. What were the independently trained IDD-PeD results?

Experiment B, strict protocol, trained from scratch on IDD-PeD only with the same
architectures and the same PIE hyperparameter configs (not re-tuned), IDD-PeD-train
normalization, 5 seeds, test touched once. τ\* fitted on IDD-PeD validation.

| model | params | AUC | PR-AUC | F1 @ τ\* | Acc | Prec | Rec | AUC 95 % cluster CI |
|---|---|---|---|---|---|---|---|---|
| **Transformer-F1** | 794,241 | **0.768 ± 0.005** | 0.224 | **0.257 ± 0.023** | 0.750 | 0.185 | 0.443 | [0.713, 0.816] |
| BiLSTM-F1 | 2,237,313 | 0.737 ± 0.023 | 0.218 | 0.233 ± 0.016 | 0.750 | 0.152 | 0.501 | [0.652, 0.800] |
| GRU-F1 | 1,678,209 | 0.721 ± 0.009 | 0.160 | 0.204 ± 0.051 | 0.789 | 0.160 | 0.406 | [0.664, 0.784] |
| Vanilla RNN-F1 | 560,001 | 0.685 ± 0.017 | 0.149 | 0.196 ± 0.047 | 0.836 | 0.181 | 0.256 | [0.616, 0.750] |
| *trivial baseline* | — | *0.500* | *0.071* | *0.133* | *0.929* | *0.071* | *1.000* | — |

Independent training beats the trivial F1 baseline (0.196–0.257 vs 0.133) and beats zero-shot
(0.129–0.131), but remains **far** below PIE's 0.844–0.852. Note that **accuracy is
meaningless here**: the do-nothing classifier scores 0.929, higher than every model.

**Ego-speed ablation on IDD-PeD (the key replication test):**

| dataset / model | 5-D AUC | 4-D bbox-only AUC | Δ AUC |
|---|---|---|---|
| **PIE (BiLSTM)** | 0.932 | 0.753 | **+0.179** |
| IDD-PeD BiLSTM-F1 | 0.737 | 0.708 | +0.029 |
| IDD-PeD Transformer-F1 | 0.768 | 0.731 | +0.037 |
| IDD-PeD GRU-F1 | 0.721 | 0.692 | +0.029 |
| IDD-PeD Vanilla RNN-F1 | 0.685 | 0.731 | **−0.045** |
| **IDD-PeD mean** | 0.728 | 0.716 | **+0.012** |

**The PIE ego-speed-dominance result does not replicate under independent IDD-PeD training.**
On PIE, dropping ego-speed costs 0.179 AUC and 0.277 F1. On IDD-PeD it costs 0.012 AUC on
average, and for the vanilla RNN the bbox-only model is *better*. See §14 for why.

**Sensitivity — what the residual leakage is worth.** Same models, same protocol, only the
event anchor differs:

| model | strict AUC | `cp_anchor` AUC | Δ | strict F1@τ\* | `cp_anchor` F1@τ\* | Δ |
|---|---|---|---|---|---|---|
| BiLSTM-F1 | 0.737 | 0.740 | +0.002 | 0.233 | 0.333 | **+0.100** |
| Transformer-F1 | 0.768 | 0.776 | +0.009 | 0.257 | 0.292 | **+0.035** |
| GRU-F1 | 0.721 | 0.764 | +0.043 | 0.204 | 0.297 | **+0.093** |
| Vanilla RNN-F1 | 0.685 | 0.713 | +0.028 | 0.196 | 0.320 | **+0.124** |
| **mean** | 0.728 | 0.748 | **+0.021** | 0.223 | 0.311 | **+0.088** |

**Accepting 29.6 % post-onset contamination inflates F1 by +0.088 and AUC by +0.021 on
average.** This is a direct, second-dataset measurement of what temporal leakage buys — the
strongest quantitative support this study produces for the paper's core methodological claim.

## 10. How do they compare with PIE?

| | PIE (in-domain) | IDD-PeD zero-shot | IDD-PeD independent |
|---|---|---|---|
| mean AUC (4 families) | **0.944** | 0.702 | 0.728 |
| mean F1 | **0.848** | 0.130 | 0.223 |
| test positive rate | 32.5 % | 7.1 % | 7.1 % |
| trivial-baseline F1 | 0.491 | 0.133 | 0.133 |
| F1 relative to its baseline | **+0.357** | **−0.003** | **+0.090** |

The last row is the fairest single comparison, since it removes the base-rate difference. PIE
models beat their trivial baseline by 0.357 F1; independently trained IDD-PeD models beat
theirs by 0.090; zero-shot models do not beat theirs at all.

Notably, **independent training barely beats zero-shot on AUC** (0.728 vs 0.702). Training on
7,318 in-domain windows with 138 positive training examples buys only ~0.026 AUC over a model
that has never seen the dataset. That is a statement about how little learnable signal the
strict IDD-PeD protocol contains, not about the models.

## 11. Did model rankings remain consistent?

**No.** (`results/family_equivalence.md`, `figures/fig4_family_equivalence.png`.)

| | PIE AUC order | IDD-PeD AUC order | Spearman ρ |
|---|---|---|---|
| zero-shot | Vanilla RNN > Transformer > GRU > BiLSTM | Transformer > BiLSTM > GRU > Vanilla RNN | −0.400 |
| independent | " | " | −0.400 |

The ordering is roughly **reversed**: the vanilla RNN, best on PIE, is worst on IDD-PeD in
both experiments. With only four models the rank-correlation p-value cannot reach significance
(minimum attainable two-sided p is 0.083), so the pairwise CIs are the substantive test:

| experiment | pairwise comparisons whose 95 % pedestrian-cluster CI excludes 0 |
|---|---|
| A — zero-shot | **1 of 6** (only Transformer vs Vanilla RNN) |
| B — independent | **5 of 6** |

**PIE's architecture-equivalence finding does not generalize.** On PIE all four families tie
on F1; under independent IDD-PeD training the Transformer significantly beats all three
recurrent families and the vanilla RNN is significantly worst (ΔAUC vs Transformer +0.072,
CI [+0.041, +0.102]). Under zero-shot the families still mostly tie, but that is weak
evidence — they are tied near a floor.

## 12. Did the main conclusions from PIE generalize?

| # | PIE conclusion | Generalizes to IDD-PeD? | Evidence |
|---|---|---|---|
| 1 | **Naive track-end anchoring leaks; event anchoring is required** | ✅ **YES, more strongly** | 81.3 % of crossing windows contaminated vs PIE's 67.9 %; leakage inflates F1 by +0.088 |
| 2 | **A leakage-free protocol is achievable** | ✅ **YES, with an adaptation** | 0.0 % contamination — but only via the strict `min(cp, onset)` anchor; the literal PIE protocol leaves 29.6 % |
| 3 | **Ego-speed is the signal that matters** | ⚠️ **PARTIAL** | ✅ in transfer (−0.165 AUC without it, boxes hurt); ❌ under independent training (+0.012 vs PIE's +0.179) |
| 4 | **Architecture / gating does not matter** | ❌ **NO** | 5 of 6 pairwise CIs exclude 0; ranking roughly reversed |
| 5 | **The model itself is accurate and deployable** | ❌ **NO** | zero-shot F1 ≤ trivial baseline; independent F1 0.196–0.257 vs PIE 0.844–0.852 |

## 13. Which conclusions did NOT generalize?

1. **Architecture equivalence (PIE finding 2) fails.** The Transformer is significantly best on
   IDD-PeD; the vanilla RNN significantly worst. On PIE they tied. So "the input matters, not
   the architecture" is **PIE-specific**, or at least does not hold when the input signal is
   weak — which is itself an interesting reading: when there is abundant signal (PIE) all
   architectures saturate and tie; when signal is scarce (IDD-PeD) capacity and inductive bias
   start to separate them.
2. **Ego-speed dominance fails under independent training.** +0.012 AUC vs PIE's +0.179.
3. **Absolute performance does not transfer at all.** −0.24 AUC zero-shot; F1 at the trivial
   baseline.

## 14. What dataset/domain differences may explain the changes?

1. **A static shortcut IDD-PeD has and PIE does not — the most likely explanation for both
   non-replications.** In the strict protocol, anchor-frame box geometry alone separates the
   classes: `bbox_xcenter` rank-biserial **+0.602**, `bbox_bottom_y` **−0.339**. PIE's audit
   found all |rb| < 0.3. This is close to tautological: IDD-PeD's positive label is *"crosses
   **in front of the ego-vehicle**"*, so positives are pedestrians near the image centre **by
   definition**. A model trained on IDD-PeD can therefore reach ~0.7 AUC from box position
   alone — which is exactly what the 4-D ablation shows (0.716 mean) — and has little need of
   ego-speed. It also explains why capacity now separates the families: they are competing on
   how well they exploit a geometric shortcut, not on modelling intent.
2. **Extreme class imbalance.** 7.1 % positive vs PIE's 32.5 %, with only **138 positive
   training windows** and 46 positive validation windows. `pos_weight` = 27.58. This alone
   destroys the transferred operating point and makes F1/accuracy nearly uninterpretable.
3. **Structural absence of pre-crossing observation.** `crossing_point == first_frame` for
   **65.4 %** of crossing tracks; median pre-event track length is **1 frame** for crossers vs
   52 for non-crossers. IDD-PeD annotators generally began crosser tracks *at* onset. Only 102
   crossing tracks survive the strict protocol.
4. **Coordinate-frame mismatch.** 29 of 33 videos are 1920×1440 (GoPro 4:3) vs PIE's
   1920×1080. Two mappings were tried; **neither is a true geometric rectification**, because
   the vertical field of view and camera mounting differ. Measured under PIE's normalization,
   `raw` leaves the y-channels at z = +1.23 / +1.77 and `rescale` at z = −2.78 / −2.98 — both
   far outside PIE's training distribution, in opposite directions. `raw` in fact scores
   *higher* (AUC 0.721–0.778 vs 0.675–0.720), and we deliberately did **not** promote it,
   because choosing a mapping on IDD-PeD test AUC would be test-set selection. This confound
   is the single largest threat to the zero-shot numbers.
5. **Genuine environment shift.** Unstructured, rule-flexible South-Asian traffic (jaywalking
   is normative; weaving through slow traffic is a labelled behaviour class) vs structured
   Toronto traffic; day **and** night vs daytime; a faster ego-vehicle (median 20 vs 16 km/h).
6. **Weaker event annotation.** `crossing_point` == true onset in 68.9 % vs PIE's 99.4 %, plus
   a handful of corrupt values far outside the frame range.

## 15. Are there any methodological limitations?

Stated plainly, all of them:

1. **Small positive N.** 138 positive training windows, 46 validation, 168 test — from 102/13/48
   pedestrian tracks. All CIs are wide and seed variance is substantial (e.g. BiLSTM zero-shot
   AUC ± 0.075). Any single number should be read with its cluster CI.
2. **Validation split is thin.** 46 positive windows drive both F1-based checkpoint selection
   and τ\* fitting. This is noisy and is a real weakness of Experiment B.
3. **The coordinate mapping is an unresolved confound** (§14.4). The zero-shot numbers should
   be read as a *band* (rescale 0.675–0.720, raw 0.721–0.778), not a point estimate.
4. **The static shortcut means IDD-PeD is partly a position-classification benchmark**, not a
   pure intention benchmark. This applies to the dataset authors' published numbers as much as
   to ours.
5. **Ego-speed is effectively 10 Hz**, upsampled to 30 fps, so a 16-frame window holds ~5.3
   independent speed samples.
6. **No LOSO.** PIE uses leave-one-set-out; IDD-PeD's positives are too unevenly distributed
   across sets (30/9/30/9/11/4/4/2/3) and `gp_set_0009` has **3 crossing tracks and zero
   non-crossing tracks**, which makes a leave-one-set-out fold undefined. The pedestrian-cluster
   bootstrap is used instead. This is a deliberate, documented deviation.
7. **Experiment B ran on MPS, not CPU.** Per the project's own Issue-12 finding, `nn.LSTM`
   training on MPS is process-history-dependent, so the three recurrent families' runs are
   **not exactly reproducible**. The device is recorded in every `final.json`. A CPU rerun
   (`--device cpu`, ~55 min) would restore bit-reproducibility and is recommended before
   submission.
8. **Hyperparameters were not re-tuned for IDD-PeD.** This is deliberate (it keeps A and B
   comparable and avoids IDD-PeD-specific search), but it means Experiment B is a *replication
   under PIE's recipe*, not IDD-PeD's best achievable model. The dataset authors report their
   own baselines dropping up to 15 % on IDD-PeD, so low absolute numbers are expected there too.
9. **The `cp_anchor` sensitivity block was run only for the 5-D input**; the 4-D × `cp_anchor`
   combination was not run (it appears in no table).
10. **Detector-in-the-loop was not run** on IDD-PeD (§Phase 9 report). No claim of cross-dataset
    detector robustness is made.

## 16. Is the experiment strong enough to support a journal claim of cross-dataset validation?

**Yes — for a claim about the *protocol and the methodology*. No — for a claim about model
generalization.**

Supportable:
- the temporal-leakage audit generalizes to a third dataset and is **more** severe there;
- a leakage-free protocol is achievable but required a documented adaptation, because
  IDD-PeD's event annotation is measurably weaker than PIE's;
- leakage inflates reported F1 by **+0.088** on a second dataset — a direct measurement of the
  cost of the practice this paper criticises;
- ego-speed is the domain-invariant channel *for transfer*;
- a full, reproducible dataset audit of an ICRA-2025 benchmark.

Not supportable:
- that the model generalizes (it does not: zero-shot F1 ≤ trivial baseline);
- that the four families tie (they do not on IDD-PeD);
- that ego-speed dominance is universal (it is not: it vanishes under independent training).

Framed as *"we validated our protocol on a second, harder dataset and report honestly where
our conclusions do and do not hold"*, this is a strong, reviewer-resistant contribution.
Framed as *"our method also works on IDD-PeD"*, it would be indefensible and a reviewer would
find it immediately.

## 17. What exact numbers should be reported in the paper?

**Temporal audit (lead with these):**
- IDD-PeD naive track-end anchor: **81.3 %** of crossing windows contain post-onset frames
  (n = 2,451); all windows 28.1 % (n = 8,651).
- `crossing_point` anchor (the literal PIE protocol): **29.6 %** (n = 537).
- Strict `min(crossing_point, onset)` anchor: **0.0 %** (n = 352 crossing, 7,318 total).
- `crossing_point` equals true onset in **68.9 %** of IDD-PeD crossers vs **99.4 %** on PIE;
  late in **19.0 %** (median 30 frames).
- Leakage inflation: **F1 +0.088**, **AUC +0.021** (mean over 4 families).

**Dataset:** 4,916 pedestrian tracks parsed (matches the authors' 3,284 + 1,632); 2,336 usable;
**7,318 windows** (train 3,944 / val 1,017 / test 2,357); test positive rate **7.1 %**;
`pos_weight` 27.58; ego-speed present on **100 %** of frames, 0 missing.

**Zero-shot (Table 4):** AUC **0.675 ± 0.075** (BiLSTM), **0.698 ± 0.075** (Transformer),
**0.713 ± 0.074** (GRU), **0.720 ± 0.069** (Vanilla RNN); PR-AUC 0.139–0.176 against a 0.071
chance line; F1 @ PIE-τ\* **0.129–0.131** against a trivial-baseline F1 of **0.133**.
Always print that baseline next to the F1.

**Independent (Table 5):** AUC **0.768 ± 0.005** (Transformer), 0.737 ± 0.023 (BiLSTM),
0.721 ± 0.009 (GRU), 0.685 ± 0.017 (Vanilla RNN); F1 @ τ\* 0.196–0.257.

**Channel ablation:** removing ego-speed from the frozen PIE models costs **−0.165 AUC**
(mean over 4 families); removing all box channels **+0.022**.

**Ego-speed ablation under independent training:** IDD-PeD **+0.012** AUC mean vs PIE
**+0.179** — report both, as a non-replication.

**Family equivalence:** **5 of 6** pairwise cluster-bootstrap CIs exclude zero under
independent training (1 of 6 zero-shot); Spearman ρ vs the PIE ordering **−0.400**.

Always report mean ± std over the 5 seeds with the pedestrian-cluster 95 % CI, never a single
seed, and never accuracy without its 0.929 do-nothing baseline.

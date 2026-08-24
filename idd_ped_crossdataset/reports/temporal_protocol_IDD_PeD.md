# Temporal protocol for IDD-PeD

The pre-registered rule that defines a valid observation window on IDD-PeD, and every
adaptation forced by differences from PIE. Written **before** any model was trained or
evaluated on IDD-PeD; the only inputs to these decisions were the schema audit
(`IDD_PeD_schema_audit.md`) and the temporal audit (`IDD_PeD_temporal_audit.md`), neither of
which involves a model.

---

## 1. Crossing-event definition

| | PIE (reference) | IDD-PeD |
|---|---|---|
| binary label | `crossing` (pedestrian crosses in front of the ego-vehicle) | `crossing` ∈ {no: 0, yes: 1} — the authors define it as *"whether the person is seen crossing in front of the ego-vehicle. Literally traversing the width of the ego-vehicle."* |
| event frame | `crossing_point` (per-pedestrian XML attribute) | `crossing_point` (per-pedestrian attribute, stored on the track labelled `POI`) |
| per-frame crossing state | `cross` attribute (recovered in Issue 1) | `CrossingBehavior` ∈ {CU, CFU, CD, CFD, CI, N/A} |

The two datasets therefore express the *same* event with the *same* annotation type. That is
what makes a like-for-like port possible at all.

**Definition used here.** A pedestrian is *crossing at frame f* iff the per-frame
`CrossingBehavior` tag at f ∈ {**CU, CFU, CD, CFD**}. **`CI` is excluded** — it means
"crossing the road but **not** in the path of the ego-vehicle", which is precisely what the
binary label calls a non-crossing. Including CI would contradict the label being predicted.

**Crossing onset** := the first frame of the track satisfying that condition.

---

## 2. The event anchor — and why IDD-PeD needs an adaptation

PIE's clean protocol anchors on `crossing_point`, which is safe there because Issue 1
verified that `crossing_point` equals the first `cross=="crossing"` frame in **516/519
crossers (99.4 %)** and is **never later** than true onset.

We ran that same verification on IDD-PeD. It does **not** hold:

| check | PIE | IDD-PeD |
|---|---|---|
| `crossing_point` == first crossing-tagged frame | 99.4 % | **68.9 %** |
| `crossing_point` ≤ first crossing-tagged frame (never late) | 100 % | **81.0 %** |
| `crossing_point` *later* than onset (would leak) | 0 % | **19.0 %**, median 30 frames late, max 291 |

Anchoring naively on `crossing_point` therefore leaves **29.6 % of crossing windows
contaminated** with post-onset frames (measured, `IDD_PeD_temporal_audit.md` §2).

**Adopted anchor (`strict`):**

```
event_frame = min( crossing_point , first frame tagged CU/CFU/CD/CFD )
```

for crossers, and `crossing_point` for pedestrians who are never tagged as crossing (there
is nothing earlier to anchor on). Because the event frame can never be later than true
onset, **no frame at or after crossing onset can enter an observation window — by
construction, not by filtering.** Measured contamination: **0.0 %** of 7,318 windows.

This is not a new temporal rule invented for IDD-PeD. On PIE the two markers coincide for
99.4 % of crossers, so `min(cp, onset)` *is* PIE's rule; taking the minimum simply restores
the semantics PIE's `crossing_point` already had, on a dataset whose annotation of the same
quantity is noisier.

The literal port (`cp_anchor`, event = `crossing_point`) is **also built and reported**, as a
disclosed sensitivity analysis that quantifies what the residual 29.6 % contamination is
worth in metric terms.

---

## 3. Label mapping

The authors' binary `crossing` attribute is used **verbatim**; no relabelling.

One documented ambiguity: **210 tracks labelled `crossing = 0` still contain CU/CFU/CD/CFD
frames.** This is *consistent*, not contradictory — the label asks whether the pedestrian
crosses **in front of the ego-vehicle**, and a pedestrian can cross the road elsewhere in the
scene. It does mean the per-frame tag is a leakage detector for positives, not a substitute
label, and it is used only in that role.

Consequence for the contamination metric: negatives that contain crossing frames are counted
in the "all windows" leakage figure but are not, strictly, label leakage. Both the
crossers-only and all-windows rates are therefore reported separately.

---

## 4. Observation length and prediction horizon

Held **identical to PIE**, no re-tuning:

| parameter | value | note |
|---|---|---|
| `obs_len` | **16 frames** | 0.53 s at 30 fps, same as PIE |
| `TTE_MIN` | **30 frames** | the last observed frame is ≥ 1.0 s before the event |
| `TTE_MAX` | **60 frames** | ≤ 2.0 s before the event |
| overlap | **0.5** (stride 8) | same as PIE |
| features | `[x1, y1, x2, y2, ego_speed]` | same order, same semantics |
| coordinates | **raw pixels** | PIE convention: never normalized to image size |

The dataset authors' own intention benchmark uses "an observation period of 0.5 s and a
time-to-event varying between 1–2 s", i.e. the same numbers — so this is not an imposition of
PIE's protocol on an incompatible dataset.

---

## 5. Frame-rate conversion

**None required, and none performed.** The paper describes two cameras (GoPro Hero 8 @ 30 fps,
DDPAI X2SPro @ 25 fps), but the `ddpai/` directories are **empty in the public release** —
all 33 released videos are GoPro at 30 fps, identical to PIE. Frame indices are therefore
directly comparable and `obs_len`/TTE carry over in frames without conversion.

---

## 6. Image resolution — the one unavoidable confound

| | PIE | IDD-PeD |
|---|---|---|
| resolution | uniformly 1920×1080 | **1920×1440 in 29 videos**, 1920×1080 in 4 |

Because the feature contract is raw pixels, this matters **only for Experiment A**
(a PIE-trained model consuming IDD-PeD coordinates). Two mappings are defined:

- **`rescale` (pre-registered primary)** — scale x by 1920/W and y by 1080/H per video, so
  every box lives in a 1920×1080 frame. Chosen *a priori* because it matches the coordinate
  convention the models were trained on.
- **`raw` (sensitivity)** — feed IDD-PeD pixels unchanged.

**Neither is a true geometric rectification, and this is disclosed as a limitation.** A 4:3
GoPro has a different vertical field of view and mounting from PIE's rig, so the same pixel
row does not correspond to the same world elevation. Measured under PIE's own training
normalization, the mean standardized value of the box channels is:

| channel | `raw` z̄ | `rescale` z̄ |
|---|---|---|
| x1 | −0.50 | −0.50 |
| y1 | **+1.23** | **−2.78** |
| x2 | −0.46 | −0.46 |
| y2 | **+1.77** | **−2.98** |
| **ego_speed** | **−0.002** | **−0.002** |

Both mappings leave the vertical channels well outside PIE's training distribution, in
opposite directions; `rescale` overshoots further. **Both are reported.** The primary is the
pre-registered `rescale`; `raw` scores higher, and we explicitly do **not** promote it to
primary, because selecting a coordinate mapping on IDD-PeD test AUC would be test-set
selection — the exact thing Experiment A forbids.

The speed channel needs **no** adaptation: both datasets record km/h, and IDD-PeD's speed
lands at z̄ = −0.002 under PIE's normalization, i.e. essentially the same distribution.

---

## 7. Splits

IDD-PeD ships a set-level **train/test** split and **no validation set**. Ours:

| split | sets | rationale |
|---|---|---|
| **train** | `gp_set_0001`, `gp_set_0004`, `gp_set_0007` | remainder of the official training pool |
| **val** | `gp_set_0002`, `gp_set_0006` | carved from the official *training* pool only |
| **test** | `gp_set_0003`, `gp_set_0005`, `gp_set_0008`, `gp_set_0009` | **the authors' official test set, untouched** |

- Splitting is at **recording-set granularity**, never a random per-window split — the same
  leakage discipline PIE uses (a pedestrian's overlapping windows can never straddle splits).
- The validation sets were chosen on **usable-track counts alone**, before any model was run,
  and without consulting the test sets. `gp_set_0002` + `gp_set_0006` give the best positive
  count available from the training pool (13 of the pool's 54 crossing tracks) at a
  train:val ratio of ≈ 79:21, closely matching PIE's ≈ 77:23.

---

## 8. Inclusion / exclusion criteria

Every rule is stated here and applied by `scripts/03_build_sequences.py`. **No suspicious
value is silently repaired.** Per-track exclusion records: `results/IDD_PeD_exclusions.csv`.

| rule | action | tracks affected (strict) |
|---|---|---|
| pedestrian has no POI attribute record | exclude (no label, no event frame) | **170** |
| no `crossing` label or no `crossing_point` | exclude | 0 (co-occurs with the above) |
| event frame lies outside any contiguous run of the track | exclude — catches both track gaps and the handful of corrupt `crossing_point` values (e.g. −8,506, 65,963) | **85** |
| track has a gap: >1 contiguous segment | keep **only** the segment containing the event frame (PIE's rule) | 29 |
| pre-event segment shorter than `obs_len + TTE_MIN` = 46 frames | exclude (no valid window exists) | **2,325** |
| any observed frame lacks an ego-speed record | exclude | 0 |
| degenerate box (x2 ≤ x1 or y2 ≤ y1) anywhere in the pre-event segment | exclude — never clipped or repaired | 0 |
| box extends outside the image bounds | **kept** — PIE feeds raw unclipped coordinates too; recorded only | — |

Result: **4,916 tracks → 2,336 usable → 7,318 windows** (train 3,944 / val 1,017 / test 2,357).

---

## 9. Missing values

- **Ego speed: none missing.** Every one of the 33 videos has exactly one OBD record per
  video frame, with contiguous ids from 0, and the OBD `id` *is* the frame index. No
  interpolation, resampling or timestamp matching is performed.
- **Underlying rate:** the OBD sensor logs at 10 Hz; the released 30 fps signal is that
  series linearly upsampled in thirds (25.8 % of values non-integer, constant-run lengths
  clustering at 3k+1). The *effective* resolution is 10 Hz, so a 16-frame window carries
  ≈ 5.3 independent speed measurements. Disclosed; the signal is consumed as published.
- **Bounding boxes:** boxes with `outside="1"` (pedestrian left the frame) are dropped at
  parse time, which is what creates the track gaps handled above.

---

## 10. Examples of valid and invalid sequences

**Valid** — `gp_set_0003 / ped X`, label 1, event frame 4,812:
window frames 4,750–4,765, anchor 4,765, TTE = 47 frames (1.57 s). Every frame is tagged
`N/A` for `CrossingBehavior` (not yet crossing); ego speed present for all 16 frames; box
non-degenerate throughout. → **admitted**.

**Invalid (a) — no lead time.** A crossing track whose `crossing_point` equals its first
annotated frame. Pre-event segment length L = 1 < 46, so no window can end ≥ 30 frames before
the event. → **excluded** (`track_too_short`). This is the single largest exclusion class and
affects **65.4 %** of crossing tracks: IDD-PeD annotators typically began crosser tracks at
onset rather than before it.

**Invalid (b) — post-onset contamination under the naive anchor.** The same crossing track,
anchored at `track_end − 45` instead of at the event, yields a window sitting *inside* the
crossing: several of its 16 frames are tagged CU. → **admitted by the naive convention
(81.3 % of crossing windows), rejected here.**

**Invalid (c) — late `crossing_point`.** A crossing track where `crossing_point` is 120 frames
*after* the first CU-tagged frame. Anchoring on `crossing_point` produces a window that still
contains crossing frames. → **admitted by `cp_anchor` (29.6 % of crossing windows), rejected
by `strict`.**

---

## 11. Summary of every deviation from the PIE protocol

| # | deviation | forced by | disclosed in |
|---|---|---|---|
| 1 | event anchor is `min(crossing_point, onset)` rather than `crossing_point` | IDD-PeD's `crossing_point` is late in 19 % of crossers | §2 |
| 2 | a validation split is carved from the training sets | IDD-PeD ships no official val split | §7 |
| 3 | `pos_weight` = 27.58 rather than PIE's 1.682 | IDD-PeD's class balance (a dataset property, not a tuned knob) | §7 |
| 4 | box coordinates are mapped into a 1920×1080 frame for Experiment A | 29 of 33 videos are 1920×1440 | §6 |

Everything else — obs_len, TTE band, overlap, feature order and semantics, raw-pixel
convention, train-only normalization, F1-first selection, 5 seeds, single test touch,
pedestrian-cluster bootstrap — is **identical to the PIE study**.

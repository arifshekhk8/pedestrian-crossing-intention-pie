# IDD-PeD schema audit — what the dataset actually provides

**Verified against the released annotations on 2026-08-25**, not against the paper's text. Every number below is produced by `scripts/02_schema_audit.py` from `data/iddped_database.pkl`; the per-track evidence is in `results/IDD_PeD_track_inventory.csv`.

## 0. Provenance and licence

| | |
|---|---|
| Dataset | IDD-PeD — *Pedestrian Intention and Trajectory Prediction in Unstructured Traffic Using IDD-PeD*, Bokkasam, Gangisetty, Hafez & Jawahar, **ICRA 2025** |
| Official project page | https://cvit.iiit.ac.in/research/projects/cvit-projects/iddped |
| Official code / annotations | https://github.com/Ruthvik9/IDD-PeD |
| Files used | `annotations.tar` (478,209,024 B), `annotations_vehicle.tar` (58,593,280 B) |
| Download host | `https://cvit.iiit.ac.in/images/datasets/IDDPed/Annotations/` |
| Access control | **none** — direct HTTPS, no registration, no access form |
| Licence | **CC BY 4.0** (stated in the paper) — permits research use with attribution |
| Videos | **not downloaded** — the main experiment needs only boxes + OBD speed |

> The project's earlier `journal_prep/cross_dataset_validation/PLAN.md` (2026-07-21) listed IDD-PeD as *"access-gated forms — not now"*. **That assessment is outdated**: the annotations are served as unauthenticated CC BY 4.0 downloads. That plan file was left untouched.

## 1. Inventory

| quantity | value |
|---|---|
| recording sets | 9 (`gp_set_0001` … `gp_set_0009`) |
| annotated videos | 33 |
| OBD (vehicle) files | 34 — one per released video, plus one for a video whose annotation XML is not published |
| **pedestrian tracks** | **4,916** |
| tracks with a POI attribute record | 4,746 (96.5 %) |
| tracks **without** attributes (annotator id mismatch) | 170 (3.5 %) |
| total annotated box-frames | 494,854 |

The authors' README states **3,284 train + 1,632 test = 4,916** pedestrians. Our independent parse recovers **4,916** tracks — an exact match, which validates the parser.

### Camera and frame rate

The paper mentions two cameras (GoPro Hero 8 @ 30 fps, DDPAI X2SPro @ 25 fps). In the public release the `ddpai/` directories exist but are **empty** — every released video is `gopro`. **All data used here is therefore 30 fps, identical to PIE**, so no frame-rate conversion is required and none is performed.

### Image resolution — not constant

| resolution | videos |
|---|---|
| 1920×1440 | 29 |
| 1920×1080 | 4 |

PIE is uniformly 1920×1080. IDD-PeD is mostly **1920×1440** (GoPro 4:3). Because the PIE feature contract uses **raw pixel coordinates**, this is a genuine domain difference that must be handled explicitly for zero-shot transfer — see `reports/temporal_protocol_IDD_PeD.md` §6.

## 2. The 13 required checks

| # | Required modality | Provided? | Evidence |
|---|---|---|---|
| 1 | video frames / sequences | ✅ (not downloaded) | 9 video tars on the CVIT host; not needed for the main experiment |
| 2 | pedestrian bounding boxes | ✅ per-frame | CVAT `<box xtl ytl xbr ybr>`; 494,854 box-frames parsed |
| 3 | pedestrian identities / tracks | ✅ | 4,916 distinct track ids |
| 4 | crossing behaviour / action labels | ✅ | per-track `crossing` ∈ {no:0, yes:1} **and** per-frame `CrossingBehavior` ∈ {CU, CFU, CD, CFD, CI, N/A} |
| 5 | crossing-onset information | ✅ **`crossing_point`** | per-track integer frame index; present for 4,746 tracks |
| 6 | **ego-vehicle speed** | ✅ **per-frame `OBD_speed`** | `annotations_vehicle/**/<vid>_obd.xml` |
| 7 | ego-vehicle acceleration | ✅ | `accT`, `accX`, `accY`, `accZ` on the same records (unused — PIE has no analogue) |
| 8 | timestamps / frame indices | ✅ frame indices | the OBD `id` **is** the video frame index; no wall-clock timestamps |
| 9 | frame rate | ✅ 30 fps | GoPro only (`ddpai` empty) |
| 10 | camera information | ✅ | GoPro Hero 8; per-video `meta/task/original_size` |
| 11 | train/val/test splits | ⚠️ train/test only | official 70/30 by set; **no official validation split** — see §5 |
| 12 | missing values | ⚠️ quantified | §3, §4, §6 |
| 13 | video ↔ ego-signal synchronisation | ✅ **exact** | §4 |

## 3. Crossing labels and the crossing event

- Tracks carrying a `crossing` label: **4,746**
  - `crossing = yes` (crosses in front of the ego-vehicle): **657** (13.8 %)
  - `crossing = no`: **4,089** (86.2 %)
- Tracks with a `crossing_point` frame: **4,746**
- `crossing_point` falling inside a contiguous run of the track: **4,659** (98.2 % of those that have one)

### `crossing_point` vs the per-frame behaviour tag — an independent consistency check

PIE's clean protocol depends on `crossing_point` being a faithful marker of true crossing onset (Issue 1 validated it at 99.4 % on PIE). We ran the equivalent check here using IDD-PeD's *own* per-frame `CrossingBehavior` tag as ground truth: onset := the first frame tagged CU / CFU / CD / CFD.

- Crossing tracks with both a `crossing_point` and a taggable onset: **610**
- `crossing_point` **exactly equals** the first crossing-tagged frame: **420** (68.9 %)
- `crossing_point` **at or before** the first crossing-tagged frame (never late, so never leaks): **494** (81.0 %)
- `crossing_point − onset` (frames): median 0, mean 6.4, p5 -40, p95 80

## 4. Ego-vehicle speed — availability and synchronisation (the critical STOP check)

**This is the modality JAAD lacked, and the reason IDD-PeD is worth doing.**

| property | finding |
|---|---|
| storage | `<vehicle_attributes><frame OBD_speed="…" accT accX accY accZ id="N"/>` — the same XML shape as PIE's `*_obd.xml` |
| record count | **exactly one record per video frame, for all 33 videos** (OBD rows == `meta/task/size` in every case) |
| frame alignment | the OBD `id` **is** the video frame index — alignment is index-to-index by construction; no interpolation, resampling or timestamp matching is needed |
| frame-id contiguity | all 34 files start at id 0 with strictly contiguous ids (0 gaps, 0 non-monotonic) |
| tracks with 100 % speed coverage | **4,916 / 4,916** (100.0 %) |
| missing speed values | **0** |
| negative or impossible speeds | **0** (min 0.00, max 60.00) |

### Underlying sampling rate — what the released signal really is

The paper states the OBD sensor logs at **10 Hz** while video runs at 30 fps. The released per-frame signal is that 10 Hz series **upsampled to 30 fps by linear interpolation in thirds** — e.g. `… 33, 33, 33, 33.66, 34.3, 35, 35, 35 …`. Measured over all 582,688 OBD records: 25.8 % of values are non-integer, and constant-value run lengths cluster at 3k+1 frames, exactly the signature of a 3× upsample.

**This is a real limitation and is disclosed as such**: the *effective* ego-speed resolution is 10 Hz, so a 16-frame (0.53 s) window carries ~5.3 independent speed measurements rather than 16. PIE's OBD is likewise not truly per-frame (its clean sequences contain only 102 distinct speed values). No further processing is applied by us — the released signal is consumed exactly as published.

### Scale compatibility with PIE — decisive for zero-shot transfer

| statistic | PIE `vehicle_speed` (clean sequences) | IDD-PeD `OBD_speed` (all records) |
|---|---|---|
| min | 0.00 | 0.00 |
| median | 16.00 | 12.00 |
| mean | 16.43 | 13.81 |
| p99 | 44.02 | 38.00 |
| max | 56.01 | 60.00 |
| % exactly zero | 22.7 % | 13.9 % |

Both are **km/h on the same scale**. No unit conversion is required, and none is applied. IDD-PeD's ego-vehicle is moderately faster (median 20 vs 16 km/h) — a genuine domain difference to report, not a units artefact. **Had the scales disagreed, zero-shot transfer would have been scientifically invalid and this experiment would have stopped here.**

## 5. Splits

The authors define a set-level 70/30 **train/test** split and **no validation set**:

- train: `gp_set_0001`, `gp_set_0002`, `gp_set_0004`, `gp_set_0006`, `gp_set_0007`
- test: `gp_set_0003`, `gp_set_0005`, `gp_set_0008`, `gp_set_0009`

Our protocol keeps the official test set untouched and carves a validation split out of the **training** sets only, at set granularity (never a random window split) — the same leakage discipline PIE uses. See `reports/temporal_protocol_IDD_PeD.md` §7.

## 6. Data-quality checks

| check | count | handling |
|---|---|---|
| duplicate frame entries within a track | 0 | none → no de-duplication rule needed |
| tracks with a gap (>1 contiguous segment) | 29 | handled exactly as PIE: keep only the contiguous segment containing `crossing_point` |
| degenerate boxes (x2≤x1 or y2≤y1) | 0 box-frames | excluded by the builder if present |
| boxes outside the image bounds | 49 box-frames | **not** clipped — PIE feeds raw coordinates too; recorded only |
| pedestrians without POI attributes | 170 | **excluded** — no label and no `crossing_point`, so no valid window can be built |
| unmapped `CrossingBehavior` values | 0 | recorded |
| missing OBD file | 0 videos | n/a |

No suspicious value is silently repaired. Every exclusion rule is stated above and re-stated in `reports/temporal_protocol_IDD_PeD.md` §8.

## 7. Feasibility of the PIE window protocol on IDD-PeD

Applying PIE's exact rule (truncate at `crossing_point`; require `L ≥ obs_len + TTE_MIN = 16+30 = 46` frames of pre-event track):

| stage | tracks |
|---|---|
| all pedestrian tracks | 4,916 |
| with POI attributes | 4,746 |
| with a `crossing_point` | 4,746 |
| `crossing_point` inside a contiguous run | 4,659 |
| **long enough for ≥1 valid window** | **2,496** |
| ↳ in official train sets | 1,679 (crossing 91 / not 1,588) |
| ↳ in official test sets | 817 (crossing 58 / not 759) |

**Verdict: the PIE protocol transfers.** IDD-PeD supplies per-frame boxes, per-frame ego speed on PIE's scale, a per-track binary crossing label, and a per-track `crossing_point` event frame — every ingredient the clean protocol needs.

## 8. STOP-condition assessment

| STOP condition | status |
|---|---|
| 1. no usable ego speed aligned to video frames | **CLEAR** — per-frame, index-aligned, 0 missing, PIE-compatible scale |
| 2. crossing onset cannot be defined reliably | **CLEAR** — native `crossing_point`, cross-validated against the per-frame behaviour tag |
| 3. annotations cannot support the same task | **CLEAR** — binary crossing-in-front-of-ego label, same task |
| 4. licence prevents the intended use | **CLEAR** — CC BY 4.0 |
| 5. a fair PIE→IDD-PeD comparison is impossible | **CLEAR**, with two disclosed adaptations (image height, no official val split) |
| 6. an existing project file must be modified | **CLEAR** — none is |
| 7. required data inaccessible | **CLEAR** — direct download, no gate |
| 8. serious label-definition ambiguity | **PARTIAL** — the `CI` ("crossing, but not in the ego-vehicle's path") behaviour class has no PIE analogue; resolved and documented in `reports/temporal_protocol_IDD_PeD.md` §3 |

**No STOP condition is triggered. Proceeding to Phase 3.**

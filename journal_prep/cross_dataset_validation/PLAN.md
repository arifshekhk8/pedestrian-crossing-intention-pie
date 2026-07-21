# PLAN — Cross-dataset validation of the PIE-trained crossing-intention models

**Goal (2026-07-21).** Strengthen the MTI paper by showing our findings generalize
beyond PIE — a **zero-shot cross-dataset test** (train on PIE, test on a second dataset
we never train on). This directly answers the "single dataset (PIE only)" reviewer risk
and is the single biggest booster now that the PSI 2.0 cross-test is paused (access form
never answered).

**Metric hierarchy: F1 → accuracy → AUC** (supervisor directive). Report F1 first; AUC/PR-AUC
are the threshold-free cross-domain metrics that survive base-rate + threshold shift, so
lead the *cross-domain* numbers with AUC/PR-AUC and report F1/Acc at a **fixed, non-re-tuned**
threshold. Never re-tune the operating point on the test dataset.

---

## 0. READ THIS FIRST — the downloadable-dataset reality (verified 2026-07-21)

I checked every candidate. The honest situation:

| Dataset | Ego-vehicle speed? | Crossing labels? | Directly downloadable? | Verdict |
|---|---|---|---|---|
| **PIE** (our source) | ✅ OBD, per-frame | ✅ | ✅ | training set — not a test |
| **PSI 2.0** | ~ (1 Hz) | ✅ | ❌ **access form, no reply** | **BLOCKED** (paused) |
| **PePScenes** (nuScenes + ped crossing labels) | ✅ (via nuScenes CAN bus) | ✅ | ❌ **repo DELETED** — `huawei-noah/PePScenes` 404s, **no Wayback snapshot, no author mirror** (verified via authenticated GitHub + archive.org) | **DEAD — do not pursue** |
| **JAAD** | ❌ **no OBD speed** (only 5 coarse driver-motion states: stopped / moving-slow / moving-fast / speeding-up / slowing-down) | ✅ | ✅ (`github.com/ykotseruba/JAAD`) | **VIABLE — methodology track** |
| **nuScenes + CAN bus** | ✅ real velocity (m/s) | ❌ **no native crossing label** (must self-derive) | ✅ (free registration) | **VIABLE — full-model track, higher effort/risk** |
| TITAN / LOKI / STIP / IDD-PeD | varies | ✅/varies | ❌ access-gated forms | not now |

**Bottom line: there is no drop-in, downloadable dataset that has BOTH ready-made crossing
labels AND ego-vehicle speed except PIE itself.** That scarcity is exactly why the field
uses PIE + JAAD, and why JAAD is used *without* ego-speed. So this plan has **two tracks**;
pick based on effort appetite (Track A recommended for MTI).

---

## 1. Two tracks (choose)

### ⭐ Track A — JAAD methodology replication (RECOMMENDED: low-risk, standard, downloadable now)
**What it proves:** our two *methodological* contributions generalize to a second dataset —
(1) the **temporal-leakage audit + clean protocol**, and (2) the **four-family "the input
signal matters, not the architecture/gating" isolation**. It does **not** test ego-speed
dominance, because JAAD structurally lacks ego-speed.
**Why that's OK and defensible:** JAAD's missing ego-motion is a *documented, citable* fact
(Rasouli & Kotseruba 2024, "Diving Deeper", explicitly note JAAD's poor ego-motion data
"causes severe degradation"). We turn the gap into supporting evidence: on the dataset
*without* ego-speed, bbox-only is the ceiling — consistent with ego-speed being the key
signal on PIE. **Effort: LOW. Label risk: NONE (standard benchmark labels).**

### Track B — nuScenes + CAN-bus, self-derived crossing labels (STRETCH: tests ego-speed too)
**What it proves:** the **full 5-D model (bbox + real ego-speed)** transfers zero-shot,
including the ego-speed signal — the only downloadable way to test that.
**The catch:** nuScenes has no native crossing-intention label, so we **derive** it from the
3D pedestrian tracks + ego trajectory (heuristic below). Self-derived labels are a reviewer
target and won't match any published nuScenes number, so frame it strictly as a *zero-shot
generalization probe* with the labeling rule fully disclosed. **Effort: HIGH. Label risk:
MODERATE (mitigate by reporting AUC/PR-AUC primarily + a hand-audited label sample).**

> **Recommendation for MTI:** do **Track A** (it cleanly generalizes the paper's core
> methodological claims at low risk). Add **Track B** only if you want the ego-speed transfer
> result and have the engineering time; if Track B's derived labels look noisy on a hand
> audit, drop it to an appendix or omit it — do not let a shaky label set undercut the
> rigor that is the paper's selling point.

---

## 2. Track A — JAAD, concrete method

**Reuse, don't reinvent:** JAAD uses the *same annotation tooling and format* as PIE (both
are handled by `github.com/ykotseruba/PedestrianActionBenchmark`), so our clean-protocol
builder and unified engine adapt with minimal change.

**Download (no gate):**
- Annotations: `git clone https://github.com/ykotseruba/JAAD` (XML annotations + interface).
- Videos: the repo's `download_clips.sh` (JAAD is 346 clips, ~5–10 s; small vs PIE).
- Use **JAAD_beh** (behavioral subset: pedestrians who are crossing/relevant, ~495 tracks)
  as the primary; note JAAD_all as the harder superset.

**Build side (leak-free, mirrors `issue2_clean_protocol/02_build_sequences_clean.py`):**
1. Parse JAAD per-frame pedestrian boxes + the per-frame `cross` behavioral tag + the
   `crossing` label. JAAD has the same `crossing_point`-style event; if absent, anchor at the
   first `cross==crossing` frame (validated the same way Issue 1 recovered PIE's).
2. Run the **leakage audit** (`issue1_leakage_audit/01_leakage_audit.py --seq-dir …`): report
   what fraction of JAAD crossers leak under the naive (track-end) anchor vs the clean
   (event) anchor. **Expected headline: JAAD has the same leakage class → our audit
   generalizes.**
3. Build clean windows: obs_len **16**, TTE ∈ [30,60] **at JAAD's 30 fps** (JAAD is 30 fps
   like PIE — no resample needed), 50% overlap, event-anchored. Feature set = **bbox-only
   (4-D)** because JAAD has no ego-speed. (Optional 5th channel: JAAD's coarse
   driver-motion state one-hot as a weak ego-motion proxy — report as a *separate* ablation,
   not the main number; it is not comparable to PIE's continuous OBD speed.)
4. Split: JAAD has no set01–06 partition; use JAAD's **official video-level train/val/test
   split** (from the benchmark repo) — never a random per-window split (same leakage
   discipline as PIE).

**Model/train side:** run **all four families** (bilstm / transformer / gru / birnn) through
the **same unified engine** (`issue12_unified_pipeline/12_unified_engine.py`), F1-first
(`--select f1`), 5 seeds, train-only z-score, pos_weight = JAAD-train neg/pos. Two experiments:
- **(A1) train-on-JAAD / test-on-JAAD** (bbox-only) — does the *architecture-isolation* hold
  on a second dataset? Expected: the four families tie on F1 again → the "input, not the
  model" finding generalizes.
- **(A2, optional) zero-shot train-on-PIE(bbox-only) / test-on-JAAD** — the true
  cross-dataset transfer of the bbox stream (standardize JAAD boxes with **PIE** train stats;
  never re-fit norm on JAAD; rescale JAAD resolution to PIE's 1920×1080 coordinate scale).

**Report:** a JAAD leakage-audit number + a four-family F1/AUC table (A1) + optional transfer
AUC/PR-AUC (A2). Append a row-block to `journal_prep/Analysis/model_comparison.md`.

---

## 3. Track B — nuScenes + CAN bus, concrete method (stretch)

**Download (free registration at nuscenes.org, no approval wait):**
- `v1.0-trainval` (or start on `v1.0-mini`, 10 scenes, to build the pipeline fast).
- The **CAN bus expansion** (separate pack) — gives ego `vehicle_monitor`/`pose` velocity in
  m/s. The **map expansion** helps the crossing-label heuristic (lane/road polygons).
- `pip install nuscenes-devkit` to load everything.

**Key mismatches vs our PIE contract (handle explicitly):**
| Our contract | nuScenes | Adaptation |
|---|---|---|
| 2-D box `[x1,y1,x2,y2]`, 1920×1080 front cam | 3-D boxes + 6 cameras; **key boxes at 2 Hz** | use the **front camera** `CAM_FRONT`; project 3-D ped box → 2-D; rescale to 1920×1080; interpolate boxes to a fixed fps |
| 30 fps windows | keyframes 2 Hz (PePScenes had augmented to 10 Hz — that augmentation is gone with the repo) | **resample/interpolate ped tracks + speed to a common fps** (e.g. 10 Hz); re-express obs_len/TTE in *seconds*, not frames (obs 0.5 s, TTE 1–2 s), then convert to whatever fps you resample to — do NOT blindly reuse "16 frames" |
| per-frame OBD speed | CAN-bus velocity (m/s), ~50 Hz | resample to the track fps; **unit-reconcile to PIE's speed scale** and standardize with **PIE** train stats |
| binary crossing label | **none** | **self-derive (below)** |

**Self-derived crossing label (disclose this rule verbatim in the paper):** for each
pedestrian track that appears in `CAM_FRONT` ahead of the ego, use nuScenes 3-D ego pose +
ped 3-D trajectory + (optionally) the map's drivable/lane polygons: label **cross = 1** if the
pedestrian's future path **intersects the ego's forward corridor within the TTE horizon**
(a lateral crossing of the ego lane), else **0** (yield/parallel). Cross-check against a
simple kinematic rule (lateral velocity sustained toward the ego path over the horizon).
**Hand-audit a random 50–100 labels** and report the agreement rate — this is what keeps the
derived-label approach credible.

**Model/eval:** load each PIE-trained F1 model + its **PIE** norm stats; run zero-shot on the
nuScenes-derived test sequences; report **AUC/PR-AUC first** (threshold-free, robust to the
different base rate), then F1/Acc at (a) τ=0.5 and (b) the PIE-val τ\* carried over (never
re-tuned on nuScenes). 5 seeds → per-seed-mean + ensemble.

---

## 4. Honest caveats (for the paper, either track)

- **Track A cannot test ego-speed** — JAAD lacks it. State plainly; cite "Diving Deeper" for
  JAAD's ego-motion limitation. The bbox-only transfer is still a real generalization result.
- **Track B labels are self-derived** — not a published benchmark; report AUC/PR-AUC primarily
  and the label hand-audit; frame as a probe, not a leaderboard number.
- **Domain shift is the point** — expect a drop (different city/camera/rig). Frame as
  *generalization*, lead with threshold-free metrics, and do **not** re-tune the threshold on
  the test dataset.
- **Ego-speed transfer (Track B) may partly reflect speed domain-shift**, not intent modeling
  (nuScenes is a different vehicle/city). This is the same honest caveat as the paused PSI plan.

---

## 5. Deliverables

`journal_prep/cross_dataset_validation/`:
- **Track A:** `01_build_jaad_sequences.py`, `02_jaad_leakage_audit.md`,
  `03_jaad_fourfamily.{py,md,csv}` (+ optional `04_pie2jaad_transfer.md`).
- **Track B (if pursued):** `01_build_nuscenes_sequences.py` (+ label-derivation +
  hand-audit report), `02_eval_pie_on_nuscenes.{py,md}`.
- A results block appended to `journal_prep/Analysis/model_comparison.md` and a 1-paragraph
  "Cross-dataset generalization" subsection drafted for the paper's Results.

---

## 6. Next action (for the executor)

1. **Decide Track A vs B** (default: A). 
2. **Track A:** clone JAAD, adapt `02_build_sequences_clean.py` to JAAD's parser, run the
   leakage audit first (fast win + confirms the anchor generalizes), then the four-family
   engine runs (CPU, F1-first, 5 seeds).
3. **Track B:** register + download nuScenes `v1.0-mini` + CAN bus, stand up the devkit,
   build the 2-D/speed/label pipeline on mini, hand-audit labels, then scale to trainval.
4. Keep the **frozen protocol** (train-only norm, test-once, F1-first, cluster bootstrap for
   CIs) identical to the PIE work so the cross-dataset numbers are directly comparable to
   `journal_prep/Analysis/`.

# PLAN — PSI 2.0 (2023) cross-dataset test of the PIE-trained models (OW 16/32/64)

**Goal (user directive, 2026-07-19):** train + validate on the **full PIE** dataset, use **PSI 2.0
(2023)** as an external **test** set, for observation windows **16, 32, 64**. Start with the
**vanilla RNN** (F1-optimised). Decisions locked: **5-D faithful** feature set (real
`[x1,y1,x2,y2,vehicle_speed]` model, PSI speed upsampled to 30 fps); download attempted now.

This is a **zero-shot cross-dataset generalization** test (train PIE → test PSI). Expect a domain
drop; that is the point — it measures how well the PIE-trained predictor transfers.

## 0. Status — BLOCKED on data access (needs the user)

The PSI 2.0 Google Drive links are dead (`~~[[Google Drive]()]~~` in both repos). Download is only
via the PSI homepage → an **access-request Google Form**
(`docs.google.com/forms/d/e/1FAIpQLSfuzL_3E8pGEU0xI0pnRfX15fGqUgks4XVu2ClPQ8V05oU0Cg`). Automated
download is therefore impossible — it requires the user's affiliation + license agreement. **User
must submit the form and drop the annotation zips into `psi_crosstest/PSI2.0/`.** We only need the
annotations (not the videos): `PSI2.0_TrainVal` + `PSI2.0_Test` → `cv_annotation/` (bboxes),
`cognitive_annotation_key_frame/` (intent), and the vehicle-speed/GPS file (for the 5-D test).

## 1. PSI 2.0 ↔ PIE compatibility (confirmed from the PSI repos)

| Our contract | PSI 2.0 | Adaptation |
|---|---|---|
| bbox `[x1,y1,x2,y2]`/frame | `cv_annotations.bbox = [[xtl,ytl,xbr,ybr],…]`, tracked | direct; rescale coords (below) |
| 30 fps window | 30 fps video | none — 16/32/64 frames = same 0.53/1.07/2.13 s |
| binary crossing label | `cognitive_annotations[a].intent ∈ {cross,not_sure,not_cross}`; vote cross=1/not_sure=.5/not_cross=0; **avg ≥ 0.5 ⇒ Crossing** | adopt PSI's documented rule verbatim |
| per-frame `vehicle_speed` | speed at **1 fps** (separate GPS/vehicle file) | **upsample 1 Hz → 30 fps** (hold-last), reconcile units to PIE stats |
| 1920×1080 pixels | **unconfirmed** (likely 1280×720) | rescale boxes by (1920/W, 1080/H) to PIE coord space — **verify on real data** |
| TTE-anchored windows | intent extended forward from key-frames | adopt PSI's observe-N→intent framing (their benchmark protocol) |

Known data gaps to skip: missing cv annotation for video_0064, 0103, 0155, 0162, 0167, 0173, 0201.
PSI2.0 official split (we use only the **test** portion): Test = Video_0147–0204.

## 2. Method

**Train side (PIE, full dataset):** retrain **RNN-F1** (`birnn`, lr1e-4 do0.2 h256 nl2, pw held) on
**all six PIE sets** re-split into train/val (PSI is now the test, so no set03 holdout). Rebuild PIE
clean sequences at OW 16/32/64 over all sets; keep the frozen engine + train-only z-score. Save the
model **and its PIE train-set norm_mean/std** — PSI features are standardized with the **PIE** stats
(a test set must never re-fit normalization).

**Test side (PSI):** `01_build_psi_sequences.py` →
1. Parse `cv_annotation` per target pedestrian → per-frame bbox; **rescale** to 1920×1080.
2. Parse extended `cognitive_annotation` → per-frame `avg_intent_vote` → binary label (≥0.5).
3. Attach ego `vehicle_speed`, upsampled 1 Hz→30 fps (hold-last), unit-reconciled to PIE.
4. Slide OW-frame windows (PSI test stride 1.0, no overlap) → `X_psi (N,OW,5)`, `y_psi (N,)`.

**Evaluate:** load PIE-trained RNN-F1 + PIE norm; probs on PSI; report AUC/PR-AUC (threshold-free,
the primary cross-domain metrics), plus F1/Acc at (a) τ=0.5 and (b) τ\* carried over from PIE val
(NOT re-tuned on PSI). 5 seeds → per-seed-mean + ensemble. Compare OW 16 vs 32 vs 64.

## 3. Risks / honest caveats (for the paper)

- **Ego-speed is our dominant feature** and PSI's is 1 Hz, different vehicle/units, in a different
  city. The 5-D transfer number may partly reflect *speed domain-shift*, not intent modeling. If PSI
  ships no per-frame speed at all, 5-D is impossible → fall back to bbox-only. (This is the main
  reason a bbox-only arm is worth adding later.)
- **Label-definition shift:** PSI = annotated crossing *intent*; PIE (training) = crossing *event*.
  Related targets, not identical.
- **Resolution/camera** differ → coordinate rescale is an approximation (assumes similar FoV).
- Frame something as **generalization**, report AUC/PR-AUC first (robust to base-rate + threshold
  shift), and do NOT re-tune the threshold on PSI.

## 4. Deliverables

`01_build_psi_sequences.py` (PSI → tensors), `02_train_pie_full.py` (RNN-F1 on full PIE, OW 16/32/64),
`03_eval_on_psi.py` (zero-shot eval), results JSON/CSV, and a cross-dataset row block appended to
`journal_prep/Analysis/model_comparison.md`. Extend to the other three families once the RNN pilot
validates the pipeline.

## 5. Next action

USER: submit the PSI request form, download the **annotations** (+ vehicle-speed file), unzip to
`psi_crosstest/PSI2.0/`. THEN: I verify the real schema (resolution, speed field/units), finish the
build script against actual files, retrain RNN-F1 on full PIE, and run the OW 16/32/64 PSI test.

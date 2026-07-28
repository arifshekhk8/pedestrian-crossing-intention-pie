# Plan — Figure 9: the model actually detecting crossing intention

**Created 2026-07-28. Status: DONE 2026-07-28 (Option A, two success panels).**

> Figure 9 is in the manuscript and Video S1 is cut. The record of what was
> built and what was decided along the way is kept below; see "Outcome" at the
> end for the final state.

The manuscript currently has eight figures and every one of them is a plot. A reader
finishes the paper without ever seeing the system look at a street. For a venue like
MTI, whose subject is multimodal interaction, that is the single most valuable missing
figure. This plan produces it.

---

## 1. Decisions locked (author, 2026-07-28)

| Decision | Choice |
|---|---|
| Faces in published frames | **Blur**, and say so in the caption and the Institutional Review Board statement |
| What the figure shows | **Successes only** |
| Supplementary video | **Yes**, a short clip |
| Model to demo | **BiLSTM-F1**, the paper's headline model |

### One flag on "successes only", recorded once and then dropped

A hand-picked all-successes figure is the one place in this paper where the selection is
not blind, and a reviewer who has just read a leakage audit is primed to notice that. The
choice is fine, but it needs the accompanying paragraph to carry the honesty that the
figure no longer does. Concretely, the text introducing Figure 9 **must** point back to the
already-measured failure rates in Section 4.10: detector recall of 88%, a dominant-track
purity of 39%, and decisions flipping in 3% of windows under detector noise. The paper then
reports its failure modes quantitatively while illustrating its behaviour qualitatively,
which is a defensible split. Step 5 below writes that sentence in. If you later want a
failure panel, it is a one-row change to the scene table in Step 1 and nothing else moves.

---

## 2. Blocking correctness issue (the reason we cannot reuse what exists)

`pipeline/demo_out/` already holds ten annotated frames per clip and two rendered videos.
**None of them can go in this paper.** They were produced with
`--weights-dir paper_and_artifacts/runs/bilstm_baseline`, which is the **legacy
leaky-protocol model** — the one whose 0.931 AUC the manuscript explicitly retracts in
Section 4.2. Publishing its output beside clean-protocol results would be an inconsistency
a careful reviewer would catch, and it would undercut the paper's central argument.

Everything must be regenerated with the clean-protocol BiLSTM-F1. The existing frames stay
useful as a **layout reference and a scene-scouting shortcut** — they tell us which frames
have interesting pedestrian activity — but not one pixel of them ships.

---

## 3. Asset inventory

Confirmed present on this machine:

| Asset | Path | Note |
|---|---|---|
| Raw clips | `PIE_clips/set03/video_0012.mp4`, `video_0016.mp4` | 2.8 GB, the only two clips downloaded; both are test-set (set03), which is correct |
| Demo script | `pipeline/10_yolo_bytetrack_demo.py` | YOLO + ByteTrack + model + overlay; `draw_box()` at line 201 |
| Headline checkpoints | `f1_optimization/runs_f1/lstm_lr1e-03_do0.3_h256_nl2/pw1.682/seed{42,0,1,2,3}/` | each has `best.pt`, `norm_mean.npy`, `norm_std.npy` |
| Ensemble threshold | `f1_optimization/05_final_arms.json` → `arms.A3.ens.tau` | **τ\* = 0.5164** |
| Ground-truth labels for these clips | `journal_prep/issue10_gt_vs_detector/10_gt_vs_detector.csv` | per pedestrian per anchor: `label`, `detected`, `miou`, `purity`, `gt_prob`, `yolo_prob` |
| Detection cache | `journal_prep/issue10_gt_vs_detector/cache_dets.pkl` | keyed by `video_0012` / `video_0016`; avoids re-running YOLO over the whole clip |
| Old frames (reference only) | `pipeline/demo_out/demo_video_00{12,16}_f*.png` | legacy model; scene-scouting only |
| Figure toolkit | `MDPI_Article_Template/figures/figstyle.py` | shared palette and chrome, so Figure 9 matches Figures 1–8 |

`journal_prep/issue10_gt_vs_detector/10_gt_vs_detector.csv` is the key to this whole plan:
it is the only file that pairs these two clips with **ground-truth crossing labels**, so it
is what lets us choose scenes on evidence instead of by eye.

---

## 4. Execution steps

### Step 0 — Teach the demo script the headline model (~30 min)

`pipeline/10_yolo_bytetrack_demo.py` currently hard-codes a single h128 checkpoint and a
0.5 threshold. Three additions, all backward compatible:

1. `--hidden` (default 128) passed through to `BiLSTMIntentPredictor`, so h256 loads.
2. `--weights-dirs` accepting a comma-separated list; if more than one is given, average
   the sigmoid outputs. This gives the **5-seed ensemble**, which is the right thing to
   demo because it is the deployable predictor and it is what the confusion matrices in
   `journal_prep/Analysis/` already report.
3. `--threshold` (default 0.5) so we can pass τ\* = 0.5164.

Write it as a new script, `pipeline/11_demo_clean_ensemble.py`, rather than editing `10_`.
The numbered scripts are a historical record and `10_` is referenced from
`pipeline/CODE_STATE.md`; a new number keeps that record intact and follows the repo's
existing convention.

**Gate before proceeding:** score one known window through the new script and confirm the
probability matches the stored `gt_prob` for that pedestrian/anchor in the issue-10 CSV to
about 1e-3. If it does not, the norm-stats or the feature order is wrong and everything
downstream would be silently incorrect.

### Step 1 — Choose the scenes on evidence (~45 min)

Write `pipeline/11a_select_demo_scenes.py` that reads the issue-10 CSV and ranks candidate
frames. Target **three panels**, and prefer this order:

| Panel | What it must show | Selection rule over `10_gt_vs_detector.csv` |
|---|---|---|
| (a) | A pedestrian correctly flagged **before** stepping into the road | `label == 1`, `detected == 1`, `gt_prob` high, `miou ≥ 0.7`, and the anchor at least 30 frames before that pedestrian's `crossing_point` |
| (b) | A pedestrian near the kerb correctly **not** flagged | `label == 0`, `detected == 1`, `gt_prob` low, and the box close to the road edge — a hard negative, not someone walking away down the pavement |
| (c) | **One frame, two verdicts** | any frame containing both a `label == 1` high-probability track and a `label == 0` low-probability track |

Panel (c) is the one worth spending time on. A single frame in which the system flags one
person and clears another standing beside them says more than any pair of separate images,
because it shows the decision is about that pedestrian's dynamics rather than about the
scene. `demo_video_0012_f07766.png` (Yonge St., several pedestrians at a kerb) is a
promising place to start scouting.

The script should emit a shortlist of `(video, frame, [track ids])` with the GT label and
stored probability beside each, so the final pick is made from a table rather than by
scrolling video.

**Record the time-to-onset for panel (a).** Annotating "1.2 s before this pedestrian stepped
into the road" is what connects the picture to the paper's central methodological claim.
Without it, the panel is just a box with a number on it.

### Step 1 RESULTS (executed 2026-07-28)

`pipeline/11_demo_clean_ensemble.py --stage verify` **passed exactly**: the five-seed
h256 ensemble reproduces the published arm to |d| = 0.00e+00 on AUC, F1, and accuracy
(0.946739 / 0.855685 / 0.905444 at τ\* = 0.5164). The demo is now provably driven by the
same model Table 3 reports.

`pipeline/11a_select_demo_scenes.py` then ranked every candidate. Outcome:

**Panel A — available and strong.** Best candidate is `video_0012`, pedestrian
`3_12_747`, anchor frame **7431**: probability **0.947**, ground-truth crosser, and the
window ends **2.00 s before** the annotated crossing point, with detector IoU 0.91 and
track purity 0.56. `3_12_750` at frame 7443 is an alternative (p = 0.950, 1.73 s ahead,
IoU 0.89, purity 0.75) and has the better track.

**Panel B — available and strong.** `video_0016` pedestrian `3_16_946` at frame 12214,
probability **0.073**, IoU 0.84, purity 0.62. Runner-up `video_0012` `3_12_753` at 8507,
p = 0.076.

**Panel C — NOT possible as specified, and the reason is worth reading.**
An exhaustive search over both clips found exactly one span where a ground-truth crosser
and a ground-truth non-crosser are annotated simultaneously: `video_0016` frames
**4415–4750**, seven pedestrians (three crossers, four non-crossers). Scoring that scene
through the ensemble gives **14 false positives out of 16 windows** — the four
non-crossers all sit at 0.49–0.63 against a threshold of 0.516. The only mixed-verdict
scene either clip contains is a failure cluster, so it cannot serve a successes-only
figure.

That cluster is scientifically interesting rather than embarrassing: those pedestrians are
standing at a kerb, and the model is not wildly wrong, it is *uncertain*, straddling the
threshold. A system that leans toward "might cross" for someone waiting at a kerb is
making the cheap error rather than the expensive one. **This is now a live decision for
the author** (see §1): panel C can be dropped, or it can become the failure panel that was
originally declined, on stronger evidence than was available when that choice was made.

**A caution about the selection filter itself.** The first pass reported TP 39 / FP 0 /
FN 3 / TN 33 over the 75 "cleanly tracked" windows — a suspiciously perfect record. That
pool was biased by its own quality filter (IoU ≥ 0.65, purity ≥ 0.5), which had quietly
excluded most of the kerb-waiting cluster. The honest figure over **all 439** windows in
these two clips is **TP 134 / FP 16 / FN 19 / TN 270**, accuracy 0.920, F1 0.884 — close
to, and slightly above, the full test set (0.905 / 0.856), so the clips are broadly
representative. Quote the 439-window numbers, never the 75-window ones.

### Step 2 — Render the frames (~20–40 min compute)

```bash
source .venv/bin/activate
python pipeline/11_demo_clean_ensemble.py \
  --video PIE_clips/set03/video_0012.mp4 --video-id video_0012 \
  --start-frame <from step 1> --max-frames 240 \
  --weights-dirs "f1_optimization/runs_f1/lstm_lr1e-03_do0.3_h256_nl2/pw1.682/seed42,\
f1_optimization/runs_f1/lstm_lr1e-03_do0.3_h256_nl2/pw1.682/seed0,\
f1_optimization/runs_f1/lstm_lr1e-03_do0.3_h256_nl2/pw1.682/seed1,\
f1_optimization/runs_f1/lstm_lr1e-03_do0.3_h256_nl2/pw1.682/seed2,\
f1_optimization/runs_f1/lstm_lr1e-03_do0.3_h256_nl2/pw1.682/seed3" \
  --hidden 256 --threshold 0.5164 \
  --dump-csv --save-frames --out-dir pipeline/demo_out_clean
```

Render **raw frames plus a per-frame CSV**, not a burned-in overlay. The overlay in `10_`
uses OpenCV's default font at 0.6 scale with 2 px boxes, which looks like a debug view and
will not survive being scaled into a two-column journal page. Step 4 draws the annotation
in matplotlib instead, at print quality and in the same visual language as Figures 1–8.

Run it on **both** clips so panel selection is not constrained to one scene.

### Step 3 — Blur faces (~30 min)

`pipeline/11b_blur_faces.py`: run a face detector (OpenCV DNN face detector, or YOLO
restricted to the upper third of each pedestrian box as a fallback) over only the chosen
frames, then apply a Gaussian blur with a kernel large enough that the region is
unrecoverable, not merely softened.

Two checks before moving on:
- **Look at every blurred frame at full resolution.** An unblurred bystander in the
  background is the failure mode here, and it is invisible at thumbnail size.
- Confirm the blur does not overlap the bounding boxes or labels in a way that obscures the
  result being shown.

Keep the unblurred renders out of the repository entirely: add `pipeline/demo_out_clean/raw/`
to `.gitignore`. Blur before anything is committed, not after.

### Step 4 — Build the publication figure (~1–1.5 h)

`MDPI_Article_Template/figures/make_fig9_qualitative.py`, importing `figstyle.py` so the
colours, ink tokens, and type match every other figure in the paper.

Design:
- **Three panels**, stacked or 2+1, at `\textwidth`.
- Bounding boxes drawn in matplotlib, 1.5 pt, using the palette's accent blue for "will
  cross" and the de-emphasis grey for "will not cross". **Not** a red/green pair: the
  existing overlay's red/orange scheme fails colour-vision-deficiency checks and clashes
  with the palette validated for the other eight figures.
- Each box labelled with the probability, in ink tokens rather than the box colour, per the
  rule already followed throughout.
- Panel (a) annotated with the time to crossing onset.
- A small caption strip under each panel giving clip, frame, and ego speed, since ego speed
  is the paper's dominant feature and showing it makes the two-stream input visible.
- No legend box if the labels carry identity; a two-entry legend if they do not.

Then **render it and look at it** before wiring it into the manuscript, exactly as was done
for Figures 2, 4, 6, and 8, each of which had a collision that only appeared on inspection.

### Step 5 — Manuscript integration (~45 min)

1. **Where it goes.** A new subsection 4.11, *Qualitative behaviour*, placed immediately
   after 4.10 (Detector-in-the-Loop Robustness). That order matters: 4.10 establishes the
   quantitative failure rates, so the qualitative figure arrives already framed by them.
2. **The honesty sentence** (see §1 above). Something close to:
   > These examples are illustrative rather than representative; the corresponding failure
   > rates are the ones quantified in Section 4.10, where detector recall is 88% and the
   > decision flips in 3% of windows under detector box noise.
3. **Caption** must state: the clean crossing-point protocol, the BiLSTM-F1 five-seed
   ensemble at τ\* = 0.5164, that boxes come from YOLO and ByteTrack rather than
   annotations, and that faces were blurred.
4. **Institutional Review Board statement** — currently "Not applicable". Extend it to note
   that the published frames come from the public PIE dataset and that faces were blurred
   prior to publication.
5. Add a `\ref{fig:qualitative}` pointer from Section 3.9 (the live pipeline) so the reader
   knows the picture is coming.
6. Recompile with `tectonic main.tex` and confirm 0 undefined references.

### Step 6 — Supplementary video (~30 min)

The existing renders are 67–160 MB, far past what a submission system will take.

```bash
ffmpeg -i pipeline/demo_out_clean/demo_video_0012.mp4 \
       -ss <start> -t 25 -vf "scale=1280:-2" -c:v libx264 -crf 26 -preset slow \
       -movflags +faststart -an paper_and_artifacts/Journal_writing/supplementary/video_s1.mp4
```

Target: 20–30 s, 1280 px wide, **under 20 MB**. Faces must be blurred in the video too,
which means running Step 3 over every frame of the chosen segment rather than over three
stills. Budget for that. Then add a Supplementary Materials block to `main.tex` describing
Video S1 and referencing it from Section 4.11.

---

## 5. Order of work and effort

| Step | Output | Effort |
|---|---|---|
| 0 | `11_demo_clean_ensemble.py` + parity gate | 30 min |
| 1 | `11a_select_demo_scenes.py` + scene shortlist | 45 min |
| 2 | clean frames + CSV, both clips | 20–40 min compute |
| 3 | `11b_blur_faces.py` + verified blurred frames | 30 min |
| 4 | `make_fig9_qualitative.py` + `fig9_qualitative.pdf` | 1–1.5 h |
| 5 | §4.11, caption, IRB update, recompile | 45 min |
| 6 | `video_s1.mp4` + Supplementary Materials block | 30 min |

Roughly **half a day**. Steps 0–2 are the ones that can surprise you; 4–6 are mechanical.

---

## 6. Things that could go wrong

- **The parity gate in Step 0 fails.** Most likely cause is feature order or norm stats. Do
  not proceed past it; every downstream artefact would be quietly wrong.
- **No frame satisfies panel (c).** Both clips are short. Fall back to two separate panels
  and drop (c) rather than staging something that is not in the data.
- **ByteTrack fragments the chosen track.** Section 4.10 already measured dominant-track
  purity at 39%, so this is likely, not merely possible. Choose a scene where the track is
  stable across the full 16-frame window; the selection script should check that and reject
  candidates that fail it.
- **Only two clips exist locally.** `PIE_clips/` holds only set03's `video_0012` and
  `video_0016`. If neither yields a good panel, more clips must be downloaded, and the York
  host throttles hard — see `PROGRESS_LOG.md` Phase 4 for the parallel-range-download
  method. Budget a day if it comes to that.
- **Scope creep into a failure panel.** Deliberately excluded per the decision above. If it
  gets added later, it is one row in the Step 1 table and one panel in Step 4.

---

## 7. Definition of done

- [ ] Figure 9 renders from a script, from the clean BiLSTM-F1 ensemble, with no legacy-model pixels
- [ ] Every face in every published frame is blurred, verified at full resolution
- [ ] Section 4.11 written, including the sentence pointing at the Section 4.10 failure rates
- [ ] IRB statement updated to cover the blurring
- [ ] `main.pdf` recompiles with 0 undefined references
- [ ] `video_s1.mp4` under 20 MB, faces blurred, Supplementary Materials block added
- [ ] `PLAN.md` and `README.md` status blocks updated to say nine figures


---

## 8. Outcome (2026-07-28)

Option A was chosen: two panels, both correct calls.

**Figure 9** is `MDPI_Article_Template/figures/fig9_qualitative.pdf`, generated by
`make_fig9_qualitative.py`. Boxes come from YOLO26 + ByteTrack and probabilities from the
five-seed BiLSTM-F1 ensemble at τ\* = 0.5164:

| Panel | Scene | Result | det IoU | track purity |
|---|---|---|---|---|
| (a) | video_0016 frame 4345, pedestrian `3_16_942` | correctly flagged at *p* = 0.71, **1.5 s before** the annotated crossing point | 0.83 | 0.88 |
| (b) | video_0012 frame 523, pedestrian `3_12_725` | a worker at the kerb beside a crosswalk, correctly left at *p* = 0.31 | 0.87 | 0.88 |

The generator **asserts** that each panel's prediction matches its ground-truth label, so
the figure cannot silently drift into showing a failure if a checkpoint or threshold ever
changes.

The pairing is the argument. Both panels show a person standing at a kerb beside a marked
crosswalk, which is what a position-based rule would flag, and the verdicts are opposite.
What separates them is the motion history in the 16-frame buffer, not where anyone stands.
Drawing the panels from different clips also makes the sentence "across the two clips shown
here", which introduces the 439-window confusion counts, literally true.

**Section 4.11 (Qualitative Behaviour)** was added after the detector-in-the-loop section,
so the reader meets the pictures already knowing the measured error rates. It states
plainly that the frames are illustrative rather than representative, repeats the Section
4.10 rates, and gives the full confusion counts for these two clips
(134/16/19/270 over 439 windows, accuracy 0.920, F1 0.884).

**Video S1** is `supplementary/video_s1.mp4`: 26 s, 1280 px wide, 3.3 MB, faces blurred,
declared through the MDPI `\supplementary{}` block.

**Privacy.** `blur_heads()` runs the detector at confidence 0.05, far below the 0.3 used
for the results, and blurs the top 24% of every person box. A spurious box costs a blurred
patch of pavement; a missed one costs a published face. Both panels were inspected at full
resolution, and the first attempt at confidence 0.25 was rejected because background
pedestrians near the shopfronts came through unblurred. The same pass runs per frame in
the video via `--blur-faces`.

**Candidate renders on disk.** `pipeline/demo_out_clean/` (gitignored) is pruned to the
seven frames any candidate panel needs: the four protocol anchors of `3_16_942`
(video_0016 f4329/4337/4345/4353, so the lead time can be changed without re-rendering)
plus the two runner-up scenes, video_0012 f5706 and f6889. Switching panels is a one-line
edit to `PANELS`, not a re-run of the detector.

### Things that went wrong, recorded so they are not repeated

**The first scene selection optimised for the wrong thing.** It ranked candidates on
confidence and detector quality and never checked whether the pedestrian was *visible*.
The top pick had a box 28 × 88 px, a speck at journal scale. Box height is now a first-class
criterion, and it turns out to trade against confidence: the model is most confident about
distant pedestrians at crosswalks and least confident about close ones.

**A scene was rejected for a miss that was not a miss.** Frame 6889 was dropped because
`3_12_744`, a large foreground pedestrian, appeared to be misclassified there. That reading
was wrong. 744's last protocol window is anchored at 6826, sixty-three frames (2.1 s)
earlier, and by 6889 they are already in the road. The protocol makes no call about them at
that instant, because mid-crossing is precisely the case Section 3.3 excludes. Judging a
frame by a window anchored seconds away is not a fair test. **The rule to use: only count a
pedestrian as right or wrong at a frame if a window is anchored within ±1 s of it;
otherwise mark them "no live call" and move on.** Applying that rule, every one of the ten
shortlisted scenes is clean, so the choice was about legibility, not correctness.

**Metrics picked a panel that looked bad.** Track purity (the fraction of the 16-frame
window owned by the dominant ByteTrack ID) is the right technical criterion, and on it
video_0012 f6889 wins outright at 0.94 against 0.62 for the scene then in the paper. It was
selected on that basis and rendered, and the picture was worse: a large unrelated pedestrian
fills the right half of the frame and the eye lands on them instead of the subject.
video_0016 f4345 keeps most of the tracking advantage (0.88) and is legible, at the cost of
0.5 s of lead time. **Render the candidate and look at it before committing; a metric that
is invisible in the image cannot outvote what the image actually shows.**

**Face blur read as a censor bar.** Blurring a hard-edged rectangle over the top 24% of a
close-up person box produces a dark block that drags the eye straight to it. `blur_heads()`
now blurs a patch 12% larger than the head box and feathers the blur back to sharp across
that added margin only, so the head box itself is still fully blurred and nothing about the
privacy guarantee weakens.

### If the supervisor asks for the failure panel

The evidence is already gathered: video_0016 frames 4415–4750 hold the only mixed-verdict
scene in either clip, and the model produces 14 false positives out of 16 windows there,
with the four non-crossers scoring 0.49–0.63 against a threshold of 0.516. It is a
cluster of pedestrians waiting at a kerb, and the model is uncertain rather than wrong.
Adding it means one more entry in `PANELS` in `make_fig9_qualitative.py`, relaxing the
success assertion for that panel, and a paragraph in Section 4.11.

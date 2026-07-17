# journal_prep/ — Reviewer-Readiness Work

Everything here fixes issues a journal reviewer would raise *before* submission.
It is **separate from the original thesis pipeline** in `pipeline/` (numbered scripts
+ `paper_and_artifacts/runs/`), which is left as the historical record
(`paper_and_artifacts/supervisor_review/` still describes it).

**Master issue list + status:** [`PLAN.md`](PLAN.md) (12 issues, 🔴/🟠/🟡 severity).

**Metric hierarchy (supervisor directive, 2026-07-12): F1 → accuracy → AUC.** The
F1-first optimization lives in top-level `f1_optimization/` (LSTM F1 0.828 → 0.844,
significant; families TIE on F1); issues 3/7/8 below carry metric-conditional notes
where their AUC-based verdicts differ under F1.

## Status at a glance (updated 2026-07-13)

| Issue | Topic | Status | Headline outcome |
|---|---|---|---|
| 1 | Temporal leakage audit | ✅ done | 🔴 leakage found: 67.9% of crossers observed mid-crossing → old AUC 0.931 inflated |
| 2 | Canonical leak-free protocol + retrain | ✅ done | clean AUC **0.932 ± 0.011** (5-seed); leak fix barely moved AUC (methodological win); **ego-speed dominant +0.18** (bbox-only collapses 0.889→0.753); attention no benefit (0.925) |
| 3 | Published-baseline comparison | ✅ done (internal) | finalized now all evidence exists: ours = **top AUC 0.932 [0.92–0.95]**, mid Acc, **2 streams, 0.575 ms/window**; positioning matrix carries a measured number in every cell. Only external PIP-Net-split verify + BibTeX remain (`issue3_baseline_comparison/`) |
| 4 | Bootstrap CIs on test AUC | ✅ done | baseline AUC 0.932, **95% CI ≈ [0.92, 0.95]**, PR-AUC 0.876 (`issue4_bootstrap_ci/`); ego-speed gap statistically unambiguous |
| 5 | Leave-one-set-out CV | ✅ done | **6-fold AUC 0.928 ± 0.041** (`issue5_loso_cv/`); set03 representative at 0.931 — answers "is set03 easy?" |
| 6 | Multi-seed window/TTE ablations | ✅ done | window **insensitive** (0.931/0.933/0.937; spread 0.006 < seed noise 0.007 = equivalent — old claim confirmed); TTE **significant decline** (0.960/0.948/0.919 @1.0/1.5/2.0s, every p≤0.008 — old "insensitive to TTE" OVERTURNED, was a leakage artifact), **confirmed on a matched cohort** (06b: same peds at all 3 horizons, sample effect ≤0.002) (`issue6_window_tte_ablation/`) |
| 7 | Hidden-size ablation | ✅ done | hidden 64/128/256 → 0.927/0.933/0.938 (5-seed); **hidden=128 justified** — 256 nominally +0.0045 but **n.s.** (p=0.34) at 3.8× params, mild non-sig. trend, kept as accuracy/cost compromise (`issue7_hidden_size/`) |
| 8 | Hyperparameter grid search | ✅ done | full 36-config grid, **val-only selection + test touched once**; **search confirms the hand-set baseline** — val-winner (lr1e-4/do0.2/h256) beats it on test by Δ+0.0006 (p=0.91, n.s.) at 3.8× params, so baseline retained; hyperparams now documented (`issue8_grid_search/`) |
| 9 | Isolated BiLSTM latency | ✅ done | **BiLSTM = 0.575 ms/window** (CPU, ~58× inside 30fps budget); CPU beats MPS at batch 1 (GPU dispatch overhead); pipeline **detection-bound** — YOLO26-M 33.7ms (93%) vs BiLSTM 1.6ms (4.5%) → 27.5 fps (`issue9_latency/`) |
| 10 | GT-box vs YOLO-box AUC drop | ✅ done | **prediction robust to box noise** — GT 0.962/0.958 vs YOLO 0.953/0.948 (drop +0.009/+0.010, 3% decision flips, N=98 peds); weak links are perception — detector recall 88%, **ByteTrack fragments badly** (track purity 39%, 59% competing-ID) (`issue10_gt_vs_detector/`) |
| 11 | Doc cleanup to match reality | ✅ done | THESIS_PLAN (file-numbering + deferred hidden-size), CODE_STATE (journal_prep Issues 1–10 entries), PROGRESS_LOG (softened demo "AUC 1.000 N=10" → superseded by Issue 10; headline = clean 0.932) |
| 12 | Unified pipeline + F1-first integration | ✅ done | **ONE model-agnostic engine** (bilstm/transformer/gru/birnn) with equivalence gates ALL PASS; single-engine CPU replication: LSTM F1 improvement + family TIE **replicate**; pedestrian-cluster bootstrap: all verdicts survive; issue3 table corrected vs sources (PedFormer 0.87 F1 split from BiPed; PIP-Net removed — custom split) (`issue12_unified_pipeline/`) |

**2026-07-13 audit sweep (multi-agent judge + inline fixes):** all issue scripts
py-compile again (post-reorg paths repaired); issues 7/8 marked metric-conditional;
issue 10's YOLO row relabeled "oracle-matched"; determinism claims scoped
(CPU context-free; LSTM-on-MPS process-history-dependent); cluster CIs added.

## Folder map

```
journal_prep/
├── PLAN.md                       master plan + per-issue DONE blocks (read first)
├── README.md                     this index
├── issue1_leakage_audit/         Issue 1 — proof the old 0.931 leaked
│   ├── 01_leakage_audit.py         audit script (now has --seq-dir/--out-dir flags)
│   ├── 01_leakage_report.md        VERDICT: LEAKAGE FOUND
│   ├── cross_state_map.pkl         recovered per-frame `cross` ground truth (cached)
│   ├── leakage_per_sequence.csv    per-sequence audit
│   └── figures/
├── issue2_clean_protocol/        Issue 2 — leak-free dataset + retrain (see its README)
│   ├── README.md
│   ├── 02_build_sequences_clean.py / 02_leakage_report_clean.md
│   ├── 03_eval_parity_check.py     / 03_eval_parity_report.md
│   ├── 04_multiseed_baseline.py    / 04_multiseed_summary.md + .csv
│   ├── 05_variant_comparison.md
│   ├── 06_multiseed_variants_kaggle.ipynb   ← run on Kaggle (bbox-only + attention)
│   ├── 06b_local_verify_seed42.py  local cross-check (CPU/MPS)
│   ├── sequences_clean/            X.npy, y.npy, meta.pkl  (upload these to Kaggle)
│   ├── runs_clean/                 single-seed checkpoints + final.json
│   └── kaggle_result/              clean multi-seed variant outputs + checkpoints
├── issue3_baseline_comparison/    Issue 3 — published-baseline table (parked; finalize at end)
│   ├── README.md
│   ├── 03_baseline_comparison.md   DRAFT comparison table + framing (figures verified)
│   └── 04_positioning_vs_prior_work.md   their-limitation → our-response matrix (seed)
├── issue4_bootstrap_ci/           Issue 4 — bootstrap 95% CIs on test AUC ✅
│   ├── README.md
│   ├── 04_bootstrap_ci.py
│   └── 04_bootstrap_ci_results.md / .csv
├── issue5_loso_cv/                Issue 5 — leave-one-set-out CV ✅
│   ├── README.md
│   ├── 05_loso_cv.py
│   └── 05_loso_results.md / .csv
└── issue6_window_tte_ablation/    Issue 6 — multi-seed window + TTE ablations ✅
    ├── README.md
    ├── 06_multiseed_ablations.py        self-contained MPS harness (build + 30 trainings)
    ├── 06_multiseed_ablation_summary.md / 06_window_multiseed.csv / 06_tte_multiseed.csv
    ├── 06_ablation_figure.png           window (flat) vs TTE (declining) AUC
    ├── 06b_matched_track_tte.py         matched-cohort TTE control (removes nested-sample confound)
    ├── 06b_matched_tte_report.md / 06b_matched_tte_results.csv / 06b_matched_tte_figure.png
    ├── sequences/<config>/ + sequences_matched/<cfg>/   per-config + matched-cohort X/y/meta
    └── runs/<config>/ + runs_matched/<cfg>/             per-run metrics
├── issue7_hidden_size/            Issue 7 — hidden-size ablation {64,128,256} ✅
│   ├── README.md
│   ├── 07_hidden_size_ablation.py       MPS harness (15 trainings), reports param counts
│   ├── 07_hidden_size_results.md / 07_hidden_size_results.csv / 07_hidden_size_figure.png
│   └── runs/h<H>/seed<k>.json           per-run metrics
└── issue8_grid_search/            Issue 8 — hyperparameter grid search ✅ (supervisor-requested)
    ├── README.md
    ├── 08_grid_search.py                full grid + val-only selection + test-once
    ├── 08_grid_full.csv / 08_candidates_multiseed.csv
    ├── 08_grid_search_summary.md / 08_grid_search_figure.png
    └── runs_grid/<cfgid>/ (val-only) + runs_final/<cfgid>/ (with test)
├── issue9_latency/                Issue 9 — isolated inference-latency benchmark ✅
│   ├── README.md
│   ├── 09_inference_latency.py          BiLSTM + YOLO + pipeline timing (--report-only to rebuild)
│   ├── 09_latency_report.md / 09_latency_results.json / 09_latency_figure.png
└── issue10_gt_vs_detector/        Issue 10 — GT-box vs YOLO-box degradation ✅
    ├── README.md
    ├── 10_gt_vs_detector_auc.py         segment YOLO+ByteTrack, IoU match, dual scoring
    ├── 10_gt_vs_detector_results.md / .csv / 10_gt_vs_detector_figure.png
    └── cache_dets.pkl                   cached YOLO detections (re-run matching without YOLO)
```

## The one-paragraph story so far

The headline AUC 0.931 was inflated: 68% of crossing pedestrians were already
mid-crossing inside the observation window (Issue 1). Rebuilding the dataset
anchored at PIE's own `crossing_point` with a canonical TTE∈[30,60] sliding
window removes 100% of that leakage and grows N from 1,389 → 4,906 (Issue 2).
The clean 5-D model still scores **0.932 ± 0.011** — so the leak fix is a
*methodological* win (genuine pre-onset prediction, believable epoch ~17), not a
deflated number. The multi-seeded variants show **ego-vehicle speed is the
dominant predictor (+0.18 AUC)**: drop it and the bbox-only model collapses
0.889 → **0.753 ± 0.020**; temporal attention adds nothing on clean data (0.925).
That clean, parity-verified 0.932 is what goes into the baseline-comparison table
next (Issue 3).

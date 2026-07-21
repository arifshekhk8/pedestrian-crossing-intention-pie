# Handover prompt — paste this into a NEW Claude Code session

> Copy everything in the fenced block below into a fresh session (run from the repo root,
> `/Users/arif/Developer/pedestrian-thesis`). It is self-contained.

```
I'm continuing a master's-thesis → MDPI MTI journal-paper project in this repo
(/Users/arif/Developer/pedestrian-thesis). This session has ONE job: run a CROSS-DATASET
VALIDATION of our PIE-trained pedestrian crossing-intention models, to answer the "single
dataset (PIE only)" reviewer risk. Read this briefing, then the files I list, then propose
first steps — don't start heavy work until I confirm.

## What the project is (1 paragraph)
We predict pedestrian crossing intention on the PIE dataset from a 16-frame (0.5 s) window of
two cheap streams — the pedestrian bounding box + the ego-vehicle OBD speed — with four model
families (BiLSTM / Transformer / GRU / vanilla RNN) trained through ONE unified engine. Our
distinctive contributions are: (1) a temporal-leakage audit + a leakage-free protocol
(re-anchored at PIE's `crossing_point`, 0% verified leakage), (2) full statistical rigor
(bootstrap + pedestrian-cluster CIs, LOSO, multi-seed), and (3) a four-family isolation showing
the INPUT signal — not the architecture or its gating — carries the task. Metric hierarchy is
F1 → accuracy → AUC (supervisor directive).

## The task this session
Execute `journal_prep/cross_dataset_validation/PLAN.md` (READ IT FIRST — it is the authoritative
plan). It has two tracks; default to **Track A (JAAD)** unless I say otherwise:
- Track A (recommended, low-risk): replicate the leakage audit + the four-family "input matters"
  isolation on JAAD (bbox-only; JAAD has NO ego-speed). Downloadable now.
- Track B (stretch): nuScenes + CAN-bus ego-speed with SELF-DERIVED crossing labels, to test the
  full 5-D model including ego-speed. Higher effort + label-quality risk.

## Access reality (already verified — do NOT waste time re-checking)
- PSI 2.0: BLOCKED (access form, no reply) — paused.
- PePScenes (nuScenes crossing labels): DEAD — repo deleted, no Wayback, no author mirror.
- JAAD: available (github.com/ykotseruba/JAAD), but NO ego-vehicle speed (only 5 coarse
  driver-motion states).
- nuScenes + CAN bus: downloadable (free registration), real ego velocity, but NO native
  crossing label (must self-derive).

## Read these first (in order)
1. journal_prep/cross_dataset_validation/PLAN.md      — the plan (both tracks, concrete steps)
2. paper_and_artifacts/Journal_writing/ProjectDescription.md — the whole project in one file
   (dataset, contract, protocol, all numbers, the four families)
3. journal_prep/issue2_clean_protocol/02_build_sequences_clean.py — the leak-free builder to adapt
4. journal_prep/issue1_leakage_audit/01_leakage_audit.py          — the leakage audit to re-run
5. journal_prep/issue12_unified_pipeline/12_unified_engine.py     — THE training engine (all 4
   families; --family bilstm|transformer|gru|birnn, --select f1, --device cpu)
6. journal_prep/Analysis/model_comparison.md          — the PIE results block to append to

## Hard constraints (keep identical to the PIE work, or the comparison isn't fair)
- Frozen protocol: train-only z-score normalization; test touched ONCE; F1-first selection
  (--select f1); 5 seeds [42,0,1,2,3]; pedestrian-cluster bootstrap for CIs.
- Never re-tune the decision threshold on the test dataset; carry PIE-val τ* over, and lead the
  cross-domain numbers with AUC/PR-AUC (threshold-free), then F1/Acc at fixed threshold.
- Recurrent families must train on CPU for bit-reproducibility (nn.LSTM on MPS is
  process-history-dependent — our own Issue-12 finding).
- Environment: `source .venv/bin/activate` (torch 2.12, sklearn 1.9, scipy present). Before any
  run that could exceed ~30 min, give me a time estimate and ask first.

## What to do first (immediate)
Confirm you've read PLAN.md + ProjectDescription.md. Then for Track A: propose the exact steps to
(a) clone JAAD + adapt the clean-protocol builder to JAAD's parser, (b) run the JAAD leakage audit
first (fast win — does our anchor generalize?), (c) then the four-family unified-engine runs
(CPU, F1-first, 5 seeds). Give me a time estimate for each before running. Do not start downloads
or training until I confirm the plan.
```

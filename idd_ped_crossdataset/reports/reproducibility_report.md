# Reproducibility report — IDD-PeD cross-dataset validation

Machine-readable companion: `manifests/experiment_manifest.json` (also mirrored to
`reports/experiment_manifest.json`).

---

## 1. Provenance of everything consumed

### Dataset

| | |
|---|---|
| Name | IDD-PeD |
| Paper | *Pedestrian Intention and Trajectory Prediction in Unstructured Traffic Using IDD-PeD* — Bokkasam, Gangisetty, Hafez & Jawahar, **ICRA 2025** (arXiv 2506.22111) |
| Project page | https://cvit.iiit.ac.in/research/projects/cvit-projects/iddped |
| Code repo | https://github.com/Ruthvik9/IDD-PeD |
| Licence | **CC BY 4.0** — no access form, no registration |
| Downloaded | 2026-08-25, from the official CVIT host |

| file | bytes | SHA-256 |
|---|---|---|
| `annotations.tar` | 478,209,024 | `bf4bea904753b1e3b26a1543bc9e37e757e3a088eb6c92a9e9cc384672e01a35` |
| `annotations_vehicle.tar` | 58,593,280 | `f5b018bf52162b61317246ff7bb0e21ac5a5a7a3c53bc8571a7db7438c30434a` |

Video tars were **not** downloaded (not needed; see `PHASE9_detector_in_the_loop.md`).
There is no published checksum from the authors to compare against; ours are recorded so a
future re-download can be verified against *this* run.

**Dataset version.** IDD-PeD publishes no version string. The identifying facts of the copy
used are: 33 annotation XMLs + 34 OBD XMLs, `ddpai/` directories present but empty,
**4,916 pedestrian tracks** — which matches the authors' stated 3,284 train + 1,632 test
exactly, and is the strongest available integrity check.

### Parser

`src/iddped_parser.py` is a self-contained re-implementation of the three parsing functions
of the authors' `Intention/iddped_interface.py`. The label→scalar maps and the XML element
paths are copied verbatim from that file. Independent validation: our parse recovers exactly
the authors' published track count (4,916).

### Frozen PIE artefacts (read, never modified)

| artefact | path |
|---|---|
| training engine | `journal_prep/issue12_unified_pipeline/12_unified_engine.py` |
| BiLSTM class | `pipeline/03_bilstm_model.py` |
| Transformer class | `transformer/phase1_setup/00_transformer_model.py` |
| GRU / vanilla-RNN class | `RecurrentIntentPredictor`, defined inside the engine |
| eval / threshold / bootstrap helpers | `f1_optimization/00_common.py` |
| PIE clean sequences | `journal_prep/issue2_clean_protocol/sequences_clean/` |
| BiLSTM-F1 checkpoints | `f1_optimization/runs_f1/lstm_lr1e-03_do0.3_h256_nl2/pw1.682/seed{42,0,1,2,3}/` |
| Transformer-F1 checkpoints | `f1_optimization/runs_f1/transformer_searched/pw1.682/seed{…}/` |
| GRU-F1 checkpoints | `gru/phase4_final/runs_final/gru_f1_winner/seed{…}/` |
| Vanilla-RNN-F1 checkpoints | `rnn/phase4_final/runs_final/rnn_f1_winner/seed{…}/` |
| parity-gate reference | `journal_prep/issue2_clean_protocol/runs_clean/multiseed/seed{…}/` |

These are loaded via `importlib` as read-only modules. Where a change was structurally
required (the engine hardcodes `input_dim=5` in its builder wrappers, for the 4-D ablation),
the module's `MODEL_REGISTRY` is monkey-patched **in memory for that process only**.
`train_run()` is called unmodified. **Zero bytes were written outside
`idd_ped_crossdataset/`.**

---

## 2. Parity gate

Before any IDD-PeD number was produced, the frozen BiLSTM's per-seed PIE **test** AUC was
regenerated from its checkpoints and compared with the stored `final.json` values:

```
seed 42: recomputed 0.913114  stored 0.913114  |Δ| 0.00e+00
seed  0: recomputed 0.933424  stored 0.933424  |Δ| 0.00e+00
seed  1: recomputed 0.943189  stored 0.943189  |Δ| 0.00e+00
seed  2: recomputed 0.936295  stored 0.936295  |Δ| 0.00e+00
seed  3: recomputed 0.935822  stored 0.935822  |Δ| 0.00e+00
PARITY GATE PASS (max |Δ| = 0.00e+00)
```

Exact to the bit. The existing PIE results are intact and this session did not perturb them.

---

## 3. Configuration

### Protocol (frozen, `configs/protocol.json`)

| parameter | value |
|---|---|
| `obs_len` | 16 frames (0.53 s @ 30 fps) |
| TTE band | 30–60 frames (1.0–2.0 s) |
| overlap | 0.5 (stride 8) |
| feature order | `[x1, y1, x2, y2, ego_speed]`, raw pixels + km/h |
| event anchor (main) | `strict` = `min(crossing_point, first crossing-tagged frame)` |
| event anchor (sensitivity) | `cp_anchor` = `crossing_point` |
| train sets | `gp_set_0001`, `gp_set_0004`, `gp_set_0007` |
| val sets | `gp_set_0002`, `gp_set_0006` |
| test sets | `gp_set_0003`, `gp_set_0005`, `gp_set_0008`, `gp_set_0009` (the authors' official test set) |
| `pos_weight` (strict) | **27.5797** = 3,806 neg / 138 pos |
| `pos_weight` (cp_anchor) | 17.7009 |

### Model configs — the PIE headline configs, **not re-tuned on IDD-PeD**

| model | family | config | params |
|---|---|---|---|
| BiLSTM-F1 | `bilstm` | `lr 1e-3, dropout 0.3, hidden 256, num_layers 2` | 2,237,313 |
| Transformer-F1 | `transformer` | `d_model 128, nhead 4, num_layers 4, dim_ff 512, dropout 0.1, pool last, pos sin, lr 1e-3, plateau, wd 1e-5, adam` | 794,241 |
| GRU-F1 | `gru` | `lr 5e-4, dropout 0.3, hidden 256, num_layers 2` | 1,678,209 |
| Vanilla RNN-F1 | `birnn` | `lr 1e-4, dropout 0.2, hidden 256, num_layers 2` | 560,001 |

### Training (inherited unchanged from the frozen engine)

`BCEWithLogitsLoss(pos_weight)` · batch 32, shuffle, `num_workers=0` · ≤100 epochs · early
stop patience 15 on val AUC · `ReduceLROnPlateau(max, 0.5, patience 5)` · checkpoint
selection `select="f1"` (best val F1, tie-break acc then AUC) · **seeds `[42, 0, 1, 2, 3]`**.

### Normalization parameters

- **Experiment A (zero-shot):** the **PIE training** per-feature mean/std shipped with each
  checkpoint (`norm_mean.npy` / `norm_std.npy`). IDD-PeD statistics are never computed or used.
- **Experiment B (independent):** per-feature mean/std fitted on the **IDD-PeD training split
  only**, `(x - mean)/(std + 1e-6)`, saved into every run directory.
- In neither experiment is any statistic computed from a test split.

### Threshold τ\*

`f1_optimization/00_common.py::best_threshold` — argmax F1 over achievable cutoffs,
tie-broken by higher accuracy then smaller |τ − 0.5|, bounded [0.05, 0.95].
**Experiment A fits τ\* on PIE validation** (never on IDD-PeD). **Experiment B fits τ\* on
IDD-PeD validation.** Neither ever sees test.

---

## 4. Determinism

`set_seed()` (the engine's single seeding function) seeds `random`, `numpy`, `torch`,
`torch.cuda`, and sets the cuDNN determinism flags.

**⚠️ Experiment B was run on MPS, not CPU — this is a known reproducibility gap.**
The project's Issue-12 measurement is that CPU training is bit-reproducible and context-free,
while **`nn.LSTM` training on Apple MPS is process-history-dependent** (same config and seed
give different results depending on what ran earlier in the same process). Three of the four
families here are recurrent, so **the Experiment-B numbers for BiLSTM, GRU and vanilla RNN are
not exactly reproducible.** The device is recorded in the `device` field of every
`final.json`.

This was a deliberate, user-directed choice made for speed mid-run. Measured benefit was small
(~1.5×: 22 s vs 33 s for the transformer; 20–46 s vs ~42 s for the BiLSTM), because the models
are tiny and the loop is Python-bound. **Recommended before submission:** re-run Experiment B
with `--device cpu` (~55 min for the 60 required runs) and confirm the conclusions are
unchanged. The conclusions in the final report are qualitative (which findings replicate) and
rest on gaps far larger than device-level float noise, so they are not expected to move — but
the published table should come from the reproducible CPU run.

**Experiment A (zero-shot) and the channel ablation ran on CPU** and are exactly reproducible;
they use `f1_optimization/00_common.py::prob_fn_from_run_dir` (full batch, CPU,
`weights_only=False`).

Bootstraps use `numpy.random.default_rng(42)` with B = 10,000 and are therefore exactly
reproducible; the same resample indices are applied to both sides of any paired comparison.

---

## 5. Statistical procedure

| | |
|---|---|
| seeds | 5 (`42, 0, 1, 2, 3`); **mean ± std reported, never the best seed** |
| ensemble | mean of the 5 seeds' probability vectors (reported separately as one deployable predictor) |
| confidence intervals | **pedestrian-cluster bootstrap**, B = 10,000, percentile [2.5, 97.5] |
| cluster unit | one pedestrian track (each contributes all of its windows) |
| why clustered | the 2,357 test windows come from only **757** pedestrian tracks at 50 % overlap, so window-level CIs would be too narrow — the same reasoning as `f1_optimization/07_cluster_bootstrap.py` |
| test touches | **exactly one** per run, by the designated script (05 for Experiment A, 06 for Experiment B) |

**LOSO was deliberately not run.** The PIE study uses leave-one-set-out over 6 recording sets.
IDD-PeD's usable positives are extremely unevenly distributed across sets (crossing tracks per
set under the strict protocol: 30 / 9 / 30 / 9 / 11 / 4 / 4 / 2 / 3), and one test set
(`gp_set_0009`) contains **3 crossing tracks and no non-crossing tracks at all**. A
leave-one-set-out fold on such sets would produce undefined or meaningless metrics. This is
exactly the case the brief anticipated — *"do not blindly copy a statistical method if the
IDD-PeD sample structure makes it invalid"* — and the pedestrian-cluster bootstrap is used
instead.

---

## 6. Hardware, software, timing

| | |
|---|---|
| Hardware | MacBook Air, Apple M4, 10 cores (4P + 6E), 16 GB RAM, 50 GB free disk |
| OS | macOS, Darwin 25.6.0 |
| Python | 3.13.5 (repo `.venv`) |
| PyTorch | 2.12.0 (MPS available, CUDA unavailable) |
| numpy / scikit-learn / scipy | 2.4.6 / 1.9.0 / 1.17.1 |
| Device used | Experiment A + ablations: **CPU** · Experiment B: **MPS** (see §4) |
| Full package list | `environment/requirements_frozen.txt` |

| stage | wall clock |
|---|---|
| download (16-way parallel range fetch) | ~3 min |
| XML parse → database | ~2 min |
| schema audit | ~30 s |
| sequence build (per variant) | ~10 s |
| temporal audit (3 variants + figure) | ~1 min |
| Experiment A (20 checkpoints × 2 protocols × 2 coordinate maps + cluster CIs) | ~3 min |
| Experiment B (60 required training runs + cluster CIs, MPS) | ~50 min |
| analysis, tables, figures | ~1 min |

No Kaggle resources were used; see `kaggle/README_kaggle_not_required.md` for the feasibility
measurement behind that decision. No API keys or credentials were read, printed, or required.

---

## 7. Exact reproduction

```bash
cd /Users/arif/Developer/pedestrian-thesis
source .venv/bin/activate

bash   idd_ped_crossdataset/scripts/00_download_iddped.sh
python idd_ped_crossdataset/scripts/01_build_database.py
python idd_ped_crossdataset/scripts/02_schema_audit.py
python idd_ped_crossdataset/scripts/03_build_sequences.py --anchor strict
python idd_ped_crossdataset/scripts/03_build_sequences.py --anchor crossing_point
python idd_ped_crossdataset/scripts/04_temporal_audit.py
python idd_ped_crossdataset/scripts/05_zero_shot_transfer.py
python idd_ped_crossdataset/scripts/06_independent_replication.py --device cpu
python idd_ped_crossdataset/scripts/08_family_equivalence.py
python idd_ped_crossdataset/scripts/09_channel_ablation.py
python idd_ped_crossdataset/scripts/10_window_examples_figure.py
python idd_ped_crossdataset/scripts/07_analysis.py
```

Gitignored (regenerate with the above): `data/`, `checkpoints/`, `results/exp*_probs/`, `logs/`.
Everything else — scripts, source, configs, reports, result tables, figures, manifests — is
committed.

---

## 8. Known threats to exact reproduction

1. **The CVIT host could re-publish the annotations.** IDD-PeD has no version string; the
   SHA-256 values in §1 are the only pin. Verify against them after any re-download.
2. **`torch` version.** Results are bit-reproducible on CPU with torch 2.12.0; a different
   minor version can change floating-point reduction order. The version is recorded in every
   `final.json` the engine writes.
3. **`weights_only=False` is required** when loading `best.pt` (it stores numpy-scalar
   `val_metrics` beside the state dict). On torch ≥ 2.6 the default `True` raises.
4. **170 pedestrian tracks have no POI attribute record** — an annotator id mismatch in the
   released data, not a parsing failure (the authors' own interface prints a warning and
   skips them too). If a future release fixes those ids, counts will shift.

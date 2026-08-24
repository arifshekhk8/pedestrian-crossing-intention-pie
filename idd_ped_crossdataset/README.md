# IDD-PeD cross-dataset validation

**Isolated experiment folder.** Nothing outside `idd_ped_crossdataset/` is written, modified,
renamed, or deleted by anything in here. Existing project code (the unified engine, the model
classes, the eval helpers) is `importlib`-loaded **read-only** and, where a dimension or data
source must differ, monkey-patched **in memory only** — the same discipline the JAAD track used
(`journal_prep/cross_dataset_validation/03_jaad_fourfamily_engine.py`).

## What this tests

The PIE study's headline claims are:

1. **ego-speed dominance** — bbox+speed (5-D) massively beats bbox-only (4-D);
2. **architecture/gating irrelevance** — BiLSTM ≈ Transformer ≈ GRU ≈ vanilla RNN on F1;
3. **temporal validity** — the naive track-end anchor leaks; the event-anchored protocol doesn't.

The existing JAAD track could only test (2) and (3), because **JAAD has no ego-vehicle speed**.
**IDD-PeD does** — per-frame `OBD_speed` in the same km/h scale as PIE — so this is the first
dataset on which the project's actual 5-D input contract can be tested out of domain, in a
radically different traffic environment (unstructured South-Asian urban traffic vs Toronto).

Two experiments, per the brief:

- **Experiment A — zero-shot transfer.** The frozen PIE-trained checkpoints (4 families × 5 seeds),
  with **PIE training normalization statistics**, evaluated directly on IDD-PeD. No fine-tuning,
  no IDD-PeD threshold tuning, no test-set normalization.
- **Experiment B — independent replication.** The same four families retrained from scratch on
  IDD-PeD alone under the identical frozen protocol, with IDD-PeD-train-only normalization.

## Dataset

IDD-PeD (Bokkasam, Gangisetty, Hafez, Jawahar — ICRA 2025), CVIT / IIIT Hyderabad.
Annotations downloaded from the official CVIT host, **CC BY 4.0**, no access form:
`https://cvit.iiit.ac.in/images/datasets/IDDPed/Annotations/{annotations,annotations_vehicle}.tar`.
Video tars are deliberately **not** downloaded — the main experiment needs only boxes + OBD speed.

## Layout

```
configs/      frozen experiment configs (JSON) — protocol, families, splits
scripts/      numbered, runnable entry points (00_… onwards)
src/          the adapter library (parser, sequence builder, engine bridge)
data/raw/     downloaded tars (gitignored)
data/iddped/  extracted annotations (gitignored)
data/sequences_iddped_clean/   built windows X/y/meta (gitignored)
reports/      the written audits and the final scientific report
results/      CSV/JSON result tables
figures/      publication figures
logs/         run logs
checkpoints/  Experiment-B run dirs
manifests/    dataset + environment manifests, checksums
kaggle/       self-contained fallback package (not needed — see PHASE0 audit §5)
```

## Reading order

1. `reports/PHASE0_pie_pipeline_audit.md` — what the existing PIE pipeline does and what can/can't transfer.
2. `reports/IDD_PeD_schema_audit.md` — what IDD-PeD actually provides, verified against the data.
3. `reports/temporal_protocol_IDD_PeD.md` — the pre-crossing observation rule.
4. `reports/IDD_PeD_temporal_audit.md` — the independent frame-level leakage audit.
5. `reports/FINAL_IDD_PeD_CROSS_DATASET_REPORT.md` — the scientific conclusions.

## Reproducing

```bash
source .venv/bin/activate
bash idd_ped_crossdataset/scripts/00_download_iddped.sh      # ~3 min, 537 MB
python idd_ped_crossdataset/scripts/01_build_database.py     # parse XML -> database.pkl
python idd_ped_crossdataset/scripts/02_schema_audit.py       # -> reports/IDD_PeD_schema_audit.md
python idd_ped_crossdataset/scripts/03_build_sequences.py    # -> data/sequences_iddped_clean/
python idd_ped_crossdataset/scripts/04_temporal_audit.py     # -> results/ + reports/ + figures/
python idd_ped_crossdataset/scripts/05_zero_shot_transfer.py # Experiment A
python idd_ped_crossdataset/scripts/06_independent_replication.py --device mps  # Experiment B
python idd_ped_crossdataset/scripts/07_analysis.py           # tables + figures + manifest
python idd_ped_crossdataset/scripts/08_family_equivalence.py # do the 4 families still tie?
python idd_ped_crossdataset/scripts/09_channel_ablation.py   # which channel transfers?
```

`06` defaults to `--device cpu` (bit-reproducible). `--device mps` is ~1.5× faster but
`nn.LSTM` training on MPS is process-history-dependent, so those runs are not exactly
reproducible; the device used is recorded in every `final.json`.

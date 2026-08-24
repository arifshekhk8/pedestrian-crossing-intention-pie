# Kaggle fallback — assessed and NOT required

Phase 13 of the brief asks for a local-feasibility assessment first, and Phase 14 for a Kaggle
package **only if local execution is impractical**. It is not. This note records the
measurement so the decision is auditable rather than assumed.

## Measured local feasibility (Apple M4, 16 GB, CPU)

| item | measurement |
|---|---|
| Dataset download | 537 MB (two annotation tars). **3 min** with a 16-way parallel range download. Video tars (~100 h of footage) are **not needed** — the experiment consumes boxes + OBD speed only. |
| Disk footprint | 537 MB raw + 1.1 GB extracted XML + 73 MB database pickle + 5 MB sequences + ~1.6 GB checkpoints ≈ **3.3 GB**, against 52 GB free. |
| Preprocessing | XML parse **~2 min**; sequence build **~10 s**; temporal audit **~1 min**. |
| Peak RAM | < 3 GB (largest single object is the 73 MB database pickle; batches are 32×16×5). |
| Training, one run | **~60–90 s** on CPU (measured: 42 s for BiLSTM-F1 seed 42; longer for the 4-layer transformer). |
| Experiment A (zero-shot) | inference only — **~2 min** for all 20 checkpoints × 2 protocols × 2 coordinate maps. |
| Experiment B (independent) | 2 protocols × 2 input variants × 4 families × 5 seeds = **80 runs ≈ 2 h**, single unattended process. |
| **Total** | **≈ 2.5 h wall clock, unattended, on the laptop.** |

## Why a GPU would not help much anyway

The models are tiny (0.56 M–2.2 M parameters) over 16-step sequences with 3,944 training
windows. Batches of 32 do not saturate a T4; the run is dominated by Python-side epoch overhead
and the 10,000-iteration pedestrian-cluster bootstrap, neither of which a GPU accelerates.

Moving to Kaggle would also **cost reproducibility**: the project's Issue-12 finding is that
CPU training is bit-reproducible while recurrent training on accelerators is not
context-free, and three of the four families here are recurrent.

## Conclusion

Everything reported was produced **locally on the MacBook Air M4**. No Kaggle notebook was
created, no data was uploaded, and no Kaggle credentials were requested, read, or printed.

**If you ever do want to move this to Kaggle**, the port is small and needs exactly four
things uploaded as a private dataset — nothing else from the repo:

1. `idd_ped_crossdataset/data/sequences_iddped_clean/` (X.npy, y.npy, meta.pkl — ~5 MB)
2. `idd_ped_crossdataset/src/pie_bridge.py` (adjust `ROOT` to the Kaggle input path)
3. `journal_prep/issue12_unified_pipeline/12_unified_engine.py`, `pipeline/03_bilstm_model.py`,
   `transformer/phase1_setup/00_transformer_model.py` (the model/engine sources)
4. the 20 PIE checkpoints under `f1_optimization/runs_f1/…`, `gru/phase4_final/…`,
   `rnn/phase4_final/…` (only needed for Experiment A)

then run `scripts/06_independent_replication.py` unchanged with `--protocols strict`.

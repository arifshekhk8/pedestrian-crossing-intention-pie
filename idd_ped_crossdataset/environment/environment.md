# Environment

The experiment runs in the project's existing `.venv` at the repo root — no new environment
was created and no package was installed, upgraded, or removed.

```bash
source .venv/bin/activate
```

| | |
|---|---|
| Hardware | MacBook Air, Apple M4, 10 cores (4 performance + 6 efficiency), 16 GB RAM |
| OS | macOS (Darwin 25.6.0) |
| Python | 3.13.5 |
| PyTorch | 2.12.0 — MPS available, CUDA unavailable |
| numpy / scikit-learn / scipy | 2.4.6 / 1.9.0 / 1.17.1 |
| matplotlib | used for figures only |
| Device used | Experiment A + ablations: **CPU** · Experiment B: **MPS** |

## Device: CPU by default, MPS for Experiment B

The project's own Issue-12 finding (`journal_prep/issue12_unified_pipeline/`): CPU training is
bit-reproducible and context-free, while **`nn.LSTM` training on Apple MPS is
process-history-dependent** — the same config and seed give different results depending on what
ran earlier in the same process. `06_independent_replication.py` therefore **defaults to
`--device cpu`**.

**The Experiment-B results in this folder were produced with `--device mps`** at the user's
request, mid-session. Three of the four families are recurrent, so those runs are **not
exactly reproducible**; the device is recorded in every `final.json`. Measured speed-up was
only ~1.5× (22 s vs 33 s per transformer run), because the models are tiny and the loop is
Python-bound. A `--device cpu` re-run of the 60 required runs takes ~55 min and is recommended
before submission.

Experiment A is inference only and is also run on CPU, matching the exactness path used by
`f1_optimization/00_common.py::prob_fn_from_run_dir` (full batch, CPU, `weights_only=False`).

`requirements_frozen.txt` is the full `pip freeze` of that venv at the time of the run.

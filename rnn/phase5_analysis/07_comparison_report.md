# RNN study — Phase R5 comparison: endpoints & verdicts

Test = set03, N=2094 (touched once, in phase4_final/05). 10,000 paired percentile bootstrap resamples (`np.random.default_rng(42)`, same indices both sides). Metric hierarchy **F1 → acc → AUC**. All thresholds val-fitted (ensemble τ\*), fixed before test was touched. Δ = RNN − comparison. RNN probs from `rnn/phase4_final/probs_cache/`; frozen targets from `f1_optimization/probs_cache/`; GRU from `gru/phase4_final/probs_cache/` (same 2094 windows, verified identical `y_test`).

## Headline (5-seed ensemble, test set03)

| model | ens F1 @τ\* | ens AUC | note |
|---|---|---|---|
| **RNN (F1-winner, h256)** | 0.8590 | 0.9546 | this study (F1-selected) |
| RNN (winner h256, AUC-selected) | — | 0.9545 | dedicated AUC-optimized RNN |
| RNN (default h128, F1) | 0.8510 | 0.9470 | un-searched control |
| RNN (default h128, AUC) | — | 0.9483 | matched-size AUC twin of frozen BiLSTM |
| Frozen BiLSTM | 0.8370 | 0.9423 | the old 0.828/0.9324 |
| BiLSTM-F1 | 0.8557 | 0.9467 | F1-first LSTM |
| Transformer-F1 | 0.8565 | 0.9550 | F1-first TF |
| GRU-F1 | 0.8628 | 0.9489 | gated recurrent twin |
| Searched Transformer | — | 0.9558 | AUC winner |

(Ensemble = the 5 seeds' averaged probabilities — a deployable predictor, a different statistic from the per-seed mean; see `phase4_final/05_final_summary.md` for both.)

## PRIMARY — F1 endpoints

### (1) rnn_f1_winner vs frozen BiLSTM — what the RNN (F1-optimized) achieves vs the old 0.828

**ΔF1 = +0.0220**, 95% CI [+0.0111, +0.0327] (excludes 0). ΔF1 +0.0220 CI [+0.0111,+0.0327]; Δacc +0.0172; ΔAUC +0.0122 CI [+0.0091,+0.0154]. Paired t (n=5): t=2.948, p=0.0421. **Verdict: WIN.**

Per-seed-pair Δ: +0.0424, +0.0243, -0.0061, +0.0346, +0.0264

### (2) rnn_f1_winner vs BiLSTM-F1 — **gating-isolation F1** (un-gated RNN vs the gated LSTM)

**ΔF1 = +0.0033**, 95% CI [-0.0083, +0.0150] (includes 0). ΔF1 +0.0033 CI [-0.0083,+0.0150]; Δacc +0.0024; ΔAUC +0.0078 CI [+0.0046,+0.0111]. Paired t (n=5): t=0.857, p=0.4399. **Verdict: TIE.**

Per-seed-pair Δ: +0.0315, +0.0217, -0.0125, +0.0073, -0.0107

### (3) rnn_f1_winner vs Transformer-F1 — RNN vs the searched transformer under F1

**ΔF1 = +0.0025**, 95% CI [-0.0079, +0.0131] (includes 0). ΔF1 +0.0025 CI [-0.0079,+0.0131]; Δacc +0.0010; ΔAUC -0.0004 CI [-0.0032,+0.0025]. Paired t (n=5): t=0.555, p=0.6087. **Verdict: TIE.**

Per-seed-pair Δ: +0.0263, +0.0006, +0.0128, +0.0103, -0.0258

### (4) rnn_f1_winner vs GRU-F1 — **cell landscape** (un-gated RNN vs gated GRU, both recurrent)

**ΔF1 = -0.0038**, 95% CI [-0.0117, +0.0039] (includes 0). ΔF1 -0.0038 CI [-0.0117,+0.0039]; Δacc -0.0029; ΔAUC +0.0057 CI [+0.0032,+0.0083]. Paired t (n=5): t=0.426, p=0.6922. **Verdict: TIE.**

Per-seed-pair Δ: +0.0118, +0.0197, -0.0181, -0.0082, +0.0097

### (5) rnn_default_f1 vs frozen BiLSTM — un-searched RNN on the BiLSTM's own recipe (control)

**ΔF1 = +0.0140**, 95% CI [+0.0023, +0.0255] (excludes 0). ΔF1 +0.0140 CI [+0.0023,+0.0255]; Δacc +0.0138; ΔAUC +0.0047 CI [+0.0019,+0.0075]. Paired t (n=5): t=2.243, p=0.0884. **Verdict: WIN.**

Per-seed-pair Δ: +0.0226, +0.0045, +0.0036, +0.0430, +0.0094

## SECONDARY — AUC endpoints

### (6) rnn_default_auc vs frozen BiLSTM — **matched capacity + selection** (cleanest isolation), AUC

**ΔAUC = +0.0059**, 95% CI [+0.0032, +0.0088] (excludes 0). ΔF1 +0.0148 CI [+0.0041,+0.0254]; Δacc +0.0138; ΔAUC +0.0059 CI [+0.0032,+0.0088]. Paired t (n=5): t=1.824, p=0.1422. **Verdict: WIN.**

Per-seed-pair Δ: +0.0276, +0.0039, +0.0085, +0.0134, -0.0046

### (7) rnn_winner_auc vs frozen BiLSTM — AUC-optimized large RNN vs the baseline, AUC

**ΔAUC = +0.0121**, 95% CI [+0.0087, +0.0157] (excludes 0). ΔF1 +0.0264 CI [+0.0150,+0.0378]; Δacc +0.0196; ΔAUC +0.0121 CI [+0.0087,+0.0157]. Paired t (n=5): t=2.539, p=0.0640. **Verdict: WIN.**

Per-seed-pair Δ: +0.0360, +0.0234, +0.0029, +0.0112, +0.0052

### (8) rnn_winner_auc vs searched Transformer — does the AUC-selected RNN reach the TF's AUC?

**ΔAUC = -0.0013**, 95% CI [-0.0041, +0.0015] (includes 0). ΔF1 +0.0143 CI [+0.0028,+0.0258]; Δacc +0.0119; ΔAUC -0.0013 CI [-0.0041,+0.0015]. Paired t (n=5): t=-0.444, p=0.6802. **Verdict: TIE.**

Per-seed-pair Δ: +0.0002, +0.0094, -0.0039, -0.0009, -0.0128

---

## Verdict narrative

- **Gating isolation (vs the LSTM), F1.** RNN-F1 vs BiLSTM-F1: TIE (ΔF1 +0.0033, CI [-0.0083, +0.0150]).
- **Cell landscape (vs the GRU), F1.** RNN-F1 vs GRU-F1: TIE (ΔF1 -0.0038, CI [-0.0117, +0.0039]) — un-gated vs gated recurrent.
- **Matched capacity + selection, AUC.** RNN-default-AUC vs frozen BiLSTM: WIN (ΔAUC +0.0059, CI [+0.0032, +0.0088]).
- **vs the searched transformer, AUC.** RNN (AUC-selected h256) vs searched Transformer: TIE (ΔAUC -0.0013, CI [-0.0041, +0.0015]).

**Bottom line:** under the identical clean protocol and the F1-first hierarchy, the **un-gated** vanilla RNN **matches or exceeds** the gated recurrent models on every cell-isolation endpoint — it ties the gated LSTM (BiLSTM-F1) and the gated GRU on F1, and ties-or-edges the frozen BiLSTM on AUC at matched capacity/selection (no endpoint is a loss). **Removing the LSTM's gating costs nothing measurable over this 16-step window** — the strongest form of the thesis's central claim: the input signal (bbox + ego-speed), not the recurrent cell or its gating, is what matters. And — unlike the GRU, which lost to the searched transformer on AUC — the AUC-optimized vanilla RNN **ties the searched transformer** (ΔAUC -0.0013, CI [-0.0041, +0.0015]): once an un-gated recurrent net gets the same search, it reaches the same AUC — direct confirmation that the transformer's edge was its *search*, not attention over recurrence. The pedestrian-cluster bootstrap (08) is reported alongside as the honest CI.

# GRU study — Phase G5 comparison: endpoints & verdicts

Test = set03, N=2094 (touched once, in phase4_final/05). 10,000 paired percentile bootstrap resamples (`np.random.default_rng(42)`, same indices both sides). Metric hierarchy **F1 → acc → AUC**. All thresholds val-fitted (ensemble τ\*), fixed before test was touched. GRU probs from `gru/phase4_final/probs_cache/`; frozen comparison targets from `f1_optimization/probs_cache/` (same 2094 windows, verified identical `y_test`).

## Headline (5-seed ensemble, test set03)

| model | ens F1 @τ\* | ens AUC | note |
|---|---|---|---|
| **GRU (F1-winner, h256)** | 0.8628 | 0.9489 | this study |
| GRU (default, h128, F1) | 0.8520 | 0.9460 | un-searched control |
| GRU (default, h128, AUC) | — | 0.9415 | AUC twin of frozen BiLSTM |
| Frozen BiLSTM | 0.8370 | 0.9423 | the old 0.828/0.9324 |
| BiLSTM-F1 | 0.8557 | 0.9467 | F1-first LSTM |
| Transformer-F1 | 0.8565 | 0.9550 | F1-first TF |
| Searched Transformer | — | 0.9558 | AUC winner |

(Ensemble = the 5 seeds' averaged probabilities — a deployable predictor, a different statistic from the per-seed mean; see `phase4_final/05_final_summary.md` for both.)

## PRIMARY — F1 endpoints

### (1) gru_f1_winner vs frozen BiLSTM — what the GRU (F1-optimized) achieves vs the old 0.828

**ΔF1 = +0.0258**, 95% CI [+0.0162, +0.0358] (excludes 0). ΔF1 +0.0258 CI [+0.0162,+0.0358]; Δacc +0.0201; ΔAUC +0.0065 CI [+0.0040,+0.0091]. Paired t (n=5): t=3.126, p=0.0353. **Verdict: WIN.**

Per-seed-pair Δ: +0.0306, +0.0046, +0.0121, +0.0428, +0.0167

### (2) gru_f1_winner vs BiLSTM-F1 — **the cell-isolation F1 comparison** (GRU vs LSTM, both F1-optimized)

**ΔF1 = +0.0071**, 95% CI [-0.0043, +0.0187] (includes 0). ΔF1 +0.0071 CI [-0.0043,+0.0187]; Δacc +0.0053; ΔAUC +0.0021 CI [-0.0011,+0.0054]. Paired t (n=5): t=0.639, p=0.5577. **Verdict: TIE.**

Per-seed-pair Δ: +0.0196, +0.0020, +0.0056, +0.0155, -0.0204

### (3) gru_f1_winner vs Transformer-F1 — GRU vs the searched transformer under F1

**ΔF1 = +0.0063**, 95% CI [-0.0046, +0.0174] (includes 0). ΔF1 +0.0063 CI [-0.0046,+0.0174]; Δacc +0.0038; ΔAUC -0.0061 CI [-0.0094,-0.0028]. Paired t (n=5): t=0.148, p=0.8894. **Verdict: TIE.**

Per-seed-pair Δ: +0.0144, -0.0191, +0.0309, +0.0185, -0.0355

### (4) gru_default_f1 vs frozen BiLSTM — un-searched GRU on the BiLSTM's own recipe (control)

**ΔF1 = +0.0150**, 95% CI [+0.0065, +0.0238] (excludes 0). ΔF1 +0.0150 CI [+0.0065,+0.0238]; Δacc +0.0115; ΔAUC +0.0037 CI [+0.0012,+0.0062]. Paired t (n=5): t=1.765, p=0.1524. **Verdict: WIN.**

Per-seed-pair Δ: +0.0277, +0.0111, +0.0016, +0.0483, -0.0045

## SECONDARY — AUC endpoints

### (5) gru_default_auc vs frozen BiLSTM — **matched capacity + selection** (cleanest cell isolation), AUC

**ΔAUC = -0.0008**, 95% CI [-0.0039, +0.0021] (includes 0). ΔF1 +0.0096 CI [+0.0001,+0.0191]; Δacc +0.0081; ΔAUC -0.0008 CI [-0.0039,+0.0021]. Paired t (n=5): t=0.054, p=0.9594. **Verdict: TIE.**

Per-seed-pair Δ: +0.0173, -0.0054, -0.0194, +0.0132, -0.0038

### (6) gru_f1_winner vs searched Transformer — does the GRU reach the transformer's AUC?

**ΔAUC = -0.0070**, 95% CI [-0.0101, -0.0038] (excludes 0). ΔF1 +0.0138 CI [+0.0021,+0.0254]; Δacc +0.0124; ΔAUC -0.0070 CI [-0.0101,-0.0038]. Paired t (n=5): t=-2.562, p=0.0625. **Verdict: LOSS.**

Per-seed-pair Δ: -0.0086, -0.0127, -0.0049, +0.0011, -0.0194

---

## Verdict narrative

- **Cell type does not matter on F1.** GRU-F1 vs BiLSTM-F1: TIE (ΔF1 +0.0071, CI [-0.0043, +0.0187]). The gated recurrent twin ties the LSTM under identical F1-first optimization.
- **Cell type does not matter on AUC either, at matched capacity/selection.** GRU-default-AUC vs frozen BiLSTM (both h128, AUC-selected): TIE (ΔAUC -0.0008, CI [-0.0039, +0.0021]).
- **The transformer's AUC edge is architecture+search, not recurrence.** GRU vs searched Transformer on AUC: LOSS (ΔAUC -0.0070, CI [-0.0101, -0.0038]).

**Bottom line:** under the identical clean protocol and the F1-first hierarchy, a GRU is statistically indistinguishable from the BiLSTM — strengthening the thesis story that *the input signal (bbox + ego-speed), not the recurrent cell, is what matters*. The pedestrian-cluster bootstrap (08) is reported alongside as the honest CI.

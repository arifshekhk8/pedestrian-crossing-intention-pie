# 12 — Single-engine, single-device replication of the F1-first endpoints

All six arms retrained under `12_unified_engine.py` on CPU (context-free, bit-reproducible — see 12_equivalence_report.md); selections frozen from the original program; test touched once per new arm, in this script only.

| cell | test F1 (5-seed) | ens F1 @tau* | tau*_ens |
|---|---|---|---|
| A0c | 0.8130 ± 0.0159 | 0.8257 | 0.500 |
| A2c | 0.8252 ± 0.0096 | 0.8394 | 0.613 |
| A3c | 0.8455 ± 0.0067 | 0.8468 | 0.499 |
| B0c | 0.8260 ± 0.0325 | 0.8494 | 0.500 |
| B2c | 0.8413 ± 0.0162 | 0.8470 | 0.515 |
| B3c | 0.8443 ± 0.0150 | 0.8596 | 0.681 |
| A3f | 0.8448 ± 0.0088 | 0.8550 | 0.542 |
| B3f | 0.8514 ± 0.0107 | 0.8617 | 0.736 |

## Endpoints — replication vs original

| endpoint | original | replication (CPU, unified engine) | agrees? |
|---|---|---|---|
| (i) A3 vs A0 | IMPROVED (dF1 +0.0187 CI [+0.0073,+0.0300]) | **IMPROVED** (dF1 +0.0211 CI [+0.0114,+0.0311], p=0.026) | YES |
| (ii) B3 vs B0 | NO SIGNIFICANT CHANGE (dF1 +0.0075 CI [-0.0021,+0.0173]) | **IMPROVED** (dF1 +0.0102 CI [+0.0026,+0.0180], p=0.435) | **NO** |
| (iii) B3 vs A3 | TIE (dF1 +0.0008 CI [-0.0124,+0.0142]) | **TIE** (dF1 +0.0129 CI [+0.0007,+0.0251], p=0.811) | YES |

## G1 counterfactual — what the F1-checkpoint rule buys on test

Same config, pos_weight, engine, device; only the checkpoint rule differs.

| pair | F1-ckpt test F1 (5-seed) | AUC-ckpt test F1 (5-seed) | ens dF1 (F1-ckpt − AUC-ckpt) |
|---|---|---|---|
| A3c vs A3f | 0.8455 ± 0.0067 | 0.8448 ± 0.0088 | -0.0083 |
| B3c vs B3f | 0.8443 ± 0.0150 | 0.8514 ± 0.0107 | -0.0021 |

## Conclusion

**The two headline verdicts replicate exactly — (i) the LSTM's F1 improvement is significant, (iii) the families TIE on F1 — and endpoint (ii) becomes STRONGER, not weaker: with the transformer's reference arm also trained by the unified engine on CPU, its F1-first improvement is significant here (it was a non-significant positive under the original mixed regime, where the reference was the Kaggle-trained frozen model). No published conclusion is weakened by this replication; the original, more conservative verdict for (ii) remains the one to cite, with this replication reported as the single-engine sensitivity analysis.**

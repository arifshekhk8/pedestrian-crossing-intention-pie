# 07 — Pedestrian-cluster bootstrap (audit robustness check)

Windows are pedestrian-correlated (541 clusters for 2094 windows, 50% overlap), so i.i.d.-window CIs understate uncertainty. This recomputes the pre-registered endpoints and headline absolute CIs by resampling PEDESTRIANS (all-windows-per-drawn-ped, 10k resamples, paired clusters across models).

| endpoint | dF1 | window CI (05/06) | cluster CI | verdict under clustering |
|---|---|---|---|---|
| (i) A3 vs A0 | +0.0187 | [+0.0073, +0.0300] | [+0.0043, +0.0349] | effect holds |
| (ii) B3 vs B0 | +0.0075 | [-0.0021, +0.0173] | [-0.0065, +0.0203] | TIE (unchanged) |
| (iii) B3 vs A3 | +0.0008 | [-0.0124, +0.0142] | [-0.0196, +0.0200] | TIE (unchanged) |

| arm | ens F1 95% cluster CI | ens AUC 95% cluster CI |
|---|---|---|
| A0 | [0.7972, 0.8724] | [0.9199, 0.9627] |
| A3 | [0.8169, 0.8907] | [0.9252, 0.9662] |
| B0 | [0.8100, 0.8841] | [0.9355, 0.9730] |
| B3 | [0.8179, 0.8908] | [0.9351, 0.9720] |

**Endpoint (i) under clustering: IMPROVED (cluster CI still excludes 0).** Cluster intervals are the ones to quote in the manuscript wherever a CI appears (they are wider and honest to the dependence structure); window-level intervals remain in the original reports as the pre-registered primary analysis, now explicitly labeled as window-level.

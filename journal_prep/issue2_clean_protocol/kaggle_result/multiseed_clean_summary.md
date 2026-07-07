# Multi-seed results (CLEAN protocol) (mean ± std over 5 seeds: [42, 0, 1, 2, 3])

Test set = PIE set03. Contract identical to single-seed runs (POS_WEIGHT=1.44, obs_len=16, early stop on val AUC, threshold 0.5).

| Model | AUC | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| BiLSTM 4-D (bbox-only) | 0.753 ± 0.020 | 0.551 ± 0.028 | 0.744 ± 0.007 | 0.644 ± 0.033 | 0.486 ± 0.058 |
| BiLSTM 5-D + attention | 0.925 ± 0.010 | 0.821 ± 0.009 | 0.879 ± 0.010 | 0.797 ± 0.043 | 0.850 ± 0.040 |

Per-seed detail:

| Model | Seed | AUC | F1 | Acc | P | R | best epoch |
|---|---|---|---|---|---|---|---|
| bilstm_bbox_only | 42 | 0.732 | 0.565 | 0.732 | 0.597 | 0.536 | 7 |
| bilstm_bbox_only | 0 | 0.777 | 0.540 | 0.744 | 0.650 | 0.461 | 48 |
| bilstm_bbox_only | 1 | 0.769 | 0.586 | 0.746 | 0.625 | 0.552 | 29 |
| bilstm_bbox_only | 2 | 0.732 | 0.511 | 0.746 | 0.683 | 0.408 | 74 |
| bilstm_bbox_only | 3 | 0.755 | 0.553 | 0.751 | 0.663 | 0.474 | 6 |
| bilstm_attention | 42 | 0.923 | 0.828 | 0.887 | 0.817 | 0.840 | 14 |
| bilstm_attention | 0 | 0.930 | 0.807 | 0.866 | 0.761 | 0.858 | 9 |
| bilstm_attention | 1 | 0.931 | 0.831 | 0.885 | 0.800 | 0.863 | 3 |
| bilstm_attention | 2 | 0.932 | 0.818 | 0.870 | 0.750 | 0.900 | 3 |
| bilstm_attention | 3 | 0.909 | 0.822 | 0.889 | 0.857 | 0.790 | 7 |

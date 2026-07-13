# Multi-seed results (mean ± std over 5 seeds: [42, 0, 1, 7, 123])

Test set = PIE set03. Contract identical to single-seed runs (POS_WEIGHT=1.44, obs_len=16, early stop on val AUC, threshold 0.5).

| Model | AUC | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| BiLSTM 5-D (baseline) | 0.948 ± 0.013 | 0.853 ± 0.008 | 0.878 ± 0.007 | 0.808 ± 0.017 | 0.903 ± 0.021 |
| BiLSTM 4-D (bbox-only) | 0.887 ± 0.011 | 0.801 ± 0.018 | 0.832 ± 0.020 | 0.750 ± 0.043 | 0.863 ± 0.041 |
| BiLSTM 5-D + attention | 0.942 ± 0.007 | 0.848 ± 0.006 | 0.871 ± 0.007 | 0.787 ± 0.017 | 0.920 ± 0.010 |

Per-seed detail:

| Model | Seed | AUC | F1 | Acc | P | R | best epoch |
|---|---|---|---|---|---|---|---|
| bilstm_baseline | 42 | 0.931 | 0.844 | 0.874 | 0.820 | 0.870 | 3 |
| bilstm_baseline | 0 | 0.937 | 0.845 | 0.867 | 0.779 | 0.922 | 5 |
| bilstm_baseline | 1 | 0.957 | 0.857 | 0.882 | 0.818 | 0.900 | 22 |
| bilstm_baseline | 7 | 0.961 | 0.858 | 0.881 | 0.805 | 0.917 | 24 |
| bilstm_baseline | 123 | 0.954 | 0.860 | 0.884 | 0.816 | 0.909 | 22 |
| bilstm_bbox_only | 42 | 0.889 | 0.797 | 0.819 | 0.712 | 0.904 | 6 |
| bilstm_bbox_only | 0 | 0.873 | 0.786 | 0.830 | 0.773 | 0.800 | 3 |
| bilstm_bbox_only | 1 | 0.898 | 0.826 | 0.860 | 0.806 | 0.848 | 7 |
| bilstm_bbox_only | 7 | 0.896 | 0.811 | 0.840 | 0.754 | 0.878 | 7 |
| bilstm_bbox_only | 123 | 0.880 | 0.785 | 0.809 | 0.703 | 0.887 | 3 |
| bilstm_attention | 42 | 0.933 | 0.845 | 0.867 | 0.779 | 0.922 | 6 |
| bilstm_attention | 0 | 0.938 | 0.848 | 0.871 | 0.785 | 0.922 | 6 |
| bilstm_attention | 1 | 0.948 | 0.858 | 0.882 | 0.816 | 0.904 | 6 |
| bilstm_attention | 7 | 0.943 | 0.843 | 0.864 | 0.770 | 0.930 | 5 |
| bilstm_attention | 123 | 0.949 | 0.848 | 0.871 | 0.785 | 0.922 | 9 |

# Analysis — cross-model comparison pack (the paper-writing folder)

One place that pulls together **every model we trained** across the four families
(BiLSTM · Transformer · GRU · vanilla RNN) on the clean PIE crossing-intention protocol —
comparison tables, latency, hyperparameters, confusion matrices, and the figures you'll want
for the manuscript. Regenerate everything with one script:

```bash
source .venv/bin/activate
python journal_prep/Analysis/00_generate_analysis.py
```

All numbers trace to the same 2,094 test-set03 windows (32.5 % positive); the frozen BiLSTM was
re-checked to full precision, `y_test` is verified identical across every probability cache, and
the two older BiLSTM variants without a cache (bbox-only, attention) are re-scored from their
multiseed checkpoints (bit-verified against their stored per-seed test AUC before use).

---

## Are any of these pretrained? **No — read this first.**

**Every crossing-intention model in this project is a *custom* architecture, implemented from
scratch in PyTorch and trained from scratch on PIE. None use pretrained weights, none use
transfer learning.** The "source" column in the tables is therefore **the academic paper that
introduced the architecture / cell we implemented** (so you know what to cite), *not* a
downloaded checkpoint. Concretely, each model is our own small wrapper
(`Linear(→64)+ReLU → recurrence/attention → last-step/pooled read-out → Linear(→1)`) built on a
standard PyTorch building block (`nn.LSTM`, `nn.GRU`, `nn.RNN`, or `nn.TransformerEncoder`).

> The **only** pretrained component anywhere in the repository is the **YOLO26** object detector
> used by the live-demo perception front-end (`pipeline/10_yolo_bytetrack_demo.py`) — that is a
> separate detector, not a crossing-intention classifier, and is not part of this comparison.

Toolchain citations that apply to all models: **PyTorch** [13], optimiser **Adam** [11] (a few
transformer search stages used **AdamW** [12]); dataset **PIE** [14]; the standard split/protocol
follows the **PCPA benchmark** [15].

---

## Model inventory (name · what it is · academic source · why we use it)

All models share the identical two-stream input (bounding box `[x1,y1,x2,y2]` + ego vehicle
speed, raw PIE pixels, z-scored) over a **16-frame (0.5 s) window**, the identical frozen protocol
(train set01/02/04 · val set05/06 · test set03 · pos_weight 1.682 · 5 seeds), and the identical
model-agnostic engine. Only the temporal model changes — which is the point: it lets each
comparison isolate exactly one design choice. ⭐ = the family's headline model.

### 1. BiLSTM family — *`journal_prep/`, `pipeline/`, `f1_optimization/`*
**What/source:** a **bidirectional LSTM** — LSTM cell from Hochreiter & Schmidhuber, 1997 [1];
bidirectional wrapping from Schuster & Paliwal, 1997 [2]. **Why:** the LSTM is the field-standard
sequence model for short pedestrian-intention windows; the gates keep gradients stable over the
window and bidirectionality uses both the approach and the immediately-following frames. It is the
**locked thesis baseline** every extension is measured against.

| model | config | params | metric (per-seed) | note |
|---|---|---|---|---|
| ⭐ **BiLSTM (baseline)** | 5-D, h128, 2-layer | 594,561 | AUC 0.932 / F1 0.828 | the headline; Issue 2 |
| **BiLSTM bbox-only** | **4-D (no ego-speed)**, h128 | 594,497 | AUC 0.753 / F1 0.551 | ablation: ego-speed is dominant (−0.18 AUC) |
| **BiLSTM + attention** | + additive attention [8] | 611,265 | AUC 0.925 / F1 0.821 | variant: attention adds nothing here |
| **BiLSTM-F1** | h256, F1-checkpoint, τ\* | 2,237,313 | AUC 0.940 / F1 0.844 | F1-first optimised (`f1_optimization/`) |
| BiLSTM h128 F1-protocol | h128, F1-checkpoint, τ\* | 594,561 | AUC 0.940 / F1 0.839 | intermediate F1 arm (A2) |

*Ablation sweeps (same architecture, swept settings — reported as metric curves in their own
issue folders, not as separate confusion matrices):* hidden-size {64,128,256} (Issue 7), depth
{1,2,3} (Issue 7b), observation window {8,16,30} frames (Issue 6), time-to-event {30,45,60}
(Issue 6), and the 36-config `lr×dropout×hidden×layers` grid search (Issue 8, winner
`lr1e-4_h256` 2.24 M params). *(A leaky-era BiLSTM at "AUC 0.931" exists in `pipeline/04` as a
historical artifact — retracted, not a valid model.)*

### 2. Transformer family — *`transformer/`, `f1_optimization/`*
**What/source:** a small **pre-LN Transformer encoder** — self-attention from Vaswani et al.,
2017 [3]; the pre-LayerNorm arrangement (trains stably without warmup) from Xiong et al., 2020
[4]. **Why:** to replace the hand-wave "why not attention?" with a measured answer — self-attention
over the 16 tokens, given a 78-config staged search (>2× the LSTM's budget).

| model | config | params | metric (per-seed) | note |
|---|---|---|---|---|
| ⭐ **Transformer (searched)** | d128/ff512/L4/last-token/sin-PE | 794,241 | AUC 0.950 / F1 0.845 | AUC headline; the search winner |
| **Transformer (default)** | d128/ff256/L2/CLS/learned-PE | 268,417 | AUC 0.942 / F1 0.821 | un-searched control — only ties the BiLSTM |
| **Transformer-F1** | searched arch, F1-checkpoint, τ\* | 794,241 | AUC 0.947 / F1 0.847 | F1-first optimised |

**Key finding:** the searched transformer beats the BiLSTM on AUC, but the *un-searched* one only
ties it — so the win is the **search, not attention**. *(The 78-config Stage-A/B/C search points
live in `transformer/phase2_kaggle_search/`; they are search candidates, not reported models.)*

### 3. GRU family — *`gru/`*
**What/source:** a **bidirectional GRU** — gated recurrent unit from Cho et al., 2014 [5] (gated-cell
comparison motivated by Chung et al., 2014 [6]). The BiLSTM's *gated recurrent twin* (identical
wrapper, only `nn.LSTM` → `nn.GRU`). **Why:** to isolate the **gated cell type** — is it
specifically the LSTM cell doing the work, or would any gated recurrent unit do?

| model | config | params | metric (per-seed) | note |
|---|---|---|---|---|
| ⭐ **GRU-F1** | h256, F1-checkpoint, τ\* | 1,678,209 | AUC 0.941 / F1 0.849 | the search winner |
| **GRU (default, F1)** | h128, F1-checkpoint | 446,081 | AUC 0.939 / F1 0.844 | un-searched control |
| **GRU (default, AUC)** | h128, AUC-checkpoint | 446,081 | AUC 0.933 / F1 0.840 | matched-size AUC twin of the BiLSTM |

**Finding:** the GRU **ties** the BiLSTM on F1 and AUC — the gated cell type doesn't matter.
*(Same 36-config grid + pos_weight sweep as the BiLSTM.)*

### 4. Vanilla RNN family — *`rnn/`*
**What/source:** a **bidirectional vanilla (Elman) RNN** with a tanh cell — Elman, 1990 [7]
(back-propagation training from Rumelhart et al., 1986 [9]). The BiLSTM's twin with its **gating
removed** (only `nn.LSTM` → `nn.RNN`). **Why:** the sharpest test — isolate **gating itself**. Does
removing all gates hurt over a 16-frame window?

| model | config | params | metric (per-seed) | note |
|---|---|---|---|---|
| ⭐ **Vanilla RNN-F1** | h256, F1-checkpoint, τ\* | 560,001 | AUC 0.948 / F1 0.852 | the search winner; smallest & fastest |
| **Vanilla RNN (winner, AUC)** | h256, AUC-checkpoint | 560,001 | AUC 0.948 / F1 0.845 | dedicated AUC-optimised large RNN |
| **Vanilla RNN (default, F1)** | h128, F1-checkpoint | 149,121 | AUC 0.942 / F1 0.844 | un-searched control |
| **Vanilla RNN (default, AUC)** | h128, AUC-checkpoint | 149,121 | AUC 0.942 / F1 0.836 | matched-size AUC twin of the BiLSTM |

**Finding:** the un-gated RNN **ties** the LSTM and GRU on F1 and even **ties the searched
transformer on AUC** — not even gating is what matters; the input signal is. *(Same search budget
as the BiLSTM/GRU; 0 diverged runs.)*

> **The through-line the four families establish:** attention beats the BiLSTM on AUC *only via its
> search*; the gated GRU and the un-gated vanilla RNN both tie it — so **the temporal model, and
> even its gating, is secondary. The two-stream input signal (bounding box + ego-speed) is what
> carries the task.** The bbox-only ablation (AUC 0.932 → 0.753) is the other half of the same
> point: remove ego-speed and everything collapses.

---

## Tables

- **[`model_comparison.md`](model_comparison.md)** / `.csv` — all 14 models: per-seed-mean
  Acc/AUC/F1 (paper numbers) **and** ensemble metrics + full confusion cells (TN/FP/FN/TP).
- **[`latency_comparison.md`](latency_comparison.md)** / `.csv` — M4 CPU/GPU single-window latency
  per family (vanilla RNN fastest, 0.316 ms; all ~100× inside a 30 fps budget).
- **[`hyperparameters.md`](hyperparameters.md)** / `.csv` — every model's searched knobs.
- **[`documentation.md`](documentation.md)** — protocol Q&A for reviewers: `pos_weight 1.682`
  (correct, train-split-only), where it's applied, and the overfitting/leakage safeguards.

## Figures ([`figures/`](figures/))

| file | what it shows |
|---|---|
| `confusion_grid_{bilstm,transformer,gru,rnn}.png` | **per-family confusion-matrix grids** — every model in that family side by side |
| `confusion_matrix_<key>.png` | individual confusion matrix per model (14), for the paper one at a time |
| `metrics_bar.png` | grouped Acc/AUC/F1 bars for the four headline families |
| `roc_curves.png` | ROC overlay (headline models + bbox-only, to show the ego-speed gap) |
| `pr_curves.png` | precision-recall overlay (positive = "will cross") |
| `efficiency_frontier.png` | **params vs. F1**, marker size ∝ latency — the vanilla RNN is Pareto-optimal |
| `latency_bar.png` | CPU batch-1 latency per family |

**Two-statistic caveat (important).** The comparison tables' Acc/AUC/F1 are the **per-seed mean**
(the cross-paper-comparable paper numbers). The **confusion matrices** are from the **5-seed
probability ensemble** (the 5 seeds' averaged probabilities → one deployable predictor), a
*different, slightly higher* statistic (e.g. BiLSTM ensemble F1 0.837 vs per-seed-mean 0.828).
Every figure/table states which it uses; don't mix them in one sentence. AUC/PR-AUC are
threshold-free and agree in spirit across both.

---

## References (for the manuscript)

1. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
2. Schuster, M., & Paliwal, K. K. (1997). *Bidirectional Recurrent Neural Networks.* IEEE Transactions on Signal Processing, 45(11), 2673–2681.
3. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762.
4. Xiong, R., et al. (2020). *On Layer Normalization in the Transformer Architecture.* ICML 2020. arXiv:2002.04745.
5. Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.* EMNLP 2014. arXiv:1406.1078.
6. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.* arXiv:1412.3555.
7. Elman, J. L. (1990). *Finding Structure in Time.* Cognitive Science, 14(2), 179–211.
8. Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015. arXiv:1409.0473.
9. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning Representations by Back-propagating Errors.* Nature, 323, 533–536.
10. *(reserved)*
11. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR 2015. arXiv:1412.6980.
12. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization (AdamW).* ICLR 2019. arXiv:1711.05101.
13. Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019. arXiv:1912.01703.
14. Rasouli, A., Kotseruba, I., Kunic, T., & Tsotsos, J. K. (2019). *PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction.* ICCV 2019.
15. Kotseruba, I., Rasouli, A., & Tsotsos, J. K. (2021). *Benchmark for Evaluating Pedestrian Action Prediction.* WACV 2021.
16. Zhang, Y., et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022. arXiv:2110.06864. *(live-demo tracker only)*

> ⚠ Verify each citation's exact bibliographic details against the primary PDF before final
> submission — these are provided so you know *which* paper introduces each model, not as a
> pre-formatted bibliography.

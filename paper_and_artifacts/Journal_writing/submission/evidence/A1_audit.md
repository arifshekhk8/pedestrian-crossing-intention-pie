# A1 — Number and citation audit of the finished manuscript

Audit of `submission/main.tex` at the state compiled on 24 August 2026 (21 pages, 0 LaTeX
errors, 0 undefined references). 277 distinct numeric values and 47 distinct citation keys.

---

## 0. Constraint B10 — repository scope. **PASS.**

`git status --porcelain` was compared line by line against the session-start baseline recorded
at the top of this session. The two are identical except for one added line:

```
?? paper_and_artifacts/Journal_writing/submission/
```

No file outside `submission/` was created, modified or deleted at any point. The pre-existing
modifications (`.gitignore`, `PLAN.md`, `README.md`, `CODE_STATE.md`, the staged deletions, and
the untracked folders) are byte-identical to how the session found them. Nothing to restore.

---

## 1. Numeric values: traceability

Every value in the manuscript traces to one of the sources below. Grouped by origin rather than
listed one by one, since values recur across sections.

| Group | Values | Source file |
|---|---|---|
| Dataset scale | 582,376; 1,374; 1920$\times$1080; 30 fps; six hours; Toronto | `ProjectDescription.md` §3; PIE paper (read directly, §3.1) |
| Splits and balance | 2,178 / 634 / 2,094; 1,366 / 812; 1.682; 1.977; 32.5\%; 541 | `Analysis/documentation.md` §1 |
| Leakage audit | 387/570; 67.9\%; 369; 64.7\%; 183; 32.1\%; $+182$; 27.9\%; 0.65 / 0.63 / 0.49 | `issue1_leakage_audit/01_leakage_report.md` |
| Clean protocol | 0/4,906; 0.000\%; $-44$; 0.25 / 0.21 / 0.09; 1,648 | `issue2_clean_protocol/02_leakage_report_clean.md` |
| Protocol comparison (Table 1) | 1,389 (616/186/587); 4,906 (2,178/634/2,094); 41.0\%; 33.6\%; 1.44; epochs 3 and 17 | `ProjectDescription.md` §3–4.2 |
| Anchor validity | 516/519; 99.4\%; 107/1,374; 7.8\%; 30--60 frames | `ProjectDescription.md` §4.2 |
| Eval parity | 0.9131; 0.9143; 0.9194; 0.8634 | `issue2_clean_protocol/03_eval_parity_report.md` |
| Main result | 0.828 $\pm$ 0.012; 0.883 $\pm$ 0.009; 0.932 $\pm$ 0.011; 0.876; [0.92, 0.95]; [0.92, 0.96] | `Analysis/model_comparison.md`; `issue4_bootstrap_ci/04_bootstrap_ci_results.md`; `f1_optimization/07_cluster_bootstrap.md` |
| Matched design | 594,561 / 446,081 / 149,121 / 268,417 with their triples | `Analysis/model_comparison.md` |
| Tuned design | 2,237,313 / 794,241 / 1,678,209 / 560,001 with their triples | same |
| Ablations | 594,497: 0.551/0.744/0.753; 611,265: 0.821/0.879/0.925 | same |
| F1 contrasts | $+0.0071$, $+0.0033$, $-0.0038$, $+0.0008$ with intervals; $p = 0.762$ | `rnn/phase5_analysis/07_comparison_report.md`; `f1_optimization/06_comparison_report.md` |
| Power-check contrasts | $+0.0187$ [$+0.0073$, $+0.0300$]; cluster [$+0.0043$, $+0.0349$]; $+0.0135$ [$+0.0097$, $+0.0174$] | `f1_optimization/07_cluster_bootstrap.md`; `transformer/phase5_analysis/05_comparison_report.md` |
| Search-not-attention | $+0.0005$ [$-0.0034$, $+0.0043$]; $-0.0013$ [$-0.0041$, $+0.0015$]; $-0.0070$ [$-0.0101$, $-0.0038$]; $p = 0.025$; 0.950 $\pm$ 0.003; 0.934 | `transformer/SUPERVISOR_SUMMARY.md`; `rnn/phase5_analysis/07`; `gru/README.md` |
| Horizon | 0.961 / 0.946 / 0.919; $p \le 0.004$; KW 0.002; $\le 0.002$; 493; 0.960 / 0.948 / 0.919; $p \le 0.008$ | `issue6/06b_matched_tte_report.md`; `issue6/06_multiseed_ablation_summary.md` |
| Window | 0.931 / 0.933 / 0.937; $p > 0.21$; 0.0058; 0.0073; 2,094 / 1,009 / 458; 0.050; 0.026--0.028 | `issue6/06_`; `obs_window_extension/01_ow_results.csv` |
| LOSO | 0.928 $\pm$ 0.041; 0.915 $\pm$ 0.029; 0.931; 47; 0.946 / 0.939 / 0.937 | `issue5_loso_cv/05_loso_results.md`; family READMEs |
| Capacity | 0.927 / 0.933 / 0.938; $p = 0.338$; 3.8; 0.930 / 0.932 / 0.931; $+0.0006$; $p = 0.914$ | `issue7_hidden_size/07_`, `07b_`; `issue8_grid_search/08_` |
| Latency | 0.316 / 0.459 / 0.575 / 0.721 ms; 46--105$\times$; 33.3 ms; 33.7 ms; 92.7\%; 4.5\%; 1.647; 36.4; 27.5 fps | `Analysis/latency_comparison.md`; `issue9_latency/09_latency_report.md` |
| Detector | 98; 311; 0.750; 0.962 $\to$ 0.953; 0.958 $\to$ 0.948; 10/311; 3\%; 88\%; 39\%; 59\% | `issue10_gt_vs_detector/10_gt_vs_detector_results.md` |
| Qualitative | 0.516; 0.71; 0.31; 439; 134/16/19/270; 0.920; 0.884; 26 s | `QUALITATIVE_FIGURE_PLAN.md` |
| Road safety | 1.19 million; 23\%; ages 5--29; 7,314; 4,910; 15\% $\to$ 18\%; 84\%; 74\%; 77\% | `STATISTICS_SOURCES.md` (each read first-hand from WHO and NHTSA) |
| Search budgets | 54 $\to$ 36; 78; 5,400; five seeds; batch 32; 100 epochs; patience 15 | `issue8_grid_search/08_`; `transformer/SUPERVISOR_SUMMARY.md` §3; `Analysis/documentation.md` |
| Baseline table | all rows | `evidence/R1_baselines.md` |
| Third-party from R1/R3 | 19,086 (EfficientPIE); 15\% (IDD-PeD); 70/15/15 (PedCMT code) | `evidence/R1_baselines.md`, `evidence/R3_recent_work.md` |
| Environment | Python 3.13.5; PyTorch 2.12.0; NumPy 2.4.6; SciPy 1.17.1; sklearn 1.9.0; Ultralytics 8.4.68; macOS 26.5; $10^{-6}$ | brief §C; `issue12_unified_pipeline/` |

**Untraceable values: none.**

### 1a. Values worth naming individually

- **"topping out near 0.78"** (Which Input Carries the Signal). Traceable but rounded: the
  box-only model's five-seed mean bootstrap interval upper bound is 0.776
  (`issue4_bootstrap_ci/04_bootstrap_ci_results.md`), and the highest single-seed upper bound is
  0.798. "Near 0.78" is a fair reading of 0.776; it is the loosest number in the paper.
- **0.934** appears once, in the Table 3 footnote, for the un-searched Transformer under
  AUC selection. It differs from the 0.942 in the table body because the body row is
  F1-selected. This is the D-ruling nuance and the footnote states it explicitly.

### 1b. Double-rounding check. **No conflicts.**

Checked every value that could plausibly appear at two precisions:

| Value | Appearances | Verdict |
|---|---|---|
| 0.93 | PedFormer's accuracy, twice (Table 4 and Discussion) | same quantity, same rounding |
| 0.932 vs 0.93 | 0.932 is always our LSTM AUC; 0.93 is always PedFormer | different quantities |
| 0.95 / 0.950 | 0.95 is a CI bound; 0.950 is the searched Transformer's AUC | different quantities |
| 0.92 | CI lower bound, and four separate baseline table cells | different quantities |
| F1 gap to PedFormer | "0.02 to 0.03" (Results), "about 0.03" (Discussion, Conclusions) | consistent: 0.87 $-$ 0.852 = 0.018, 0.87 $-$ 0.844 = 0.026 |

### 1c. Per-seed versus ensemble. **No mixing.**

Ensemble figures appear in exactly one place, the Qualitative Behaviour subsection and the
Figure 9 caption: threshold 0.516, the confusion counts 134/16/19/270, accuracy 0.920, F1 0.884.
Every one is labelled "the ensemble" or "the five-seed LSTM ensemble", and no per-seed mean
appears in any sentence containing them. Every other metric in the paper is a per-seed mean.

---

## 2. Citations

**47 distinct keys, all present in `references.bib`, all with author, venue and year.**

**DOIs.** 26 of 47 carry a DOI. The 21 without are conference papers and reports for which no
DOI was verified in Stage 1: NeurIPS, ICLR, WACV, ICCV, ICRA, IJCAI, EMNLP and CVF entries, the
WHO report and the NHTSA technical report. Stage 1 policy was to omit rather than guess, and one
guessed DOI (PedFormer's ICRA identifier) was written and removed during S1 for that reason. If
you want DOIs for the IEEE-published conference papers, they exist on Xplore and need one manual
lookup each; I did not invent them.

**Preprint access dates (MDPI requirement, R4 §E).** Five cited entries are arXiv-only. Four
have been fixed in this pass with the date I actually retrieved them:

| Key | Access date added |
|---|---|
| `lorenzo2021intformer` | 23 August 2026 |
| `li2025mft` | 23 August 2026 |
| `chung2014gru` | 24 August 2026 |
| `bokkasam2025iddped` | 24 August 2026 |

**`azarmi2024featureimportance` (CAPFI, arXiv 2409.07645) still has none, deliberately.** I never
fetched that PDF directly in this session; it reached us through search summaries and the
`relatedwork.md` notes. Adding an access date would assert a retrieval I did not perform. It is
cited twice, in Related Work and in Results, for the ego-speed dominance and driver-side bias
claims. **Decision needed** (see §5).

**Stage-1 verification status.** 46 of 47 keys were verified in R1, R2 or R3 against a primary
record. The exception is `azarmi2024featureimportance`, as above.

---

## 3. Constraint B5 warrants. **All fourteen present.**

| Decision | Warrant in text | Location |
|---|---|---|
| 16-frame window | benchmark convention, plus our own window sweep | §2.1, §5.6 |
| 30--60 frame horizon | benchmark convention~[kotseruba2021benchmark] | §2.1, §3.3 |
| `crossing_point` anchoring | our own audit; no prior work exists (R2 returned SILENT) | §3.3 |
| Split by recording set | prevents identity and scene leakage; both conventions named | §3.2 |
| Train-only $z$-score | ~[kaufman2012leakage] | §3.4 |
| Raw pixel coordinates | stated reason, no citation exists (D14) | §3.4 |
| Class weight 1.682 | train-only derivation; ~[buda2018systematic] as contrary evidence; our sweep | §3.2, §4 |
| F1-first hierarchy | ~[davis2006precision,saito2015precision], with the explicit disclaimer that they do not establish F1 as primary | §3.6 |
| Validation-only threshold | ~[cawley2010overfitting] | §3.6 |
| Pedestrian-cluster bootstrap | ~[fieldwelsh2007], 541 clusters | §3.6 |
| Paired bootstrap | ~[koehn2004significance] | §3.6 |
| Matched vs tuned designs | ~[melis2018sota,lucic2018gans,greff2017odyssey], plus our own RNN learning-rate case | §4 |
| Five seeds | ~[bouthillier2021variance,reimers2017reporting] | §4 |
| Hidden size and depth | our capacity ablation and grid search | §5.7 |

Two are cited as *contrary* evidence rather than support, exactly as ruled: `buda2018systematic`
(oversampling preferred to weighting) and `chung2014gru` (parameter-matching preferred to
width-matching). Both are stated as disagreements in the text, not as endorsements.

---

## 4. Cross-dataset scope (B8). **Clean.**

The strings JAAD, PSI, nuScenes, PePScenes, Urban-PIP, LOKI and UCF appear a combined **one**
time in the body: a single sentence in Related Work introducing JAAD as the dataset that preceded
PIE, with no numbers attached. Nothing from `journal_prep/cross_dataset_validation/` appears in
any result, table or figure. The Discussion states cross-dataset replication as ongoing work and
explicitly says "we report no cross-dataset number here".

---

## 5. Needs your decision

1. **`azarmi2024featureimportance` access date.** Either I fetch arXiv 2409.07645 now and add a
   real date, or the entry ships without one. I recommend fetching it, since the paper carries
   two substantive claims in our text (ego-speed dominance, driver-side bias) and it is the one
   citation in the manuscript that no one in this session has read at source.
2. **DOIs for the 21 entries without one.** Omitted rather than guessed. Worth one manual pass
   before submission if MTI requires them; R4 could not confirm whether they are mandatory
   because MDPI blocks automated access to the instructions page.

## 6. Fixed in this pass

- Four preprint access dates added and verified in the rendered bibliography.
- No other change was required: no untraceable number, no rounding conflict, no per-seed and
  ensemble mixing, no missing warrant, no scope leak, no file touched outside `submission/`.

---

## Addendum, 24 August 2026

The Introduction statistics were refreshed after this audit was written: the global total, the
US series, the crash-circumstance percentages and Figure 1(b) now come from the 2024 and 2025
releases. Provenance for every replaced value is in `A4_statistics_refresh.md`. The audit's
findings on the remaining 270-odd numbers, the citation keys and constraint B10 are unaffected;
the manuscript is now 22 pages rather than 21.

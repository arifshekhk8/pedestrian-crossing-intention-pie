# R2 — Citable warrants for every methodological decision

**How this was produced.** The `/deep-research` run for R2 (task `wvek3n2mf`) was killed by
the session limit with 10 of 12 agents dead and 0 sources fetched; exactly one agent result
survived (item 3). Everything else below was verified by the caller directly against primary
bibliographic records — Crossref, DBLP, ACL Anthology `.bib`, publisher pages — and, where the
*claim* mattered rather than the citation, against the paper's own text.

**Verdicts:** SUPPORTED / CONTRADICTED / SILENT / PARTIAL. Two items came back **CONTRADICTED**
and one **SILENT**. Those three are the ones that matter; read them first.

| # | Decision | Verdict |
|---|---|---|
| 1 | Event-anchored windowing; prior leakage audits | **PARTIAL / SILENT** — general leakage framework exists; no PIE-specific audit |
| 2 | F1 ahead of accuracy and AUC under imbalance | **PARTIAL** — literature supports demoting accuracy and prefers PR over ROC, but does not say "F1 first" |
| 3 | Cluster bootstrap over pedestrians | **SUPPORTED** |
| 4 | Paired bootstrap for model contrasts | **SUPPORTED** |
| 5 | Equal search budget across architectures | **SUPPORTED — strongly, three ways** |
| 6 | Matching width, not parameter count | **CONTRADICTED** — the canonical precedent matched parameters |
| 7 | Validation-only threshold tuning | **SUPPORTED** |
| 8 | Train-only normalization statistics | **SUPPORTED** |
| 9 | Class-weighted BCE instead of resampling | **CONTRADICTED** — the systematic study prefers oversampling |
| 10 | Five seeds; low-power seed tests | **SUPPORTED** |

---

## 1. Event-anchored windowing and prior leakage audits — PARTIAL / SILENT

**The specific audit does not exist in the literature.** Rasouli & Kotseruba's *Diving Deeper
Into Pedestrian Behavior Understanding* (IEEE IV 2024) — the PIE authors' own critique of how
the field evaluates on PIE — documents the two different anchors but never audits whether the
observation window contains the crossing. Verbatim from its Figure 2 caption:

> "Intention labels are represented by aggregated votes of human observers who watched videos
> of pedestrians from experiment start up to the critical point. Action labels are based on
> the observed action of crossing in front of the ego-vehicle. Sequences for action prediction
> task are sampled so that the observations end between 1-3s TTE."

and, on sample alignment: *"not all samples have overlaps (as shown in Figure 2)"*. That is an
acknowledgment that the anchors differ — it is not a leakage measurement, and no percentage of
contaminated windows appears anywhere in the paper.

**Methods-ready warrant (general principle):** *Leakage — the presence of information in the
training or evaluation data that would be unavailable at prediction time — is a recognised and
recurring failure mode in applied machine learning, and detecting it requires explicitly
checking the temporal relationship between features and the target event [kaufman2012leakage].*

**Methods-ready warrant (the anchor):** *The PIE benchmark's own authors note that intention
clips terminate at a `critical_point` while action samples are drawn at 1–3 s time-to-event,
so the two tasks do not share an anchor [rasouli2024diving]; we make the anchor explicit by
sampling every window relative to the `crossing_point` event and verifying the result.*

→ **Consequence:** the leakage audit remains ours to claim. Phrase it as "to our knowledge, no
prior work measures observation-window contamination on PIE", which R3 was meant to confirm and
has not yet been able to.

## 2. F1 ahead of accuracy and ROC-AUC — PARTIAL, and the honest phrasing matters

No source found says "use F1 as the primary metric". What the literature does support is
(a) demoting accuracy under imbalance and (b) preferring precision-recall to ROC under skew.

- **Davis & Goadrich, ICML 2006** (verified: pp. 233–240, DOI 10.1145/1143844.1143874) —
  establishes the formal relationship between the two curve families and that a large change
  in a PR curve can correspond to a small change in ROC space under class skew.
- **Saito & Rehmsmeier, PLoS ONE 2015** (verified: 10(3):e0118432,
  DOI 10.1371/journal.pone.0118432) — the title states the finding directly: the
  precision-recall plot is more informative than the ROC plot when evaluating binary
  classifiers on imbalanced datasets.
- Supporting note, from **Buda et al. 2018**'s abstract: ROC AUC was chosen there *"since
  overall accuracy metric is associated with notable difficulties in the context of imbalanced
  data."*

**Methods-ready warrant:** *Under class imbalance, overall accuracy is a poor summary and
ROC-based summaries can be optimistic relative to precision-recall analysis
[davis2006precision,saito2015precision]; we therefore lead with F1, the operating-point
harmonic mean of precision and recall, and report accuracy and threshold-free AUC alongside it.*

→ **Do not write** that the literature establishes F1 as the primary metric. The hierarchy is a
supervisor directive; the literature supports only that accuracy alone is inadequate here.

## 3. Cluster bootstrap over pedestrians — SUPPORTED

**Field & Welsh, "Bootstrapping clustered data", *J. R. Statist. Soc. B* 69(3):369–390, 2007,
DOI 10.1111/j.1467-9868.2007.00593.x** (verified on the Oxford Academic publisher record).
It compares bootstrap schemes for one-way clustered arrays and is the source that names
cluster-level resampling as the appropriate scheme when observations within a cluster are
dependent.

**Methods-ready warrant:** *Because the 2,094 test windows are drawn from 541 pedestrian tracks
with 50% overlap, windows within a track are not independent; we therefore resample whole
pedestrians rather than windows, the standard treatment for one-way clustered data
[fieldwelsh2007], and report those wider intervals wherever a confidence interval appears.*

## 4. Paired bootstrap for model contrasts — SUPPORTED

**Koehn, "Statistical Significance Tests for Machine Translation Evaluation", EMNLP 2004,
pp. 388–395** (verified from the ACL Anthology `.bib`). This is the standard citation for
paired bootstrap resampling as a system-comparison procedure on a shared test set.

**Methods-ready warrant:** *Model contrasts are computed by paired bootstrap resampling, in
which both systems are scored on identical resamples so that the interval describes the
difference rather than each system's separate variability [koehn2004significance].*

## 5. Equal search budget across architectures — SUPPORTED, strongly

This is the best-supported decision in the paper, and all three sources say something usefully
different. Abstracts read directly.

- **Melis, Dyer & Blunsom, ICLR 2018** — differing code bases and limited compute are
  *"uncontrolled sources of experimental variation"*; with large-scale black-box tuning,
  *"standard LSTM architectures, when properly regularised, outperform more recent models."*
  This is the closest analogue to our finding that a searched Transformer beats an un-searched
  LSTM while an un-searched Transformer does not.
- **Lucic et al., NeurIPS 2018** — *"most models can reach similar scores with enough
  hyperparameter optimization and random restarts. This suggests that improvements can arise
  from a higher computational budget and tuning more than fundamental algorithmic changes."*
  This is almost exactly our conclusion, transposed to GANs.
- **Greff et al., IEEE TNNLS 28(10):2222–2232, 2017** — hyperparameters of all eight LSTM
  variants were *"optimized separately using random search"*, 5,400 runs, and *"none of the
  variants can improve upon the standard LSTM architecture significantly."* This licenses both
  the per-family search and the null result.

**Methods-ready warrant:** *Architecture comparisons are confounded when candidates receive
unequal tuning effort, since apparent architectural gains often reflect search budget rather
than design [melis2018sota,lucic2018gans]; we therefore gave every family an identical,
pre-registered search budget and tuned each separately, following the protocol of
[greff2017odyssey].*

## 6. Matching width rather than parameter count — CONTRADICTED

**The canonical cell-comparison paper did it the other way.** Chung, Gulcehre, Cho & Bengio
(arXiv 1412.3555, 2014), §"Experiments Settings", verbatim:

> "As the primary objective of these experiments is to compare all three units fairly, we
> choose the size of each model so that each model has approximately the same number of
> parameters. We intentionally made the models to be small enough in order to avoid
> overfitting which can easily distract the comparison."

They matched **parameter count**, not hidden size. Our matched block holds width fixed at
h128, which gives the LSTM 594,561 parameters against the vanilla RNN's 149,121 — a 4×
capacity advantage to the gated cell.

**This is not fatal, and the direction is in our favour.** Matched width hands the LSTM more
capacity, so the un-gated RNN tying it is a *conservative* result: the smaller model matches
the larger one. But the manuscript must state the choice, name Chung et al.'s different
convention, and say why ours is defensible — a bare "matched configuration" claim will not
survive a reviewer who knows this paper. The tuned block independently addresses it, since
there each family chose its own width under an equal search budget.

**Methods-ready warrant:** *Cell-type comparisons must fix some notion of capacity; Chung et
al. equalised parameter count [chung2014gru], whereas we hold the recurrent width fixed, which
grants the gated cells more parameters than the un-gated one and therefore biases the matched
comparison against our conclusion rather than towards it. The tuned comparison removes the
choice altogether by letting each family select its own width under an identical search budget.*

## 7. Validation-only threshold tuning — SUPPORTED

**Cawley & Talbot, *JMLR* 11:2079–2107, 2010** (verified via DBLP). Selecting any quantity —
hyperparameters or a decision threshold — using the same data on which performance is reported
produces an optimistically biased estimate.

**Methods-ready warrant:** *The decision threshold is selected on the validation split only,
because selection performed on the evaluation data biases the reported performance
[cawley2010overfitting]; the test set is scored exactly once per experiment with the threshold
already fixed.*

## 8. Train-only normalization statistics — SUPPORTED

**Kaufman, Rosset & Perlich**, both versions verified: KDD 2011, pp. 556–563,
DOI 10.1145/2020408.2020496; and the extended *ACM TKDD* 6(4):1–21, 2012,
DOI 10.1145/2382577.2382579 (with Stitelman as fourth author). Cite the TKDD version.

**Methods-ready warrant:** *Per-feature standardisation statistics are computed on the training
split alone and applied unchanged to validation and test, since fitting any preprocessing step
on data used for evaluation constitutes leakage [kaufman2012leakage].*

## 9. Class-weighted BCE instead of resampling — CONTRADICTED

**Buda, Maki & Mazurowski, *Neural Networks* 106:249–259, 2018**
(DOI 10.1016/j.neunet.2018.07.011) is the systematic study, and it does **not** support our
choice. Verbatim conclusion: *"the method of addressing class imbalance that emerged as
dominant in almost all analyzed scenarios was oversampling"*, and *"oversampling should be
applied to the level that completely eliminates the imbalance"*.

**Scope caveat that makes this survivable:** their study is CNN image classification (MNIST,
CIFAR-10, ImageNet) at imbalance ratios far more severe than ours. Our training split is
1,366:812, a ratio of 1.68:1 — mild by their standards. They also find that *"thresholding
should be applied to compensate for prior class probabilities"*, which is precisely what our
validation-tuned threshold does.

**Recommended handling:** state the choice plainly, cite Buda et al. as the contrary evidence,
and justify from our own experiment — the GRU and RNN studies both swept
`pos_weight ∈ {1.0, 1.3, 1.682, 2.1, 2.5}` on validation and confirmed 1.682, so the operating
point is empirically selected rather than assumed. Do **not** cite Buda et al. as if it
supported weighting.

## 10. Five seeds and the limits of small-sample seed tests — SUPPORTED

- **Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks", MLSys 2021**
  (arXiv 2103.03098) — *"variance due to data sampling, parameter initialization and
  hyperparameter choice impact markedly the results"*, and analyses how standard comparison
  methods behave under that variance.
- **Reimers & Gurevych, EMNLP 2017, DOI 10.18653/v1/D17-1035** (verified from ACL Anthology) —
  reporting score distributions rather than single scores changes conclusions for
  LSTM sequence taggers.

**Methods-ready warrant:** *Five seeds is too few to support a well-powered significance test
across training runs [bouthillier2021variance,reimers2017reporting]; we therefore report the
seed-to-seed spread descriptively and rest every comparative claim on bootstrap resampling over
test instances, where the sample size is 2,094 windows from 541 pedestrians rather than 5.*

---

## Bibliographic discrepancy found

The pre-existing `lucic2018gans` entry gave pages **700–709**; DBLP records **698–707** for the
NeurIPS 2018 proceedings entry. I have set the entry to DBLP's range. If the published
proceedings PDF disagrees, DBLP is the one to re-check — flagging rather than silently choosing.

## Entries added to references.bib in this pass

`davis2006precision`, `saito2015precision`, `fieldwelsh2007`, `koehn2004significance`,
`cawley2010overfitting`, `kaufman2012leakage`, `buda2018systematic`, `bouthillier2021variance`,
`reimers2017reporting`. Already present and verified as correct: `greff2017odyssey`,
`melis2018sota`, `lucic2018gans` (pages corrected), `chung2014gru`, `rasouli2024diving`.

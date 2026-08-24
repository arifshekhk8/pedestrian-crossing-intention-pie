# R5 — Reconciliation of the master brief against Stage-1 evidence

Every item below is a **decision for the author**. Nothing in the brief has been changed.
Items are numbered `D1`, `D2`, … so you can answer them by number.

Severity: 🔴 blocks writing a specific sentence · 🟠 changes wording or adds a footnote ·
🟡 record-keeping, no prose impact.

---

# 1. Numbers in section C that R1 or R3 contradicts

## D1 🔴 PedCMT's accuracy of 0.92 cannot be printed

- **Brief section C:** `PedCMT  0.92 / -- / --  box+ego-speed (2)`
- **R1:** PedCMT is closed-access with no green OA copy in any indexed repository; the official
  repo publishes no results. The 0.92 traces to the abstract's qualitative "on par with" claim,
  not to a printed table. The only sourced figure found is **third-party** — ESIA
  (arXiv 2604.23728) Table I gives PedCMT **Acc 0.93, F1 0.87, AUC 0.92**, which does not match
  0.92 accuracy either.
- Also from R1: PedCMT's released code uses a **random 70/15/15 split**, so even a recovered
  number would not be standard-protocol.

**Options.** (a) Drop the row; keep PedCMT as the prose concession only. (b) Keep the row with
ESIA's 0.93/0.92/0.87 marked third-party and footnote the random split. **Recommend (a)** — the
concession does not need a number, and a third-party row for a paper with a non-standard split
invites the exact criticism we are trying to pre-empt.

## D2 🔴 "The highest ROC-AUC in the standard-protocol table" is no longer true

- **Brief sections A and C** state this in the thesis sentence and the framing paragraph.
- **R1 verified** MFT at AUC **0.94** and **PIP-Net at 0.94 AUC / 0.88 F1** — the latter also
  above our F1 band of 0.844–0.852.
- See D4 and D12; this is the single most consequential wording change in Stage 1.

**Proposed replacement wording** (also answers section 4 below):
> "…reach F1 0.844–0.852 and ROC-AUC 0.940–0.950, at the top of the range reported on this
> benchmark, using two input streams rather than the three to seven of the methods alongside
> them."

## D3 🔴 PIP-Net's exclusion rationale is wrong

- **Brief section C:** *"Do not tabulate PIP-Net (custom random split, prose context only)."*
- **R1 §2a:** PIP-Net's "approximately 50% (880 samples) … 40% (719) for testing, and 10% (243)
  for validation" are **exactly PCPA's printed PIE track counts** (880 train / 243 val / 719
  test). It is the benchmark split described as proportions, not a randomisation. Verified
  against PCPA's own paper and PIP-Net's accepted T-ITS manuscript.
- Its verified row is **0.92 / 0.94 / 0.88** (Table III, at ETC = 0.5 s), which would sit at or
  above us on both AUC and F1.

**Options.** (a) Re-include PIP-Net with an ETC = 0.5 s footnote and 7-stream count.
(b) Keep it out, but on the honest ground — a different operating point and a seven-stream
pipeline — not on a split claim we now know is false. **Recommend (a).** Excluding the strongest
comparable result weakens us more than including it: we still lead on streams and latency, and
including it demonstrates we are not curating the table.

## D4 🟠 MFT is verified but sits on the *other* standard split

- **Brief section C:** `MFT (2025) 0.90 / 0.94 / 0.83  4 context streams`. R1 confirms the
  numbers first-hand from its main results table — the "AUC only in a cost table" suspicion was
  wrong.
- But MFT states verbatim: *"Following [5], we use set01, set02, and set06 for training, set04
  and set05 for validation, and set03 for testing."* That is the PCPA split, not ours.

**Decision:** print the row, and footnote the split. Ties into D5.

## D5 🔴 There are two "standard" PIE splits, and the brief assumes one

- **Brief section C:** *"Split by recording set: train set01/02/04, validation set05/06, test
  set03. No random split."* — correct for us, and it is PIE's own toolkit default.
- **R1 §2:** PCPA's paper — the benchmark everyone cites — states verbatim: *"videos from set01,
  set02 and set06 are used for training, set04 and set05 for validation and set03 for testing.
  The number of pedestrian tracks in PIE is 880, 243 and 719."* **set04 and set06 are swapped.**
- Who uses which: **toolkit split** — us, GTransPDM's Table I, Pedestrian Graph+'s code;
  **PCPA split** — PCPA, MFT, PIP-Net; **neither** — PedCMT (random 70/15/15), BiPed (its own
  sampling, 3,980 sequences).

**Decision:** how to handle in Materials and Methods. **Recommend:** one sentence stating both
conventions, noting test = set03 in both, and that we follow the dataset toolkit's default. This
converts a hidden comparability problem into a visible piece of protocol rigor.

## D6 🟠 Not all tabulated AUCs are the same statistic

- **R1 §2b:** the PCPA benchmark computes `roc_auc_score(labels, np.round(preds))` — AUC on
  hard 0/1 decisions. PedCMT's released code does the same. Ours is computed on sigmoid
  probabilities.
- The brief nowhere mentions this.

**Decision:** add to the Discussion's comparability caveats, and reference it when we state our
AUC position. Strengthens D2's softened wording rather than undermining it.

## D7 🟠 Two tabulated methods used a different configuration, now confirmed from their own texts

- **Brief section C** carries GTransPDM's footnote as something to "verify vs originals".
  R1 verified it against the originals: **BiPed** clips at the crossing event, samples at 50%
  overlap over TTE 30–60, yields 3,980 training sequences and re-ran competitors padded to *its*
  setup; **Pedestrian Graph+** uses **three classes** (crossing / not-crossing / irrelevant) and
  **observation length 32**, not 16, per its released code.

**Decision:** footnote both rows in the baseline table. This is now first-party evidence, not
GTransPDM's assertion.

## D8 🟡 IntFormer's protocol deviation is documented

R1 verified IntFormer's own words: it halves the benchmark's 16 frames, *"sampling the input
time interval at 15 fps instead of 30 fps, keeping sequences with N/2 = 8 evenly spaced frames"*
— scoped to image data. Its 0.89/0.92/0.81 is first-party and correct. Optional footnote.

## D9 🟡 BiPed now has its own citation

Brief amendment 4 confirmed: BiPed is Rasouli, Rohani & Luo, ICCV 2021, pp. 15600–15610,
arXiv 2012.03298. `rasouli2021biped` added; the misleading note on the PedFormer entry removed.
No prose decision.

## D10 🟡 GTransPDM version trap

v1 prints 91.21 / 88.13 / 81.61 for the full model and contains **no 92% row**; the 0.92
without-pose ablation appears only in v2/SPL. Cite v2/SPL. Already reflected in the bib.

## D11 🟡 PCPA's own row settles a three-way disagreement

PCPA's Table 3 final model prints **0.87 / 0.86 / 0.77**. GTransPDM transcribes it correctly;
MFT's 0.87/0.85/0.78 blends two of PCPA's tables; BiPed's 0.86/0.84/0.76 matches no row;
IntFormer prints 0.86/0.86/0.77. Use PCPA's own. No prose decision.

## D12 🔴 New competitor the brief predates: EfficientPIE (IJCAI 2025)

- **R3:** *EfficientPIE: Real-Time Prediction on Pedestrian Crossing Intention with Sole
  Observation*, IJCAI 2025. Input is a **single 300×300 crop** — no sequence, no ego-speed.
  Reports PIE 0.92 / 0.92 / 0.95 at 0.21 ms.
- **Not tabulatable:** its 19,086 single-frame samples are a different extraction from the
  ~1,842-track benchmark, and the numbers are internally implausible under a 32.5%-positive test
  set (F1 0.95 with precision 0.96 implies accuracy ≈ 0.97, not the 0.92 printed).

**Decision:** add it to the Introduction/Discussion concession list as a fourth minimal-input
precedent, with the protocol caveat. It makes our concession paragraph more credible, and its
0.21 ms undercuts any "fastest" claim we might be tempted to make.

## D13 🟡 Unresolved: PIE's "~6 h" and "Toronto"

Still not sourced from PIE's own paper. MFT states *"PIE dataset consists of 6 hours of HD video
recorded in Toronto"* — third-party. Either cite PIE (Rasouli et al. 2019) directly after
checking it, or drop both details. One check, five minutes, when we write Methods.

---

# 2. B5 decisions with no citation — must be justified from our own experiments

R2 covered ten items. These are the B5 decisions **R2 found no external warrant for**, so the
manuscript must carry an explicit stated reason plus the supporting experiment.

## D14 🔴 Raw pixel coordinates rather than image-normalized

**No citation exists, and R2 did not find one.** This is a pure engineering choice. The only
defence is our own: the inference contract is fixed and the normalization is a per-feature
train-split z-score, which makes the absolute pixel scale irrelevant after standardisation.
**Decision:** state it as a contract detail with that reason, or run a small ablation. I would
state it — it is not a claim anyone would contest, and an ablation spends pages for nothing.

## D15 🔴 The matched-width convention (contradicted, see also D18)

Chung et al. matched **parameter count**. We match width. R2 item 6 gives the defence — matched
width favours the gated cells, so the un-gated RNN's tie is conservative — but this must be
written explicitly, naming Chung's different convention. **Decision:** confirm you want the
"conservative direction" defence rather than adding parameter-matched runs.

## D16 🟠 pos_weight = 1.682 as a value

The train-split ratio is arithmetic, not a choice. The *choice to weight rather than resample*
is contradicted (D19). The value is defended by our own symmetric sweep over
{1.0, 1.3, 1.682, 2.1, 2.5} in the GRU and RNN studies. **Decision:** cite the sweep, not a paper.

## D17 🟠 F1 as *primary* metric

R2: the literature supports demoting accuracy under imbalance and preferring precision-recall to
ROC under skew, but **nothing says F1 first**. **Decision:** attribute the hierarchy honestly —
it is a stated reporting choice appropriate to an imbalanced safety-critical decision, supported
by the imbalance literature but not dictated by it. Do not imply the field ranks it this way.

## D18 🟠 Crossing-point anchoring itself

R2 item 1 came back **SILENT** — no prior work audits this. Kaufman et al. supplies only the
general leakage framework; Rasouli & Kotseruba note the anchor difference without measuring it.
**This is good news** (it is our contribution) but it means the anchoring rule is justified by
our own audit alone. **Decision:** none needed; just be aware the warrant is internal.

## D19 🟠 Class weighting instead of resampling — actively contradicted

Buda et al.: *"the method of addressing class imbalance that emerged as dominant in almost all
analyzed scenarios was oversampling."* **Decision:** cite Buda as contrary evidence, note the
scope difference (severe CNN image imbalance vs our 1.68:1 sequences), and rest on our sweep.
Do **not** cite Buda as if it supported us.

## D20 🟡 Items that turned out to be adequately warranted

For the record, these need no internal justification: observation window of 16 frames and the
30–60 frame horizon (both fixed by PCPA's benchmark protocol, verified in R1 — PCPA even
motivates the TTE range by traffic studies); split by recording set (PIE toolkit + PCPA);
train-only normalization (Kaufman); validation-only threshold (Cawley & Talbot); cluster
bootstrap (Field & Welsh); paired bootstrap (Koehn); equal search budgets (Melis, Lucic, Greff);
five seeds as a *limitation* (Bouthillier, Reimers & Gurevych); hidden size and depth (our
Issues 7–8, plus Greff for the per-architecture search principle).

---

# 3. MDPI MTI requirements the current plan does not satisfy

## D21 🔴 Generative-AI disclosure is owed, in Materials and Methods

R4: the template requires disclosure *"where applicable"* of how GenAI was used, and exempts
only *"superficial text editing (e.g., grammar, spelling, punctuation, and formatting)"*. This
manuscript is being drafted with AI assistance well past that threshold. **Decision required:
the wording is yours.** I will not draft a disclosure that describes your process on your
behalf. Tell me what it should say and where — Methods, Acknowledgments, or both.

## D22 🟠 The reference list currently violates the static-content rule

R4: *"The citation list should contain only references to static content… Content that does not
fulfil these criteria may be listed directly in the main text and might include company
websites, or websites to track project development (such as github)."*
- `jocher2025yolo` is a `@misc` pointing at `github.com/ultralytics/ultralytics` → must move
  into the text or Data Availability.
- `ec2025roadfatalities` is a news URL → borderline; check when we write the Introduction.
- Our own repository URL (B9 placeholder) belongs in Data Availability, not the bibliography.
- ByteTrack keeps its ECCV citation, which is static and fine.

## D23 🟠 arXiv-only entries need access dates

R4 quotes the ACS preprint pattern: *"arXiv 2004, arXiv:physics/0402096. Available online: URL
(accessed on DD Month YYYY)."* Affected: `lorenzo2021intformer`, `li2025mft`,
`bokkasam2025iddped`, and any other `@misc`/arXiv entry. Mechanical fix at bibliography time.

## D24 🔴 Institutional Review Board statement — open decision

Figure 9 publishes blurred pedestrian faces from PIE. Options: a waiver justification (secondary
analysis of a public dataset, faces blurred) or "Not applicable". Either way the blurring gets
declared. **This is the one I flagged earlier and still need from you.**

## D25 🟠 Captions must be self-contained

R4 quotes the Style Guide's explicit contrast between a useless and a good caption, and the
reason: captions appear online detached from the article. Consequence: the baseline-table
caption must carry the protocol caveat *inside itself* (which B7 already wants), and every other
caption must be readable alone. Affects how I write all 11 captions.

## D26 🟡 MTI imposes no length limit

*"No restriction on the maximum length of the papers, number of figures or colors."* The 15–17
page ceiling in B2 is your editorial constraint, not the journal's. Flagging only so that if a
section needs another half page, you know the cost is yours to weigh, not a submission risk.

## D27 🟡 Six requirements still unverified

MDPI 403s automated access. Unconfirmed: whether DOIs are mandatory; MTI-specific keyword count;
preprint policy; graphical-abstract expectations; APC; submission-system checklist. One manual
browser visit closes all six.

---

# 4. Section A claims that are no longer defensible as stated

## D28 🔴 The thesis sentence

**Current (section A):** *"…models consuming only a pedestrian bounding box and the ego-vehicle's
speed reach F1 0.844-0.852 and the highest ROC-AUC in the standard-protocol table…"*

**Why it fails:** PIP-Net's verified 0.94 AUC / 0.88 F1 and MFT's verified 0.94 AUC. On AUC we
are level with the best, not above it; on F1, PIP-Net is above us.

**Proposed replacement:**
> "On a leakage-free re-extraction of the PIE benchmark, models consuming only a pedestrian
> bounding box and the ego-vehicle's speed reach F1 0.844–0.852 and ROC-AUC 0.940–0.950 — at the
> top of the range reported for this benchmark, with two input streams rather than three to
> seven — and four very different temporal encoders trained through one identical engine are
> statistically indistinguishable on the primary metric, so the input signal, not the
> architecture, carries this task."

## D29 🟠 The MANDATORY FRAMING RULE list should gain a fourth name

**Current:** PedCMT, the pose-free GTransPDM ablation, and Achaji et al.
**Add:** EfficientPIE (IJCAI 2025), which claims a *single image* suffices. Its existence makes
the concession stronger, not weaker — the minimal-input direction is demonstrably active, and
our contribution is the leakage-free protocol and the causal explanation, exactly as the brief
already frames it.

## D30 🟠 Contribution 2's implicit "we lead" framing

**Current section A:** contribution 2 is fine as written, but any Results or Discussion sentence
implying we top the table needs the D28 treatment. Specifically the section-C framing paragraph
*"ours = top AUC (0.94–0.95), near-ceiling F1"* must become "at the top of the reported range"
and "within 0.02–0.04 of the best reported F1" (PIP-Net 0.88, not PedFormer 0.87, is now the
number to measure against).

## D31 ✅ Claims that R3 confirms are safe

- **Contribution 1 (leakage audit and re-extraction):** no competing audit found; the negative
  is reproducible from the search terms recorded in R3 §1. Strongest claim in the paper.
- **Contribution 3 (four-family isolation):** no controlled architecture comparison exists for
  this task or its close neighbours. Safe.
- **The ego-speed dominance result:** already conceded as confirmatory in the brief; CAPFI
  corroborates both the dominance and the driver-side bias. R3 adds a concrete future-work
  option — CAPFI's proximity-change-rate representation — to replace a vague mitigation promise.

---

# Decision summary

**Need your answer before I can write the affected sections:**
D1 (drop or third-party PedCMT), D3 (re-include PIP-Net?), D5 (how to present two splits),
D15 (accept the conservative-direction defence?), D21 (GenAI disclosure wording), D24 (IRB),
D28 (accept the replacement thesis sentence?).

**I can proceed on my own judgement unless you object:**
D2, D4, D6, D7, D8, D12, D16, D17, D18, D19, D22, D23, D25, D29, D30.

**Record-keeping only:** D9, D10, D11, D13, D20, D26, D27, D31.

**Repo debt outside `submission/`, not actioned this session:**
`journal_prep/issue3_baseline_comparison/03_baseline_comparison.md` carries the wrong PIP-Net
rationale (D3) and a stale RNN-vs-LSTM confidence interval; `relatedwork.md` §4 lists BiPed
under PedFormer's arXiv identifier (D9).

---

# Author rulings (recorded; supersede the options above)

**D1 — PedCMT row dropped.** Concession stays in prose in the Introduction and the Discussion,
with one sentence noting that its released code uses a random 70/15/15 split, so its numbers are
not standard-protocol comparable. The value 0.92 is never printed.

**D3 — PIP-Net row re-included.** Receipts verified by the caller before the ruling took effect:
- PCPA, WACV 2021 p. 1260: *"The number of pedestrian tracks in PIE is 880, 243 and 719 in
  train, validation and test sets."*
- PIP-Net (arXiv 2402.12810 v3 = accepted T-ITS): *"We utilised approximately 50% (880 samples)
  of the dataset for training, 40% (719) for testing, and 10% (243) for validation as per the
  same split proportion as [12]."*
- PIP-Net Table III: `PIP-Net-α (Ours) 0.92 0.94 0.88 0.89 0.88`.
Identical counts, identical roles. **`journal_prep/issue3_baseline_comparison/03_baseline_comparison.md`
is wrong to call this a custom random split**; that correction is recorded here and not applied
to the original document.

**New finding that arrived with the receipt — the horizon differs.** PIP-Net's Table III carries
the footnote *"All models are evaluated at ETC = 0.5s"*, and its text confirms ETC is the
prediction horizon (*"the model can predict the pedestrians' estimated time to cross (ETC), 1 to
4 seconds in advance"*), with performance falling as it grows (*"our model shows a 6.6% decrease
in the AUC from ETC = 1 to 2"*). PIP-Net's 0.94/0.88 is therefore measured at 0.5 s; ours is at
TTE 30--60 frames, i.e. 1--2 s. PedFormer is horizon-comparable to us (0.5 s observation, TTE
1--2 s).

**D28 as superseded by the author.** No paper is designated "the ceiling". Table~4 gains a
**horizon column** beside the streams column, populated from R1 for every row and marked "not
reported" where R1 could not verify it. Two factual sentences, neither ranked: the highest F1 in
the table is PIP-Net's 0.88, obtained at a 0.5 s horizon with seven input streams; the highest
among methods evaluated at a horizon comparable to ours is PedFormer's 0.87. The horizon caveat
is justified by our own matched-cohort ablation (AUC 0.961 at 1.0 s falling to 0.919 at 2.0 s),
stated in the text beside the table, which makes the caveat self-supporting rather than asserted.
Our AUC of 0.948 is **not** to be leaned on against PIP-Net's 0.94: the same horizon caveat that
protects the F1 comparison forbids claiming the AUC one. State it descriptively, claim nothing.
"Within about 0.03" survives, stated against PedFormer at a comparable horizon and labelled as
such. **No sentence anywhere claims the top of the table on any metric.**

**D21 — disclosure wording fixed.** Tool and version: "Claude Code (Anthropic), Claude Opus".
Scope widened to the record: drafting and editing text; writing and debugging analysis and
figure-generation code; analysis and interpretation of results; literature search and
verification. MDPI's closing sentence kept verbatim.

**D24 — "Not applicable"**, with the blurring declared in the same statement.

**D14, D15, D19 — as recommended:** one sentence for raw pixel coordinates with no ablation;
Chung et al. cited as matching parameter count with the conservative-direction argument made
explicitly and parameter counts kept in every table; Buda et al. cited as contrary evidence with
our own pos\_weight sweep carrying the choice.

All fifteen minor items (D2, D4, D6, D7, D8, D12, D16, D17, D18, D22, D23, D25, D29, D30, plus
EfficientPIE in the concession list) approved as proposed.

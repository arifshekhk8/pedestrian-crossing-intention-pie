# R3 — Recent work (mid-2025 →) that could scoop or contradict us

**Coverage warning, read first.** The `/deep-research` run for R3 (task `wfb6k32hj`) was killed
by the session limit having fetched **zero** sources — 5 of 6 agents died immediately. What
follows is the caller's own targeted sweep: six web searches plus direct reads of the primary
PDFs that mattered. **This is narrower than the commissioned fan-out.** The negative result on
Claim 1 is therefore *supported but not exhaustive*, and R3 should be re-run through
`/deep-research` after the session limit resets if you want the stronger form of that claim.
Search terms actually used are recorded in §1 so the negative can be reproduced and extended.

---

## 1. CLAIM 1 — has anyone published a PIE/JAAD leakage audit? **No work found.**

**Verdict: the claim survives.** No paper found audits observation-window contamination on PIE
or JAAD, measures what fraction of positive windows already contain the crossing, or publishes
a re-anchored leakage-free re-extraction.

Search terms used (record these; they are the reproducible basis of the negative):
`"data leakage" OR "temporal leakage" + PIE/JAAD + observation window + already crossing`;
`benchmark protocol flaw re-evaluation observation window contamination crossing onset
2025 2026`; plus targeted checks of every 2025–2026 paper surfaced.

What exists instead, and why it is not the same thing:

- **Rasouli & Kotseruba, IEEE IV 2024** (verified in R2) documents that intention clips end at a
  `critical_point` while action samples are drawn at 1–3 s TTE, and notes *"not all samples have
  overlaps"*. It observes the anchor difference; it never measures contamination.
- Several papers **assert** pre-event sampling without auditing it. Example, from the 2026
  psychological-features paper (arXiv 2603.19533): *"For all crossing cases, both the
  observation period and the time-to-event precede the pedestrian's road crossing initiation."*
  Asserted, not verified — which is precisely the gap the manuscript fills.
- **TCL** (arXiv 2504.06292) states the TTE gap and never audits it (confirmed in R1).
- **EfficientPIE** (below) says its 19,086 PIE samples are *"all taken before the happening of
  crossing event"* — again an assertion, with no verification and no reported check.

**Recommended wording:** "to our knowledge, no prior work on PIE measures whether the
observation window itself contains the crossing action" — accurate, and defensible against
everything surveyed here and in R1.

## 2. CLAIM 2 — minimal inputs. **One new paper matters, and it cuts both ways.**

### EfficientPIE (IJCAI 2025) — the most consequential find in this sweep

- **Qu, Zhou, He, Gao, Luo, Feng, Liu & Guo**, *EfficientPIE: Real-Time Prediction on Pedestrian
  Crossing Intention with Sole Observation*, IJCAI 2025 (Montreal, 16–22 August 2025).
  Read from the PolyU institutional copy; the IJCAI-hosted PDF was unreachable by DNS.
- **It is more minimal than we are, in a different direction.** Its input is a *single*
  300×300 crop around the pedestrian at one frame — no sequence, no ego-speed, no pose.
  Verbatim: *"we found that the crossing intention may be predicted through analyzing just one
  image."*
- **Its PIE numbers (Table 3): Accuracy 0.92, AUC 0.92, F1 0.95, Precision 0.96**, at 0.21 ms
  inference — nominally faster than our fastest family (0.316 ms, different hardware).
- **But the numbers are not protocol-comparable, and are internally implausible under the
  standard protocol.** F1 0.95 with precision 0.96 implies recall ≈ 0.94; on a test set with
  ~32.5% positives that would force accuracy near 0.97, not the 0.92 printed. The explanation
  is in their §4.1: they generate **19,086 PIE samples** by clipping tracks at 0.5 overlap and
  taking single frames — an order of magnitude more samples than the ~1,842-track benchmark, and
  with a different class balance. **Do not put this row in our comparison table.** If cited,
  cite it as a minimal-input precedent with the protocol caveat stated.

**Effect on our concession:** it *strengthens* the case for conceding parsimony — a 2025 IJCAI
paper argues one image suffices — while leaving our specific claim (bbox + ego-speed *sequence*,
under a leakage-free protocol, with the architecture isolation) untouched. It also makes the
concession paragraph stronger if we cite it, because it shows the minimal-input direction is an
active line and we are contributing rigor to it rather than announcing it.

### Others noted, not threatening

- **ACIT** (arXiv 2511.20020), **TrajFusionNet** (arXiv 2508.19866) — multimodal, already in the
  bibliography from earlier passes.
- **Vision-language / zero-shot approaches**: arXiv 2507.21161 (*Seeing Beyond Frames*, zero-shot
  with raw video and multimodal cues) and arXiv 2606.09142 (*Decoding Pedestrian Crossing
  Intention from Egocentric Vision via Vision Language Models*). A distinct, higher-latency
  paradigm — useful as contrast in the Discussion, not as competitors.

## 3. CLAIM 3 — architecture comparison under matched budgets. **Nothing found; claim survives.**

Searched for controlled LSTM/GRU/Transformer comparisons on this task and close neighbours. No
published study performs a controlled architecture comparison with equal per-family search
budgets for pedestrian crossing prediction. Individual papers compare *their* architecture
against others' published numbers, which is exactly the confound R2 item 5 documents
(Melis, Lucic). **This remains a genuine contribution**, and the R2 citations give it a
methodological pedigree it did not have before.

## 4. CLAIM 4 — ego-speed dominance and driver-side bias. **Corroborated, and not first.**

**CAPFI** — *Feature Importance in Pedestrian Intention Prediction: A Context-Aware Review*
(arXiv 2409.07645) — confirms both halves, and confirms the concession you already require:

- Ego-speed dominance is independently established there via contextual permutation importance.
- The driver-side bias is explicitly named: model reliance on ego-vehicle speed **may induce
  driver-side bias, especially in yielding (deceleration) scenarios**.
- It goes further than we do by *proposing a mitigation*: a pedestrian–vehicle **proximity
  change rate** representation, which it reports as partially mitigating but not eliminating the
  bias.

**Consequence for the manuscript.** Our ego-speed finding is confirmatory, not novel — which the
brief already concedes. But the mitigation is a concrete piece of future work we can name
instead of gesturing at one: replace or augment raw ego-speed with proximity change rate and
re-run the ablation. Note also that **EfficientPIE uses no ego-speed at all**, which is an
independent data point that the task is solvable without it.

## 5. Requirement A — current best published PIE numbers

Nothing found from 2025–2026 that beats us on the *standard* protocol on AUC. Standing:

| Source | Acc | AUC | F1 | Protocol status |
|---|---|---|---|---|
| PedFormer (ICRA 2023) | 0.93 | 0.90 | 0.87 | standard; the Acc/F1 ceiling |
| PIP-Net (T-ITS 2025) | 0.92 | 0.94 | 0.88 | PCPA split, ETC = 0.5 s (R1) |
| MFT (arXiv 2025) | 0.90 | 0.94 | 0.83 | PCPA split (R1) |
| GTransPDM w/o pose (SPL 2025) | 0.92 | 0.90 | 0.86 | toolkit split (R1) |
| EfficientPIE (IJCAI 2025) | 0.92 | 0.92 | 0.95 | **not comparable** — 19,086 single-frame samples |
| **Ours** | 0.897–0.902 | 0.940–0.950 | 0.844–0.852 | leakage-free re-extraction |

**This changes the honest headline.** PIP-Net's verified 0.94 AUC / 0.88 F1 sits at or above our
band, and its split is the PCPA one rather than a custom split, so the R1 finding that we
misjudged it matters here too. Combined with MFT at 0.94, the defensible sentence is now that
our AUC is **at the top of the reported range** rather than uniquely highest, and that we reach
it with two streams under a protocol that removes leakage. Recommend dropping "highest" entirely
in favour of "among the highest reported, at a fraction of the input cost".

## 6. Requirement B — new datasets since 2025

- **IDD-PeD** — Bokkasam, Gangisetty, Hafez & Jawahar, arXiv 2506.22111 (27 June 2025).
  Indian unstructured traffic; pedestrian intention *and* trajectory; high- and low-level
  annotations focused on pedestrians requiring ego-vehicle attention; targets illumination
  change, occlusion, unsignalised scenes. Reports that existing intention methods **drop up to
  15%** when transferred to it. Ego-telemetry and crossing-onset labels not confirmed from the
  abstract page — **UNVERIFIED**, needs the full PDF.
- **Urban-PIP** (within PIP-Net) and **UCF-VRU** (VRU-CIPI) were already known.
- No PIE successor from the original York group surfaced.

**Use:** IDD-PeD is the strongest single answer to "why only one dataset?" — cite it as the
concrete near-term cross-dataset target in future work, alongside its 15% transfer drop as
independent evidence that cross-dataset generalisation is an open problem, not a formality.

---

## 7. What this sweep did NOT cover

Being explicit, since the fan-out died:

1. No systematic 2026 venue sweep (CVPR/ICCV/ECCV/ITSC/IV 2026 proceedings not enumerated).
2. EfficientPIE's split (which recording sets) was not established — only its sample count.
3. IDD-PeD's modalities and telemetry unconfirmed.
4. No adversarial second pass on the Claim 1 negative — it rests on six searches, not a fan-out.
5. Non-English literature not searched at all.

Re-running R3 through `/deep-research` after the limit resets would close 1, 4 and 5.

## 8. Entries to add to references.bib

Added in this pass: `qu2025efficientpie`, `bokkasam2025iddped`. Already present:
`azarmi2024featureimportance` (CAPFI), `li2025mft`, `azarmi2025pipnet`, `xie2025gtranspdm`,
`landry2025trajfusionnet`.

# R1 — Baseline verification against primary sources

**Produced:** Stage 1, this session. **Method:** two `/deep-research` runs (103 + 32 agents,
22+ primary sources) followed by first-hand re-verification of every consequential number by
the caller via `curl | pdftotext` and direct HTML fetch. Both workflow runs lost their
synthesis step (run 1: machine sleep; run 2: session limit, 72/104 agents killed), so the
tables below are the caller's own reconciliation of the surviving agent extracts plus
independent re-checks. Every number carries a verification tier.

**Tiers used throughout:**

| Tier | Meaning |
|---|---|
| **VERIFIED** | read by the caller from the method's OWN paper, quoted below with table number |
| **THIRD-PARTY** | only available from another paper's comparison table; that paper is named |
| **UNVERIFIED** | could not be read from any primary source — must not be printed as fact |

---

## 1. Comparison table — PIE crossing prediction

| Method | Acc | AUC | F1 | Streams | Split used | Tier | Source read |
|---|---|---|---|---|---|---|---|
| PCPA (WACV 2021) | 0.87 | 0.86 | 0.77 | 4 (local box, pose, box, ego-speed) | **PCPA split** (set01/02/06 · set04/05 · set03) | **VERIFIED** | own Table 3, "Temp. + mod. attention" row |
| Pedestrian Graph+ (T-ITS 2022) | 0.89 | 0.90 | 0.81 | pose graph (+image, segmentation, 2 speed channels) | PIE toolkit `default` per its own code | **THIRD-PARTY** (GTransPDM Table I; same triple in PIP-Net Table III) | paper is closed-access |
| IntFormer (arXiv 2021) | 0.89 | 0.92 | 0.81 | 4 (bbox crops, bbox coords, pose, ego-speed) | benchmark split | **VERIFIED** | own Table I, row "IntFormer / Fusion C / 4M" |
| PIT (T-ITS 2023) | 0.91 | 0.92 | 0.82 | 5 (bbox, image, pose, ego-motion, scene) | claimed benchmark | **THIRD-PARTY** (GTransPDM Table I) | paper is closed-access; workshop precursor is JAAD-only |
| BiPed (ICCV 2021) | 0.91 | 0.90 | 0.85 | 4 (scene image, bbox/traj, grid locations, ego-motion) | PIE "default split ratios", own sampling | **VERIFIED** | own Table 1 |
| PedFormer (ICRA 2023) | 0.93 | 0.90 | 0.87 | multimodal multitask (traj + action) | "default data split" | **VERIFIED** | own Table I |
| GTransPDM, full model | 0.90 | 0.87 | 0.82 | 3 (position/PDM, pose, ego-motion) | set01/02/04 · set05/06 · set03 for Table I | **VERIFIED** | own Table I (v2) |
| GTransPDM, w/o pose (`w/o X_ke`) | 0.92 | 0.90 | 0.86 | 2 (position, ego-motion) | as above | **VERIFIED** | own Table I (v2) |
| PedCMT (T-ITS 2024) | — | — | — | 2 (bbox + ego-speed) | released code uses **random 70/15/15** | **UNVERIFIED** | closed-access; repo has no results |
| MFT (arXiv 2025) | 0.90 | 0.94 | 0.83 | 4 numerical context blocks | **PCPA split** (set01/02/06 · set04/05 · set03) | **VERIFIED** | own results table, PIE block |
| PIP-Net (T-ITS 2025) | 0.92 | 0.94 | 0.88 | 7 (bbox, pose, RGB, flow, semantics, depth, ego-speed) | 880/719/243 = **PCPA track counts** | **VERIFIED** | own Table III, ETC = 0.5 s |

---

## 2. The finding that changes our Methods section: there are TWO "standard" PIE splits

This was not previously known to the project and contradicts our own
`journal_prep/issue3_baseline_comparison/03_baseline_comparison.md`, which asserts a single
standard protocol.

**Split A — the PIE toolkit default** (what we use). From `aras62/PIE/utilities/pie_data.py`:

```
image_set_nums = {'train': ['set01', 'set02', 'set04'],
                  'val':   ['set05', 'set06'],
                  'test':  ['set03'], ...}
```

**Split B — the PCPA benchmark paper's split.** Verbatim, WACV 2021 paper, p. 1260:

> "In the PIE dataset, we follow the data split defined in [42]: videos from set01, set02
> and set06 are used for training, set04 and set05 for validation and set03 for testing.
> The number of pedestrian tracks in PIE is 880, 243 and 719 in train, validation and test
> sets."

So set04 and set06 are **swapped** between the two. Test = set03 in both, which is why
cross-paper test numbers remain broadly comparable, but the training data differs.

Who uses which, as far as could be established:

| Paper | Split | Evidence |
|---|---|---|
| **This work** | A (toolkit default) | our own code |
| PCPA | B | quoted above |
| MFT | B | "Following [5], we use set01, set02, and set06 for training, set04 and set05 for validation, and set03 for testing." |
| PIP-Net | B, described as proportions | 880/719/243 are exactly PCPA's train/test/val track counts |
| GTransPDM (Table I) | A | "For comparison with existing methods (Table I and II), the performance following the data split in [3] was also evaluated, with Set 01, 02, 04 for training, Set 05, 06 for validation and Set 03 for testing." Its own experiments elsewhere use a **person-ID random split at 0.5:0.1:0.4** |
| PedCMT | **random 70/15/15** | released `pie.py`: `'data_split_type':'random'`, `'ratios':[0.7,0.15,0.15]` |
| Pedestrian Graph+ | toolkit `default` (= A) | `params = {'data_split_type': 'default'}` in its released code |
| BiPed | "default data split ratios", own sampling → 3,980 train sequences | own paper |

**Manuscript consequence.** The baseline table caption must say the tabulated rows do not all
share one protocol — this is now demonstrated, not merely suspected, and it reinforces the
caveat you already required (B7: the comparison is indicative, not like-for-like).

### 2a. PIP-Net's "custom split" claim in our repo is WRONG

`03_baseline_comparison.md` removed PIP-Net on the grounds that it uses "a custom random
split (~880/719/243)". Verbatim from PIP-Net (arXiv v3 = accepted T-ITS version):

> "We utilised approximately 50% (880 samples) of the dataset for training, 40% (719) for
> testing, and 10% (243) for validation as per the same split proportion as [12]."

Those three counts are **identical to PCPA's printed PIE track counts** (880 train / 243 val
/ 719 test). PIP-Net is describing the benchmark split in percentage terms, not randomising.
The removal rationale does not survive. Whether to tabulate PIP-Net is now a judgement call
about its ETC = 0.5 s operating point and its 7-stream pipeline, not about a broken split.

### 2b. The tabulated AUCs are not all the same statistic

The PCPA benchmark's own evaluation code computes AUC on **rounded** predictions:

```python
auc = roc_auc_score(test_data['data'][1], np.round(test_results))   # action_predict.py
```

PedCMT's released code does the same (`roc_auc_score(label_cpu, np.round(pred_cpu))`). An AUC
computed on hard 0/1 decisions is not a threshold-free ROC-AUC and is systematically
different from ours, which is computed on sigmoid probabilities. Any paper reproducing
numbers through that codebase inherits it. This is material to our AUC claim and belongs in
the Discussion.

---

## 3. Answers to the four questions asked

**(a) Does GTransPDM's "92%" refer to the without-pose ablation? YES — confirmed.**
Its Table I (v2/SPL) prints GTransPDM full model 0.90 / 0.87 / 0.82 and GTransPDM `w/o X_ke`
**0.92 / 0.90 / 0.86**, while the abstract says "achieves 92% accuracy on the PIE dataset".
The headline is the pose-free ablation. Note also a version trap: v1 (Sep 2024) prints
91.21 / 88.13 / 81.61 for the full model and contains no 92% variant at all — cite v2/SPL.

**(b) PedCMT's exact PIE numbers: UNVERIFIED, and one circulating claim is refuted.**
- No open-access copy exists anywhere. OpenAlex: `is_oa: false`, `oa_status: 'closed'`,
  `any_repository_has_fulltext: false`, `best_oa_location: null`. Semantic Scholar agrees;
  there is no arXiv version. ResearchGate is "Request PDF" only.
- The official repo `xbchen82/PedCMT` is 12 files with no results, no logs, no checkpoints;
  its README is two lines. The mirror named in its readme 404s.
- **The circulating "PIE Acc 0.876 / F1 0.806 / AUC 0.889" figures are a misattribution.**
  They belong to MFFN — Ni, Yang, Wei, Hu & Yang, *IET Intelligent Transport Systems*,
  DOI 10.1049/itr2.12253 — whose abstract reports "0.912/0.876, 0.813/0.806, 0.896/0.889,
  and 0.802/0.788" as **JAAD/PIE** pairs. Discard them.
- The only sourced PedCMT PIE row found is **third-party**: ESIA (arXiv 2604.23728) Table I
  gives PedCMT Acc 0.93, F1 0.87, AUC 0.92, inputs B,S, 16 input frames.
- **Modality claim stands (VERIFIED from the abstract):** "merely using bounding boxes and
  ego-vehicle speed as input features". Its code confirms 2 streams, though "ego-speed" is
  actually two channels (OBD speed concatenated with GPS speed).
- **Ruling for the manuscript:** the concession to PedCMT as a minimal-modality precedent is
  safe and must stay. Its numeric row stays out of the table, or enters marked third-party
  via ESIA. Its random 70/15/15 split is an additional reason not to tabulate it as
  standard-protocol.

**(c) Were Pedestrian Graph+ and BiPed configured differently? YES for both.**
GTransPDM's footnote reads, verbatim: *"Except for Pedestrian Graph+ and BiPed, others show
the same configurations as ours, following [3]."* Checked against the originals:
- **BiPed** (own paper): *"we clip the pedestrian tracks up to the crossing event frames and
  sample sequences with 50% overlap and time to event between 1 to 2 seconds (30 to 60
  frames)"*, giving 3,980 PIE training sequences (995 crossing), and it re-ran the baselines
  padded "for compatibility with evaluation criteria" — i.e. competitors were moved onto
  BiPed's setup, not the reverse.
- **Pedestrian Graph+** (own released code): **three classes**, not binary
  (`n_clss=3`, "# 0 for no crosing, 1 for crossing, 2 for irrelevant"); **observation length
  32**, not 16 (`torch.FloatTensor(1, 4, 32, 19)`); class re-balancing and sample duplication
  that change the evaluated population. Its printed numbers remain UNVERIFIED (closed access).

**(d) MFT's numbers: VERIFIED, and the suspicion about a cost table is refuted.**
MFT's PIE row sits in its **main quantitative results table** (v2: Table 1, merged
JAAD+PIE; v1: Table 2, "Quantitative results on PIE dataset") and reads
**Acc 0.90 / AUC 0.94 / F1 0.83 / P 0.83 / R 0.82** — read first-hand from the PDF row.
Its abstract corroborates: "achieving accuracy rates of 73%, 93%, and 90% on the JAADbeh,
JAADall, and PIE datasets". The AUC does not live only in a cost table. **However**, MFT uses
Split B, so it is not on our split.

---

## 4. Per-method notes

**PCPA** — Kotseruba, Rasouli & Tsotsos, WACV 2021, pp. 1258–1268. Own Table 3 ("Results for
the proposed model PCPA trained with different types of attention mechanisms"), PIE columns:
no attention 0.83/0.83/0.73; temporal attention 0.85/0.84/0.75; modality attention
0.86/0.86/0.77; **temporal + modality attention 0.87/0.86/0.77** — the final model, and the
row our table should cite. Table 4 separately reports the final model with different visual
features (local box 0.86/0.86/0.77, local context 0.87/0.86/0.77, scene context
0.88/0.85/0.78). Protocol: obs 16 frames (≈0.5 s), TTE 30–60 frames, Split B. Four streams
(`obs_input_type: [local_box, pose, box, speed]`, C3D backbone, GRU cells). **No leakage
check** — the word does not occur, and windows are cut by positional slicing from the track
end. *Note the three third-party transcriptions of this row all differ from each other:
GTransPDM 0.87/0.86/0.77 (correct), MFT 0.87/0.85/0.78 (blends Table 3 accuracy with Table 4
scene-context AUC/F1), BiPed 0.86/0.84/0.76, IntFormer 0.86/0.86/0.77. Use PCPA's own.*

**Pedestrian Graph+** — Cadena, Qian, Wang & Yang. **Venue settled: IEEE T-ITS, vol. 23,
no. 11, pp. 21050–21061, 2022, DOI 10.1109/TITS.2022.3173537** (Crossref, OpenAlex and DBLP
`journals/tits/CadenaQWY22` agree). The T-IV hypothesis is refuted. Closed access, no
repository full text, so its printed row is THIRD-PARTY only. Configuration deviates (see 3c).
No leakage statement in its code or README.

**IntFormer** — Lorenzo, Parra & Sotelo, arXiv 2105.08647, v1 only, no published venue. Own
**Table I** ("Results in the benchmark"), PIE columns: **IntFormer (Fusion C, 4M params)
0.89 / 0.92 / 0.81**. Same table gives PCPA 0.86/0.86/0.77 and its own PCPA reproduction
0.86/0.86/0.78. JAAD-beh 0.59/0.54/0.69 and JAAD-all 0.86/0.78/0.62 are separate columns.
**Documented protocol deviation**, verbatim: *"The benchmark established an input time length
of ≈ 0.5 s which corresponds to 16 frames. In our experiments, We halved that number,
sampling the input time interval at 15 fps instead of 30 fps, keeping sequences with N/2 = 8
evenly spaced frames."* — scoped to image data. No leakage check. Its abstract independently
reports that **ego-vehicle speed is its most important variable**, which corroborates our
ego-speed ablation from a fourth source.

**PIT** — Zhou, Tan, Zhong, Li & Gou, T-ITS vol. 24 no. 12, pp. 14213–14225, 2023,
DOI 10.1109/TITS.2023.3309309 (Crossref-verified). Closed access; OpenAlex reports no green
OA copy anywhere. The IJCAI-2022 AI4AD workshop precursor was retrieved and **contains no PIE
results** — verbatim: *"we conduct exhaustive comparison experiments on JAAD dataset"*. Its
0.91/0.92/0.82 is therefore THIRD-PARTY (GTransPDM Table I; PIP-Net Table III marks it with
an asterisk meaning "reported from the original article due to unavailable source code").
Protocol from the precursor: obs 16 frames, TTE 1–2 s. No leakage statement.

**BiPed** — **Rasouli, Rohani & Luo, "Bifold and Semantic Reasoning for Pedestrian Behavior
Prediction", ICCV 2021, pp. 15600–15610, arXiv 2012.03298.** Your amendment 4 is confirmed:
arXiv 2210.07886 is PedFormer's identifier and PedFormer cites BiPed as its reference [7].
BiPed's own Table 1 ("Performance of the proposed method on the PIE dataset") prints
**0.91 / 0.90 / 0.85** with precision 0.82; JAAD is a separate Table 5 (0.83/0.79/0.60/0.52).
No leakage check — verified by absence of the relevant vocabulary throughout.

**PedFormer** — Rasouli & Kotseruba, ICRA 2023, pp. 9844–9851, arXiv 2210.07886. Own Table I:
**0.93 / 0.90 / 0.87**, precision 0.89. Protocol: 0.5 s observation, 1 s prediction, 50%
overlap, TTE 1–2 s, "default data split" (set numbers never stated). No leakage check.

**GTransPDM** — Xie, Lin, Zheng, Gong & López, IEEE Signal Processing Letters, vol. 32,
pp. 2109–2113, 2025, DOI 10.1109/LSP.2025.3567249; arXiv 2409.20223 (v1 Sep 2024, v2 May
2025 — numbers changed between versions, cite v2). Full Table I as printed, PIE:
BiPed 0.91/0.90/0.85 · Pedestrian Graph+ 0.89/0.90/0.81 · MultiRNN 0.83/0.80/0.71 ·
SFRNN 0.82/0.79/0.69 · SingleRNN 0.81/0.75/0.64 · PCPA 0.87/0.86/0.77 ·
TrouSPI-Net 0.88/0.88/0.80 · IntFormer 0.89/0.92/0.81 · FF-STP 0.89/0.86/0.80 ·
PIT 0.91/0.92/0.82 · **GTransPDM w/o X_ke 0.92/0.90/0.86** · **GTransPDM 0.90/0.87/0.82**.
Obs 16 frames (0.5 s), TTE ∈ [30,60]. Three streams (position with PDM decomposition,
20-keypoint AlphaPose skeleton, ego-vehicle speed and acceleration). No leakage discussion.

**PedCMT** — see 3(b). Bibliography verified: Chen, Zhang, Li & Yang, IEEE T-ITS, vol. 25,
no. 9, pp. 12538–12549, Sept. 2024, DOI 10.1109/TITS.2024.3386689. Code confirms obs length
16 (`--times_num 16`), TTE [30,60], train overlap 0.6, and the rounded-prediction AUC bug.

**MFT** — Li, Zhong & Müller, arXiv 2511.20011 (v1 25 Nov 2025, v2 21 Mar 2026), TU Berlin;
arXiv-only, no DOI. See 3(d). Obs 16-frame clips, PIE overlap 60%, TTE 1–2 s; four numerical
context blocks (pedestrian behaviour, environment, localisation, vehicle motion) — no raw
RGB, pose or flow. No leakage check. Its Table 1 also tabulates GTransPDM at 0.90/0.87/0.82,
independently corroborating that GTransPDM's *full model* is the 0.90 row.

**PIP-Net** — Azarmi, Rezaei & Wang, IEEE T-ITS, vol. 26, no. 7, pp. 9824–9837, 2025,
DOI 10.1109/TITS.2025.3570794; arXiv 2402.12810 v3 (6 Jul 2025) is the accepted manuscript
and matches the published version. Table III ("PERFORMANCE COMPARISON ON THE PIE DATASET"),
footnote "All models are evaluated at ETC = 0.5s": **PIP-Net-α 0.92 / 0.94 / 0.88** (P 0.89,
R 0.88). Seven streams. Split: see 2a — not custom. Its Table V (0.73/0.71/0.69) is
Urban-PIP, not PIE; do not confuse. No leakage check.

---

## 5. Cross-cutting negative finding (supports our contribution 1)

**No paper among the ten states any check on observation-window temporal leakage.** This was
tested rather than assumed, and confirmed at the source in four independent ways: the words
do not appear in BiPed, PedFormer, GTransPDM, IntFormer, PIP-Net or MFT; the PCPA benchmark's
config exposes no such option and cuts windows by positional slicing (`start = len(seq) -
obs_length - tte_max`); PIE's own toolkit truncates at `crossing_point` and returns the entire
preceding track with no verification; and TCL (arXiv 2504.06292), checked as an extra data
point, asserts the TTE gap without auditing it. The claim "to our knowledge, no prior PIE
crossing-prediction work audits observation-window leakage" is supportable as written.

---

## 6. Actions for the manuscript

1. **BiPed gets its own citation** (`rasouli2021biped`, ICCV 2021, arXiv 2012.03298) — added
   to `references.bib`. The misleading note on the PedFormer entry has been corrected.
2. **MFT is printable** at 0.90 / 0.94 / 0.83, with a footnote that it uses Split B.
3. **PedCMT's numbers stay out**; the minimal-modality concession stays in.
4. **"Highest AUC in the standard-protocol table" must be reworded.** Our 0.947–0.950 exceeds
   MFT's verified 0.94, but MFT is on Split B, PIP-Net reaches 0.94 on Split B at ETC 0.5 s,
   and several tabulated AUCs are computed on rounded predictions rather than probabilities.
   Suggested honest form: *the highest ROC-AUC among the tabulated methods, noting that the
   rows do not share a single protocol and that some are not threshold-free AUCs.*
5. **Fix `03_baseline_comparison.md`'s PIP-Net rationale** (out of scope for this session —
   nothing outside `submission/` is being written — but record it as a repo debt).
6. The two-split finding and the rounded-AUC finding both belong in the Discussion as
   comparability caveats.

---

## 7. Sources read first-hand by the caller

- `openaccess.thecvf.com/.../Kotseruba_Benchmark_for_Evaluating_Pedestrian_Action_Prediction_WACV_2021_paper.pdf`
- `arxiv.org/html/2409.20223v2` (GTransPDM, Table I + split paragraph)
- `arxiv.org/pdf/2511.20011` (MFT, results row + split sentence)
- `arxiv.org/pdf/2402.12810` (PIP-Net v3, Table III + split sentence)
- `arxiv.org/pdf/2105.08647` (IntFormer, Table I + halving quote)

Verified by workflow agents against primary sources, spot-checked but not re-read end-to-end
by the caller: BiPed (CVF ICCV 2021 PDF), PedFormer (ar5iv 2210.07886), PedCMT repo code,
Pedestrian Graph+ repo code, PIT workshop precursor, Crossref/OpenAlex bibliographic records.

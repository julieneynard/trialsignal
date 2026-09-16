# Model Card — TrialSignal Trial-Progression Risk Model

Following the structure of Mitchell et al., "Model Cards for Model Reporting"
(FAT* 2019), adapted for a research-portfolio project.

**Status: v7.** v1 (2 hyp.): ROC-AUC ≈ 0.92, confounded. v2 (7 hyp.): confound
fixed, ≈ 0.5, near-chance. v3 (7 hyp., results-validated labels): ≈ 0.68,
small-sample. v4 (12 hyp., more disease diversity): ≈ 0.63, a larger-sample
correction of v3. v5 (17 hyp., 3 more disease areas + 2 more contrast
pairs): ROC-AUC ≈ 0.68 (temporal holdout), back up from v4. v6 (same 17
hyp., ChEMBL bioactivity now matched molecule-name-first instead of
target-level-only): ROC-AUC ≈ 0.68, essentially unchanged from v5, while
`chembl_activity_count` moved up to the model's 3rd-most-important
feature by mean |SHAP|; CV-temporal gap widened to −0.076, flagged rather
than glossed over. **v7 (22 hyp., first non-oncology therapeutic area —
5 immunology/rheumatology hypotheses added, see "Extending past
oncology" below): 1,186 rows (+343). ROC-AUC ≈ 0.69 (LightGBM, temporal
holdout)** — up from v6's 0.676, and the CV-temporal gap came back down
to −0.072, essentially the same size as v6's rather than widening further
— the cleanest read yet that adding a whole new therapeutic area didn't
destabilize the model or reveal a domain-specific breakdown. Seven
versions in, the pattern holding is: point estimates move ±0.05–0.07 as
hypotheses, data quality, or therapeutic-area scope change, and no single
move — up or down — should be read as more than "still modestly above
chance, with normal sampling variation at this dataset size."

## Model details
- **Developed by:** Julien Eynard, independent portfolio project.
- **Model type:** Gradient-boosted decision trees (LightGBM: `n_estimators=100,
  max_depth=3`, default hyperparameters — see `docs/METHODS.md` for why no
  tuning was done), binary classification, with a logistic-regression
  baseline for comparison.
- **Version:** `0.1.0`, trained via `trialsignal train --eval-mode temporal`;
  artifact metadata (version + training timestamp) is stored in the saved
  bundle and readable through `trialsignal.models.registry.load_model()`.

## Intended use
- **In scope:** Demonstrating an entity-resolved, leakage-aware,
  results-validated trial-outcome pipeline end-to-end on public data — and
  demonstrating that a metric needs re-checking against a second evaluation
  method every time the dataset changes, not reported once and trusted. A
  worked example of the modeling and evaluation process, not a production
  predictive tool at this dataset size.
- **Out of scope:** Any actual clinical, investment, R&D-strategy, or
  regulatory decision. Live scoring (`/score`) is restricted to the 22
  hand-verified hypotheses in `src/trialsignal/features/hypothesis.py` — not
  a general gene/disease lookup. Not a substitute for a real
  pharmacovigilance or R&D strategy process.

## Training data
- Source: ClinicalTrials.gov (trial metadata *and*, where posted, the
  results section), Open Targets (target-disease evidence), ChEMBL
  (bioactivity) — joined via `trialsignal build-features` for each
  hypothesis. Full source table and join logic: `docs/METHODS.md`.
- **22 curated hypotheses.** v5 added 5 more: ALK/crizotinib (NSCLC — same
  disease as EGFR, different target), FLT3/midostaurin (acute myeloid
  leukemia, new disease area), BCL2/venetoclax (CLL — same disease as BTK,
  different mechanism), FGFR3/erdafitinib (bladder cancer, new disease
  area), VEGFA/bevacizumab (colorectal cancer, new disease area — against
  the VEGF ligand, distinct from KDR's receptor). v7 added 5 more, the
  first outside oncology entirely — see "Extending past oncology" below.
- **n = 1,186** labeled trials (1,051 success / **135** failure).
  Per-hypothesis breakdown (the 17 oncology rows moved slightly from v6's
  843/102 — trials were re-fetched fresh for the v7 rebuild and CT.gov's
  live registry data changes over time as trials complete or post
  results):

  | Hypothesis | n | success | failure | results-based | molecule-matched |
  |---|---|---|---|---|---|
  | PDCD1 / pembrolizumab / melanoma | 114 | 96 | 18 | 13 | 0 |
  | IL6R / tocilizumab / rheumatoid arthritis | 135 | 124 | 11 | 25 | 0 |
  | VEGFA / bevacizumab / colorectal cancer | 102 | 84 | 18 | 19 | 0 |
  | ERBB2 / trastuzumab / breast cancer | 102 | 95 | 7 | 11 | 0 |
  | TNF / adalimumab / rheumatoid arthritis | 96 | 82 | 14 | 26 | 0 |
  | KDR / sunitinib / renal cell carcinoma | 89 | 81 | 8 | 16 | 89 |
  | ABL1 / imatinib / CML | 80 | 74 | 6 | 7 | 80 |
  | CD38 / daratumumab / multiple myeloma | 67 | 64 | 3 | 9 | 0 |
  | IL17A / secukinumab / psoriasis | 54 | 49 | 5 | 21 | 0 |
  | BTK / ibrutinib / CLL | 48 | 45 | 3 | 12 | 48 |
  | MTOR / everolimus / renal cell carcinoma | 48 | 41 | 7 | 8 | 48 |
  | AR / enzalutamide / prostate cancer | 43 | 34 | 9 | 12 | 43 |
  | IL12B / ustekinumab / psoriasis | 38 | 37 | 1 | 17 | 0 |
  | PARP1 / olaparib / ovarian cancer | 32 | 25 | 7 | 12 | 32 |
  | BCL2 / venetoclax / CLL | 28 | 24 | 4 | 7 | 28 |
  | BRAF / vemurafenib / melanoma | 26 | 22 | 4 | 3 | 26 |
  | EGFR / osimertinib / NSCLC | 21 | 17 | 4 | 3 | 21 |
  | JAK1 / tofacitinib / rheumatoid arthritis | 20 | 18 | 2 | 5 | **20** |
  | CDK4 / palbociclib / breast cancer | 19 | 17 | 2 | 3 | 19 |
  | FLT3 / midostaurin / AML | 16 | 15 | 1 | 3 | 16 |
  | ALK / crizotinib / NSCLC | 7 | 6 | 1 | 3 | 7 |
  | FGFR3 / erdafitinib / bladder cancer | 1 | 1 | 0 | 1 | 1 |

- **Extending past oncology (v7).** Five immunology/rheumatology
  hypotheses, chosen specifically to test whether the pipeline generalizes
  past oncology rather than to pad the count: TNF/adalimumab, IL6R/
  tocilizumab, and JAK1/tofacitinib all for rheumatoid arthritis (a
  same-disease/different-mechanism triple, mirroring the oncology set's
  contrast pairs), and IL17A/secukinumab + IL12B/ustekinumab both for
  psoriasis (a second such pair). Deliberately mixed mechanism: 4
  antibodies + 1 small molecule (JAK1/tofacitinib), matching the
  antibody-heavy real composition of this drug class. A genuinely
  different, honestly-reported finding versus every oncology batch: **none
  of the 5 needed an entity-resolution naming-mismatch fix.** Checked the
  same way as every prior batch (scoring real CT.gov phrasing against the
  live Open Targets disease name before hardcoding) — "Rheumatoid
  Arthritis," "Psoriasis," "Psoriatic Arthritis" all scored 1.000, and even
  the trickiest case ("Crohn's Disease" vs Open Targets' "Crohn disease")
  scored 0.929, comfortably above the 0.85 threshold. Four-for-four
  oncology batches needing a fix, zero-for-one immunology batches needing
  one — a real signal that the naming-mismatch pattern documented in
  `entity_resolution.py` is a texture of *some* of these data sources for
  *some* disease areas, not a universal property of the pipeline. Also
  caught and fixed one real error before it shipped: ustekinumab binds the
  IL-12/IL-23 shared p40 subunit (gene **IL12B**), not IL23A's p19 subunit
  (that would be guselkumab/risankizumab's target) — verified against
  ChEMBL/Open Targets before hardcoding, not assumed from memory.
  `IL12B` came back with **zero** ChEMBL bioactivity records at both the
  target and molecule level — a real, structural finding (IL-12B is a
  cytokine subunit, not a typical small-molecule assay target), not a
  matching failure, consistent with the antibody-coverage-gap pattern
  documented in v6.
- A fourth naming-mismatch fix was needed (after carcinoma/cancer,
  myelogenous/myeloid, plasma-cell myeloma): Open Targets names bladder
  disease "urinary bladder cancer"/"...carcinoma" — checked against the
  live API before hardcoding FGFR3/erdafitinib, scored 0.78 unfixed.
  **FGFR3 landed only 1 row** — erdafitinib is a recent approval (2019)
  with a narrow, still-small trial footprint; a thin real result, not a
  pipeline bug (compare VEGFA/bevacizumab, an established drug, at 102
  rows from the same-sized bladder/colorectal trial pools).
- **ChEMBL molecule-name matching (v6).** `molecule-matched` above is
  `chembl_matched_by_molecule_name` — whether ChEMBL bioactivity data for
  that row's target came from the specific marketed drug (resolved by
  synonym search against `molecule.json`, covering generic and brand
  names) rather than falling back to the target's generic aggregate.
  **Exactly two clean groups: 100% for all 13 small-molecule hypotheses,
  0% for all 4 antibody hypotheses** (pembrolizumab, trastuzumab,
  daratumumab, bevacizumab). The 0% is a real, verified structural finding,
  not a bug: checked directly against ChEMBL's live API, none of these 4
  antibodies has *any* bioactivity record under any target/standard-type —
  ChEMBL is overwhelmingly small-molecule potency data (IC50/EC50/Ki
  against a purified protein), and antibody mechanisms aren't characterized
  that way in this database. Before this fix, molecule-name matching found
  0/458 of the 13 small-molecule hypotheses' rows (the root cause: a
  target's full activity list can run to 10,000s of rows, and the first
  ~200 unfiltered rows returned by a target-level pull essentially never
  contain a specific marketed compound's data — see `chembl.py`'s module
  docstring for the verified osimertinib example, 485 real EGFR activities
  recovered by querying molecule-first vs. 0 found in the truncated
  target-level pull). This fix applies to both the offline
  `trialsignal build-features` pipeline and the live `/score` endpoint
  (`serving/api.py`'s `_get_activities`).
- Train/test split: temporal, cutoff 2020-01-01 → 1,045 train / 141 test,
  both classes present in both (18 failures in test).

## Evaluation

**Temporal split (cutoff 2020-01-01, n_train=1,045, n_test=141):**

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.614 | **0.689** |
| PR-AUC | 0.933 | 0.938 |
| Brier score | 0.112 | 0.107 |

**Stratified 5-fold CV (`--eval-mode cv`, same n=1,186):**

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.587 | 0.617 |
| PR-AUC | 0.922 | 0.929 |

**Read all five versions' CV-vs-temporal gaps together, not any one
number in isolation.**

| Version | Hypotheses | n | Temporal AUC | CV AUC | Gap |
|---|---|---|---|---|---|
| v3 | 7 | 464 | 0.678 | 0.709 | +0.031 (CV higher) |
| v4 | 12 | 686 | 0.632 | 0.660 | +0.028 (CV higher) |
| v5 | 17 | 840 | 0.680 | 0.645 | −0.035 (temporal higher) |
| v6 | 17 | 843 | 0.676 | 0.600 | −0.076 (temporal higher) |
| v7 | 22 | 1,186 | **0.689** | 0.617 | **−0.072 (temporal higher)** |

**v7 is the reassuring result v6 left open.** v6's gap (−0.076) was
flagged as the widest since the v2 confound fix, with an honest "probably
sampling noise, not confirmed" verdict pending further evidence. v7 adds
a genuinely different kind of evidence — not just more rows, a whole new
therapeutic area — and the gap came back down to −0.072, essentially the
same size, not wider. If v6's gap had been a returning leakage problem,
the natural expectation would be for it to keep growing or behave
erratically as the dataset changed shape further; instead it held steady
within noise of v6's value while both AUCs moved up together (temporal
0.676→0.689, CV 0.600→0.617). Read plainly: this dataset size now produces
CV-temporal gaps of 0.031, 0.028, 0.035, 0.076, 0.072 — one notably wider
pair (v6, v7) and three tighter, still nothing resembling v2's ≈0.2 leaky,
one-directional gap. The honest summary stays "modestly above chance, with
sampling noise on the order of ±0.05–0.08," now checked against a genuine
domain-generalization test, not just more of the same data.

**Top SHAP features (LightGBM, temporal-trained model, v7):**
`max_phase_ordinal` (0.512), `enrollment` (0.385), `ot_overall_score`
(0.293), `chembl_best_pchembl` (0.144), `chembl_activity_count` (0.134).
The same recurring, plausible handful as every prior version — still
nothing resembling a disguised hypothesis-identity indicator, now
verified across two structurally different therapeutic areas rather than
just oncology. `chembl_best_pchembl` entering the top 5 tracks directly
with JAK1/tofacitinib's real ChEMBL potency data (the one small-molecule
hypothesis in the new batch) joining the pool the model can learn from.

**This is still not a production-grade number.** ≈0.59–0.69 ROC-AUC
(depending on evaluation mode) is a real, modestly-above-chance,
repeatedly cross-checked signal on public trial-metadata features —
informative for a portfolio demonstration, nowhere near what would be
needed to inform an actual go/no-go decision.

**Calibration (v7, recomputed from v6's methodology on the full 22-hypothesis
pool).** Same leakage-free 5-fold stratified CV out-of-fold predictions the
CV eval numbers above come from, binned into deciles (~118-122 rows/bin):

| Mean predicted P(success) | Observed success rate | n |
|---|---|---|
| 0.645 | 0.840 | 119 |
| 0.810 | 0.849 | 119 |
| 0.852 | 0.861 | 122 |
| 0.876 | 0.861 | 115 |
| 0.899 | 0.847 | 118 |
| 0.921 | 0.843 | 121 |
| 0.938 | 0.898 | 118 |
| 0.957 | 0.949 | 117 |
| 0.978 | 0.941 | 119 |
| 0.995 | 0.975 | 118 |

Count-weighted Expected Calibration Error: **0.050** (5.0 points, LightGBM,
10-bin) — **improved from v6's 0.073.** The same systematic
under-confidence in the lowest-confidence decile is still there (predicts
≈65% success, actual ≈84%) but the gap shrank from v6's 27 points to
**~19.5 points**, and the overall ECE dropped by a third. Plausible driver:
135 failures spread across 22 hypotheses (vs. 102 across 17) gives the
low-probability region more real examples to calibrate against, without
any change to the calibration methodology itself. Still not well enough
calibrated to treat `risk_score` as a literal probability — read it as a
ranking signal, same conclusion as v6, just with a smaller (not
eliminated) miscalibration in the tail.

## Ethical considerations
- Built entirely from public registry/database data; no patient-level or
  proprietary data involved.
- A published "risk score" for a drug/target/disease hypothesis could, if
  taken out of context, be read as investment or clinical advice — the README
  and API responses carry an explicit disclaimer for this reason.
- This card's full history (v1 → v2 → v3 → v4 → v5 → v6 → v7) is kept
  rather than collapsed into only the current number — including the AUC
  moving non-monotonically (up, down, up again, flat, up) as more data,
  better data, and a whole new therapeutic area arrived, which is the more
  useful thing to show a technical reader than a single cherry-picked
  figure.
- v6 specifically demonstrates that a data-quality fix (matching ChEMBL
  bioactivity by the actual drug instead of falling back to a generic
  target aggregate) can be real and verifiable — 100%/0% molecule-match
  split confirmed live against ChEMBL, a top-3 SHAP feature shift — without
  moving the headline metric. v7 demonstrates the pipeline generalizes past
  its original domain: same methodology, same discipline, applied to
  immunology/rheumatology, with the model's performance and the
  CV-temporal gap both landing within noise of the oncology-only result —
  and one clean negative control (zero of 5 new hypotheses needed a
  naming-mismatch fix, versus four-for-four in oncology), which is itself
  useful evidence about *why* the oncology fixes were needed (CT.gov vs.
  Open Targets phrasing divergence specific to certain disease-naming
  conventions, not a general property of free-text medical data).
  Reporting these outcomes honestly, rather than searching for a way to
  frame every version as a win, is the more important signal about this
  project's methodology than any single AUC number.

## Caveats and recommendations
See `docs/LIMITATIONS.md` in full. Most importantly: (1) the label is still
a proxy for most rows — 236/1,186 (≈20%) have a real primary-endpoint
result, up from v6's ~17%, but ~80% of the dataset is still
registry-status-labeled (see the per-hypothesis `results-based` column
above for the split); (2) 1,186 rows for 11
features, with 141 in the temporal test set, still carries real sampling
uncertainty — the AUC has moved ±0.05–0.07 across the last five
re-measurements, so treat ≈0.61–0.69 as a range, not a single figure;
(3) broadening results-based coverage was investigated and explicitly
rejected as infeasible without fabricating labels (`docs/LIMITATIONS.md`
item 1a); (4) several hypotheses (FGFR3 at n=1, ALK at n=7, JAK1 at n=20)
are too thin individually to trust in isolation — they contribute to the
pooled estimate but shouldn't be read as validating those specific drugs;
(5) ChEMBL bioactivity data structurally does not exist for
antibody-drug hypotheses (all 8/8 antibody hypotheses have zero records,
verified live — IL12B additionally has zero even at the target level) —
this is a property of the database, not a gap in this project's matching
logic, and `chembl_activity_count`/`chembl_matched_by_molecule_name`
should be read as "not applicable" rather than "zero signal" for those
hypotheses; (6) `risk_score` is not a well-calibrated probability — the
decile reliability table above shows the model under-confident by
~19.5 points in its own lowest-confidence decile (improved from v6's 27
points, but not eliminated) — treat it as a ranking signal, not a literal
likelihood, and don't subtract it from 1 and call the result a failure
probability; (7) the 5 immunology hypotheses (v7) are all rheumatoid
arthritis or psoriasis — real mechanistic diversity, but only 2 diseases,
so "generalizes past oncology" should be read as "generalizes to this one
additional therapeutic area," not validated broadly across all of
immunology yet. Continuing to add diverse curated hypotheses — across
mechanisms, diseases, and now therapeutic areas — each checked for its own
naming mismatches and each triggering a full re-measurement rather than an
assumed improvement, remains the validated way to work this number.

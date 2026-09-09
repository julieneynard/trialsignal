# Model Card — TrialSignal Trial-Progression Risk Model

Following the structure of Mitchell et al., "Model Cards for Model Reporting"
(FAT* 2019), adapted for a research-portfolio project.

**Status: v6.** v1 (2 hyp.): ROC-AUC ≈ 0.92, confounded. v2 (7 hyp.): confound
fixed, ≈ 0.5, near-chance. v3 (7 hyp., results-validated labels): ≈ 0.68,
small-sample. v4 (12 hyp., more disease diversity): ≈ 0.63, a larger-sample
correction of v3. v5 (17 hyp., 3 more disease areas + 2 more contrast
pairs): ROC-AUC ≈ 0.68 (temporal holdout), back up from v4. v6 (same 17
hyp., ChEMBL bioactivity now matched molecule-name-first instead of
target-level-only — see "ChEMBL molecule-name matching" below): **ROC-AUC
≈ 0.68 (LightGBM, temporal holdout)** — essentially unchanged from v5
(0.680 → 0.676), while `chembl_activity_count` moved up to the model's
3rd-most-important feature by mean |SHAP| (previously 4th). Read together:
the ChEMBL fix changed *what data the model sees* for 13 of 17 hypotheses
without moving the headline metric — a legitimate outcome to report as-is,
not a win to spin or a regression to explain away. Six versions in, the
pattern holding is: point estimates move ±0.05 as hypotheses or data
quality change, and neither move should be read as more than "still
modestly above chance, with normal sampling variation at this dataset
size" — see "Evaluation" for the one CV-temporal gap number in v6 that
doesn't fit that pattern as cleanly as v3–v5's did.

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
  regulatory decision. Live scoring (`/score`) is restricted to the 17
  hand-verified hypotheses in `src/trialsignal/features/hypothesis.py` — not
  a general gene/disease lookup. Not a substitute for a real
  pharmacovigilance or R&D strategy process.

## Training data
- Source: ClinicalTrials.gov (oncology trials — trial metadata *and*, where
  posted, the results section), Open Targets (target-disease evidence),
  ChEMBL (bioactivity) — joined via `trialsignal build-features` for each
  hypothesis. Full source table and join logic: `docs/METHODS.md`.
- **17 curated hypotheses.** v5 added 5 more: ALK/crizotinib (NSCLC — same
  disease as EGFR, different target), FLT3/midostaurin (acute myeloid
  leukemia, new disease area), BCL2/venetoclax (CLL — same disease as BTK,
  different mechanism), FGFR3/erdafitinib (bladder cancer, new disease
  area), VEGFA/bevacizumab (colorectal cancer, new disease area — against
  the VEGF ligand, distinct from KDR's receptor).
- **n = 843** labeled trials (741 success / **102** failure). Per-hypothesis
  breakdown (moved slightly from v5's 840/97 — trials were re-fetched fresh
  for the v6 rebuild, and CT.gov's live registry data changes over time as
  trials complete or post results; the `molecule-matched` column is new in
  v6):

  | Hypothesis | n | success | failure | results-based | molecule-matched |
  |---|---|---|---|---|---|
  | PDCD1 / pembrolizumab / melanoma | 114 | 96 | 18 | 13 | 0 |
  | VEGFA / bevacizumab / colorectal cancer | 102 | 84 | 18 | 19 | 0 |
  | ERBB2 / trastuzumab / breast cancer | 102 | 95 | 7 | 11 | 0 |
  | KDR / sunitinib / renal cell carcinoma | 89 | 81 | 8 | 16 | 89 |
  | ABL1 / imatinib / CML | 80 | 74 | 6 | 7 | 80 |
  | CD38 / daratumumab / multiple myeloma | 67 | 64 | 3 | 9 | 0 |
  | BTK / ibrutinib / CLL | 48 | 45 | 3 | 12 | 48 |
  | MTOR / everolimus / renal cell carcinoma | 48 | 41 | 7 | 8 | 48 |
  | AR / enzalutamide / prostate cancer | 43 | 34 | 9 | 12 | 43 |
  | PARP1 / olaparib / ovarian cancer | 32 | 25 | 7 | 12 | 32 |
  | BCL2 / venetoclax / CLL | 28 | 24 | 4 | 7 | 28 |
  | BRAF / vemurafenib / melanoma | 26 | 22 | 4 | 3 | 26 |
  | EGFR / osimertinib / NSCLC | 21 | 17 | 4 | 3 | 21 |
  | CDK4 / palbociclib / breast cancer | 19 | 17 | 2 | 3 | 19 |
  | FLT3 / midostaurin / AML | 16 | 15 | 1 | 3 | 16 |
  | ALK / crizotinib / NSCLC | 7 | 6 | 1 | 3 | 7 |
  | FGFR3 / erdafitinib / bladder cancer | 1 | 1 | 0 | 1 | 1 |

  A fourth naming-mismatch fix was needed (after carcinoma/cancer,
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
- Train/test split: temporal, cutoff 2020-01-01 → 753 train / 90 test, both
  classes present in both (17 failures in test).

## Evaluation

**Temporal split (cutoff 2020-01-01, n_train=753, n_test=90):**

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.545 | **0.676** |
| PR-AUC | 0.855 | 0.909 |
| Brier score | 0.159 | 0.154 |

**Stratified 5-fold CV (`--eval-mode cv`, same n=843):**

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.585 | 0.600 |
| PR-AUC | 0.912 | 0.922 |

**Read all four versions' CV-vs-temporal gaps together, not any one
number in isolation.**

| Version | Hypotheses | n | Temporal AUC | CV AUC | Gap |
|---|---|---|---|---|---|
| v3 | 7 | 464 | 0.678 | 0.709 | +0.031 (CV higher) |
| v4 | 12 | 686 | 0.632 | 0.660 | +0.028 (CV higher) |
| v5 | 17 | 840 | 0.680 | 0.645 | −0.035 (temporal higher) |
| v6 | 17 | 843 | **0.676** | 0.600 | **−0.076 (temporal higher)** |

**v6's gap is honestly the odd one out** — at −0.076 it's roughly double
v5's −0.035, the largest of any version since the v2 confound fix, though
still far short of v2's original ≈0.2 leaky-and-consistent-direction gap.
Read in context: v6 changed the underlying ChEMBL data (not the trial
labels or hypothesis count) for 13 of 17 hypotheses, so this isn't a
like-for-like re-measurement of v5 the way v3→v4→v5 were of each other —
some gap growth from a genuinely different feature distribution is
plausible, not necessarily a return of the leakage v2 had. It's also worth
flagging rather than dismissing: this dataset size (n≈843) has now
produced CV-temporal gaps of 0.031, 0.028, 0.035, and 0.076 — three tight
and one notably wider — and the honest read is "still probably sampling
noise, but the widest instance seen since the confound was fixed," not a
confident "definitely fine." The baseline logreg tells a related story:
its temporal AUC dropped sharply from v5's 0.623 to v6's 0.545, but its CV
AUC only moved from 0.628 to 0.585 — a much smaller change — consistent
with the drop being concentrated in specifics of this particular 90-row
temporal test set rather than a real degradation in the baseline model
itself. LightGBM continues to clearly outperform the baseline in both
evaluation modes. The headline LightGBM number itself has now been
0.5 → 0.68 → 0.63 → 0.68 → 0.68 across five re-measurements; the honest
summary remains "reliably modestly-above-chance, ±0.05–0.08," not any
single point estimate — and v6 is the first version where that error bar
should probably be read as closer to the wider end.

**Top SHAP features (LightGBM, temporal-trained model):** `max_phase_ordinal`
(0.600), `enrollment` (0.502), `chembl_activity_count` (0.412),
`ot_overall_score` (0.305), `ot_clinical_score` (0.112). The same
recurring, plausible handful as every prior version, still nothing
resembling a disguised hypothesis-identity indicator — but
`chembl_activity_count` moved from 4th to **3rd** in v6, the first
externally-visible effect of the molecule-name-matching fix: the feature
now reflects real drug-specific bioactivity data for 13/17 hypotheses
instead of always falling back to a target-level aggregate, and the model
leans on it more as a result, even though the aggregate AUC barely moved.

**This is still not a production-grade number.** ≈0.60–0.68 ROC-AUC
(depending on evaluation mode) is a real, modestly-above-chance,
repeatedly cross-checked signal on public trial-metadata features —
informative for a portfolio demonstration, nowhere near what would be
needed to inform an actual go/no-go decision.

No calibration reliability diagram is included — 102 failures across 17
hypotheses (several with single-digit failure counts) is still thin for
dense per-bin calibration counts.

## Ethical considerations
- Built entirely from public registry/database data; no patient-level or
  proprietary data involved.
- A published "risk score" for a drug/target/disease hypothesis could, if
  taken out of context, be read as investment or clinical advice — the README
  and API responses carry an explicit disclaimer for this reason.
- This card's full history (v1 → v2 → v3 → v4 → v5 → v6) is kept rather
  than collapsed into only the current number — including the AUC moving
  non-monotonically (up, down, up again, flat) as more data and better
  data arrived, which is the more useful thing to show a technical reader
  than a single cherry-picked figure.
- v6 specifically demonstrates that a data-quality fix (matching ChEMBL
  bioactivity by the actual drug instead of falling back to a generic
  target aggregate) can be real and verifiable — 100%/0% molecule-match
  split confirmed live against ChEMBL, a top-3 SHAP feature shift — without
  moving the headline metric. Reporting that outcome honestly, rather than
  searching for a way to frame it as a win, is the more important signal
  about this project's methodology than the AUC number itself.

## Caveats and recommendations
See `docs/LIMITATIONS.md` in full. Most importantly: (1) the label is still
a proxy for ~83% of rows (only ~17% of this dataset has a real
primary-endpoint result); (2) 843 rows for 11 features, with 90 in the
temporal test set, still carries real sampling uncertainty — the AUC has
moved ±0.05 across the last four re-measurements, and v6's CV-temporal gap
(−0.076) is the widest since the v2 confound fix, so treat ≈0.60–0.68 as a
range with a wider-than-usual error bar right now, not a single figure;
(3) broadening results-based coverage beyond ~17% was investigated and
explicitly rejected as infeasible without fabricating labels
(`docs/LIMITATIONS.md` item 1a); (4) several hypotheses (FGFR3 at n=1, ALK
at n=7) are too thin individually to trust in isolation — they contribute
to the pooled estimate but shouldn't be read as validating those specific
drugs; (5) ChEMBL bioactivity data structurally does not exist for
antibody-drug hypotheses (0/4 have any record, verified live) — this is a
property of the database, not a gap in this project's matching logic, and
`chembl_activity_count`/`chembl_matched_by_molecule_name` should be read
as "not applicable" rather than "zero signal" for those 4 hypotheses.
Continuing to add diverse curated hypotheses, each checked for its own
naming mismatches and each triggering a full re-measurement rather than an
assumed improvement, remains the validated way to work this number.

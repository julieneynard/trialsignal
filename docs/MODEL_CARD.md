# Model Card — TrialSignal Trial-Progression Risk Model

Following the structure of Mitchell et al., "Model Cards for Model Reporting"
(FAT* 2019), adapted for a research-portfolio project.

**Status: v2 — the hypothesis set was deliberately expanded from 2 to 7 to
fix a confound v1 had (see "Evaluation" below and `docs/LIMITATIONS.md`).
The honest headline result is that the model performs near chance once that
confound is removed — that is the actual finding of this iteration, not a
regression to fix.**

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
- **In scope:** Demonstrating an entity-resolved, leakage-aware trial-outcome
  pipeline end-to-end on public data, and demonstrating *how to detect and
  correct* a confound in a small training set. A worked example of the
  modeling approach, not a validated predictive tool at this dataset size.
- **Out of scope:** Any actual clinical, investment, R&D-strategy, or
  regulatory decision. Live scoring (`/score`) is restricted to the 7
  hand-verified hypotheses in `src/trialsignal/features/hypothesis.py` — not
  a general gene/disease lookup. Not a substitute for a real
  pharmacovigilance or R&D strategy process.

## Training data
- Source: ClinicalTrials.gov (oncology trials), Open Targets (target-disease
  evidence), ChEMBL (bioactivity) — joined via `trialsignal build-features`
  for each hypothesis. Full source table and join logic: `docs/METHODS.md`.
- **7 curated hypotheses**, chosen for mechanistic diversity (not just
  headcount) to prevent any single feature from separating hypotheses
  trivially: EGFR/osimertinib (NSCLC), ABL1/imatinib (CML), BRAF/vemurafenib
  (melanoma), ERBB2/trastuzumab — a monoclonal antibody (breast cancer),
  KDR/sunitinib (renal cell carcinoma), PARP1/olaparib (ovarian cancer),
  PDCD1/pembrolizumab — a PD-1 immune checkpoint inhibitor, also an antibody
  (melanoma).
- **n = 405** labeled trials (378 success / **27** failure) after label
  construction and entity resolution, from ~24,000 total pulled trials
  across all 7 condition queries. Per-hypothesis label counts:

  | Hypothesis | n | success | failure |
  |---|---|---|---|
  | PDCD1 / pembrolizumab / melanoma | 108 | 95 | 13 |
  | ERBB2 / trastuzumab / breast cancer | 100 | 98 | 2 |
  | KDR / sunitinib / renal cell carcinoma | 87 | 83 | 4 |
  | ABL1 / imatinib / CML | 42 | 42 | 0 |
  | PARP1 / olaparib / ovarian cancer | 24 | 23 | 1 |
  | BRAF / vemurafenib / melanoma | 26 | 22 | 4 |
  | EGFR / osimertinib / NSCLC | 18 | 15 | 3 |

  Failures are now spread across 5 of 7 hypotheses (v1 had all 3 failures in
  a single hypothesis) — this is the actual fix, not just a larger n.
- Train/test split: **temporal now works**, unlike v1. Cutoff 2020-01-01 →
  361 train / 44 test, with both classes present in both splits (9 failures
  in test, across EGFR/PARP1/PDCD1) — `InsufficientClassDiversityError` no
  longer fires on this dataset.
- PDCD1's bioactivity coverage from ChEMBL is sparse (3 records pulled vs.
  hundreds for the small-molecule targets) — expected, not a bug: ChEMBL is
  overwhelmingly small-molecule potency data, and pembrolizumab/PD-1 blockade
  isn't measured that way. `chembl_matched_by_molecule_name` correctly
  reports this per-row rather than hiding it.

## Evaluation

**Temporal split (the methodologically correct evaluation — cutoff
2020-01-01, n_train=361, n_test=44):**

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.492 | 0.532 |
| PR-AUC | 0.818 | 0.854 |
| Brier score | 0.180 | 0.172 |

**This is the honest number, and it is close to chance (0.5).** Once the
hypothesis-identity confound is removed (see below), the model shows no
meaningful ability to predict which trials, tested after 2020, will succeed
or fail — using only these public, target/chemistry-level features. That is
a real, informative negative result, not a bug to chase: it means either (a)
more data is needed, (b) the feature set is missing the signal that actually
drives trial outcomes (patient-level data, protocol design detail, real
endpoint results — none of which are in scope here, see
`docs/LIMITATIONS.md`), or most likely both.

**Stratified 5-fold CV (`--eval-mode cv`, same n=405, NOT a temporal
holdout):**

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.718 | 0.656 |
| PR-AUC | 0.969 | 0.964 |

**The CV number is higher, and that gap is itself the finding worth
reading, not a discrepancy to reconcile in the model's favor.** CV folds
are random, not time-ordered: rows from the same hypothesis end up on both
sides of most folds, so the model can still partly learn "this row's
hypothesis-correlated feature values look like other rows from that
hypothesis" even without an explicit hypothesis-identity feature —
residual leakage the temporal split, by construction, doesn't allow. The
~0.2 AUC gap between CV and temporal evaluation on the *same* data is a
concrete illustration of why `docs/METHODS.md` insists temporal splitting
is the correct methodology for this problem, not a stylistic preference.

**v1 comparison, for context:** the 2-hypothesis version of this dataset
reported ROC-AUC ≈ 0.92 (CV; temporal was impossible then — see git
history). That number was real in the sense that the code computed it
correctly, and meaningless in the sense that it mostly measured "can the
model tell EGFR trials from ABL1 trials" (see `ot_tractable_antibody` being
perfectly collinear with hypothesis identity in that dataset). This
version's lower, honest number is a better result, not a worse one.

**Top SHAP features (LightGBM, temporal-trained model):** `enrollment`,
`ot_clinical_score`, `ot_tractable_small_molecule`, `ot_safety_liability_count`,
`max_phase_ordinal` — notably, no longer dominated by a feature that's a
disguised hypothesis-identity indicator, which is itself a sign the fix
worked at the mechanism level, even though the resulting AUC is weak.

No calibration reliability diagram is included — with 27 total failures
spread thin across 7 hypotheses, per-bin calibration counts still aren't
dense enough to be meaningful.

## Ethical considerations
- Built entirely from public registry/database data; no patient-level or
  proprietary data involved.
- A published "risk score" for a drug/target/disease hypothesis could, if
  taken out of context, be read as investment or clinical advice — the README
  and API responses carry an explicit disclaimer for this reason.
- Reporting a near-chance AUC honestly, instead of quietly keeping the
  inflated v1 number, is itself the point of this section: the confound
  described above is exactly the kind of thing that produces a
  superficially impressive model a non-technical reader would overtrust.

## Caveats and recommendations
See `docs/LIMITATIONS.md` in full. Most importantly: (1) the label is a
proxy (registry status) for scientific/clinical success, not a direct
measurement of it; (2) even with the confound fixed, 405 rows is still a
small dataset for 11 features, and the near-chance temporal result should be
read as "not yet enough signal," not "public data can't predict trial
risk." Growing `CURATED_HYPOTHESES` further, and — per the roadmap — parsing
CT.gov's trial *results* section for real endpoint-based labels instead of
the registry-status proxy, are the two changes most likely to move this
number.

# Model Card — TrialSignal Trial-Progression Risk Model

Following the structure of Mitchell et al., "Model Cards for Model Reporting"
(FAT* 2019), adapted for a research-portfolio project.

**Status: v1, trained on a deliberately small curated dataset. Read
"Evaluation" below before citing the AUC — it is not evidence of a
generalizable model, and this card says so on purpose.**

## Model details
- **Developed by:** Julien Eynard, independent portfolio project.
- **Model type:** Gradient-boosted decision trees (LightGBM: `n_estimators=100,
  max_depth=3`, default hyperparameters — see `docs/METHODS.md` for why no
  tuning was done), binary classification, with a logistic-regression
  baseline for comparison.
- **Version:** `0.1.0`, trained via `trialsignal train`; artifact metadata
  (version + training timestamp) is stored in the saved bundle and readable
  through `trialsignal.models.registry.load_model()`.

## Intended use
- **In scope:** Demonstrating an entity-resolved, leakage-aware trial-outcome
  pipeline end-to-end on public data. A worked example of the modeling
  approach, not a validated predictive tool at this dataset size.
- **Out of scope:** Any actual clinical, investment, R&D-strategy, or
  regulatory decision. Not validated for use beyond the two curated
  hypotheses (EGFR/osimertinib/NSCLC, ABL1/imatinib/CML) it was trained on.
  Not a substitute for a real pharmacovigilance or R&D strategy process.

## Training data
- Source: ClinicalTrials.gov (oncology trials), Open Targets (target-disease
  evidence), ChEMBL (bioactivity) — joined via `trialsignal build-features`
  for the two hypotheses in `src/trialsignal/features/hypothesis.py`. Full
  source table and join logic: `docs/METHODS.md`.
- **n = 60** labeled trials (57 success / **3** failure) after label
  construction and entity resolution, from 5,773 total pulled trials across
  both hypotheses' condition queries (3,977 NSCLC + 1,796 CML — see
  `docs/LIMITATIONS.md` for the coverage funnel; this yield is expected at
  this data scope, not a data-quality bug).
- Date range: 1999-08-09 to 2024-01-08. All 3 failure-labeled trials are
  dated 2021-06-29 or later — a real pattern, not an artifact (osimertinib
  is still mid-lifecycle; most of its trials that stopped early were
  excluded, not labeled FAILURE, by the leakage guard in `labels.py`).
- Train/test split: **cross-validation, not temporal.** A temporal split
  (`--eval-mode temporal`) raises `InsufficientClassDiversityError` on this
  dataset, because every failure postdates 2021 — no cutoff puts both
  classes on both sides of the split. Reported numbers use 3-fold
  stratified CV (`--eval-mode cv`), the documented small-data fallback; see
  `docs/METHODS.md` for exactly what that trades away.

## Evaluation

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.918 | 0.915 |
| PR-AUC | 0.995 | 0.995 |
| Brier score | 0.036 | 0.047 |

(3-fold stratified CV, out-of-fold predictions, n=60.)

**Read this before trusting the AUC.** All 3 failures in the dataset are
EGFR trials; all 42 ABL1 trials are successes. The top SHAP feature,
`ot_tractable_antibody`, is 0 for every ABL1 row and 1 for every EGFR row —
perfectly collinear with *which hypothesis* a trial belongs to, not an
independent biological signal in this dataset. A model can reach ROC-AUC
≈ 0.92 here largely by learning "is this an EGFR trial or an ABL1 trial,"
which trivially predicts the label given the current class split, without
learning anything that would generalize to a third hypothesis. See
`docs/METHODS.md`, section "What v1's real numbers actually mean," for the
full reasoning. The baseline and LightGBM scoring almost identically is
consistent with this: there's little non-linear structure to find in a
dataset this size, of this shape.

No calibration reliability diagram is included — with only 3 positive
examples of the minority class, per-bin calibration counts would not be
meaningful.

## Ethical considerations
- Built entirely from public registry/database data; no patient-level or
  proprietary data involved.
- A published "risk score" for a drug/target/disease hypothesis could, if
  taken out of context, be read as investment or clinical advice — the README
  and API responses carry an explicit disclaimer for this reason.
- The hypothesis-identity confound described above is itself an ethical
  consideration for a tool like this: a superficially good AUC is exactly
  the kind of number that could mislead a non-technical reader into
  overtrusting the model. Documenting it here, rather than omitting it, is
  the point of a model card.

## Caveats and recommendations
See `docs/LIMITATIONS.md` in full. Most importantly: (1) the label is a
proxy (registry status) for scientific/clinical success, not a direct
measurement of it; (2) this dataset is too small and too dominated by
hypothesis identity for its evaluation numbers to indicate real-world
predictive validity. Meaningful evaluation requires expanding
`CURATED_HYPOTHESES` to enough targets/drugs that no single feature
perfectly separates them — that is the actual v2 milestone, not
hyperparameter tuning on the current data.

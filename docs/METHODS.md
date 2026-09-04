# Methods

## Problem framing

Given a (target, disease) pair — optionally anchored to a specific drug — estimate
the probability that a clinical trial testing that hypothesis progresses to a
successful outcome, and surface the features driving that estimate.

This is framed as **binary classification on individual trials** (`success` /
`failure`, see [labels.py](../src/trialsignal/features/labels.py)), aggregated
to the target/disease/drug level at inference time. Trials are the unit of
labeling because that's the unit the label actually exists at; target/disease
"repurposing scores" are a downstream aggregation over the trials sharing that
hypothesis, not a separately labeled quantity.

## Scope constraint

v1 is restricted to **oncology** trials (CT.gov condition query scoped to
cancer/neoplasm terms). This is the therapeutic area with the richest
annotation across all three source databases — most complete `why_stopped`
text, most Open Targets genetic evidence, most ChEMBL bioactivity coverage —
and the scoping decision is one sentence, not an arbitrary cut. Other
therapeutic areas are the natural v2 extension once the pipeline is validated
here.

## Data sources and the join

| Source | Provides | Native ID |
|---|---|---|
| ClinicalTrials.gov API v2 | Trial status, phase, dates, enrollment, `why_stopped` (**the label**) | NCT ID, free-text conditions/interventions |
| Open Targets GraphQL | Target-disease genetic/clinical evidence, tractability, safety liabilities | Ensembl gene ID, EFO disease ID |
| ChEMBL API | Bioactivity (IC50/EC50/Ki), mechanism of action, ADMET | ChEMBL compound/target ID |

None of these IDs agree with each other. [`entity_resolution.py`](../src/trialsignal/data/entity_resolution.py)
is the module that turns "NSCLC" (CT.gov), `ENSG00000146648` (Open Targets),
and `CHEMBL203` (ChEMBL) into one resolved entity — with an explicit
similarity score and confidence threshold rather than a silent string-equality
join. See its module docstring for why this matters.

## Label construction

**Not** `status == COMPLETED → success, everything else → failure`. See
[`labels.py`](../src/trialsignal/features/labels.py)'s module docstring for
the full reasoning — in short: a trial terminated for funding/enrollment
reasons carries no signal about the drug and must be excluded from training,
not mislabeled as a failure. The stop-reason classifier and its test suite
(`tests/unit/test_labels.py`) are the part of this repo worth reading first.

## Modeling

- **Baseline**: logistic regression on the feature set below — exists to
  check the gradient-boosted model is actually earning its complexity, not
  just curve-fitting a small, noisy dataset. On v1's real data the two
  models perform almost identically, which is itself informative (see
  "What v1's numbers actually mean" below) — a real gap would suggest
  LightGBM is finding non-linear structure logistic regression can't; no
  gap suggests the dataset is too small/simple for that distinction to show.
- **Primary model**: LightGBM
  ([`train.py`](../src/trialsignal/models/train.py)), default
  hyperparameters (`n_estimators=100, max_depth=3`). No hyperparameter
  search — with tens of rows, tuning would fit noise, not signal; this is a
  documented tradeoff, not an oversight.
- **Split — two modes, and the choice matters:**
  - `train_and_evaluate` / `--eval-mode temporal` (the methodologically
    correct approach): trials starting before a cutoff date train the
    model, trials starting after it are held out. A random split leaks
    future information (a drug's later trials informing predictions about
    its earlier ones) and produces an inflated, meaningless validation
    score — the single most common mistake in trial-outcome modeling
    papers. It requires enough data that both classes appear on both sides
    of the cutoff; when they don't (see below), it raises
    `InsufficientClassDiversityError` rather than silently doing something
    else.
  - `cross_validate_lightgbm` / `--eval-mode cv` (the documented small-data
    fallback): stratified k-fold, out-of-fold predictions. This reintroduces
    the leakage risk temporal splitting exists to prevent (folds are
    random, not time-ordered) — used only because v1's real dataset cannot
    support a valid temporal split at all (see below), not because it's
    preferred.
- **Interpretability**: SHAP values via `shap.TreeExplainer`, surfaced
  through `top_shap_features` in the training report.
- **Calibration**: Brier score computed on every run. Reliability-diagram
  plotting is not yet built (v1's sample size is too small for the
  per-bin counts to mean anything — see below).

## What v1's real numbers actually mean (read before citing the AUC)

The current curated dataset is 2 hypotheses (EGFR/osimertinib/NSCLC,
ABL1/imatinib/CML), 60 labeled trials total, 57 success / **3** failure — see
`docs/LIMITATIONS.md` for exactly how that number was arrived at. A real
training run (`trialsignal train ... --eval-mode cv`, 3-fold stratified CV,
forced by `InsufficientClassDiversityError` on the temporal split — every
failure is dated 2021+) produced ROC-AUC ≈ 0.92 for both models.

**That number is not evidence the model has learned a generalizable
trial-risk signal, and should not be read as one.** All 3 failures are EGFR
trials; all 42 ABL1 trials are successes. With only 2 hypotheses in the
data, any feature that differs systematically between EGFR and ABL1 — and
several genuinely do, for real biological reasons (ABL1 is an intracellular
kinase, not antibody-tractable; EGFR is a cell-surface receptor, and is) —
is statistically indistinguishable from "which of the two drugs is this,"
a trivial predictor with zero generalization value. The run's own top SHAP
feature, `ot_tractable_antibody`, is a clean illustration: it is 0 for every
ABL1 row and 1 for every EGFR row, i.e. perfectly collinear with hypothesis
identity in this dataset. A high AUC built substantially on that kind of
feature says "the model can tell EGFR trials from ABL1 trials," not "the
model predicts trial risk." Evaluation only becomes meaningful once the
curated hypothesis list (`features/hypothesis.py`) is large enough that no
single feature perfectly separates hypotheses — that's the real bar for
v2, not a higher AUC on the current data.

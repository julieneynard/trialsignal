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

## Modeling (planned — feature pipeline is the current milestone)

- **Baseline**: logistic regression on a small, interpretable feature set —
  exists to make sure the gradient-boosted model is earning its complexity,
  not just curve-fitting noise.
- **Primary model**: LightGBM, tuned via nested cross-validation.
- **Split**: **temporal**, not random. Trials starting before a cutoff date
  train the model; trials starting after it are held out. A random split
  leaks future information (e.g. a drug's later trials informing predictions
  about its earlier ones) and would produce an inflated, meaningless
  validation score — the single most common mistake in trial-outcome
  modeling papers.
- **Interpretability**: SHAP values on every prediction, surfaced through the
  `/score` API and the demo UI — not just a static feature-importance plot in
  a notebook.
- **Calibration**: reliability diagram + Brier score. A 0.7 "risk score" that
  isn't actually 70% empirically is worse than an honest, uncalibrated rank.

## Evaluation

Beyond ROC-AUC/PR-AUC on the held-out temporal split: calibration, and a
comparison against a naive baseline (e.g. "phase alone" — later-phase trials
succeed more often almost by construction, so the model must beat that, not
just beat chance).

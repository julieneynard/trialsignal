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

**How good is that proxy, really?** `clinicaltrials.py` also extracts each
trial's real primary-endpoint p-value from CT.gov's results section, when
one is posted and structured enough to parse (`TrialRecord.primary_pvalue`)
— see its module docstring for exactly how sparse that is (~5-10% in the
initial exploratory sample; ~14% — 65/464 — in the current full dataset,
see below). `trialsignal validate-labels`
([`label_validation.py`](../src/trialsignal/features/label_validation.py))
cross-checks the registry-status label against that independent signal
wherever both exist. Run on the pre-fix dataset (405 rows): 65.8% agreement
on the 38 checkable rows, with every disagreement in the same direction — a
`COMPLETED` trial (labeled `success`) whose primary endpoint did not reach
significance. That's an empirical measurement of the proxy's real error
rate, not a caveat left as a guess — see `docs/LIMITATIONS.md` item 1 for
the full numbers.

**That finding is now acted on, not just measured.**
`labels.resolve_trial_label` — what the feature pipeline actually calls —
prefers the real primary-endpoint result over the registry-status proxy
whenever one exists (`TrialFeatureRow.label_source` records which, per
row), including for trials the proxy alone would have excluded entirely.
Coverage is still ~14% (too sparse to be the *only* label source — most
rows still rely on the proxy), but for that ~14% the label is now ground
truth, not an approximation. `validate-labels` remains in the repo as a
general auditing tool — it's what found this problem, and re-running it
after the fix reports 100% agreement by construction (the label already
came from the same p-value it's checking against), confirming the fix
applied correctly rather than silently leaving the old values in place.

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

## What the real numbers actually mean (read before citing the AUC)

**v1 (2 hypotheses, superseded).** The original dataset was EGFR/osimertinib
(NSCLC) and ABL1/imatinib (CML) — 60 labeled trials, 57 success / 3 failure,
all 3 failures EGFR. A temporal split was literally impossible (every
failure postdated 2021), so evaluation used 3-fold CV, reporting ROC-AUC ≈
0.92. That number was not evidence of a working model: with only 2
hypotheses, any feature differing systematically between EGFR and ABL1 for
real biological reasons (e.g. `ot_tractable_antibody`, 0 for every ABL1 row
and 1 for every EGFR row) was statistically indistinguishable from "which
drug is this," a trivial, non-generalizing predictor.

**v2 (7 hypotheses, current).** `CURATED_HYPOTHESES` was expanded to 7,
chosen for mechanistic diversity specifically to eliminate that shortcut
(see [`hypothesis.py`](../src/trialsignal/features/hypothesis.py)'s module
docstring) — 405 labeled trials, 378 success / 27 failure, with failures
now spread across 5 of 7 hypotheses. Two things changed as a direct result:

1. **A temporal split now works.** Cutoff 2020-01-01 gives 361 train / 44
   test with both classes present on both sides —
   `InsufficientClassDiversityError` no longer fires. This alone confirms
   v1's data problem (all failures clustered post-2021) was a symptom of
   having too few, too similar hypotheses, not an unrelated bug.

2. **The temporal ROC-AUC dropped to ≈ 0.49–0.53 — chance level.** This is
   the honest result, not a regression. Once the model can no longer win by
   learning hypothesis identity, it shows no real ability to predict
   which post-2020 trials succeed from public target/chemistry-level
   features alone. Compare against the same data evaluated by 5-fold CV
   (ROC-AUC ≈ 0.66–0.72, *higher*): CV folds are random, not time-ordered,
   so same-hypothesis rows still land on both sides of most folds, letting
   the model partially learn hypothesis-correlated patterns even without an
   explicit identity feature — leakage the temporal split is specifically
   designed to prevent. That ~0.2 AUC gap between the two evaluation modes,
   on identical data, is the clearest concrete evidence in this repo for
   why temporal splitting is the correct methodology and CV is a fallback,
   not a matter of preference.

**v3 (same 7 hypotheses, results-validated labels, current).**
`labels.resolve_trial_label` now prefers each trial's own primary-endpoint
p-value over the registry-status proxy wherever one is posted and
parseable (~14% of trials) — both correcting proxy-mislabeled rows and
rescuing trials the proxy alone would have excluded. Dataset grew from 405
to **464 rows, 27 to 54 failures** — ABL1 alone went from 42 rows/0
failures to 80 rows/6 failures, meaning v2's "imatinib never fails"
pattern was a labeling artifact of the proxy, not a real property of the
drug. Retraining on this dataset (same 7 hypotheses, same features, same
temporal cutoff): **ROC-AUC ≈ 0.68 (LightGBM, temporal holdout)** — up
from v2's ≈0.53, and for the first time LightGBM clearly beats the
logistic-regression baseline (0.68 vs 0.57), suggesting there's now real
non-linear structure to find, not just noise both models fit equally
badly. The CV-vs-temporal gap that flagged v2's numbers as leaky (~0.2
AUC) has also shrunk to ~0.03-0.04 here — two independently-computed
numbers converging is evidence this result is real, not an artifact of
either evaluation choice.

**The chain matters as much as the destination.** v1's 0.92 was wrong for
one reason (hypothesis confound); v2's fix revealed a second, unrelated
problem (label noise) that had been masked by the first; only fixing both,
in sequence, with a measurement after each step, produced a number worth
trusting. Full numbers, the per-hypothesis breakdown including how many
rows per hypothesis are results-based vs. proxy-based, and what's still
open: `docs/MODEL_CARD.md`.

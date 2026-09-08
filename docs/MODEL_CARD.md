# Model Card — TrialSignal Trial-Progression Risk Model

Following the structure of Mitchell et al., "Model Cards for Model Reporting"
(FAT* 2019), adapted for a research-portfolio project.

**Status: v4.** v1 (2 hypotheses): ROC-AUC ≈ 0.92, confounded. v2 (7
hypotheses): confound fixed, ≈ 0.5, near-chance. v3 (7 hypotheses,
results-validated labels): ≈ 0.68, a real but small-sample estimate. v4 (12
hypotheses, same label fix, more data): **ROC-AUC ≈ 0.63 (temporal
holdout)** — *lower* than v3's headline number, and that's the finding this
version actually reports: v3's 0.68 was measured on a 48-row temporal test
set and was likely optimistic; a larger, more diverse 77-row test set pulls
the honest estimate down, while the CV-vs-temporal agreement gets *tighter*
(≈0.02–0.03 gap, down from ≈0.03–0.04), meaning ≈0.63 is the more trustworthy
number, not a regression to explain away.

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
  demonstrating that a metric from a small test set needs re-checking as
  more data arrives, not just reported once and trusted. A worked example
  of the modeling and evaluation process, not a production predictive tool
  at this dataset size.
- **Out of scope:** Any actual clinical, investment, R&D-strategy, or
  regulatory decision. Live scoring (`/score`) is restricted to the 12
  hand-verified hypotheses in `src/trialsignal/features/hypothesis.py` — not
  a general gene/disease lookup. Not a substitute for a real
  pharmacovigilance or R&D strategy process.

## Training data
- Source: ClinicalTrials.gov (oncology trials — trial metadata *and*, where
  posted, the results section), Open Targets (target-disease evidence),
  ChEMBL (bioactivity) — joined via `trialsignal build-features` for each
  hypothesis. Full source table and join logic: `docs/METHODS.md`.
- **12 curated hypotheses**, chosen for mechanistic *and* disease diversity:
  EGFR/osimertinib (NSCLC), ABL1/imatinib (CML), BRAF/vemurafenib
  (melanoma), ERBB2/trastuzumab — antibody (breast cancer), KDR/sunitinib
  (renal cell carcinoma), PARP1/olaparib (ovarian cancer),
  PDCD1/pembrolizumab — PD-1 checkpoint inhibitor, antibody (melanoma),
  AR/enzalutamide — hormone receptor antagonist (prostate cancer, new
  disease area), BTK/ibrutinib (chronic lymphocytic leukemia, new disease
  area), CD38/daratumumab — antibody (multiple myeloma, new disease area),
  MTOR/everolimus (renal cell carcinoma — same disease as KDR, different
  mechanism), CDK4/palbociclib — cell-cycle inhibitor (breast cancer — same
  disease as ERBB2, different mechanism).
- **n = 686** labeled trials (613 success / **73** failure). Per-hypothesis
  breakdown, including how many rows per hypothesis are graded on a real
  primary-endpoint result (`label_source="results"`) versus the
  registry-status proxy:

  | Hypothesis | n | success | failure | results-based |
  |---|---|---|---|---|
  | PDCD1 / pembrolizumab / melanoma | 114 | 96 | 18 | 13 |
  | ERBB2 / trastuzumab / breast cancer | 102 | 95 | 7 | 11 |
  | KDR / sunitinib / renal cell carcinoma | 89 | 81 | 8 | 16 |
  | ABL1 / imatinib / CML | 80 | 74 | 6 | 7 |
  | CD38 / daratumumab / multiple myeloma | 67 | 64 | 3 | 9 |
  | BTK / ibrutinib / CLL | 48 | 45 | 3 | 12 |
  | MTOR / everolimus / renal cell carcinoma | 46 | 44 | 2 | 0 |
  | AR / enzalutamide / prostate cancer | 43 | 34 | 9 | 12 |
  | PARP1 / olaparib / ovarian cancer | 32 | 25 | 7 | 12 |
  | BRAF / vemurafenib / melanoma | 26 | 22 | 4 | 3 |
  | EGFR / osimertinib / NSCLC | 21 | 17 | 4 | 3 |
  | CDK4 / palbociclib / breast cancer | 18 | 16 | 2 | 0 |

  Two new naming-mismatch fixes were needed to get here (same pattern as
  v2's carcinoma/cancer and CML myelogenous/myeloid gaps): Open Targets
  names multiple myeloma "plasma cell myeloma" — without the fix,
  CD38/daratumumab would have silently returned 0 rows, the same failure
  mode that hit ABL1 before it was caught (`docs/LIMITATIONS.md` item 3).
- Train/test split: temporal, cutoff 2020-01-01 → 609 train / 77 test, both
  classes present in both (15 failures in test).
- MTOR and CDK4 have 0 results-based rows (both landed on trials where a
  primary p-value happened not to be posted/parseable) — a reminder that
  ~14% overall coverage means some individual hypotheses get none of the
  label-quality improvement, purely by chance of which trials exist.

## Evaluation

**Temporal split (cutoff 2020-01-01, n_train=609, n_test=77):**

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.613 | **0.632** |
| PR-AUC | 0.880 | 0.898 |
| Brier score | 0.164 | 0.180 |

**Stratified 5-fold CV (`--eval-mode cv`, same n=686):**

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.621 | 0.660 |
| PR-AUC | 0.928 | 0.940 |

**Read this alongside v3's numbers, not instead of them.** v3 (7
hypotheses, n=464): temporal 0.678, CV 0.709, gap 0.031. v4 (12 hypotheses,
n=686): temporal 0.632, CV 0.660, gap 0.028. The point estimate went down,
but the CV-temporal agreement — the signal used throughout this project to
judge whether a number is trustworthy, not just high — stayed just as tight,
actually slightly tighter. The straightforward read: v3's 48-row temporal
test set gave an estimate with real sampling noise, and it happened to land
high; v4's 77-row test set is a better estimate of the same underlying
quantity, not evidence the added hypotheses hurt the model. Reporting v3's
number without this follow-up would have been the more common mistake —
publishing a good-looking result from a small holdout and not checking if
it holds up.

**LightGBM's edge over the baseline shrank** (+0.02 temporal here vs. +0.11
in v3) — with more, more varied hypotheses, a large part of what LightGBM
was fitting in v3 may have been small-sample structure logistic regression
couldn't reach, not necessarily generalizable non-linear signal. Both
models land in the same ~0.61–0.66 range now, which is itself informative:
neither is finding much the other one can't.

**Top SHAP features (LightGBM, temporal-trained model):** `enrollment`,
`max_phase_ordinal`, `ot_overall_score`, `ot_safety_liability_count`,
`ot_clinical_score` — the same handful of plausible features as v3, in a
different order, still nothing resembling a disguised hypothesis-identity
indicator.

**This is still not a production-grade number.** ≈0.63 ROC-AUC is a real,
modestly-above-chance, cross-validated signal on public trial-metadata
features — informative for a portfolio demonstration, nowhere near what
would be needed to inform an actual go/no-go decision.

No calibration reliability diagram is included — 73 failures across 12
hypotheses is still thin for dense per-bin calibration counts.

## Ethical considerations
- Built entirely from public registry/database data; no patient-level or
  proprietary data involved.
- A published "risk score" for a drug/target/disease hypothesis could, if
  taken out of context, be read as investment or clinical advice — the README
  and API responses carry an explicit disclaimer for this reason.
- This card's full history (v1 → v2 → v3 → v4) is kept rather than
  rewritten to show only the current number — including v3's 0.68 being
  revised down in v4, which is the more uncomfortable but more honest
  thing to publish than quietly updating a single "current AUC" figure.

## Caveats and recommendations
See `docs/LIMITATIONS.md` in full. Most importantly: (1) the label is still
a proxy for ~86% of rows (only ~14% of this dataset has a real
primary-endpoint result); (2) 686 rows for 11 features, with 77 in the
temporal test set, still carries real sampling uncertainty — treat 0.63 as
"modestly above chance, reasonably estimated," not a precise figure, and
expect it to keep moving as more hypotheses are added; (3) broadening
results-based coverage beyond ~14% was investigated and explicitly rejected
as not currently feasible without fabricating labels (`docs/LIMITATIONS.md`
item 1a) — parsing more of CT.gov's results section further is not the
next lever. Continuing to add diverse curated hypotheses remains the
validated way to move this number, understanding that each addition should
be re-measured, not assumed to help.

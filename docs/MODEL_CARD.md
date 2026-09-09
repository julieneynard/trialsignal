# Model Card — TrialSignal Trial-Progression Risk Model

Following the structure of Mitchell et al., "Model Cards for Model Reporting"
(FAT* 2019), adapted for a research-portfolio project.

**Status: v5.** v1 (2 hyp.): ROC-AUC ≈ 0.92, confounded. v2 (7 hyp.): confound
fixed, ≈ 0.5, near-chance. v3 (7 hyp., results-validated labels): ≈ 0.68,
small-sample. v4 (12 hyp., more disease diversity): ≈ 0.63, a larger-sample
correction of v3. v5 (17 hyp., 3 more disease areas + 2 more contrast
pairs): **ROC-AUC ≈ 0.68 (temporal holdout)**, back up from v4 — checked
against CV before trusting the direction this time too (see "Evaluation").
Five versions in, the pattern holding is: point estimates move ±0.05 as
hypotheses are added, the CV-temporal gap stays in the ≈0.03–0.04 band
either way, and neither move should be read as more than "still modestly
above chance, with normal sampling variation at this dataset size."

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
- **n = 840** labeled trials (743 success / **97** failure). Per-hypothesis
  breakdown:

  | Hypothesis | n | success | failure | results-based |
  |---|---|---|---|---|
  | PDCD1 / pembrolizumab / melanoma | 114 | 96 | 18 | 13 |
  | VEGFA / bevacizumab / colorectal cancer | 102 | 84 | 18 | 19 |
  | ERBB2 / trastuzumab / breast cancer | 102 | 95 | 7 | 11 |
  | KDR / sunitinib / renal cell carcinoma | 89 | 81 | 8 | 16 |
  | ABL1 / imatinib / CML | 80 | 74 | 6 | 7 |
  | CD38 / daratumumab / multiple myeloma | 67 | 64 | 3 | 9 |
  | BTK / ibrutinib / CLL | 48 | 45 | 3 | 12 |
  | MTOR / everolimus / renal cell carcinoma | 46 | 44 | 2 | 0 |
  | AR / enzalutamide / prostate cancer | 43 | 34 | 9 | 12 |
  | PARP1 / olaparib / ovarian cancer | 32 | 25 | 7 | 12 |
  | BCL2 / venetoclax / CLL | 28 | 24 | 4 | 7 |
  | BRAF / vemurafenib / melanoma | 26 | 22 | 4 | 3 |
  | EGFR / osimertinib / NSCLC | 21 | 17 | 4 | 3 |
  | CDK4 / palbociclib / breast cancer | 18 | 16 | 2 | 0 |
  | FLT3 / midostaurin / AML | 16 | 15 | 1 | 3 |
  | ALK / crizotinib / NSCLC | 7 | 6 | 1 | 0 |
  | FGFR3 / erdafitinib / bladder cancer | 1 | 1 | 0 | 1 |

  A fourth naming-mismatch fix was needed (after carcinoma/cancer,
  myelogenous/myeloid, plasma-cell myeloma): Open Targets names bladder
  disease "urinary bladder cancer"/"...carcinoma" — checked against the
  live API before hardcoding FGFR3/erdafitinib, scored 0.78 unfixed.
  **FGFR3 landed only 1 row** — erdafitinib is a recent approval (2019)
  with a narrow, still-small trial footprint; a thin real result, not a
  pipeline bug (compare VEGFA/bevacizumab, an established drug, at 102
  rows from the same-sized bladder/colorectal trial pools).
- Train/test split: temporal, cutoff 2020-01-01 → 750 train / 90 test, both
  classes present in both (17 failures in test).

## Evaluation

**Temporal split (cutoff 2020-01-01, n_train=750, n_test=90):**

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.623 | **0.680** |
| PR-AUC | 0.889 | 0.911 |
| Brier score | 0.155 | 0.151 |

**Stratified 5-fold CV (`--eval-mode cv`, same n=840):**

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.628 | 0.645 |
| PR-AUC | 0.925 | 0.931 |

**Read all three versions' CV-vs-temporal gaps together, not any one
number in isolation.**

| Version | Hypotheses | n | Temporal AUC | CV AUC | Gap |
|---|---|---|---|---|---|
| v3 | 7 | 464 | 0.678 | 0.709 | +0.031 (CV higher) |
| v4 | 12 | 686 | 0.632 | 0.660 | +0.028 (CV higher) |
| v5 | 17 | 840 | **0.680** | 0.645 | **−0.035 (temporal higher)** |

v5 is the first version where temporal beats CV rather than the reverse —
a small, direction-flipped gap of similar magnitude to v3/v4's. Read
together with v2's original ≈0.2 AUC gap (the confounded, leaky case this
project used to calibrate what "untrustworthy" looks like), a gap that
stays in the ≈0.03 band and wanders in sign as the dataset grows is
consistent with ordinary sampling variation at n≈800–900, not a
resurfacing of the leakage problem v2 diagnosed. The headline number
itself has now been 0.5 → 0.68 → 0.63 → 0.68 across four re-measurements;
the honest summary is "reliably modestly-above-chance, ±0.05," not any
single point estimate.

**Top SHAP features (LightGBM, temporal-trained model):** `max_phase_ordinal`,
`enrollment`, `ot_overall_score`, `chembl_activity_count`,
`ot_safety_liability_count` — the same recurring, plausible handful across
every version, still nothing resembling a disguised hypothesis-identity
indicator.

**This is still not a production-grade number.** ≈0.65–0.68 ROC-AUC is a
real, modestly-above-chance, repeatedly cross-checked signal on public
trial-metadata features — informative for a portfolio demonstration,
nowhere near what would be needed to inform an actual go/no-go decision.

No calibration reliability diagram is included — 97 failures across 17
hypotheses (several with single-digit failure counts) is still thin for
dense per-bin calibration counts.

## Ethical considerations
- Built entirely from public registry/database data; no patient-level or
  proprietary data involved.
- A published "risk score" for a drug/target/disease hypothesis could, if
  taken out of context, be read as investment or clinical advice — the README
  and API responses carry an explicit disclaimer for this reason.
- This card's full history (v1 → v2 → v3 → v4 → v5) is kept rather than
  collapsed into only the current number — including the AUC moving
  non-monotonically (up, down, up again) as more data arrived, which is
  the more useful thing to show a technical reader than a single
  cherry-picked figure.

## Caveats and recommendations
See `docs/LIMITATIONS.md` in full. Most importantly: (1) the label is still
a proxy for ~85% of rows (only ~15% of this dataset has a real
primary-endpoint result); (2) 840 rows for 11 features, with 90 in the
temporal test set, still carries real sampling uncertainty — the AUC has
moved ±0.05 across the last three re-measurements alone, so treat ≈0.65–0.68
as a range, not a figure; (3) broadening results-based coverage beyond
~15% was investigated and explicitly rejected as infeasible without
fabricating labels (`docs/LIMITATIONS.md` item 1a); (4) several hypotheses
(FGFR3 at n=1, ALK at n=7) are too thin individually to trust in isolation
— they contribute to the pooled estimate but shouldn't be read as
validating those specific drugs. Continuing to add diverse curated
hypotheses, each checked for its own naming mismatches and each triggering
a full re-measurement rather than an assumed improvement, remains the
validated way to work this number.

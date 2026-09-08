# Model Card — TrialSignal Trial-Progression Risk Model

Following the structure of Mitchell et al., "Model Cards for Model Reporting"
(FAT* 2019), adapted for a research-portfolio project.

**Status: v3 — two independent fixes, applied in sequence, each measured
before moving to the next.** v1 (2 hypotheses) reported a confounded
ROC-AUC ≈ 0.92. v2 (7 hypotheses, fixing the confound) reported an honest
but near-chance ROC-AUC ≈ 0.5. v3 (same 7 hypotheses, now with real
primary-endpoint results substituted for the registry-status proxy label
wherever available) reports **ROC-AUC ≈ 0.68 (temporal holdout)** — a real,
above-chance, reproducible result. The full chain of reasoning that got here
is preserved below and in git history because the sequence itself is the
point: each fix was validated against real data before being trusted, not
assumed to help.

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
  demonstrating the diagnostic process (find a confound, fix it, measure the
  result, find the next problem, fix that, measure again) at least as much
  as the resulting number. A worked example of the modeling approach, not a
  production predictive tool at this dataset size.
- **Out of scope:** Any actual clinical, investment, R&D-strategy, or
  regulatory decision. Live scoring (`/score`) is restricted to the 7
  hand-verified hypotheses in `src/trialsignal/features/hypothesis.py` — not
  a general gene/disease lookup. Not a substitute for a real
  pharmacovigilance or R&D strategy process.

## Training data
- Source: ClinicalTrials.gov (oncology trials — trial metadata *and*, where
  posted, the results section), Open Targets (target-disease evidence),
  ChEMBL (bioactivity) — joined via `trialsignal build-features` for each
  hypothesis. Full source table and join logic: `docs/METHODS.md`.
- **7 curated hypotheses**, chosen for mechanistic diversity (not just
  headcount): EGFR/osimertinib (NSCLC), ABL1/imatinib (CML),
  BRAF/vemurafenib (melanoma), ERBB2/trastuzumab — a monoclonal antibody
  (breast cancer), KDR/sunitinib (renal cell carcinoma), PARP1/olaparib
  (ovarian cancer), PDCD1/pembrolizumab — a PD-1 immune checkpoint
  inhibitor, also an antibody (melanoma).
- **n = 464** labeled trials (410 success / **54** failure). Per-hypothesis
  breakdown, including how many rows per hypothesis are graded on a real
  primary-endpoint result (`label_source="results"`) versus the
  registry-status proxy:

  | Hypothesis | n | success | failure | results-based |
  |---|---|---|---|---|
  | PDCD1 / pembrolizumab / melanoma | 114 | 96 | 18 | 13 |
  | ERBB2 / trastuzumab / breast cancer | 102 | 95 | 7 | 11 |
  | KDR / sunitinib / renal cell carcinoma | 89 | 81 | 8 | 16 |
  | ABL1 / imatinib / CML | 80 | 74 | 6 | 7 |
  | PARP1 / olaparib / ovarian cancer | 32 | 25 | 7 | 12 |
  | BRAF / vemurafenib / melanoma | 26 | 22 | 4 | 3 |
  | EGFR / osimertinib / NSCLC | 21 | 17 | 4 | 3 |

  This is up from v2's 405 rows / 27 failures for two reasons, not one:
  more failures were found (54 vs. 27) *and* more rows overall (464 vs.
  405) — `resolve_trial_label` (labels.py) doesn't just correct labels
  where the proxy and the real result disagree, it also rescues trials the
  registry-status proxy alone would have EXCLUDED (ambiguous stop reason,
  still-active status) but that have since posted a clear primary-endpoint
  result. ABL1 is the clearest example: 42 rows/0 failures under the pure
  proxy (see v2 history) → 80 rows/6 failures once real results are
  consulted — the "imatinib never fails" pattern in v2 was a labeling
  artifact, not a real property of the drug.
- Train/test split: temporal, cutoff 2020-01-01 → 416 train / 48 test, both
  classes present in both (12 failures in test).
- PDCD1's bioactivity coverage from ChEMBL is sparse (a handful of records
  vs. hundreds for the small-molecule targets) — expected, not a bug:
  ChEMBL is overwhelmingly small-molecule potency data, and
  pembrolizumab/PD-1 blockade isn't measured that way.
  `chembl_matched_by_molecule_name` reports this per-row.

## Evaluation

**Temporal split (cutoff 2020-01-01, n_train=416, n_test=48):**

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.569 | **0.678** |
| PR-AUC | 0.802 | 0.889 |
| Brier score | 0.201 | 0.212 |

**Stratified 5-fold CV (`--eval-mode cv`, same n=464):**

| | Baseline (logreg) | LightGBM |
|---|---|---|
| ROC-AUC | 0.608 | 0.709 |
| PR-AUC | 0.905 | 0.942 |

**Read the CV-vs-temporal gap, not just the numbers.** In v2 (confounded
hypotheses, proxy labels), CV scored ~0.2 AUC higher than the temporal
holdout on identical data — strong evidence the CV number was inflated by
residual leakage. Here, after fixing both the hypothesis-diversity confound
*and* the label-noise problem, that gap has shrunk to ~0.03–0.04. Two
independent evaluation methodologies converging this closely is itself
evidence the ≈0.68–0.71 LightGBM result reflects real, reproducible
structure in the data — not an artifact of either evaluation choice. This
is a materially different, and more trustworthy, situation than v2's near-
chance-but-inconsistent numbers.

**LightGBM clearly beats the logistic-regression baseline for the first
time** (+0.11 temporal, +0.10 CV) — in v1 and v2 the two models scored
almost identically, suggesting there was no non-linear structure worth a
gradient-boosted model. That gap appearing now, alongside the accuracy
gain, is consistent with the label-quality fix having surfaced real signal
a noisier target obscured, rather than just changing which rows are in the
dataset.

**Top SHAP features (LightGBM, temporal-trained model):** `enrollment`,
`ot_overall_score`, `max_phase_ordinal`, `ot_safety_liability_count`,
`ot_clinical_score` — plausible on their face (bigger, later-phase trials
with strong target-disease evidence tend to succeed more), and still not
led by anything resembling a disguised hypothesis-identity feature.

**This is still not a production-grade number, and should not be read as
one.** 0.68 ROC-AUC is a real, above-chance, cross-validated signal on
public trial-metadata features — nowhere near what would be needed to
inform an actual go/no-go decision. See "Caveats and recommendations."

No calibration reliability diagram is included — 54 failures across 7
hypotheses is still thin for dense per-bin calibration counts.

## Ethical considerations
- Built entirely from public registry/database data; no patient-level or
  proprietary data involved.
- A published "risk score" for a drug/target/disease hypothesis could, if
  taken out of context, be read as investment or clinical advice — the README
  and API responses carry an explicit disclaimer for this reason.
- This card's history (v1 → v2 → v3, each with the honest number for its
  stage rather than only the final one) is deliberately kept rather than
  rewritten, so a reader can see how the model got here, not just where it
  ended up.

## Caveats and recommendations
See `docs/LIMITATIONS.md` in full. Most importantly: (1) the label is still
a proxy for ~90% of rows (only ~14% of this dataset has a real
primary-endpoint result — see the "results-based" column above); (2) 464
rows for 11 features, with only 48 in the temporal test set, means these
numbers carry real sampling uncertainty — treat 0.68 as "meaningfully above
chance," not as a precise estimate. The most likely next lever, per the
roadmap, is finding a second results-derived signal with broader coverage
than the primary-endpoint p-value (~50% of trials have posted *some*
results, vs. ~14% with a parseable primary p-value) to extend the v3 fix's
benefit to more of the dataset.

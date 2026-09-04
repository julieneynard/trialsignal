# Model Card — TrialSignal Trial-Progression Risk Model

Following the structure of Mitchell et al., "Model Cards for Model Reporting"
(FAT* 2019), adapted for a research-portfolio project. **Status: template —
populated once training (`trialsignal train`) has run; see field notes below
for what each section will contain.**

## Model details
- **Developed by:** Julien Eynard, independent portfolio project.
- **Model type:** Gradient-boosted decision trees (LightGBM), binary
  classification, with a logistic-regression baseline for comparison.
- **Version:** unreleased — populated with a semantic version + training date
  from `trialsignal.models.registry.ModelBundle` once an artifact exists.

## Intended use
- **In scope:** Exploratory research signal for target/disease repurposing
  hypotheses and relative trial-risk ranking, built entirely on public data.
- **Out of scope:** Any actual clinical, investment, or regulatory decision.
  Not validated for use beyond the oncology trials it was trained on (see
  `LIMITATIONS.md`). Not a substitute for a real pharmacovigilance or R&D
  strategy process.

## Training data
- ClinicalTrials.gov (oncology trials, v2 API), Open Targets (target-disease
  evidence), ChEMBL (bioactivity/ADMET) — see `docs/METHODS.md` for the full
  source table and the entity-resolution join between them.
- **To be filled in after training:** trial count, date range, class balance
  (success/failure/excluded), train/test split date cutoff.

## Evaluation
- **To be filled in after training:** ROC-AUC / PR-AUC on the temporal
  held-out split, calibration (Brier score + reliability diagram), and
  comparison against the "phase alone" naive baseline described in
  `METHODS.md`.

## Ethical considerations
- Built entirely from public registry/database data; no patient-level or
  proprietary data involved.
- A published "risk score" for a drug/target/disease hypothesis could, if
  taken out of context, be read as investment or clinical advice — the README
  and API responses carry an explicit disclaimer for this reason.

## Caveats and recommendations
See `docs/LIMITATIONS.md` in full — most importantly, the label is a proxy
(registry status) for scientific/clinical success, not a direct measurement
of it.

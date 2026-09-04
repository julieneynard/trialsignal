# Limitations

Stated plainly, up front, rather than discovered by a reviewer.

1. **"Completed" is a proxy for success, not proof of it.** A trial can run
   to completion and still miss its primary endpoint. CT.gov's registry
   *status* field alone can't distinguish that — it would require parsing the
   trial *results* section (effect sizes / p-values against the declared
   primary outcome), which is a documented extension, not part of v1's label.
   Treat `risk_score` as "probability the trial isn't abandoned for cause,"
   not "probability the drug works."

2. **Stop-reason classification is keyword-based, not a trained classifier.**
   `classify_stop_reason` (see `labels.py`) uses regex pattern matching over
   `why_stopped` free text. It's precision-oriented by design — ambiguous
   text is excluded rather than guessed — but it will still misclassify some
   trials, and the keyword list reflects oncology-trial language specifically.

3. **Entity resolution has a hard confidence cutoff, and the drug-matching
   step is a bigger coverage bottleneck than it looks.** Measured on the
   real dataset the current model was trained on: pulling all
   ClinicalTrials.gov trials for "non-small cell lung cancer" (3,977 trials)
   and "chronic myeloid leukemia" (1,796 trials) yielded only 18
   osimertinib/NSCLC rows and 42 imatinib/CML rows (60 total) after
   intervention matching, label resolution, and disease matching. Most
   pulled trials test a different drug entirely (expected — these are broad
   disease-level pulls), and a meaningful share of the remainder are still
   RECRUITING/ACTIVE_NOT_RECRUITING (expected for osimertinib specifically,
   which is still mid-lifecycle) rather than resolution failures. The
   `build-features` CLI reports this ratio (`N/M pulled trials matched`)
   every run rather than hiding it. Separately, ChEMBL bioactivity matching
   by molecule `pref_name` found zero marketed-drug-name matches in the
   pulled activity pages for either target — `chembl_matched_by_molecule_name`
   is `False` for every row in the trained dataset, meaning ChEMBL features
   fell back to target-level aggregates throughout. A molecule-name-first
   ChEMBL query (search by compound name, then pull its activities directly,
   rather than filtering a large target-level activity page) is the natural
   fix and is on the roadmap, not yet built.

4. **60 labeled rows from 2 hypotheses is not enough to evaluate a model
   on, and the trained v1 model's ROC-AUC (~0.92) should not be read as
   evidence it works.** All 3 failure-labeled trials in the dataset are
   EGFR trials; all 42 ABL1 trials are successes — meaning class label is
   almost perfectly confounded with *which of the two curated hypotheses*
   a row belongs to. The trained model's top SHAP feature,
   `ot_tractable_antibody`, is 0 for every ABL1 row and 1 for every EGFR
   row: a real biological fact (ABL1 is intracellular, EGFR is a
   cell-surface receptor) that in this dataset is indistinguishable from a
   literal "is this an EGFR trial" indicator. A model can score well here
   by learning to tell the two hypotheses apart, which is not the same
   claim as "predicts trial risk." See `docs/METHODS.md` ("What v1's real
   numbers actually mean") and `docs/MODEL_CARD.md` for the full reasoning
   — this is the single most important caveat on the current model, more
   important than any of the others in this document. It resolves only by
   adding enough distinct hypotheses that no one feature separates them.

5. **The real dataset's failures are clustered in time (all 3 postdate
   2021-06-29), which made a temporal train/test split impossible.**
   `train_and_evaluate` (the methodologically correct `--eval-mode temporal`)
   raises `InsufficientClassDiversityError` on this data — no cutoff date
   produces a training split containing both classes. The reported v1
   numbers use `--eval-mode cv` (stratified k-fold) instead, a documented
   downgrade that reintroduces the temporal-leakage risk `temporal_split`
   exists to prevent (see `docs/METHODS.md`). This is a direct consequence
   of limitation 4 above (too little data, too concentrated) rather than a
   separate root cause.

6. **Public-data-only.** No access to unpublished internal pharma trial data,
   proprietary ADMET assays, or FDA advisory committee deliberations — all of
   which materially affect real go/no-go decisions. This is a research-grade
   signal built entirely from what's publicly disclosed, not a production
   pharma decision tool.

7. **Therapeutic-area scope.** v1 is oncology-only (see METHODS.md). Findings
   and feature importances should not be assumed to generalize to other
   disease areas without re-validation — oncology trial dynamics (fast
   biomarker-driven attrition, adaptive designs) differ meaningfully from,
   say, chronic disease trials.

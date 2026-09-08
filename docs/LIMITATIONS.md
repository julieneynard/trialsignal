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
   step is a bigger coverage bottleneck than it looks.** Also: this class of
   naming mismatch (colloquial trial phrasing vs. formal Open Targets/EFO
   phrasing) is not a one-off — after the carcinoma/cancer gap (found while
   building the join pipeline), building the live `/score` endpoint
   surfaced a second, independent instance: CT.gov trials say "chronic
   myeloid leukemia," Open Targets' actual CML entry is "chronic
   myelogenous leukemia, BCR-ABL1 positive," and without a targeted fix
   (`_MYELOGENOUS_SYNONYM` + `_BIOMARKER_QUALIFIER` in
   `entity_resolution.py`) this silently broke the *entire* ABL1/imatinib
   hypothesis — `build_live_feature_vector` returned `None` for every real
   CML query. The pattern (a small, curated, documented synonym table
   rather than a general fuzzy-matching threshold change) generalizes; the
   next hypothesis added to `CURATED_HYPOTHESES` should expect to need its
   own naming check against the live API before trusting it works. Measured on the
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

   The 5 hypotheses added for v2 show the same funnel pattern, at varying
   yield: BRAF/vemurafenib/melanoma 26/3,728, ERBB2/trastuzumab/breast
   cancer 100/3,988, KDR/sunitinib/renal cell carcinoma 87/2,710,
   PARP1/olaparib/ovarian cancer 24/3,990, PDCD1/pembrolizumab/melanoma
   108/3,728 — consistent with this being a structural property of broad
   disease-level pulls, not specific to the original two. PDCD1's ChEMBL
   pull returned only 3 bioactivity records total (vs. hundreds for every
   small-molecule target) — expected, not a bug: ChEMBL is overwhelmingly
   small-molecule potency data, and PD-1 blockade by an antibody isn't
   measured that way.

4. **[Fixed in v2, kept here as a worked example] 60 labeled rows from 2
   hypotheses was not enough to evaluate a model on, and v1's ROC-AUC
   (~0.92) was not evidence it worked.** All 3 failure-labeled trials were
   EGFR trials; all 42 ABL1 trials were successes — class label was almost
   perfectly confounded with *which curated hypothesis* a row belonged to.
   `CURATED_HYPOTHESES` was expanded from 2 to 7 specifically to fix this
   (see `hypothesis.py`'s module docstring), chosen for mechanistic
   diversity (a monoclonal antibody, a PARP inhibitor, a PD-1 checkpoint
   inhibitor, across 6 diseases) rather than just more of the same shape of
   data. Result: 405 rows, 27 failures spread across 5 of 7 hypotheses, and
   the temporal ROC-AUC dropped to ≈0.5 (chance) — the honest number, not a
   regression. Full reasoning: `docs/METHODS.md` ("What the real numbers
   actually mean") and `docs/MODEL_CARD.md`. **This is still the most
   important thing to understand about this model**: the confound is fixed,
   but the near-chance result means the current public feature set doesn't
   yet predict trial risk well — that's a different, now-honest limitation
   (see #4a below), not the one described here.

4a. **Even with the confound fixed, near-chance accuracy means the current
   features likely don't capture what actually drives trial outcomes.**
   405 rows for 11 features, evaluated on a genuine 44-row temporal holdout,
   is still a small, noisy estimate — but the CV-vs-temporal gap (≈0.66-0.72
   vs ≈0.49-0.53 ROC-AUC on identical data, see METHODS.md) is concrete
   evidence the temporal number reflects real difficulty, not just sample
   noise pulling toward 0.5. The features available (target-disease
   association scores, coarse bioactivity aggregates, trial phase/enrollment)
   are target/chemistry-level and never see the things that plausibly drive
   most real trial outcomes: protocol design quality, patient selection
   criteria, competitive landscape, or the actual efficacy/safety data
   itself (see limitation #1 — the label is a registry-status proxy, not a
   results-based outcome). Closing this gap is a data problem before it's a
   modeling problem.

5. **[Fixed in v2] The v1 dataset's failures were clustered in time (all 3
   postdated 2021-06-29), which made a temporal train/test split
   impossible** — `train_and_evaluate` raised `InsufficientClassDiversityError`
   on that data, with no cutoff able to put both classes on both sides.
   This turned out to be a symptom of limitation 4 (too few, too similar
   hypotheses), not an independent problem: with 7 hypotheses and failures
   spread more broadly across drugs and time, a temporal split (cutoff
   2020-01-01) now produces 361 train / 44 test rows with both classes
   present in both. `cross_validate_lightgbm` / `--eval-mode cv` remains in
   the codebase as the documented fallback for whenever a future dataset
   subset doesn't support a temporal split — this fix doesn't make that
   fallback obsolete, just unnecessary for the current full dataset.

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

8. **Live `/score` trades coverage for latency, and has no in-flight
   request de-duplication.** Measured against the live APIs: fetching
   EGFR's full Open Targets association list (10 pages, 6,459 rows) took
   ~79s for a single cold request and once hit a transient ChEMBL 5xx that
   exhausted the client's retry budget. `/score` caps pagination at 2 pages
   per source instead (a cache-miss request now takes single-digit seconds
   — see `serving/api.py`'s module docstring for the full reasoning and the
   sorted-by-score argument for why this is safe across all 7 current
   hypotheses). Two concurrent cache-miss requests for the *same*
   never-yet-cached hypothesis will both independently hit the live APIs
   rather than one waiting on the other's in-flight fetch — harmless
   (redundant work, not incorrect results) at this project's traffic level,
   but a real gap a production version would need a per-key lock for.

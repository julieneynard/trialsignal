# Limitations

Stated plainly, up front, rather than discovered by a reviewer.

1. **["Completed" is a proxy, not proof — measured, then fixed for the rows
   it could be fixed for.]** `trialsignal validate-labels`
   (`features/label_validation.py`) found this empirically before it was
   fixed: run against the pre-fix 405-row dataset, only 38 rows (~9%) had a
   usable primary-endpoint p-value, and among those, agreement with the
   registry-status label was only **65.8%** — all 13 disagreements the same
   direction, a trial labeled `success` (it was `COMPLETED`, not
   terminated) whose primary endpoint did **not** reach statistical
   significance (p-values from 0.08 to 0.98). `labels.resolve_trial_label`
   now acts on that finding: it substitutes the real result for the proxy
   label wherever one exists, and rescues trials the proxy alone would
   have excluded. Net effect on the full 7-hypothesis dataset: 405 → **464**
   rows, 27 → **54** failures — ABL1 alone went from 42 rows/0 failures to
   80 rows/6 failures, meaning "imatinib never fails" was a labeling
   artifact, not a real property of the drug (see `docs/MODEL_CARD.md`).
   **This remains a real, live limitation for the ~86% of rows still
   proxy-labeled** — coverage of the real signal (~14% of trials) is the
   bottleneck now, not the substitution logic. Treat `risk_score` as
   "probability grounded in a real result where `label_source="results"`,
   probability the trial wasn't abandoned for cause otherwise" — the two
   are not the same claim, and the CSV output distinguishes them per row.

1a. **Investigated and rejected: broadening results-based coverage from
   ~14% to the ~50% of trials with `hasResults=True`.** The natural next
   idea after limitation 1's fix — use trials with posted results but no
   parseable primary-endpoint p-value, inferring success/failure from raw
   arm-level measurements instead. Checked against 133 real examples (every
   `hasResults=True`/no-p-value trial across 4 hypotheses' feature tables)
   before writing any extraction code, the same way every other join/label
   decision in this pipeline was checked against real data first. Result:
   this is not a coverage-vs-noise tradeoff, it's mostly not a comparative
   signal at all —
     - **39/133 (29%)** have a safety/toxicity primary endpoint (adverse
       event counts, dose-limiting toxicities) — doesn't speak to efficacy
       regardless of parsing quality.
     - **54/133 (41%)** are single-arm (dose-escalation cohorts, Phase
       1/1b safety-lead-ins) — no comparator to measure "better than" at
       all; judging success would require an external, disease- and
       line-of-therapy-specific historical benchmark this project has no
       verified source for.
     - **38/133 (29%)** have 2+ arms but no interpretable control/placebo
       label (e.g. dose cohorts, or arms named "Arm 1"/"Arm 2") — inferring
       which arm is the comparator would be guessing, not extracting.
     - **2/133 (1.5%)** have an explicit placebo/control-labeled arm — too
       few to be worth building a directional-inference pipeline for, and
       even these still require a per-outcome-type "which direction is
       better" table to interpret without a significance test.

   **Decision: not pursued.** Building a heuristic on this data would mean
   guessing on ~99% of the addressable rows (single-arm trials and
   unlabeled multi-arm cohorts), which is exactly the failure mode this
   project has repeatedly found and fixed elsewhere (see limitation 1
   itself, and the carcinoma/cancer and CML naming fixes in item 3) — a
   real coverage number bought with a fabricated label is a worse trade
   than the current ~14%. If this is revisited, it needs either genuine
   oncology domain input (historical ORR/response-rate benchmarks per
   disease and treatment line) or a narrower, still-conservative rule
   (e.g. only 2-arm trials with an explicit placebo/SoC label, accepting
   the ~1.5% coverage that leaves) — not a general arm-comparison heuristic.

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
   own naming check against the live API before trusting it works. A third
   instance was found the same way when adding v4's hypotheses: Open
   Targets names multiple myeloma "plasma cell myeloma," not "multiple
   myeloma" — checked against the live API before hardcoding
   CD38/daratumumab (per the project's standing rule), scored 0.63
   unfixed, fixed with `_PLASMA_CELL_MYELOMA_SYNONYM` (deliberately the
   literal phrase, not a general "myeloma" rule, since "smoldering plasma
   cell myeloma" is a distinct precursor condition that must not collapse
   into it). A fourth was found adding v5's hypotheses: Open Targets names
   bladder disease "urinary bladder cancer"/"...carcinoma" — checked
   against the live API before hardcoding FGFR3/erdafitinib, scored 0.78
   unfixed, fixed with `_URINARY_BLADDER_SYNONYM`. Four for four now:
   every batch of new hypotheses added to this project has needed its own
   naming check, without exception — treat that as the expected cost of
   adding the next one, not a risk that might not materialize. Measured on
   the real dataset
   the current model was trained on (post item-1 fix — numbers below
   include results-based rescues): pulling all ClinicalTrials.gov trials
   for "non-small cell lung cancer" (3,977 trials) and "chronic myeloid
   leukemia" (1,796 trials) yielded 21 osimertinib/NSCLC rows and 80
   imatinib/CML rows after intervention matching, label resolution, and
   disease matching. Most pulled trials test a different drug entirely
   (expected — these are broad disease-level pulls), and a meaningful
   share of the remainder are still RECRUITING/ACTIVE_NOT_RECRUITING
   (expected for osimertinib specifically, which is still mid-lifecycle)
   rather than resolution failures. The `build-features` CLI reports this
   ratio (`N/M pulled trials matched`) every run rather than hiding it.
   Separately, ChEMBL bioactivity matching by molecule `pref_name` found
   zero marketed-drug-name matches in the pulled activity pages for either
   target — `chembl_matched_by_molecule_name` was `False` for every
   EGFR/ABL1 row, meaning ChEMBL features fell back to target-level
   aggregates throughout. **Fixed in v6** (see item 3a below): a
   molecule-name-first ChEMBL query, resolving the drug name to a ChEMBL
   molecule ID first and querying `activity.json` filtered by both
   `molecule_chembl_id` and `target_chembl_id`, now finds the exact
   compound-target data directly instead of filtering a truncated
   target-level pull.

   The 5 hypotheses added for v2 show the same funnel pattern, at varying
   yield: BRAF/vemurafenib/melanoma 26/3,728, ERBB2/trastuzumab/breast
   cancer 102/3,988, KDR/sunitinib/renal cell carcinoma 89/2,710,
   PARP1/olaparib/ovarian cancer 32/3,990, PDCD1/pembrolizumab/melanoma
   114/3,728 — consistent with this being a structural property of broad
   disease-level pulls, not specific to the original two. PDCD1's ChEMBL
   pull returned only 3 bioactivity records total (vs. hundreds for every
   small-molecule target) — expected, not a bug: ChEMBL is overwhelmingly
   small-molecule potency data, and PD-1 blockade by an antibody isn't
   measured that way. The 5 hypotheses added for v4 continue the pattern:
   AR/enzalutamide/prostate cancer 43/3,984, BTK/ibrutinib/CLL 48/2,601,
   CD38/daratumumab/multiple myeloma 67/3,967, MTOR/everolimus/renal cell
   carcinoma 46/2,710, CDK4/palbociclib/breast cancer 18/3,988. The 5
   hypotheses added for v5: ALK/crizotinib/NSCLC 7/3,977 (reuses the
   existing NSCLC pool — a much lower yield than EGFR's 21/3,977 from the
   same pool, since ALK-positive NSCLC is a smaller trial-eligible
   subpopulation), FLT3/midostaurin/AML 16/3,976, BCL2/venetoclax/CLL
   28/2,601, FGFR3/erdafitinib/bladder cancer **1/2,293** (erdafitinib's
   2019 approval means a genuinely small real trial history, not a
   pipeline problem — see `docs/MODEL_CARD.md`), VEGFA/bevacizumab/
   colorectal cancer 102/3,996 (bevacizumab is a long-established drug;
   contrast with FGFR3 shows the funnel yield tracks real-world trial
   volume, not a fixed rate).

3a. **[Fixed in v6] Molecule-name-first ChEMBL matching, and a genuine
   database-coverage gap it revealed.** `find_molecule_ids_by_synonym`
   (`data/chembl.py`) resolves a hypothesis's `drug_aliases` to ChEMBL
   molecule IDs via `molecule_synonyms__molecule_synonym__iexact` search
   (covers generic and brand names — "osimertinib" and "Tagrisso" both
   resolve to `CHEMBL3353410`), then `iter_activities_for_molecule` queries
   `activity.json` filtered by both `molecule_chembl_id` and
   `target_chembl_id`. Verified against the live API before writing the
   fix: this recovers 485 real osimertinib/EGFR bioactivity records that
   the old target-level-pull-then-filter approach found zero of. Rebuilding
   all 17 hypotheses' feature tables with this fix (`build_feature_table`,
   and mirrored in the live `/score` endpoint's `_get_activities`) moved
   `chembl_matched_by_molecule_name` from ~0% to **exactly 100%** for all
   13 small-molecule hypotheses. It did **not** move the 4 antibody
   hypotheses (pembrolizumab, trastuzumab, daratumumab, bevacizumab) off
   0% — checked directly against ChEMBL's API and confirmed these 4 drugs
   have no bioactivity record under any target or standard-type at all, in
   or out of this project's pipeline. This is a real, structural property
   of ChEMBL (small-molecule IC50/EC50/Ki potency data, not how antibody
   mechanisms are characterized), not a residual matching gap — nothing
   left to fix here for those 4 hypotheses. Net model effect: temporal
   ROC-AUC essentially unchanged (0.680 → 0.676), but
   `chembl_activity_count` moved to the model's 3rd-most-important SHAP
   feature (from 4th) — the fix demonstrably changed what real data the
   model sees without moving the headline metric, and is reported as
   exactly that rather than reframed as a win. See `docs/METHODS.md`'s v6
   section for the full retraining discussion, including a CV-temporal gap
   (−0.076, vs. −0.035 in v5) that widened enough to flag rather than wave
   off — still nowhere near v2's leaky ≈0.2 gap, but the largest since.

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
   and (see #4a below, kept current as the model has been re-measured
   through v3, v4, and v5) the label-quality fix and further hypothesis
   growth have since moved accuracy from near-chance to a real, modestly
   above-chance result that keeps getting re-checked, not just asserted
   once — read #4a's current text for the up-to-date number, not the
   "near-chance" framing this note originally shipped with.

4a. **[Current as of v6 — no longer near-chance, but still far from
   decision-grade, and the number has moved four times since this note
   was first written.]** Confound fixed (limitation 4) + results-validated
   labels (limitation 1) took temporal ROC-AUC from ≈0.5 (v2) to ≈0.68 (v3,
   464 rows, 48-row test). 5 more hypotheses for disease diversity (v4, 686
   rows, 77-row test) brought it to ≈0.63 — down. 5 more again (v5, 840
   rows, 90-row test) brought it back to ≈0.68. Fixing ChEMBL molecule-name
   matching (v6, limitation 3a; same 17 hypotheses, 843 rows, 90-row test)
   left it essentially flat at **≈0.68 (0.676)** — a real fix that changed
   real feature data for 13/17 hypotheses without moving this metric,
   reported as exactly that. Each move was checked against CV before being
   trusted: the CV-temporal gap stayed in the ≈0.03 band across v3/v4/v5
   (0.031, 0.028, 0.035 — v5's flipped in direction, temporal now scoring
   higher than CV, but the *magnitude* stayed consistent), unlike v2's
   ≈0.2 gap in one fixed direction, which is what real leakage looked like
   when this project had it. **v6 breaks that tight pattern**: its gap is
   −0.076, roughly double v5's — still nowhere near v2's ≈0.2 leaky gap,
   and plausibly explained by v6 changing the underlying ChEMBL feature
   distribution rather than being a like-for-like re-measurement, but
   flagged here rather than folded into "normal ±0.05 wandering" without
   comment. It is still a modest dataset for 11 features, and the feature
   set is still target/chemistry-level, missing protocol design quality,
   patient selection criteria, and competitive landscape — real drivers of
   trial outcomes this pipeline has no source for. ~83% of rows are still
   registry-status-labeled, not results-based (limitation 1) — but
   broadening that coverage was investigated and rejected (limitation 1a);
   further hypothesis growth, re-measured every time rather than assumed
   to help, is the validated lever left.

5. **[Fixed in v2] The v1 dataset's failures were clustered in time (all 3
   postdated 2021-06-29), which made a temporal train/test split
   impossible** — `train_and_evaluate` raised `InsufficientClassDiversityError`
   on that data, with no cutoff able to put both classes on both sides.
   This turned out to be a symptom of limitation 4 (too few, too similar
   hypotheses), not an independent problem: with more hypotheses and
   failures spread more broadly across drugs and time, a temporal split
   (cutoff 2020-01-01) now produces 753 train / 90 test rows (v6, 17
   hypotheses; was 750/90 in v5, 609/77 with 12 hypotheses, 416/48 with 7
   hypotheses post limitation-1 fix, 361/44 with 7 hypotheses pre-fix) with
   both classes present in both. `cross_validate_lightgbm` / `--eval-mode cv` remains in
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
   sorted-by-score argument for why this is safe across all 17 current
   hypotheses). Two concurrent cache-miss requests for the *same*
   never-yet-cached hypothesis will both independently hit the live APIs
   rather than one waiting on the other's in-flight fetch — harmless
   (redundant work, not incorrect results) at this project's traffic level,
   but a real gap a production version would need a per-key lock for.

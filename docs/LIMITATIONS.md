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
   step is a bigger coverage bottleneck than it looks.** Measured on a real
   pull (`EGFR / osimertinib / NSCLC`, 1,989 ClinicalTrials.gov trials for
   "non-small cell lung cancer"): 16 trials mentioned osimertinib in their
   interventions, but only 10 both matched a confident disease and had a
   resolved (non-EXCLUDED) outcome label — most of the drug's trials are
   still RECRUITING/ACTIVE_NOT_RECRUITING, which is expected for a drug
   still mid-lifecycle (osimertinib), not a resolution failure. The
   `build-features` CLI command reports this ratio (`N/M pulled trials
   matched`) every run rather than hiding it — a small feature table for an
   actively-trialed drug is the correct, honest output, not a bug to chase.
   Separately, in that same run none of the 300 pulled ChEMBL bioactivity
   records had a `pref_name` matching the drug (`chembl_matched_by_molecule_name`
   was False for every row) — the marketed-drug-name activities exist in
   ChEMBL but weren't in the first 300 (of 26,600+) IC50 records pulled for
   the target; a molecule-name-first ChEMBL query is the natural fix and is
   on the roadmap, not yet built.

4. **Public-data-only.** No access to unpublished internal pharma trial data,
   proprietary ADMET assays, or FDA advisory committee deliberations — all of
   which materially affect real go/no-go decisions. This is a research-grade
   signal built entirely from what's publicly disclosed, not a production
   pharma decision tool.

5. **Therapeutic-area scope.** v1 is oncology-only (see METHODS.md). Findings
   and feature importances should not be assumed to generalize to other
   disease areas without re-validation — oncology trial dynamics (fast
   biomarker-driven attrition, adaptive designs) differ meaningfully from,
   say, chronic disease trials.

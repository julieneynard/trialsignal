"""Joins the three source systems into one training-ready table, per
Hypothesis.

The join has two steps, both handled by entity_resolution rather than by
naive equality, because naive equality is exactly what silently corrupts a
dataset like this:

  1. Which trials are actually testing this hypothesis? A trial's
     `interventions` list is free text ("Osimertinib", "AZD9291 80mg
     tablet") matched against `Hypothesis.drug_aliases`, and its
     `conditions` list is free text matched against the Open Targets
     disease name for this target via `resolve_condition_to_efo` — a trial
     whose condition doesn't clear the confidence threshold is dropped, not
     force-matched.

  2. Which ChEMBL activities describe *this* drug specifically, versus the
     target in general? Matched by `pref_name` against the same
     `drug_aliases`. When no molecule-specific match exists (common — most
     ChEMBL activities are for research compounds, not the marketed drug
     name), aggregates fall back to *all* activities for the target, and
     `chembl_matched_by_molecule_name=False` records that this happened —
     a "how potent is this exact drug" question silently answered with "how
     druggable is this target in general" is a different claim, and callers
     need to be able to tell the two apart.
"""

from __future__ import annotations

from trialsignal.data.entity_resolution import resolve_condition_to_efo
from trialsignal.data.schemas import (
    ChemblActivity,
    TargetDiseaseAssociation,
    TrialFeatureRow,
    TrialRecord,
)
from trialsignal.features.hypothesis import Hypothesis
from trialsignal.features.labels import TrialOutcome, build_trial_outcome_label

_DISEASE_MATCH_THRESHOLD = 0.85


def _matches_drug(intervention_or_name: str, drug_aliases: list[str]) -> bool:
    normalized = intervention_or_name.strip().lower()
    return any(alias.lower() in normalized for alias in drug_aliases)


def _trial_matches_hypothesis(trial: TrialRecord, hypothesis: Hypothesis) -> bool:
    return any(_matches_drug(i, hypothesis.drug_aliases) for i in trial.interventions)


def _aggregate_chembl(
    activities: list[ChemblActivity], drug_aliases: list[str]
) -> tuple[int, float | None, bool]:
    """Returns (activity_count, best_pchembl, matched_by_molecule_name)."""
    molecule_matched = [
        a for a in activities if a.pref_name and _matches_drug(a.pref_name, drug_aliases)
    ]
    pool = molecule_matched if molecule_matched else activities
    pchembl_values = [a.pchembl_value for a in pool if a.pchembl_value is not None]
    best = max(pchembl_values) if pchembl_values else None
    return len(pool), best, bool(molecule_matched)


def build_feature_table(
    hypothesis: Hypothesis,
    trials: list[TrialRecord],
    target_diseases: list[TargetDiseaseAssociation],
    activities: list[ChemblActivity],
) -> list[TrialFeatureRow]:
    """Build one TrialFeatureRow per trial that (a) tests a drug matching
    `hypothesis.drug_aliases`, (b) has a resolvable success/failure label
    (see labels.py — in-progress and ambiguously-stopped trials are
    excluded), and (c) has a condition confidently resolvable to one of
    `target_diseases`' disease entries.

    A trial failing any of those three checks is silently dropped, by
    design — this is the leakage/mismatch guard described in the module
    docstring, not a bug. The gap between `len(trials)` and
    `len(build_feature_table(...))` is exactly the pipeline's real coverage,
    and callers should report it rather than treat it as free.
    """
    disease_candidates = [(td.disease_id, td.disease_name) for td in target_diseases]
    target_disease_by_id = {td.disease_id: td for td in target_diseases}

    rows: list[TrialFeatureRow] = []
    for trial in trials:
        if not _trial_matches_hypothesis(trial, hypothesis):
            continue

        label = build_trial_outcome_label(trial)
        if label == TrialOutcome.EXCLUDED:
            continue

        best_match = None
        for condition in trial.conditions:
            match = resolve_condition_to_efo(
                condition, disease_candidates, confidence_threshold=_DISEASE_MATCH_THRESHOLD
            )
            is_better = best_match is None or (match is not None and match.score > best_match.score)
            if match is not None and match.confident and is_better:
                best_match = match
        if best_match is None:
            continue

        target_disease = target_disease_by_id[best_match.efo_id]
        matched_drug = next(
            (i for i in trial.interventions if _matches_drug(i, hypothesis.drug_aliases)),
            hypothesis.drug_aliases[0],
        )
        activity_count, best_pchembl, matched_by_name = _aggregate_chembl(
            activities, hypothesis.drug_aliases
        )

        rows.append(
            TrialFeatureRow(
                nct_id=trial.nct_id,
                label=label.value,
                gene_symbol=hypothesis.gene_symbol,
                disease_name=target_disease.disease_name,
                drug_name=matched_drug,
                max_phase=trial.max_phase,
                enrollment=trial.enrollment,
                start_date=trial.start_date,
                ot_overall_score=target_disease.overall_score,
                ot_genetic_association_score=target_disease.datatype_scores.get("genetic_association"),
                ot_clinical_score=target_disease.datatype_scores.get("clinical"),
                ot_tractable_small_molecule=target_disease.tractable_small_molecule,
                ot_tractable_antibody=target_disease.tractable_antibody,
                ot_safety_liability_count=target_disease.safety_liability_count,
                chembl_activity_count=activity_count,
                chembl_best_pchembl=best_pchembl,
                chembl_matched_by_molecule_name=matched_by_name,
            )
        )
    return rows

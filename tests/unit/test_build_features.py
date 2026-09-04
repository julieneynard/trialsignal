from datetime import date

from trialsignal.data.schemas import (
    ChemblActivity,
    TargetDiseaseAssociation,
    TrialPhase,
    TrialRecord,
    TrialStatus,
)
from trialsignal.features.build_features import build_feature_table
from trialsignal.features.hypothesis import Hypothesis

HYPOTHESIS = Hypothesis(
    name="EGFR / osimertinib / NSCLC",
    gene_symbol="EGFR",
    ensembl_target_id="ENSG00000146648",
    chembl_target_id="CHEMBL203",
    drug_aliases=["osimertinib", "tagrisso", "azd9291"],
    ctgov_condition_query="non-small cell lung cancer",
)

TARGET_DISEASE = TargetDiseaseAssociation(
    target_id="ENSG00000146648",
    target_symbol="EGFR",
    disease_id="MONDO_0005233",
    disease_name="non-small cell lung carcinoma",
    overall_score=0.85,
    datatype_scores={"genetic_association": 0.74, "clinical": 0.99},
    tractable_small_molecule=True,
    tractable_antibody=True,
    safety_liability_count=21,
)

UNRELATED_DISEASE = TargetDiseaseAssociation(
    target_id="ENSG00000146648",
    target_symbol="EGFR",
    disease_id="EFO_9999999",
    disease_name="psoriatic arthritis",
    overall_score=0.1,
)


def _trial(
    nct_id: str,
    *,
    status: TrialStatus = TrialStatus.COMPLETED,
    interventions: list[str] | None = None,
    conditions: list[str] | None = None,
    why_stopped: str | None = None,
    phases: list[TrialPhase] | None = None,
) -> TrialRecord:
    return TrialRecord(
        nct_id=nct_id,
        title="test",
        status=status,
        interventions=interventions or ["Osimertinib"],
        conditions=conditions or ["Non-Small Cell Lung Cancer"],
        why_stopped=why_stopped,
        phases=phases or [TrialPhase.PHASE2],
        enrollment=100,
        start_date=date(2020, 1, 1),
    )


def test_matching_trial_produces_one_feature_row() -> None:
    trials = [_trial("NCT001")]
    rows = build_feature_table(HYPOTHESIS, trials, [TARGET_DISEASE], [])

    assert len(rows) == 1
    row = rows[0]
    assert row.nct_id == "NCT001"
    assert row.label == "success"
    assert row.gene_symbol == "EGFR"
    assert row.disease_name == "non-small cell lung carcinoma"
    assert row.ot_overall_score == 0.85
    assert row.ot_genetic_association_score == 0.74
    assert row.ot_tractable_small_molecule is True


def test_trial_testing_a_different_drug_is_dropped() -> None:
    trials = [_trial("NCT002", interventions=["Erlotinib"])]
    rows = build_feature_table(HYPOTHESIS, trials, [TARGET_DISEASE], [])
    assert rows == []


def test_excluded_label_trial_is_dropped() -> None:
    """A trial stopped for business reasons (EXCLUDED label, see labels.py)
    must never reach the training table — the whole point of the leakage
    guard is that it happens before features are ever built."""
    trials = [
        _trial(
            "NCT003",
            status=TrialStatus.TERMINATED,
            why_stopped="Terminated due to sponsor business decision",
        )
    ]
    rows = build_feature_table(HYPOTHESIS, trials, [TARGET_DISEASE], [])
    assert rows == []


def test_trial_with_unresolvable_condition_is_dropped() -> None:
    trials = [_trial("NCT004", conditions=["Some Unrelated Rare Disease"])]
    rows = build_feature_table(HYPOTHESIS, trials, [TARGET_DISEASE], [])
    assert rows == []


def test_condition_matches_correct_disease_among_multiple_candidates() -> None:
    trials = [_trial("NCT005")]
    rows = build_feature_table(HYPOTHESIS, trials, [UNRELATED_DISEASE, TARGET_DISEASE], [])

    assert len(rows) == 1
    assert rows[0].disease_name == "non-small cell lung carcinoma"


def test_chembl_activities_matched_by_molecule_name() -> None:
    activities = [
        ChemblActivity(
            molecule_chembl_id="CHEMBL1",
            pref_name="OSIMERTINIB",
            target_chembl_id="CHEMBL203",
            standard_type="IC50",
            pchembl_value=8.5,
        ),
        ChemblActivity(
            molecule_chembl_id="CHEMBL2",
            pref_name="SOME OTHER COMPOUND",
            target_chembl_id="CHEMBL203",
            standard_type="IC50",
            pchembl_value=9.9,  # higher, but not the drug in question — must not win
        ),
    ]
    rows = build_feature_table(HYPOTHESIS, [_trial("NCT006")], [TARGET_DISEASE], activities)

    assert len(rows) == 1
    assert rows[0].chembl_matched_by_molecule_name is True
    assert rows[0].chembl_activity_count == 1
    assert rows[0].chembl_best_pchembl == 8.5


def test_chembl_falls_back_to_target_level_when_no_molecule_match() -> None:
    activities = [
        ChemblActivity(
            molecule_chembl_id="CHEMBL2",
            pref_name="RESEARCH COMPOUND X",
            target_chembl_id="CHEMBL203",
            standard_type="IC50",
            pchembl_value=7.0,
        ),
    ]
    rows = build_feature_table(HYPOTHESIS, [_trial("NCT007")], [TARGET_DISEASE], activities)

    assert len(rows) == 1
    assert rows[0].chembl_matched_by_molecule_name is False
    assert rows[0].chembl_activity_count == 1
    assert rows[0].chembl_best_pchembl == 7.0


def test_no_activities_at_all_yields_zero_count_and_none_pchembl() -> None:
    rows = build_feature_table(HYPOTHESIS, [_trial("NCT008")], [TARGET_DISEASE], [])
    assert rows[0].chembl_activity_count == 0
    assert rows[0].chembl_best_pchembl is None


def test_multiple_matching_trials_all_produce_rows() -> None:
    trials = [_trial("NCT009"), _trial("NCT010", interventions=["Tagrisso"])]
    rows = build_feature_table(HYPOTHESIS, trials, [TARGET_DISEASE], [])
    assert {r.nct_id for r in rows} == {"NCT009", "NCT010"}

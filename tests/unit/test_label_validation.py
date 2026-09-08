from trialsignal.data.schemas import TrialRecord, TrialStatus
from trialsignal.features.label_validation import validate_labels


def _trial(
    nct_id: str, pvalue: float | None, analysis_type: str | None = "SUPERIORITY"
) -> TrialRecord:
    return TrialRecord(
        nct_id=nct_id,
        title="t",
        status=TrialStatus.COMPLETED,
        has_results=pvalue is not None,
        primary_pvalue=pvalue,
        primary_analysis_type=analysis_type if pvalue is not None else None,
    )


def test_agreement_when_significant_pvalue_matches_success_label() -> None:
    rows = [{"nct_id": "NCT1", "label": "success"}]
    trials = {"NCT1": _trial("NCT1", 0.001)}

    report = validate_labels(rows, trials)

    assert report.n_with_usable_signal == 1
    assert report.n_agree == 1
    assert report.agreement_rate == 1.0
    assert report.disagreements == []


def test_agreement_when_nonsignificant_pvalue_matches_failure_label() -> None:
    rows = [{"nct_id": "NCT1", "label": "failure"}]
    trials = {"NCT1": _trial("NCT1", 0.4)}

    report = validate_labels(rows, trials)

    assert report.n_agree == 1
    assert report.agreement_rate == 1.0


def test_disagreement_recorded_when_significant_but_labeled_failure() -> None:
    """A trial that hit statistical significance (p<0.05) on its primary
    endpoint but was labeled FAILURE by the registry-status heuristic (e.g.
    terminated for a business reason after already showing signal) — a real
    case worth surfacing, not silently discarding."""
    rows = [{"nct_id": "NCT1", "label": "failure"}]
    trials = {"NCT1": _trial("NCT1", 0.01)}

    report = validate_labels(rows, trials)

    assert report.n_agree == 0
    assert report.agreement_rate == 0.0
    assert len(report.disagreements) == 1
    assert report.disagreements[0].nct_id == "NCT1"
    assert report.disagreements[0].primary_pvalue == 0.01


def test_rows_without_usable_signal_are_excluded_from_rate_not_counted_as_disagreement() -> None:
    rows = [
        {"nct_id": "NCT1", "label": "success"},  # has a p-value
        {"nct_id": "NCT2", "label": "success"},  # no p-value at all
        {"nct_id": "NCT3", "label": "success"},  # not in trials_by_nct_id (fetch failed/skipped)
    ]
    trials = {
        "NCT1": _trial("NCT1", 0.001),
        "NCT2": _trial("NCT2", None),
    }

    report = validate_labels(rows, trials)

    assert report.n_rows_checked == 3
    assert report.n_with_usable_signal == 1
    assert report.n_agree == 1
    assert report.disagreements == []


def test_empty_input_returns_none_agreement_rate_not_zero_division() -> None:
    report = validate_labels([], {})
    assert report.n_with_usable_signal == 0
    assert report.agreement_rate is None

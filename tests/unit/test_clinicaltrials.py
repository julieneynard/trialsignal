"""parse_study() is the boundary between "whatever CT.gov's API returns
today" and the typed schema everything else relies on. It's tested against
fixture JSON, never against the live API — the live client (retry/pagination
behavior) is covered separately with a mocked transport, so CI never depends
on ClinicalTrials.gov being up.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from trialsignal.data.clinicaltrials import _extract_primary_result, _parse_pvalue, parse_study
from trialsignal.data.schemas import TrialPhase, TrialStatus

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_parses_terminated_study_with_why_stopped() -> None:
    record = parse_study(_load("ctgov_terminated_study.json"))

    assert record is not None
    assert record.nct_id == "NCT01234567"
    assert record.status == TrialStatus.TERMINATED
    assert record.phases == [TrialPhase.PHASE2]
    assert record.max_phase == TrialPhase.PHASE2
    assert record.conditions == ["Non-Small Cell Lung Cancer"]
    assert record.interventions == ["Drug X", "Placebo"]
    assert record.sponsor == "Example Pharma Inc."
    assert record.sponsor_class == "INDUSTRY"
    assert record.enrollment == 120
    assert record.why_stopped == "Terminated due to lack of efficacy at planned interim analysis"
    # "2019-03" (no day) normalizes to the 1st of the month, not a parse error.
    assert record.start_date == date(2019, 3, 1)
    assert record.primary_completion_date == date(2021, 7, 15)


def test_missing_nct_id_returns_none_instead_of_raising() -> None:
    """A single malformed study in a 300-record page must not blow up the
    whole pull — it's skipped and counted, not raised."""
    assert parse_study(_load("ctgov_malformed_study.json")) is None


def test_empty_payload_returns_none() -> None:
    assert parse_study({}) is None


def test_terminated_study_fixture_has_no_results() -> None:
    """The pre-existing terminated-study fixture has no resultsSection at
    all — has_results/primary_pvalue must default safely, not KeyError."""
    record = parse_study(_load("ctgov_terminated_study.json"))
    assert record is not None
    assert record.has_results is False
    assert record.primary_pvalue is None
    assert record.primary_analysis_type is None


def test_parses_primary_result_preferring_superiority_analysis() -> None:
    """Real fixture data from NCT02296125 (AURA3): the primary outcome has
    two analyses (global cohort = SUPERIORITY, China cohort = OTHER) — the
    SUPERIORITY one is the trial's main powered comparison and must win,
    and the SECONDARY outcome measure's p-value must be ignored entirely."""
    record = parse_study(_load("ctgov_results_section_aura3.json"))

    assert record is not None
    assert record.has_results is True
    assert record.primary_pvalue == pytest.approx(0.0001)
    assert record.primary_analysis_type == "SUPERIORITY"


def test_has_results_true_but_no_parseable_pvalue() -> None:
    """The common real case: hasResults=True but the primary outcome only
    reports raw measurements, no statistical analysis at all — this must
    not be silently treated as either a significant or non-significant
    result."""
    record = parse_study(_load("ctgov_results_section_no_pvalue.json"))

    assert record is not None
    assert record.has_results is True
    assert record.primary_pvalue is None
    assert record.primary_analysis_type is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<0.0001", 0.0001),
        (">0.9999", 0.9999),
        ("0.023", 0.023),
        ("  0.05  ", 0.05),
        ("=0.01", 0.01),
        (None, None),
        ("", None),
        ("NS", None),
        ("Not significant", None),
    ],
)
def test_parse_pvalue(raw: str | None, expected: float | None) -> None:
    result = _parse_pvalue(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_extract_primary_result_ignores_non_primary_measures() -> None:
    results_section = {
        "outcomeMeasuresModule": {
            "outcomeMeasures": [
                {"type": "SECONDARY", "analyses": [{"pValue": "0.001"}]},
                {"type": "OTHER", "analyses": [{"pValue": "0.002"}]},
            ]
        }
    }
    assert _extract_primary_result(results_section) == (None, None)


def test_extract_primary_result_falls_back_to_first_when_no_superiority() -> None:
    results_section = {
        "outcomeMeasuresModule": {
            "outcomeMeasures": [
                {
                    "type": "PRIMARY",
                    "analyses": [
                        {"pValue": "0.04", "nonInferiorityType": "NON_INFERIORITY"},
                        {"pValue": "0.08", "nonInferiorityType": "OTHER"},
                    ],
                }
            ]
        }
    }
    pvalue, analysis_type = _extract_primary_result(results_section)
    assert pvalue == pytest.approx(0.04)
    assert analysis_type == "NON_INFERIORITY"


def test_extract_primary_result_handles_missing_module() -> None:
    assert _extract_primary_result(None) == (None, None)
    assert _extract_primary_result({}) == (None, None)

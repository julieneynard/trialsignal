"""parse_study() is the boundary between "whatever CT.gov's API returns
today" and the typed schema everything else relies on. It's tested against
fixture JSON, never against the live API — the live client (retry/pagination
behavior) is covered separately with a mocked transport, so CI never depends
on ClinicalTrials.gov being up.
"""

import json
from datetime import date
from pathlib import Path

from trialsignal.data.clinicaltrials import parse_study
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

"""The label logic is the part of this project most likely to silently lie
to the model, so it gets the most exhaustive tests: every stop-reason
category, every trial status, and the ambiguous/missing-text edge cases that
real ClinicalTrials.gov data is full of.
"""

from datetime import date

import pytest

from trialsignal.data.schemas import TrialRecord, TrialStatus
from trialsignal.features.labels import (
    StopReason,
    TrialOutcome,
    build_trial_outcome_label,
    classify_stop_reason,
)


@pytest.mark.parametrize(
    ("why_stopped", "expected"),
    [
        ("Trial stopped due to serious adverse events in the treatment arm", StopReason.SAFETY),
        ("Halted for safety concerns identified by the DSMB", StopReason.SAFETY),
        ("Terminated due to lack of efficacy at interim analysis", StopReason.EFFICACY),
        ("Study did not meet its primary endpoint", StopReason.EFFICACY),
        ("Terminated due to slow accrual", StopReason.ENROLLMENT),
        ("Insufficient participant enrollment", StopReason.ENROLLMENT),
        (
            "Sponsor decision due to business/portfolio reprioritization",
            StopReason.BUSINESS_ADMINISTRATIVE,
        ),
        ("Study halted due to COVID-19 pandemic", StopReason.BUSINESS_ADMINISTRATIVE),
        (None, StopReason.AMBIGUOUS),
        ("", StopReason.AMBIGUOUS),
        ("   ", StopReason.AMBIGUOUS),
        ("Other", StopReason.AMBIGUOUS),
    ],
)
def test_classify_stop_reason(why_stopped: str | None, expected: StopReason) -> None:
    assert classify_stop_reason(why_stopped) == expected


def test_safety_language_wins_over_administrative_when_both_present() -> None:
    # Mixed language should resolve to the clinically meaningful category,
    # not the administrative one, since that's the signal worth training on.
    text = "Terminated for safety concerns; also cites funding constraints"
    assert classify_stop_reason(text) == StopReason.SAFETY


def _trial(status: TrialStatus, why_stopped: str | None = None) -> TrialRecord:
    return TrialRecord(
        nct_id="NCT00000000",
        title="test trial",
        status=status,
        start_date=date(2020, 1, 1),
    ).model_copy(update={"why_stopped": why_stopped})


def test_completed_trial_is_success() -> None:
    assert build_trial_outcome_label(_trial(TrialStatus.COMPLETED)) == TrialOutcome.SUCCESS


def test_terminated_for_efficacy_is_failure() -> None:
    trial = _trial(TrialStatus.TERMINATED, "Terminated due to lack of efficacy")
    assert build_trial_outcome_label(trial) == TrialOutcome.FAILURE


def test_terminated_for_safety_is_failure() -> None:
    trial = _trial(TrialStatus.TERMINATED, "Stopped due to unacceptable toxicity")
    assert build_trial_outcome_label(trial) == TrialOutcome.FAILURE


def test_terminated_for_business_reason_is_excluded_not_failure() -> None:
    """This is the leakage guard: a funding-driven termination must NOT be
    scored as a drug failure just because the trial stopped early."""
    trial = _trial(TrialStatus.TERMINATED, "Terminated for business reasons")
    assert build_trial_outcome_label(trial) == TrialOutcome.EXCLUDED


def test_terminated_with_no_stated_reason_is_excluded() -> None:
    trial = _trial(TrialStatus.TERMINATED, None)
    assert build_trial_outcome_label(trial) == TrialOutcome.EXCLUDED


def test_withdrawn_for_enrollment_is_excluded() -> None:
    trial = _trial(TrialStatus.WITHDRAWN, "Withdrawn due to inability to recruit participants")
    assert build_trial_outcome_label(trial) == TrialOutcome.EXCLUDED


@pytest.mark.parametrize(
    "status",
    [
        TrialStatus.RECRUITING,
        TrialStatus.ACTIVE_NOT_RECRUITING,
        TrialStatus.NOT_YET_RECRUITING,
        TrialStatus.UNKNOWN,
        TrialStatus.ENROLLING_BY_INVITATION,
    ],
)
def test_in_progress_trials_are_excluded(status: TrialStatus) -> None:
    assert build_trial_outcome_label(_trial(status)) == TrialOutcome.EXCLUDED

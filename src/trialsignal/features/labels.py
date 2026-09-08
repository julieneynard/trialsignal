"""Label construction: turning a raw trial status into a training-usable
outcome.

The naive approach — status == COMPLETED is a success, everything else is a
failure — is wrong and will silently poison the model. A large fraction of
TERMINATED/WITHDRAWN trials stop for reasons that have nothing to do with the
drug (funding lapses, enrollment failure, COVID, sponsor restructuring). If
those get labeled FAILURE, the model partly learns "which sponsors ran out of
money" instead of "which drugs don't work" — a classic leakage/confounding
trap in trial-outcome modeling, and the single biggest way this kind of
project quietly produces a meaningless number.

This module classifies `why_stopped` free text into a reason category and
only assigns a SUCCESS/FAILURE label when the stated reason is actually about
efficacy or safety. Everything else is EXCLUDED rather than guessed — a
smaller, honest training set beats a larger, contaminated one.

`build_trial_outcome_label` is the registry-status proxy described above.
It's a proxy because CT.gov's status field can't tell "ran to completion"
apart from "ran to completion AND met its primary endpoint" — confirmed
empirically, not just argued: `trialsignal validate-labels` found the proxy
disagrees with the trial's own reported primary-endpoint p-value on ~1/3 of
the rows where a real result exists to check against (see
docs/LIMITATIONS.md item 1). `resolve_trial_label` is what the feature
pipeline actually calls: it prefers that real result whenever
`TrialRecord.primary_pvalue` is available (including for trials the
status-based proxy would otherwise have excluded as ambiguous — a trial
can be in a messy administrative state and still have posted a clear
result), and falls back to the proxy only when no real result exists.
"""

from __future__ import annotations

import re
from enum import StrEnum

from trialsignal.data.schemas import TrialRecord, TrialStatus


class StopReason(StrEnum):
    EFFICACY = "efficacy"
    SAFETY = "safety"
    ENROLLMENT = "enrollment"
    BUSINESS_ADMINISTRATIVE = "business_administrative"
    AMBIGUOUS = "ambiguous"


class TrialOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    EXCLUDED = "excluded"


# Order matters: checked in sequence, first match wins. Safety and efficacy
# language is checked before the broad administrative bucket because stop
# reasons often mix genres ("terminated due to lack of efficacy and funding")
# and the clinically meaningful part should take priority.
_PATTERNS: list[tuple[StopReason, re.Pattern[str]]] = [
    (
        StopReason.SAFETY,
        re.compile(
            r"\b(safety|adverse event|toxicit\w*|serious adverse|dsmb|"
            r"data safety monitoring|risk[- ]benefit)\b",
            re.IGNORECASE,
        ),
    ),
    (
        StopReason.EFFICACY,
        re.compile(
            r"\b(lack of efficacy|futility|efficacy|failed to (meet|show)|"
            r"did not (meet|show)|no (clinical )?benefit|interim analysis)\b",
            re.IGNORECASE,
        ),
    ),
    (
        StopReason.ENROLLMENT,
        re.compile(
            r"\b(enroll(ment|ing)?|accrual|recruit(ment|ing)?|slow accrual)\b",
            re.IGNORECASE,
        ),
    ),
    (
        StopReason.BUSINESS_ADMINISTRATIVE,
        re.compile(
            r"\b(business|funding|financ|sponsor decision|strategic|"
            r"portfolio|restructur|covid|pandemic|manufactur|supply|"
            r"study design|protocol amendment|administrative)\b",
            re.IGNORECASE,
        ),
    ),
]

_TERMINAL_NON_COMPLETED = {TrialStatus.TERMINATED, TrialStatus.WITHDRAWN, TrialStatus.SUSPENDED}

# Standard two-sided significance threshold. Matches trialsignal.features.
# label_validation.SIGNIFICANCE_THRESHOLD — kept here as the canonical
# definition since this is now where it drives an actual labeling decision,
# not just an audit; label_validation imports it from here.
RESULTS_SIGNIFICANCE_THRESHOLD = 0.05


def classify_stop_reason(why_stopped: str | None) -> StopReason:
    """Classify a `why_stopped` free-text field into a reason category.

    Missing text is AMBIGUOUS by construction — a trial with no stated
    reason cannot be attributed to efficacy or safety, so it must not be
    scored as a failure just because it stopped early.
    """
    if not why_stopped or not why_stopped.strip():
        return StopReason.AMBIGUOUS
    for reason, pattern in _PATTERNS:
        if pattern.search(why_stopped):
            return reason
    return StopReason.AMBIGUOUS


def build_trial_outcome_label(trial: TrialRecord) -> TrialOutcome:
    """Map a trial to a training label.

    Known limitation (see docs/LIMITATIONS.md): COMPLETED is treated as a
    SUCCESS proxy. A trial can run to completion and still miss its primary
    endpoint — CT.gov's registry status alone can't distinguish that. Doing
    so properly requires the trial *results* section (p-values / effect
    sizes), which is a documented v2 extension, not part of this label.
    """
    if trial.status == TrialStatus.COMPLETED:
        return TrialOutcome.SUCCESS

    if trial.status in _TERMINAL_NON_COMPLETED:
        reason = classify_stop_reason(trial.why_stopped)
        if reason in (StopReason.EFFICACY, StopReason.SAFETY):
            return TrialOutcome.FAILURE
        return TrialOutcome.EXCLUDED

    # RECRUITING / ACTIVE_NOT_RECRUITING / NOT_YET_RECRUITING / UNKNOWN / ...
    # — outcome hasn't resolved yet, can't be used as a label.
    return TrialOutcome.EXCLUDED


def resolve_trial_label(trial: TrialRecord) -> tuple[TrialOutcome, bool]:
    """The label the feature pipeline actually uses. Returns (label,
    is_results_based) — the second value is surfaced in
    `TrialFeatureRow.label_source` so a reviewer can see, per row, which
    ~9% were graded on the trial's own statistics versus the ~91% still
    relying on the registry-status proxy; it is not hidden inside an
    aggregate label.

    A real primary-endpoint p-value wins whenever it's available, including
    for trials `build_trial_outcome_label` would have EXCLUDED (e.g. status
    ACTIVE_NOT_RECRUITING with a posted interim result, or TERMINATED for
    an ambiguous reason that nonetheless reported a clear primary-endpoint
    analysis) — that data point is real signal about the drug regardless of
    the trial's current administrative status, and dropping it would throw
    away exactly the rows this fix is meant to add.
    """
    if trial.primary_pvalue is not None:
        significant = trial.primary_pvalue < RESULTS_SIGNIFICANCE_THRESHOLD
        return (TrialOutcome.SUCCESS if significant else TrialOutcome.FAILURE), True
    return build_trial_outcome_label(trial), False

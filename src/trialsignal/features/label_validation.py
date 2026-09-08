"""Cross-checks the registry-status label (labels.py) against the
results-section primary-endpoint p-value (clinicaltrials.py), where both
are available.

This is deliberately a validation tool, not a replacement label pipeline —
see clinicaltrials.py's module docstring for why: a usable primary p-value
exists for only ~5-10% of trials, far too sparse to train on directly. What
it can do is answer a real question about the *existing* status-based
label's quality: when we can independently check (via a real statistical
result) whether a trial actually met its primary endpoint, does our proxy
label (COMPLETED -> success) agree?
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trialsignal.data.schemas import TrialRecord

SIGNIFICANCE_THRESHOLD = 0.05


@dataclass
class Disagreement:
    nct_id: str
    existing_label: str
    primary_pvalue: float
    primary_analysis_type: str | None


@dataclass
class LabelValidationReport:
    n_rows_checked: int
    n_with_usable_signal: int
    n_agree: int
    disagreements: list[Disagreement] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float | None:
        if self.n_with_usable_signal == 0:
            return None
        return self.n_agree / self.n_with_usable_signal


def validate_labels(
    existing_rows: list[dict[str, str]], trials_by_nct_id: dict[str, TrialRecord]
) -> LabelValidationReport:
    """`existing_rows` are dicts with at least `nct_id` and `label`
    ("success"/"failure") — i.e. rows from a `build-features` CSV output,
    read directly via `csv.DictReader` (no need to round-trip through
    TrialFeatureRow for this). `trials_by_nct_id` should come from
    `ClinicalTrialsClient.fetch_by_nct_ids` on those same NCT IDs, freshly
    fetched so the results-section fields are populated.

    A row contributes to `agreement_rate` only when the fresh fetch found a
    usable primary p-value — rows without one are counted in
    `n_rows_checked` but silently excluded from the rate, same
    drop-don't-guess philosophy as the rest of this pipeline.
    """
    n_with_signal = 0
    n_agree = 0
    disagreements: list[Disagreement] = []

    for row in existing_rows:
        trial = trials_by_nct_id.get(row["nct_id"])
        if trial is None or trial.primary_pvalue is None:
            continue

        n_with_signal += 1
        statistically_significant = trial.primary_pvalue < SIGNIFICANCE_THRESHOLD
        existing_success = row["label"] == "success"

        if statistically_significant == existing_success:
            n_agree += 1
        else:
            disagreements.append(
                Disagreement(
                    nct_id=row["nct_id"],
                    existing_label=row["label"],
                    primary_pvalue=trial.primary_pvalue,
                    primary_analysis_type=trial.primary_analysis_type,
                )
            )

    return LabelValidationReport(
        n_rows_checked=len(existing_rows),
        n_with_usable_signal=n_with_signal,
        n_agree=n_agree,
        disagreements=disagreements,
    )

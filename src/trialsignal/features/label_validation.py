"""Cross-checks a label against the results-section primary-endpoint
p-value (clinicaltrials.py), where both are available.

This is the tool that found the finding documented in docs/LIMITATIONS.md
item 1: run against the registry-status proxy label
(`labels.build_trial_outcome_label`) on the full dataset, it showed only
65.8% agreement on the 38/405 rows with a real result to check against, all
disagreements in the same direction. `labels.resolve_trial_label` now acts
on that finding directly — the feature pipeline (`build_features.py`)
prefers the real result over the proxy whenever one exists, so a freshly
built feature table's `label` column is already the corrected value for
those rows. This tool remains useful for auditing any label source
(including re-verifying the fix, or checking an older/external dataset)
against ground truth, independent of whichever labeling function produced
it — it takes plain label strings, not a specific label function's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trialsignal.data.schemas import TrialRecord
from trialsignal.features.labels import RESULTS_SIGNIFICANCE_THRESHOLD as SIGNIFICANCE_THRESHOLD


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

"""Training-pipeline tests use small synthetic feature tables — CSV files
shaped exactly like `trialsignal build-features` output, written to
tmp_path — rather than fixtures pulled from real hypotheses, since the goal
here is to exercise the pipeline mechanics (type coercion, temporal split,
degenerate-split handling) deterministically, not to validate real-world
predictive performance (see docs/LIMITATIONS.md for that honesty).
"""

import csv
from pathlib import Path

import pandas as pd
import pytest

from trialsignal.models.train import (
    FEATURE_COLUMNS,
    InsufficientClassDiversityError,
    cross_validate_lightgbm,
    evaluate,
    load_feature_table,
    temporal_split,
    train_and_evaluate,
)

_FIELDNAMES = [
    "nct_id",
    "label",
    "gene_symbol",
    "disease_name",
    "drug_name",
    "max_phase",
    "enrollment",
    "start_date",
    "ot_overall_score",
    "ot_genetic_association_score",
    "ot_clinical_score",
    "ot_tractable_small_molecule",
    "ot_tractable_antibody",
    "ot_safety_liability_count",
    "chembl_activity_count",
    "chembl_best_pchembl",
    "chembl_matched_by_molecule_name",
]


def _row(nct_id: str, label: str, start_date: str, **overrides: object) -> dict:
    base = {
        "nct_id": nct_id,
        "label": label,
        "gene_symbol": "EGFR",
        "disease_name": "non-small cell lung carcinoma",
        "drug_name": "osimertinib",
        "max_phase": "PHASE2",
        "enrollment": "100",
        "start_date": start_date,
        "ot_overall_score": "0.85",
        "ot_genetic_association_score": "0.74",
        "ot_clinical_score": "0.99",
        "ot_tractable_small_molecule": "True",
        "ot_tractable_antibody": "True",
        "ot_safety_liability_count": "21",
        "chembl_activity_count": "300",
        "chembl_best_pchembl": "9.7",
        "chembl_matched_by_molecule_name": "False",
    }
    base.update(overrides)
    return base


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_load_feature_table_coerces_booleans_and_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_csv(
        path,
        [
            _row("NCT1", "success", "2019-01-01"),
            _row(
                "NCT2",
                "failure",
                "2020-01-01",
                ot_genetic_association_score="",  # missing, as a real empty optional field would be
                chembl_matched_by_molecule_name="True",
            ),
        ],
    )
    df = load_feature_table([path])

    assert df["label_binary"].tolist() == [1, 0]
    assert df["max_phase_ordinal"].tolist() == [2, 2]
    assert df.loc[0, "chembl_matched_by_molecule_name"] == 0.0
    assert df.loc[1, "chembl_matched_by_molecule_name"] == 1.0
    assert pd.isna(df.loc[1, "ot_genetic_association_score"])
    assert df["start_date"].dt.year.tolist() == [2019, 2020]


def test_load_feature_table_concatenates_multiple_files(tmp_path: Path) -> None:
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    _write_csv(path_a, [_row("NCT1", "success", "2019-01-01")])
    _write_csv(path_b, [_row("NCT2", "failure", "2020-01-01", gene_symbol="ABL1")])

    df = load_feature_table([path_a, path_b])
    assert len(df) == 2
    assert set(df["gene_symbol"]) == {"EGFR", "ABL1"}


def test_temporal_split_partitions_by_cutoff_date(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_csv(
        path,
        [
            _row("NCT_early", "success", "2018-01-01"),
            _row("NCT_late", "failure", "2022-01-01"),
        ],
    )
    df = load_feature_table([path])
    train, test = temporal_split(df, cutoff="2021-01-01")

    assert train["nct_id"].tolist() == ["NCT_early"]
    assert test["nct_id"].tolist() == ["NCT_late"]


def test_temporal_split_puts_unknown_dates_in_train(tmp_path: Path) -> None:
    """A trial can't leak into the held-out evaluation just because its
    start_date failed to parse — see temporal_split's docstring."""
    path = tmp_path / "features.csv"
    _write_csv(path, [_row("NCT_unknown", "success", start_date="")])
    df = load_feature_table([path])
    train, test = temporal_split(df, cutoff="2021-01-01")

    assert train["nct_id"].tolist() == ["NCT_unknown"]
    assert test.empty


def _synthetic_table(tmp_path: Path, n_per_class: int = 8) -> Path:
    rows = []
    for i in range(n_per_class):
        rows.append(
            _row(
                f"NCT_S{i}",
                "success",
                f"201{i % 9}-01-01",
                ot_overall_score=str(0.6 + 0.03 * i),
                chembl_best_pchembl=str(7.0 + 0.1 * i),
            )
        )
        rows.append(
            _row(
                f"NCT_F{i}",
                "failure",
                f"201{i % 9}-06-01",
                ot_overall_score=str(0.2 + 0.02 * i),
                chembl_best_pchembl=str(4.0 + 0.1 * i),
            )
        )
    path = tmp_path / "synthetic_features.csv"
    _write_csv(path, rows)
    return path


def test_evaluate_reports_none_for_single_class_test_split(tmp_path: Path) -> None:
    path = _synthetic_table(tmp_path)
    df = load_feature_table([path])
    # cutoff far in the future -> everything lands in train, test split is empty
    baseline, _, report = train_and_evaluate(df, cutoff="2099-01-01")
    # With an empty test split, train_and_evaluate falls back to evaluating
    # on the training data itself (both classes present) rather than crashing.
    assert report.n_test == 0
    assert report.baseline_eval.roc_auc is not None


def test_train_and_evaluate_end_to_end_produces_valid_report(tmp_path: Path) -> None:
    path = _synthetic_table(tmp_path, n_per_class=10)
    df = load_feature_table([path])

    baseline, lightgbm_model, report = train_and_evaluate(df, cutoff="2015-01-01")

    assert report.n_total == 20
    assert report.n_train + report.n_test == report.n_total
    assert report.n_train > 0
    for result in (report.baseline_eval, report.lightgbm_eval):
        assert 0 <= result.n_test_success + result.n_test_failure == result.n_test
        if result.roc_auc is not None:
            assert 0.0 <= result.roc_auc <= 1.0
    assert len(report.top_shap_features) > 0
    assert all(name in FEATURE_COLUMNS for name, _ in report.top_shap_features)

    # Both pipelines must be usable for prediction on held-in data.
    x = df[FEATURE_COLUMNS]
    proba = lightgbm_model.predict_proba(x)
    assert proba.shape == (20, 2)


def test_missing_feature_column_raises_keyerror(tmp_path: Path) -> None:
    """A malformed feature table (wrong CSV shape) should fail loudly at
    the point of use, not silently train on a subset of features."""
    path = tmp_path / "bad.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write("nct_id,label\nNCT1,success\n")
    df = pd.read_csv(path)
    df["label_binary"] = 1
    with pytest.raises(KeyError):
        evaluate(_dummy_pipeline(), df)


def _time_clustered_table(tmp_path: Path) -> Path:
    """Mirrors the real dataset's actual shape (EGFR/osimertinib +
    ABL1/imatinib, see docs/LIMITATIONS.md): plenty of successes spread
    across many years, but every failure clustered in one recent window —
    so no single cutoff date can put both classes on both sides of the split."""
    rows = [_row(f"NCT_old_S{i}", "success", f"20{10 + i % 10:02d}-01-01") for i in range(10)]
    rows += [_row(f"NCT_new_F{i}", "failure", "2022-06-01") for i in range(3)]
    path = tmp_path / "time_clustered.csv"
    _write_csv(path, rows)
    return path


def test_train_and_evaluate_raises_clear_error_when_train_split_is_single_class(
    tmp_path: Path,
) -> None:
    path = _time_clustered_table(tmp_path)
    df = load_feature_table([path])

    # Every failure is dated 2022-06-01; any cutoff before that leaves the
    # training split all-success. This must fail with our own clear error,
    # not sklearn's opaque "needs samples of at least 2 classes" ValueError.
    with pytest.raises(InsufficientClassDiversityError, match="only one"):
        train_and_evaluate(df, cutoff="2021-01-01")


def test_cross_validate_lightgbm_handles_time_clustered_labels(tmp_path: Path) -> None:
    """The exact dataset shape that breaks temporal_split must still
    produce a real evaluation via the documented CV fallback."""
    path = _time_clustered_table(tmp_path)
    df = load_feature_table([path])

    model, report = cross_validate_lightgbm(df)

    assert report.n_total == 13
    assert report.lightgbm_eval.roc_auc is not None
    assert "stratified CV" in report.lightgbm_eval.note
    assert "temporal" not in report.cutoff or "no temporal cutoff" in report.cutoff
    proba = model.predict_proba(df[FEATURE_COLUMNS])
    assert proba.shape == (13, 2)


def test_cross_validate_lightgbm_raises_when_minority_class_too_small(tmp_path: Path) -> None:
    rows = [_row(f"NCT_S{i}", "success", "2018-01-01") for i in range(10)]
    rows += [_row("NCT_F0", "failure", "2018-01-01")]  # only 1 example of the minority class
    path = tmp_path / "tiny_minority.csv"
    _write_csv(path, rows)
    df = load_feature_table([path])

    with pytest.raises(InsufficientClassDiversityError, match="at least 2"):
        cross_validate_lightgbm(df)


def _dummy_pipeline():
    from sklearn.linear_model import LogisticRegression

    from trialsignal.models.train import _build_pipeline

    return _build_pipeline(LogisticRegression())

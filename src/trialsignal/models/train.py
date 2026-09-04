"""Training pipeline: temporal-split baseline vs LightGBM on the
TrialFeatureRow table(s) produced by `trialsignal build-features`.

Reads the feature table from CSV rather than trusting pandas' automatic type
inference on it. That matters here specifically: `csv.DictWriter` (used by
the CLI) round-trips booleans as the strings "True"/"False" and None as an
empty string — and on a feature table this small, an optional numeric column
can easily be *entirely* empty for one run, which would make
`pd.read_csv`'s inference call it `object` or `float64` unpredictably rather
than the nullable numeric/boolean types the model actually needs. Every
column is coerced explicitly instead.

Given the real dataset size this pipeline runs against in v1 (tens of rows,
not thousands — see docs/LIMITATIONS.md), nested cross-validation or
hyperparameter search would be tuning noise, not signal. Both models use
library defaults; that tradeoff is documented, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: list[str] = [
    "max_phase_ordinal",
    "enrollment",
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

# NA is deliberately -1, distinct from "phase unknown" (also -1 today) —
# CT.gov's own NA phase means "not applicable" (e.g. observational studies),
# which is a real, different category, but v1 doesn't have enough rows to
# support a fourth ordinal bucket for it, documented as a simplification.
_PHASE_ORDINAL: dict[str, int] = {
    "EARLY_PHASE1": 0,
    "PHASE1": 1,
    "PHASE2": 2,
    "PHASE3": 3,
    "PHASE4": 4,
    "NA": -1,
}
_BOOL_COLUMNS = [
    "ot_tractable_small_molecule",
    "ot_tractable_antibody",
    "chembl_matched_by_molecule_name",
]
_NUMERIC_COLUMNS = [
    "enrollment",
    "ot_overall_score",
    "ot_genetic_association_score",
    "ot_clinical_score",
    "ot_safety_liability_count",
    "chembl_activity_count",
    "chembl_best_pchembl",
]


def load_feature_table(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate one or more `build-features` CSV outputs,
    coercing every column to its real type explicitly."""
    frames = [pd.read_csv(p, dtype=str) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return _coerce_types(df)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_binary"] = (df["label"] == "success").astype(int)
    df["max_phase_ordinal"] = df["max_phase"].map(_PHASE_ORDINAL).fillna(-1).astype(int)
    for col in _BOOL_COLUMNS:
        df[col] = df[col].map({"True": 1.0, "False": 0.0})
    for col in _NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    return df


def temporal_split(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by `start_date`, not randomly — a trial's outcome must never be
    predictable from a later trial testing the same drug (see METHODS.md).
    Rows with an unknown start_date go to train (conservative: they can't
    leak into the held-out evaluation)."""
    cutoff_ts = pd.Timestamp(cutoff)
    train = df[(df["start_date"].isna()) | (df["start_date"] < cutoff_ts)]
    test = df[df["start_date"] >= cutoff_ts]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _build_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("estimator", estimator),
        ]
    )


@dataclass
class EvalResult:
    n_test: int
    n_test_success: int
    n_test_failure: int
    roc_auc: float | None
    pr_auc: float | None
    brier_score: float
    note: str = ""


def evaluate(pipeline: Pipeline, test_df: pd.DataFrame) -> EvalResult:
    """Never crash on a degenerate test split — a single-class test set
    (very likely at this dataset's size) makes ROC-AUC/PR-AUC undefined, and
    that's reported as `None` with an explanatory note rather than raising,
    since a training run failing outright on a small, real, expected data
    shape would be a worse outcome than an honestly incomplete metric."""
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label_binary"]
    proba = pipeline.predict_proba(x_test)[:, 1]

    n_success = int(y_test.sum())
    n_failure = len(y_test) - n_success
    brier = float(brier_score_loss(y_test, proba))

    if n_success == 0 or n_failure == 0:
        return EvalResult(
            n_test=len(y_test),
            n_test_success=n_success,
            n_test_failure=n_failure,
            roc_auc=None,
            pr_auc=None,
            brier_score=brier,
            note="Test split has only one class present — ROC-AUC/PR-AUC undefined.",
        )

    return EvalResult(
        n_test=len(y_test),
        n_test_success=n_success,
        n_test_failure=n_failure,
        roc_auc=float(roc_auc_score(y_test, proba)),
        pr_auc=float(average_precision_score(y_test, proba)),
        brier_score=brier,
    )


class InsufficientClassDiversityError(Exception):
    """Raised when a split doesn't contain at least one example of each
    class. sklearn's own error here is an opaque solver-internals message
    ("needs samples of at least 2 classes") that doesn't tell the caller
    *why* — usually because the dataset's positive/negative examples are
    clustered in time (see docs/LIMITATIONS.md) and no cutoff date can
    separate them into a valid train/test split. `cross_validate_lightgbm`
    is the documented fallback for exactly this case."""


@dataclass
class TrainingReport:
    n_total: int
    n_train: int
    n_test: int
    cutoff: str
    baseline_eval: EvalResult
    lightgbm_eval: EvalResult
    top_shap_features: list[tuple[str, float]] = field(default_factory=list)


def train_and_evaluate(df: pd.DataFrame, cutoff: str) -> tuple[Pipeline, Pipeline, TrainingReport]:
    train_df, test_df = temporal_split(df, cutoff)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label_binary"]

    if y_train.nunique() < 2:
        counts = y_train.value_counts().to_dict()
        raise InsufficientClassDiversityError(
            f"Training split (start_date < {cutoff}, n={len(train_df)}) contains only one "
            f"class (label counts: {counts}). No single cutoff can fix this if successes and "
            f"failures are clustered in different time ranges — try cross_validate_lightgbm() "
            f"instead, or `trialsignal train --eval-mode cv`."
        )

    baseline = _build_pipeline(LogisticRegression(max_iter=1000))
    baseline.fit(x_train, y_train)

    import lightgbm as lgb

    lightgbm_model = _build_pipeline(
        lgb.LGBMClassifier(n_estimators=100, max_depth=3, min_child_samples=1, verbosity=-1)
    )
    lightgbm_model.fit(x_train, y_train)

    eval_df = test_df if len(test_df) else train_df
    baseline_eval = evaluate(baseline, eval_df)
    lightgbm_eval = evaluate(lightgbm_model, eval_df)

    top_shap = _top_shap_features(lightgbm_model, x_train)

    report = TrainingReport(
        n_total=len(df),
        n_train=len(train_df),
        n_test=len(test_df),
        cutoff=cutoff,
        baseline_eval=baseline_eval,
        lightgbm_eval=lightgbm_eval,
        top_shap_features=top_shap,
    )
    return baseline, lightgbm_model, report


def _cross_val_eval(estimator: object, x: pd.DataFrame, y: pd.Series, n_splits: int) -> EvalResult:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipeline = _build_pipeline(estimator)
    proba = cross_val_predict(pipeline, x, y, cv=skf, method="predict_proba")[:, 1]
    return EvalResult(
        n_test=len(y),
        n_test_success=int(y.sum()),
        n_test_failure=len(y) - int(y.sum()),
        roc_auc=float(roc_auc_score(y, proba)),
        pr_auc=float(average_precision_score(y, proba)),
        brier_score=float(brier_score_loss(y, proba)),
        note=(
            f"{n_splits}-fold stratified CV, out-of-fold predictions — NOT a temporal "
            "holdout (folds are random, not time-ordered); see docs/LIMITATIONS.md."
        ),
    )


def cross_validate_lightgbm(
    df: pd.DataFrame, n_splits: int | None = None
) -> tuple[Pipeline, TrainingReport]:
    """Stratified k-fold evaluation — the documented fallback for a dataset
    too small/imbalanced for `train_and_evaluate`'s temporal split to
    produce two-class train AND test partitions (see
    InsufficientClassDiversityError and docs/LIMITATIONS.md). This is a
    deliberate methodological downgrade: cross-validation folds are random,
    so a later trial CAN inform an earlier one across folds, reintroducing
    exactly the leakage risk `temporal_split` exists to prevent. Use this
    only to get an evaluation number out of a dataset this small; prefer
    `train_and_evaluate` once there's enough data per class per time
    period."""
    import lightgbm as lgb

    x = df[FEATURE_COLUMNS]
    y = df["label_binary"]
    minority_count = int(y.value_counts().min())
    if minority_count < 2:
        raise InsufficientClassDiversityError(
            f"Minority class has only {minority_count} example(s) in the full dataset "
            f"(n={len(df)}) — cross-validation needs at least 2 of each class."
        )
    if n_splits is None:
        n_splits = max(2, min(5, minority_count))

    baseline_eval = _cross_val_eval(LogisticRegression(max_iter=1000), x, y, n_splits)
    lightgbm_eval = _cross_val_eval(
        lgb.LGBMClassifier(n_estimators=100, max_depth=3, min_child_samples=1, verbosity=-1),
        x,
        y,
        n_splits,
    )

    # The saved/returned model is fit on the FULL dataset (no holdout left
    # over once every row has been used as a CV test fold) — its SHAP
    # values describe what it learned, not an unbiased held-out evaluation.
    final_model = _build_pipeline(
        lgb.LGBMClassifier(n_estimators=100, max_depth=3, min_child_samples=1, verbosity=-1)
    )
    final_model.fit(x, y)
    top_shap = _top_shap_features(final_model, x)

    report = TrainingReport(
        n_total=len(df),
        n_train=len(df),
        n_test=len(df),
        cutoff=f"(no temporal cutoff — {n_splits}-fold stratified CV)",
        baseline_eval=baseline_eval,
        lightgbm_eval=lightgbm_eval,
        top_shap_features=top_shap,
    )
    return final_model, report


def _top_shap_features(
    pipeline: Pipeline, x_train: pd.DataFrame, top_n: int = 5
) -> list[tuple[str, float]]:
    import shap

    imputed = pipeline.named_steps["impute"].transform(x_train)
    scaled = pipeline.named_steps["scale"].transform(imputed)
    explainer = shap.TreeExplainer(pipeline.named_steps["estimator"])
    shap_values = explainer.shap_values(scaled)
    # Verified directly against the installed shap/lightgbm versions: a
    # binary LGBMClassifier returns a single (n_samples, n_features) ndarray
    # here, not a per-class list — despite shap's own UserWarning text
    # suggesting otherwise. The isinstance branch is defensive for other
    # shap/lightgbm version combinations that do return a per-class list.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]  # positive-class SHAP values
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(FEATURE_COLUMNS, mean_abs, strict=True), key=lambda kv: kv[1], reverse=True)
    return [(name, float(value)) for name, value in ranked[:top_n]]


def save_model_bundle(pipeline: Pipeline, output: Path, model_version: str) -> None:
    import joblib

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "feature_names": FEATURE_COLUMNS,
            "model_version": model_version,
            "trained_at": datetime.now(UTC).isoformat(),
        },
        output,
    )

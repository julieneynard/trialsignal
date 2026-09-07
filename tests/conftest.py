"""Shared fixtures. `trained_model_path` trains a real (tiny, synthetic)
LightGBM pipeline once per test session and saves it via
`save_model_bundle` — several test modules need an actual model artifact on
disk (API tests especially), and training one per test would make the
suite noticeably slower for no extra coverage."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tests.unit.test_train import _synthetic_table, load_feature_table
    from trialsignal.models.train import cross_validate_lightgbm, save_model_bundle

    tmp_dir = tmp_path_factory.mktemp("model")
    csv_path = _synthetic_table(tmp_dir, n_per_class=10)
    df = load_feature_table([csv_path])
    model, _ = cross_validate_lightgbm(df)

    model_path = tmp_dir / "trialsignal_model.joblib"
    save_model_bundle(model, model_path, model_version="test-0.0.1")
    return model_path

from pathlib import Path

import httpx
import respx

from trialsignal.models.registry import load_model

MODEL_URL = "https://example.com/releases/trialsignal_model.joblib"


def test_load_model_returns_none_when_url_unset_and_path_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TRIALSIGNAL_MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    monkeypatch.delenv("TRIALSIGNAL_MODEL_URL", raising=False)

    assert load_model() is None


@respx.mock
def test_load_model_downloads_from_url_when_local_path_missing(
    tmp_path: Path, monkeypatch, trained_model_path: Path
) -> None:
    target_path = tmp_path / "downloaded" / "trialsignal_model.joblib"
    monkeypatch.setenv("TRIALSIGNAL_MODEL_PATH", str(target_path))
    monkeypatch.setenv("TRIALSIGNAL_MODEL_URL", MODEL_URL)
    respx.get(MODEL_URL).mock(
        return_value=httpx.Response(200, content=trained_model_path.read_bytes())
    )

    bundle = load_model()

    assert bundle is not None
    assert bundle.model_version == "test-0.0.1"
    assert target_path.exists()  # downloaded artifact is cached on disk


@respx.mock
def test_load_model_returns_none_when_download_fails(tmp_path: Path, monkeypatch) -> None:
    target_path = tmp_path / "trialsignal_model.joblib"
    monkeypatch.setenv("TRIALSIGNAL_MODEL_PATH", str(target_path))
    monkeypatch.setenv("TRIALSIGNAL_MODEL_URL", MODEL_URL)
    respx.get(MODEL_URL).mock(return_value=httpx.Response(404))

    assert load_model() is None
    assert not target_path.exists()


def test_load_model_prefers_existing_local_file_over_url(
    tmp_path: Path, monkeypatch, trained_model_path: Path
) -> None:
    monkeypatch.setenv("TRIALSIGNAL_MODEL_PATH", str(trained_model_path))
    # No respx.mock active here: a real request to this URL would error the
    # test, proving the local file short-circuits the download entirely.
    monkeypatch.setenv("TRIALSIGNAL_MODEL_URL", MODEL_URL)

    bundle = load_model()

    assert bundle is not None
    assert bundle.model_version == "test-0.0.1"

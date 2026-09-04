from fastapi.testclient import TestClient

from trialsignal.serving.api import app


def test_health_reports_no_model_loaded_when_artifact_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRIALSIGNAL_MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["model_version"] is None


def test_score_returns_503_when_no_model_is_loaded(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRIALSIGNAL_MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    with TestClient(app) as client:
        response = client.post(
            "/score",
            json={"gene_symbol": "EGFR", "disease_name": "non-small cell lung carcinoma"},
        )

    assert response.status_code == 503
    assert "trialsignal train" in response.json()["detail"]

import json
from pathlib import Path

import httpx
import respx
from fastapi.testclient import TestClient

from trialsignal.data.chembl import BASE_URL as CHEMBL_URL
from trialsignal.data.chembl import MOLECULE_URL
from trialsignal.data.open_targets import BASE_URL as OPEN_TARGETS_URL
from trialsignal.serving.api import app

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _mock_upstream_apis() -> None:
    respx.post(OPEN_TARGETS_URL).mock(
        return_value=httpx.Response(200, json=_load("opentargets_egfr_response.json"))
    )
    # Empty molecule-synonym match: exercises the real molecule-first lookup
    # path (_get_activities now calls this before the target-level pull) but
    # keeps this fixture's "falls back to target-level data" scenario intact.
    respx.get(MOLECULE_URL).mock(return_value=httpx.Response(200, json={"molecules": []}))
    # chembl_last_page.json has page_meta.next=null — a single-page response,
    # so pagination in the client stops after one call instead of looping
    # up to --max-pages against a mock that would otherwise "next" forever.
    respx.get(CHEMBL_URL).mock(
        return_value=httpx.Response(200, json=_load("chembl_last_page.json"))
    )


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


def test_score_returns_400_for_unknown_gene_symbol(trained_model_path, monkeypatch) -> None:
    monkeypatch.setenv("TRIALSIGNAL_MODEL_PATH", str(trained_model_path))
    with TestClient(app) as client:
        response = client.post(
            "/score", json={"gene_symbol": "NOTAREALGENE", "disease_name": "some disease"}
        )

    assert response.status_code == 400
    assert "NOTAREALGENE" in response.json()["detail"]
    assert "EGFR" in response.json()["detail"]  # lists the available curated hypotheses


@respx.mock
def test_score_returns_422_for_unresolvable_disease(trained_model_path, monkeypatch) -> None:
    monkeypatch.setenv("TRIALSIGNAL_MODEL_PATH", str(trained_model_path))
    _mock_upstream_apis()

    with TestClient(app) as client:
        response = client.post(
            "/score", json={"gene_symbol": "EGFR", "disease_name": "totally unrelated condition"}
        )

    assert response.status_code == 422
    assert "unrelated condition" in response.json()["detail"]


@respx.mock
def test_score_returns_valid_response_for_known_hypothesis(trained_model_path, monkeypatch) -> None:
    monkeypatch.setenv("TRIALSIGNAL_MODEL_PATH", str(trained_model_path))
    _mock_upstream_apis()

    with TestClient(app) as client:
        response = client.post(
            "/score",
            json={"gene_symbol": "egfr", "disease_name": "non-small cell lung carcinoma"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["model_version"] == "test-0.0.1"
    assert len(body["top_contributions"]) > 0
    for contribution in body["top_contributions"]:
        assert set(contribution) == {"feature", "value", "shap_contribution"}
    # The mocked ChEMBL fixture has no activity whose pref_name matches an
    # EGFR drug alias, so the fallback-to-target-level warning must fire.
    assert any("ChEMBL" in w for w in body["warnings"])


@respx.mock
def test_score_caches_upstream_calls_across_requests(trained_model_path, monkeypatch) -> None:
    monkeypatch.setenv("TRIALSIGNAL_MODEL_PATH", str(trained_model_path))
    ot_route = respx.post(OPEN_TARGETS_URL).mock(
        return_value=httpx.Response(200, json=_load("opentargets_egfr_response.json"))
    )
    molecule_route = respx.get(MOLECULE_URL).mock(
        return_value=httpx.Response(200, json={"molecules": []})
    )
    chembl_route = respx.get(CHEMBL_URL).mock(
        return_value=httpx.Response(200, json=_load("chembl_last_page.json"))
    )

    with TestClient(app) as client:
        payload = {"gene_symbol": "EGFR", "disease_name": "non-small cell lung carcinoma"}
        first = client.post("/score", json=payload)
        second = client.post("/score", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    # Second request must reuse the in-process cache, not re-fetch. EGFR has
    # 3 drug_aliases, so the first request alone makes 3 molecule-search
    # calls (one per alias) — what matters is that the *second* request adds
    # none of them, not that there's only one.
    assert ot_route.call_count == 1
    assert molecule_route.call_count == 3
    assert chembl_route.call_count == 1


@respx.mock
def test_score_returns_502_when_upstream_is_persistently_down(
    trained_model_path, monkeypatch
) -> None:
    """A real gap found by hand-testing against the live APIs: an upstream
    5xx that exhausts the client's own retry budget must surface as a clean
    502, not an unhandled 500 with a raw traceback. This exercises the real
    ChemblClient retry path (tenacity), so it genuinely takes ~15s — that
    cost buys confidence the fix actually works end-to-end, not just that
    the wrapper function looks right in isolation."""
    monkeypatch.setenv("TRIALSIGNAL_MODEL_PATH", str(trained_model_path))
    respx.post(OPEN_TARGETS_URL).mock(
        return_value=httpx.Response(200, json=_load("opentargets_egfr_response.json"))
    )
    respx.get(MOLECULE_URL).mock(return_value=httpx.Response(200, json={"molecules": []}))
    respx.get(CHEMBL_URL).mock(return_value=httpx.Response(500))

    with TestClient(app) as client:
        response = client.post(
            "/score",
            json={"gene_symbol": "EGFR", "disease_name": "non-small cell lung carcinoma"},
        )

    assert response.status_code == 502
    assert "ChEMBL" in response.json()["detail"]

"""Fixture content here mirrors a real response captured from the live Open
Targets API for EGFR (see module docstring in open_targets.py) — not a
guessed shape, so these tests catch drift against reality, not just against
whatever the parser happens to assume.
"""

import json
from pathlib import Path

import httpx
import pytest
import respx

from trialsignal.data.open_targets import (
    BASE_URL,
    OpenTargetsClient,
    OpenTargetsQueryError,
    parse_target_diseases,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_parses_associated_diseases_with_tractability_and_safety() -> None:
    target = _load("opentargets_egfr_response.json")["data"]["target"]
    rows = parse_target_diseases(target)

    assert len(rows) == 2
    nsclc = next(r for r in rows if r.disease_id == "MONDO_0005233")
    assert nsclc.target_symbol == "EGFR"
    assert nsclc.disease_name == "non-small cell lung carcinoma"
    assert nsclc.overall_score == 0.8525670184292347
    assert nsclc.datatype_scores["genetic_association"] == 0.7464039237342681
    assert nsclc.datatype_scores["clinical"] == 0.9959475193895682
    # Both SM and AB "Approved Drug" tractability rows are true in the fixture.
    assert nsclc.tractable_small_molecule is True
    assert nsclc.tractable_antibody is True
    assert nsclc.safety_liability_count == 3


def test_missing_target_returns_empty_list() -> None:
    assert parse_target_diseases(None) == []
    assert parse_target_diseases({}) == []


def test_target_with_no_tractability_data_returns_none_not_false() -> None:
    """A target Open Targets has no tractability assessment for must not be
    silently reported as "not tractable" — that's a different claim from
    "unknown", and the two must stay distinguishable downstream."""
    target = {
        "id": "ENSG00000000000",
        "approvedSymbol": "TEST",
        "associatedDiseases": {
            "count": 1,
            "rows": [{"score": 0.5, "disease": {"id": "EFO_0000001", "name": "test disease"}}],
        },
    }
    rows = parse_target_diseases(target)
    assert rows[0].tractable_small_molecule is None
    assert rows[0].tractable_antibody is None
    assert rows[0].safety_liability_count == 0


@respx.mock
def test_iter_target_diseases_stops_when_fewer_rows_than_page_size() -> None:
    respx.post(BASE_URL).mock(
        return_value=httpx.Response(200, json=_load("opentargets_egfr_response.json"))
    )

    client = OpenTargetsClient()
    rows = list(client.iter_target_diseases("ENSG00000146648", page_size=50))

    assert len(rows) == 2


@respx.mock
def test_graphql_errors_field_raises_without_retrying() -> None:
    route = respx.post(BASE_URL).mock(
        return_value=httpx.Response(200, json={"errors": [{"message": "boom"}]})
    )

    client = OpenTargetsClient()
    with pytest.raises(OpenTargetsQueryError):
        list(client.iter_target_diseases("ENSG00000146648", max_pages=1))

    # A GraphQL-level error must not trigger the retry-on-HTTPError logic —
    # it's the wrong failure mode to retry (see OpenTargetsQueryError docstring).
    assert route.call_count == 1

"""Fixture data mirrors a real response captured from the live ChEMBL API for
target CHEMBL203 (EGFR) — including the detail that trips a naive parser:
`standard_value` / `pchembl_value` are JSON strings, not numbers.
"""

import json
from pathlib import Path

import httpx
import respx

from trialsignal.data.chembl import BASE_URL, ChemblClient, parse_activity

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_parses_string_typed_numeric_fields() -> None:
    raw = _load("chembl_egfr_activities.json")["activities"][0]
    record = parse_activity(raw)

    assert record is not None
    assert record.molecule_chembl_id == "CHEMBL68920"
    assert record.target_chembl_id == "CHEMBL203"
    assert record.standard_type == "IC50"
    # These are strings ("41.0", "7.39") in the raw payload — must come out as float.
    assert record.standard_value == 41.0
    assert isinstance(record.standard_value, float)
    assert record.pchembl_value == 7.39
    assert record.standard_units == "nM"


def test_null_pref_name_and_numeric_fields_do_not_crash_parsing() -> None:
    raw = _load("chembl_last_page.json")["activities"][1]
    record = parse_activity(raw)

    assert record is not None
    assert record.pref_name is None
    assert record.standard_value is None
    assert record.pchembl_value is None


def test_missing_identifying_field_returns_none() -> None:
    assert parse_activity({"standard_type": "IC50"}) is None
    assert parse_activity({}) is None


@respx.mock
def test_iter_activities_follows_pagination() -> None:
    route = respx.get(BASE_URL)
    route.side_effect = [
        httpx.Response(200, json=_load("chembl_egfr_activities.json")),
        httpx.Response(200, json=_load("chembl_last_page.json")),
    ]

    client = ChemblClient()
    records = list(client.iter_activities("CHEMBL203", page_size=2))

    # 2 + 2 raw activities, but one in the last page has no standard_value —
    # still parses (identifying fields are present), so all 4 come through.
    assert len(records) == 4
    assert route.call_count == 2


@respx.mock
def test_iter_activities_respects_max_pages() -> None:
    route = respx.get(BASE_URL)
    route.side_effect = [httpx.Response(200, json=_load("chembl_egfr_activities.json"))]

    client = ChemblClient()
    records = list(client.iter_activities("CHEMBL203", page_size=2, max_pages=1))

    assert len(records) == 2
    assert route.call_count == 1

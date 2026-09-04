"""The client's pagination and retry behavior against a mocked transport —
this must never depend on ClinicalTrials.gov being reachable in CI.
"""

import httpx
import respx

from trialsignal.data.clinicaltrials import BASE_URL, ClinicalTrialsClient

_PAGE_1 = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Study A"},
                "statusModule": {"overallStatus": "COMPLETED"},
            }
        }
    ],
    "nextPageToken": "page2token",
}

_PAGE_2 = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000002", "briefTitle": "Study B"},
                "statusModule": {"overallStatus": "TERMINATED"},
            }
        }
    ]
}


@respx.mock
def test_iter_studies_follows_pagination() -> None:
    route = respx.get(BASE_URL)
    route.side_effect = [
        httpx.Response(200, json=_PAGE_1),
        httpx.Response(200, json=_PAGE_2),
    ]

    client = ClinicalTrialsClient()
    records = list(client.iter_studies("lung cancer"))

    assert [r.nct_id for r in records] == ["NCT00000001", "NCT00000002"]
    assert route.call_count == 2


@respx.mock
def test_iter_studies_respects_max_pages() -> None:
    route = respx.get(BASE_URL)
    route.side_effect = [httpx.Response(200, json=_PAGE_1)]

    client = ClinicalTrialsClient()
    records = list(client.iter_studies("lung cancer", max_pages=1))

    assert len(records) == 1
    assert route.call_count == 1


@respx.mock
def test_transient_5xx_is_retried_then_succeeds() -> None:
    route = respx.get(BASE_URL)
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(200, json=_PAGE_2),
    ]

    client = ClinicalTrialsClient()
    records = list(client.iter_studies("lung cancer"))

    assert [r.nct_id for r in records] == ["NCT00000002"]
    assert route.call_count == 2

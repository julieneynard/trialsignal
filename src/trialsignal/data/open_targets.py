"""Client for the Open Targets Platform GraphQL API (public, unauthenticated).

Docs: https://platform-docs.opentargets.org/data-access/graphql-api

Query shape and field names below were verified against the live API
(`api.platform.opentargets.org/api/v4/graphql`) for EGFR/ABL1 before writing
this, not assumed from documentation — see the module docstring on
`TargetDiseaseAssociation` for the two things that turned out to differ from
the naive assumption (mixed EFO/MONDO disease IDs, and the real datatype-score
names).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trialsignal.data.schemas import TargetDiseaseAssociation

BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_PAGE_SIZE = 50


class OpenTargetsQueryError(Exception):
    """A GraphQL-level error (HTTP 200, but an `errors` field in the body) —
    e.g. a malformed query or an unknown ID. Deliberately NOT an
    httpx.HTTPError subclass: those trigger the client's retry logic, and
    retrying a query that's wrong won't make it right. A 5xx or connection
    error is worth retrying; a bad query is not."""

_QUERY = """
query TargetDiseases($ensemblId: String!, $index: Int!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    tractability {
      label
      modality
      value
    }
    safetyLiabilities {
      event
    }
    associatedDiseases(page: { index: $index, size: $size }) {
      count
      rows {
        score
        disease {
          id
          name
        }
        datatypeScores {
          id
          score
        }
      }
    }
  }
}
"""


def _tractability_flag(tractability: list[dict[str, Any]], modality: str) -> bool | None:
    """Open Targets returns tractability as a flat list of {label, modality,
    value} rows rather than a single score — "Approved Drug" is the row that
    answers "can this modality already reach this target," which is the
    binary signal worth keeping; the rest (pocket quality, structural
    evidence, etc.) is out of scope for v1's feature set."""
    for row in tractability:
        if row.get("label") == "Approved Drug" and row.get("modality") == modality:
            value = row.get("value")
            return bool(value) if isinstance(value, bool) else None
    return None


def parse_target_diseases(raw_target: dict[str, Any] | None) -> list[TargetDiseaseAssociation]:
    """Convert one `target` GraphQL response into one row per associated
    disease. Returns an empty list (never raises) for a target with no data —
    a target simply not being in Open Targets is a normal outcome, not a
    pipeline failure."""
    if not raw_target:
        return []

    target_id = raw_target.get("id")
    target_symbol = raw_target.get("approvedSymbol")
    if not target_id or not target_symbol:
        return []

    tractability = raw_target.get("tractability") or []
    tractable_sm = _tractability_flag(tractability, "SM")
    tractable_ab = _tractability_flag(tractability, "AB")
    safety_count = len(raw_target.get("safetyLiabilities") or [])

    rows: list[TargetDiseaseAssociation] = []
    for row in (raw_target.get("associatedDiseases") or {}).get("rows", []):
        disease = row.get("disease") or {}
        disease_id = disease.get("id")
        disease_name = disease.get("name")
        score = row.get("score")
        if not disease_id or not disease_name or score is None:
            continue

        datatype_scores = {
            entry["id"]: entry["score"]
            for entry in row.get("datatypeScores") or []
            if "id" in entry and "score" in entry
        }

        rows.append(
            TargetDiseaseAssociation(
                target_id=target_id,
                target_symbol=target_symbol,
                disease_id=disease_id,
                disease_name=disease_name,
                overall_score=score,
                datatype_scores=datatype_scores,
                tractable_small_molecule=tractable_sm,
                tractable_antibody=tractable_ab,
                safety_liability_count=safety_count,
            )
        )
    return rows


class OpenTargetsClient:
    """Thin, retrying wrapper around the Open Targets GraphQL API."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=30.0)

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
    )
    def _post(self, variables: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(
            BASE_URL, json={"query": _QUERY, "variables": variables}, timeout=30.0
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        if "errors" in payload:
            raise OpenTargetsQueryError(str(payload["errors"]))
        return payload

    def iter_target_diseases(
        self,
        ensembl_id: str,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> Iterator[TargetDiseaseAssociation]:
        """Paginate through every disease associated with `ensembl_id`."""
        index = 0
        pages_seen = 0
        while True:
            payload = self._post({"ensemblId": ensembl_id, "index": index, "size": page_size})
            target = payload.get("data", {}).get("target")
            rows = parse_target_diseases(target)
            yield from rows

            pages_seen += 1
            total_count = ((target or {}).get("associatedDiseases") or {}).get("count", 0)
            fetched_so_far = (index + 1) * page_size
            if (
                not rows
                or fetched_so_far >= total_count
                or (max_pages is not None and pages_seen >= max_pages)
            ):
                return
            index += 1

    def close(self) -> None:
        self._client.close()

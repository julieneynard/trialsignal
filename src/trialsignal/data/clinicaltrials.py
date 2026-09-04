"""Client for the ClinicalTrials.gov API v2 (public, unauthenticated).

Docs: https://clinicaltrials.gov/data-api/api

This is the label source: trial phase progression / termination is what the
model is trained to predict, so parsing here has to be conservative about
malformed/missing fields rather than silently defaulting them.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trialsignal.data.schemas import TrialPhase, TrialRecord, TrialStatus

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_PAGE_SIZE = 100


def _parse_date(struct: dict[str, Any] | None) -> date | None:
    if not struct or "date" not in struct:
        return None
    raw = struct["date"]
    # CT.gov emits YYYY-MM or YYYY-MM-DD; normalize to the 1st of the month for the former.
    parts = raw.split("-")
    try:
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
    except ValueError:
        return None
    return None


def _to_enum(value: str | None, enum_cls: type) -> Any | None:
    if value is None:
        return None
    try:
        return enum_cls(value)
    except ValueError:
        return None


def parse_study(raw: dict[str, Any]) -> TrialRecord | None:
    """Convert one raw `protocolSection` study payload into a TrialRecord.

    Returns None (rather than raising) for studies missing the fields a
    label/feature needs — logged upstream as a data-quality metric, not a
    pipeline failure. A single malformed study must never abort a full pull.
    """
    section = raw.get("protocolSection", {})
    ident = section.get("identificationModule", {})
    status_mod = section.get("statusModule", {})
    design = section.get("designModule", {})
    conditions_mod = section.get("conditionsModule", {})
    arms = section.get("armsInterventionsModule", {})
    sponsors = section.get("sponsorCollaboratorsModule", {})

    nct_id = ident.get("nctId")
    status = _to_enum(status_mod.get("overallStatus"), TrialStatus)
    if nct_id is None or status is None:
        return None

    phases = [p for p in (_to_enum(p, TrialPhase) for p in design.get("phases", [])) if p]
    enrollment_info = design.get("enrollmentInfo", {}) or {}
    lead_sponsor = sponsors.get("leadSponsor", {}) or {}

    return TrialRecord(
        nct_id=nct_id,
        title=ident.get("briefTitle", ""),
        status=status,
        phases=phases,
        conditions=conditions_mod.get("conditions", []) or [],
        interventions=[i.get("name", "") for i in arms.get("interventions", []) if i.get("name")],
        sponsor=lead_sponsor.get("name"),
        sponsor_class=lead_sponsor.get("class"),
        enrollment=enrollment_info.get("count"),
        start_date=_parse_date(status_mod.get("startDateStruct")),
        primary_completion_date=_parse_date(status_mod.get("primaryCompletionDateStruct")),
        why_stopped=status_mod.get("whyStopped"),
        study_type=design.get("studyType"),
    )


class ClinicalTrialsClient:
    """Thin, retrying wrapper around the CT.gov v2 REST API."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        # No base_url: httpx concatenates base_url + "" into a trailing-slash
        # URL that's easy to mismatch against in tests/mocks, so each request
        # targets BASE_URL directly instead.
        self._client = client or httpx.Client(timeout=30.0)

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
    )
    def _get_page(self, params: dict[str, Any]) -> dict[str, Any]:
        response = self._client.get(BASE_URL, params=params)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload

    def iter_studies(
        self,
        condition: str,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> Iterator[TrialRecord]:
        """Paginate through every study matching `condition`, yielding parsed
        TrialRecords. Malformed studies are skipped, not raised."""
        params: dict[str, Any] = {"query.cond": condition, "pageSize": page_size}
        pages_seen = 0
        while True:
            payload = self._get_page(params)
            for raw_study in payload.get("studies", []):
                record = parse_study(raw_study)
                if record is not None:
                    yield record

            pages_seen += 1
            next_token = payload.get("nextPageToken")
            if not next_token or (max_pages is not None and pages_seen >= max_pages):
                return
            params["pageToken"] = next_token

    def close(self) -> None:
        self._client.close()

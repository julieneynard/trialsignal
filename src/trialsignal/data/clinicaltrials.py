"""Client for the ClinicalTrials.gov API v2 (public, unauthenticated).

Docs: https://clinicaltrials.gov/data-api/api

This is the label source: trial phase progression / termination is what the
model is trained to predict, so parsing here has to be conservative about
malformed/missing fields rather than silently defaulting them.

The list endpoint (`iter_studies` uses `/studies?query.cond=...`) returns a
full `resultsSection` per study for free — no extra per-trial request needed
— when a trial has posted results, so `parse_study` also extracts the
primary-endpoint statistical result from it (see `_extract_primary_result`).
Measured against the live API on 226 real trials across 3 hypotheses:
~50% have `hasResults=True` (posted *something*), but only ~5-10% have a
parseable p-value on a PRIMARY-type outcome measure's analysis — most
posted results report raw arm-level data without a structured significance
test. That coverage is far too sparse to use as the primary training label
(it would cut the dataset by ~90%+), so it's exposed as a secondary field
(`TrialRecord.primary_pvalue`) for cross-validating the registry-status
label in `labels.py`, not a replacement for it — see `docs/METHODS.md`.
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


def _parse_pvalue(raw: str | None) -> float | None:
    """CT.gov reports p-values as free-text strings ("<0.0001", "0.023",
    sometimes "NS" or empty) rather than numbers — strip a leading
    comparison operator and parse what's left, returning None for anything
    that doesn't parse rather than guessing."""
    if not raw:
        return None
    text = raw.strip().lstrip("<>=").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _extract_primary_result(
    results_section: dict[str, Any] | None,
) -> tuple[float | None, str | None]:
    """Returns (p_value, analysis_type) for the primary endpoint's main
    statistical analysis, or (None, None) if there isn't one.

    A single outcome measure can carry multiple `analyses` entries (e.g. one
    per subgroup/cohort) — SUPERIORITY is preferred when present since
    that's typically the trial's main powered comparison; otherwise the
    first parseable analysis is used. Secondary/other-type outcome measures
    are ignored entirely: only PRIMARY measures speak to "did this trial
    meet its main goal," which is the question labels.py's proxy label is
    trying to approximate.
    """
    outcome_measures = ((results_section or {}).get("outcomeMeasuresModule") or {}).get(
        "outcomeMeasures", []
    )
    candidates: list[tuple[float, str | None]] = []
    for measure in outcome_measures:
        if measure.get("type") != "PRIMARY":
            continue
        for analysis in measure.get("analyses", []):
            pvalue = _parse_pvalue(analysis.get("pValue"))
            if pvalue is not None:
                candidates.append((pvalue, analysis.get("nonInferiorityType")))

    if not candidates:
        return None, None
    superiority = [c for c in candidates if c[1] == "SUPERIORITY"]
    return superiority[0] if superiority else candidates[0]


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
    primary_pvalue, primary_analysis_type = _extract_primary_result(raw.get("resultsSection"))

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
        has_results=bool(raw.get("hasResults", False)),
        primary_pvalue=primary_pvalue,
        primary_analysis_type=primary_analysis_type,
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

    def fetch_by_nct_ids(self, nct_ids: list[str], *, batch_size: int = 50) -> list[TrialRecord]:
        """Re-fetch specific trials by ID, batched via an `AREA[NCTId](A OR
        B OR ...)` query term rather than one request per ID — verified
        against the live API that per-trial single-record fetches
        (`/studies/{nctId}`) rate-limit hard (429) after a handful of rapid
        requests, while this same list endpoint used by `iter_studies`
        happily returns 100+ full records (results section included) in one
        call. `batch_size` caps the OR clause length per request; 50 is
        comfortably under whatever URL/query-complexity limit CT.gov has,
        verified empirically rather than assumed."""
        records: list[TrialRecord] = []
        for i in range(0, len(nct_ids), batch_size):
            batch = nct_ids[i : i + batch_size]
            term = "AREA[NCTId](" + " OR ".join(batch) + ")"
            payload = self._get_page({"query.term": term, "pageSize": batch_size})
            for raw_study in payload.get("studies", []):
                record = parse_study(raw_study)
                if record is not None:
                    records.append(record)
        return records

    def close(self) -> None:
        self._client.close()

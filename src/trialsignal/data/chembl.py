"""Client for the ChEMBL REST API (public, unauthenticated).

Docs: https://www.ebi.ac.uk/chembl/api/data/docs

Verified against the live API (target CHEMBL203 / EGFR) before writing this.
The one surprising thing worth flagging: `standard_value` and `pchembl_value`
come back as **strings** ("41.0", "7.39"), not numbers — `activity_comment`
and `molecule_pref_name` are commonly null. Parsing this as if it were a
clean typed API silently produces either a crash or (worse) a pydantic
coercion that happens to work today and breaks on the first malformed row.

`iter_activities` pulls a target's activity list *unfiltered by compound* —
for a well-studied target like EGFR that's 26,000+ rows, of which the first
few hundred (all `fetch-activities` pulls by default, see LIMITATIONS.md)
are essentially never the specific marketed drug a hypothesis cares about;
they're overwhelmingly anonymous research-compound entries. Verified
directly: querying `activity.json` filtered by BOTH `molecule_chembl_id`
and `target_chembl_id` returns exactly the drug-specific activities (485
real IC50 records for osimertinib against EGFR) that a target-level pull
was silently missing. `find_molecule_ids_by_synonym` +
`iter_activities_for_molecule` are that targeted path — resolving a drug
name (generic or brand; ChEMBL's `molecule_synonyms` table covers both,
verified with "osimertinib" and "Tagrisso" both resolving to
CHEMBL3353410) to its ChEMBL ID first, then querying activities for that
exact compound-target pair instead of guessing from a truncated general
pull.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trialsignal.data.schemas import ChemblActivity

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
DEFAULT_PAGE_SIZE = 100


def _to_float(value: Any) -> float | None:
    """ChEMBL numeric fields arrive as strings, or null, or occasionally an
    already-numeric value — never assume which."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_activity(raw: dict[str, Any]) -> ChemblActivity | None:
    """Convert one raw activity record into a ChemblActivity. Returns None
    (never raises) for a record missing the identifying fields — a
    malformed row must not abort the whole pull."""
    molecule_id = raw.get("molecule_chembl_id")
    target_id = raw.get("target_chembl_id")
    standard_type = raw.get("standard_type")
    if not molecule_id or not target_id or not standard_type:
        return None

    return ChemblActivity(
        molecule_chembl_id=molecule_id,
        pref_name=raw.get("molecule_pref_name"),
        target_chembl_id=target_id,
        standard_type=standard_type,
        standard_value=_to_float(raw.get("standard_value")),
        standard_units=raw.get("standard_units"),
        pchembl_value=_to_float(raw.get("pchembl_value")),
    )


class ChemblClient:
    """Thin, retrying wrapper around the ChEMBL activity and molecule
    endpoints."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=30.0)

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
    )
    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self._client.get(url, params=params)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload

    def _iter_activities(
        self, params: dict[str, Any], *, page_size: int, max_pages: int | None
    ) -> Iterator[ChemblActivity]:
        """Shared pagination loop for any activity.json query — target-level
        (`iter_activities`) and molecule+target-level
        (`iter_activities_for_molecule`) differ only in which filter params
        they pass in, not in how pagination/parsing works."""
        pages_seen = 0
        while True:
            payload = self._get(BASE_URL, params)
            for raw_activity in payload.get("activities", []):
                record = parse_activity(raw_activity)
                if record is not None:
                    yield record

            pages_seen += 1
            page_meta = payload.get("page_meta") or {}
            if not page_meta.get("next") or (max_pages is not None and pages_seen >= max_pages):
                return
            params["offset"] = params["offset"] + page_size

    def iter_activities(
        self,
        target_chembl_id: str,
        *,
        standard_type: str = "IC50",
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> Iterator[ChemblActivity]:
        """Paginate through every bioactivity record for a target, filtered
        to one `standard_type` (IC50/EC50/Ki/...) at a time — mixing types
        without normalizing units first is how a feature pipeline quietly
        averages nanomolar and micromolar values together.

        Unfiltered by compound — for a well-studied target this is mostly
        anonymous research compounds, not the marketed drug a hypothesis
        cares about (see module docstring). Use
        `iter_activities_for_molecule` when a specific drug is known;
        this method is for general target-level exploration."""
        params: dict[str, Any] = {
            "target_chembl_id": target_chembl_id,
            "standard_type": standard_type,
            "limit": page_size,
            "offset": 0,
        }
        yield from self._iter_activities(params, page_size=page_size, max_pages=max_pages)

    def find_molecule_ids_by_synonym(self, name: str) -> list[str]:
        """Resolve a drug name (generic or brand — ChEMBL's synonym table
        covers both, see module docstring) to its ChEMBL molecule ID(s).
        Usually zero or one match; occasionally more than one (e.g. a free
        base and a salt form sharing a trade name) — callers should query
        activities for all of them, not just the first."""
        payload = self._get(
            MOLECULE_URL,
            {"molecule_synonyms__molecule_synonym__iexact": name, "limit": 25},
        )
        return [
            m["molecule_chembl_id"]
            for m in payload.get("molecules", [])
            if m.get("molecule_chembl_id")
        ]

    def iter_activities_for_molecule(
        self,
        molecule_chembl_id: str,
        target_chembl_id: str,
        *,
        standard_type: str = "IC50",
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> Iterator[ChemblActivity]:
        """Every activity for one specific compound-target pair — small and
        exact, unlike `iter_activities`' unfiltered target-level pull."""
        params: dict[str, Any] = {
            "molecule_chembl_id": molecule_chembl_id,
            "target_chembl_id": target_chembl_id,
            "standard_type": standard_type,
            "limit": page_size,
            "offset": 0,
        }
        yield from self._iter_activities(params, page_size=page_size, max_pages=max_pages)

    def close(self) -> None:
        self._client.close()

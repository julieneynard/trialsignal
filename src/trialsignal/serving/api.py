"""FastAPI application serving trial-progression risk scores.

Design choices worth flagging:

- The model artifact is optional at boot. A freshly cloned repo (or a CI
  build) must be able to start this service and answer /health honestly
  even before anyone has run the training pipeline — returning a clear 503
  from /score is correct behavior, not a bug, until `trialsignal train` has
  produced `models/trialsignal_model.joblib`.

- Live scoring is restricted to the curated hypotheses in
  `features/hypothesis.py`. There is no general "any gene, any disease"
  lookup — the entity-resolution join this project is built around only
  works because each hypothesis's drug/target identifiers were verified by
  hand against the live APIs (see hypothesis.py's module docstring); a
  gene symbol typed into a request with no matching Hypothesis has no
  ChEMBL target ID or drug-alias list to resolve against.

- Open Targets/ChEMBL data for a hypothesis is fetched live on first use
  and cached in-process for the life of the server — not re-fetched per
  request (would be slow and pointless: this reference data doesn't change
  request-to-request), and not baked into the repo (would go stale). The
  synchronous httpx-based clients run via `asyncio.to_thread` so a slow
  upstream call doesn't block the event loop for other requests.

- Pagination depth is capped small (2 pages) for both upstream sources.
  Measured against the live APIs: fetching EGFR's full 6,459-row Open
  Targets association list (10 pages) took ~79s end-to-end for a single
  cold request and once hit a transient ChEMBL 5xx that exhausted the
  client's retry budget — unacceptable latency for a request a user is
  waiting on, for data where only the single best-scoring disease match
  actually matters. Open Targets returns `associatedDiseases` sorted by
  score descending, so the confident match for any disease worth asking
  about is almost always on the first page or two; ChEMBL bioactivity
  aggregates are directional either way at this dataset's size. A cache
  miss is still slow-ish (a few seconds) but not tens of seconds, and a
  genuine upstream outage now surfaces as a clean 502, not an unhandled
  500 (see `_fetch_upstream`).

- ChEMBL activities are fetched molecule-first, not just target-first:
  `_get_activities` resolves each of a hypothesis's `drug_aliases` to a
  ChEMBL molecule ID (via `find_molecule_ids_by_synonym`, which covers
  both generic and brand names) and fetches that specific compound-target
  pair's activities before falling back to the generic target-level pull.
  Verified this matters, not just in theory: a target-level pull alone
  found zero of osimertinib's 485 real EGFR activities (they're not in the
  first ~200 rows of EGFR's 26,000+ unfiltered activity list); the
  molecule-first query finds them directly. See chembl.py's module
  docstring for the full reasoning.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

import httpx
from fastapi import FastAPI, HTTPException
from tenacity import RetryError

from trialsignal.data.chembl import ChemblClient
from trialsignal.data.open_targets import OpenTargetsClient, OpenTargetsQueryError
from trialsignal.data.schemas import ChemblActivity, TargetDiseaseAssociation
from trialsignal.features.build_features import build_live_feature_vector
from trialsignal.features.hypothesis import CURATED_HYPOTHESES, Hypothesis
from trialsignal.models.registry import ModelBundle, load_model
from trialsignal.models.train import FEATURE_COLUMNS, explain_instance, feature_row_to_frame
from trialsignal.serving.schemas import (
    FeatureContribution,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)

_ACTIVITIES_MAX_PAGES = 2
_TARGET_DISEASES_MAX_PAGES = 2

_T = TypeVar("_T")

_state: dict[str, ModelBundle | None] = {"model": None}
_target_disease_cache: dict[str, list[TargetDiseaseAssociation]] = {}
_activity_cache: dict[str, list[ChemblActivity]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _state["model"] = load_model()
    yield
    _state["model"] = None
    _target_disease_cache.clear()
    _activity_cache.clear()


app = FastAPI(
    title="TrialSignal API",
    description="Clinical trial progression risk scoring for drug/target/disease hypotheses.",
    version="0.1.0",
    lifespan=lifespan,
)


def _find_hypothesis(gene_symbol: str) -> Hypothesis:
    hypothesis = next(
        (h for h in CURATED_HYPOTHESES if h.gene_symbol.lower() == gene_symbol.lower()),
        None,
    )
    if hypothesis is None:
        available = ", ".join(h.gene_symbol for h in CURATED_HYPOTHESES)
        raise HTTPException(
            status_code=400,
            detail=(
                f"No curated hypothesis for gene_symbol={gene_symbol!r}. "
                f"Available: {available}. See features/hypothesis.py — live scoring is "
                "restricted to hand-verified hypotheses, not an open gene/disease lookup."
            ),
        )
    return hypothesis


async def _fetch_upstream(source_name: str, fetch: Callable[[], _T]) -> _T:
    """Run a blocking upstream client call off the event loop, converting
    its failure modes into a clean 502 instead of an unhandled 500. Both
    ClinicalTrialsClient-family clients already retry transient errors
    internally (see their `tenacity` config) — a `RetryError` here means
    that budget was exhausted, i.e. the upstream API is genuinely down or
    persistently failing right now, not a one-off blip worth retrying again
    at this layer."""
    try:
        return await asyncio.to_thread(fetch)
    except (RetryError, httpx.HTTPError, OpenTargetsQueryError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"{source_name} is currently unavailable or returned an error: {exc}",
        ) from exc


async def _get_target_diseases(hypothesis: Hypothesis) -> list[TargetDiseaseAssociation]:
    cached = _target_disease_cache.get(hypothesis.ensembl_target_id)
    if cached is not None:
        return cached

    def _fetch() -> list[TargetDiseaseAssociation]:
        client = OpenTargetsClient()
        try:
            return list(
                client.iter_target_diseases(
                    hypothesis.ensembl_target_id, max_pages=_TARGET_DISEASES_MAX_PAGES
                )
            )
        finally:
            client.close()

    rows = await _fetch_upstream("Open Targets", _fetch)
    _target_disease_cache[hypothesis.ensembl_target_id] = rows
    return rows


async def _get_activities(hypothesis: Hypothesis) -> list[ChemblActivity]:
    cached = _activity_cache.get(hypothesis.chembl_target_id)
    if cached is not None:
        return cached

    def _fetch() -> list[ChemblActivity]:
        client = ChemblClient()
        try:
            molecule_ids: set[str] = set()
            for alias in hypothesis.drug_aliases:
                molecule_ids.update(client.find_molecule_ids_by_synonym(alias))

            molecule_activities: list[ChemblActivity] = []
            for molecule_id in molecule_ids:
                molecule_activities.extend(
                    client.iter_activities_for_molecule(
                        molecule_id,
                        hypothesis.chembl_target_id,
                        max_pages=_ACTIVITIES_MAX_PAGES,
                    )
                )
            target_activities = list(
                client.iter_activities(hypothesis.chembl_target_id, max_pages=_ACTIVITIES_MAX_PAGES)
            )
            # Molecule-specific rows first: _aggregate_chembl (build_features.py)
            # filters this combined pool by pref_name, so real drug-specific
            # activities (present here whenever ChEMBL has any) take over from
            # the generic target-level pool automatically — no extra logic
            # needed to prefer one over the other.
            return molecule_activities + target_activities
        finally:
            client.close()

    rows = await _fetch_upstream("ChEMBL", _fetch)
    _activity_cache[hypothesis.chembl_target_id] = rows
    return rows


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    model = _state["model"]
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_version=model.model_version if model else None,
    )


@app.post("/score", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    model = _state["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model artifact found. Run `trialsignal train` to produce "
                "models/trialsignal_model.joblib before calling /score."
            ),
        )

    hypothesis = _find_hypothesis(request.gene_symbol)
    target_diseases, activities = await asyncio.gather(
        _get_target_diseases(hypothesis), _get_activities(hypothesis)
    )

    feature_row = build_live_feature_vector(
        hypothesis, target_diseases, activities, request.disease_name
    )
    if feature_row is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Could not confidently resolve disease_name={request.disease_name!r} against "
                f"{hypothesis.gene_symbol}'s known Open Targets disease associations "
                f"({len(target_diseases)} candidates checked)."
            ),
        )

    x_row = feature_row_to_frame(feature_row)[FEATURE_COLUMNS]
    risk_score = float(model.model.predict_proba(x_row)[0, 1])
    contributions = explain_instance(model.model, x_row)

    warnings: list[str] = []
    if not feature_row.chembl_matched_by_molecule_name:
        warnings.append(
            "No ChEMBL bioactivity record matched this drug by name — chembl_* features "
            "reflect the target in general, not this specific drug (see build_features.py)."
        )

    return ScoreResponse(
        risk_score=risk_score,
        model_version=model.model_version,
        top_contributions=[
            FeatureContribution(feature=name, value=value, shap_contribution=shap)
            for name, value, shap in contributions
        ],
        warnings=warnings,
    )

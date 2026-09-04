"""FastAPI application serving trial-progression risk scores.

Design choice worth flagging: the model artifact is optional at boot. A
freshly cloned repo (or a CI build) must be able to start this service and
answer /health honestly even before anyone has run the training pipeline —
returning a clear 503 from /score is correct behavior, not a bug, until
`trialsignal train` has produced `models/trialsignal_model.joblib`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from trialsignal.models.registry import ModelBundle, load_model
from trialsignal.serving.schemas import HealthResponse, ScoreRequest, ScoreResponse

_state: dict[str, ModelBundle | None] = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _state["model"] = load_model()
    yield
    _state["model"] = None


app = FastAPI(
    title="TrialSignal API",
    description="Clinical trial progression risk scoring for drug/target/disease hypotheses.",
    version="0.1.0",
    lifespan=lifespan,
)


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

    # Feature construction (Open Targets + ChEMBL lookup for the requested
    # gene/disease/drug, resolved through trialsignal.data.entity_resolution)
    # is the next milestone — see README roadmap. Wiring it here is deferred
    # until the training pipeline has a stable feature schema to match.
    raise HTTPException(
        status_code=501,
        detail="Feature construction for live scoring is not yet implemented — see README roadmap.",
    )

"""Loads the trained model artifact for serving.

Kept deliberately dumb: training (trialsignal.models.train) produces a
versioned joblib bundle {model, feature_names, metadata}; this module only
ever reads it. No training logic belongs here — the API process should never
be able to accidentally retrain or mutate the artifact it serves.

`data/` and `models/*.joblib` are deliberately gitignored (reproducible from
the source clients, not versioned — see README's repo layout) — that's the
right call for development, but it means a freshly built deployment
container has no model file at all. Rather than break that design principle
for a deployment concern, `load_model` can fetch the artifact from a URL
(a GitHub Release asset in practice) via `TRIALSIGNAL_MODEL_URL` when the
local path doesn't exist. Unset, this is a complete no-op — local dev and
CI never set it, so they exercise the exact same "no model" path they always
have.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_MODEL_PATH = Path("models/trialsignal_model.joblib")


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    feature_names: list[str]
    model_version: str
    trained_at: str


def model_path() -> Path:
    return Path(os.environ.get("TRIALSIGNAL_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def _maybe_download_model(path: Path) -> None:
    """Best-effort fetch of the model artifact from `TRIALSIGNAL_MODEL_URL`.
    Silent no-op if the variable isn't set or the download fails for any
    reason — `load_model`'s caller already treats a missing artifact as a
    normal, expected state (a clean 503, not a crash), so a failed download
    should degrade to that same safe state rather than take the process down
    at startup over what is, for this project, a demo-hosting concern."""
    url = os.environ.get("TRIALSIGNAL_MODEL_URL")
    if not url:
        return
    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)


def load_model() -> ModelBundle | None:
    """Returns None (never raises) when no trained artifact exists yet and
    none could be fetched.

    The API surface is expected to run — and return a clear 503 — before a
    model has ever been trained, so a missing artifact is a normal state to
    handle, not an error condition.
    """
    path = model_path()
    if not path.exists():
        _maybe_download_model(path)
    if not path.exists():
        return None

    # Local import: keeps joblib off the hot import path for callers that never load a model.
    import joblib

    bundle = joblib.load(path)
    return ModelBundle(
        model=bundle["model"],
        feature_names=bundle["feature_names"],
        model_version=bundle["model_version"],
        trained_at=bundle["trained_at"],
    )

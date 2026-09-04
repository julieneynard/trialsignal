"""Loads the trained model artifact for serving.

Kept deliberately dumb: training (trialsignal.models.train) produces a
versioned joblib bundle {model, feature_names, metadata}; this module only
ever reads it. No training logic belongs here — the API process should never
be able to accidentally retrain or mutate the artifact it serves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_PATH = Path("models/trialsignal_model.joblib")


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    feature_names: list[str]
    model_version: str
    trained_at: str


def model_path() -> Path:
    return Path(os.environ.get("TRIALSIGNAL_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def load_model() -> ModelBundle | None:
    """Returns None (never raises) when no trained artifact exists yet.

    The API surface is expected to run — and return a clear 503 — before a
    model has ever been trained, so a missing artifact is a normal state to
    handle, not an error condition.
    """
    path = model_path()
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

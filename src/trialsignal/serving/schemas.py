from __future__ import annotations

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    gene_symbol: str = Field(..., examples=["EGFR"], description="Target gene symbol (HGNC).")
    disease_name: str = Field(..., examples=["non-small cell lung carcinoma"])
    drug_name: str | None = Field(default=None, examples=["osimertinib"])


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_contribution: float


class ScoreResponse(BaseModel):
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted probability of trial progression/success."
    )
    model_version: str
    top_contributions: list[FeatureContribution]
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None

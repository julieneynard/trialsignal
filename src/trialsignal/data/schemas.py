"""Typed schemas for every external data source.

Each source (ClinicalTrials.gov, Open Targets, ChEMBL) ships its own ID system
and its own notion of "what a drug/target/disease is". These models are the
contract between raw API responses and the rest of the pipeline: nothing
downstream should touch a raw dict from an API client.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class TrialPhase(StrEnum):
    EARLY_PHASE1 = "EARLY_PHASE1"
    PHASE1 = "PHASE1"
    PHASE2 = "PHASE2"
    PHASE3 = "PHASE3"
    PHASE4 = "PHASE4"
    NA = "NA"


class TrialStatus(StrEnum):
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    RECRUITING = "RECRUITING"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    COMPLETED = "COMPLETED"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    WITHDRAWN = "WITHDRAWN"
    UNKNOWN = "UNKNOWN"


class TrialRecord(BaseModel):
    """One ClinicalTrials.gov study, normalized to the fields the labeling
    and feature pipelines actually consume."""

    nct_id: str
    title: str
    status: TrialStatus
    phases: list[TrialPhase] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    sponsor: str | None = None
    sponsor_class: str | None = None  # INDUSTRY / NIH / OTHER, drives the leakage filter
    enrollment: int | None = None
    start_date: date | None = None
    primary_completion_date: date | None = None
    why_stopped: str | None = None  # free text, only present for TERMINATED/WITHDRAWN
    study_type: str | None = None

    @property
    def max_phase(self) -> TrialPhase | None:
        order = list(TrialPhase)
        ranked = [p for p in self.phases if p in order]
        return max(ranked, key=order.index) if ranked else None


class TargetDiseaseAssociation(BaseModel):
    """One Open Targets target<->disease evidence row."""

    target_id: str  # Ensembl gene ID, e.g. ENSG00000146648
    target_symbol: str
    disease_id: str  # EFO ID
    disease_name: str
    overall_score: float = Field(ge=0.0, le=1.0)
    genetic_association_score: float | None = Field(default=None, ge=0.0, le=1.0)
    known_drug_score: float | None = Field(default=None, ge=0.0, le=1.0)
    safety_liability_count: int | None = None


class ChemblActivity(BaseModel):
    """One ChEMBL bioactivity measurement for a compound-target pair."""

    molecule_chembl_id: str
    pref_name: str | None = None
    target_chembl_id: str
    standard_type: str  # IC50 / EC50 / Ki / ...
    standard_value: float | None = None
    standard_units: str | None = None
    pchembl_value: float | None = None  # -log10(activity in M), the comparable form


class ResolvedEntity(BaseModel):
    """Output of entity resolution: one drug/target/disease anchored across
    all three ID systems. This is what features and labels are joined on —
    never join raw source records directly against each other."""

    canonical_name: str
    gene_symbol: str | None = None
    ensembl_target_id: str | None = None
    chembl_target_id: str | None = None
    efo_disease_id: str | None = None
    ctgov_condition_terms: list[str] = Field(default_factory=list)

"""A `Hypothesis` is the unit the feature pipeline actually joins on: one
(target, drug, disease) triple, plus the identifiers/aliases needed to find
it in each of the three source systems.

Deliberately out of scope for v1: automatically *discovering* which
target/drug/disease triples exist from the raw data. ClinicalTrials.gov gives
free-text drug names and free-text conditions with no target attached at
all — recovering "this trial is testing EGFR inhibition" from "Osimertinib"
and "NSCLC" alone requires either a drug->target mechanism lookup (ChEMBL has
this, via the `mechanism` endpoint, not yet wired in) or a curated mapping.
This module is that curated mapping: an explicit, human-reviewed, small list
of hypotheses to start from, not a claim that every trial's target is known.
Automating the drug->target step is the natural v2 (see README roadmap).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    """One target/drug/disease hypothesis to build a training set around."""

    name: str
    gene_symbol: str
    ensembl_target_id: str
    chembl_target_id: str
    drug_aliases: list[str] = Field(
        ..., description="Intervention-name / molecule-name variants to match, case-insensitive."
    )
    ctgov_condition_query: str = Field(
        ..., description="CT.gov query.cond value used to pull the trial pool for this hypothesis."
    )


# A small, human-reviewed starting set — chosen because they're well-known,
# well-documented drug/target/disease relationships (easy to sanity-check the
# pipeline's output against what's publicly known), not an attempt at
# covering oncology broadly. Extending this list is the fastest way to grow
# the training set once the pipeline is validated on these.
CURATED_HYPOTHESES: list[Hypothesis] = [
    Hypothesis(
        name="EGFR / osimertinib / NSCLC",
        gene_symbol="EGFR",
        ensembl_target_id="ENSG00000146648",
        chembl_target_id="CHEMBL203",
        drug_aliases=["osimertinib", "tagrisso", "azd9291"],
        ctgov_condition_query="non-small cell lung cancer",
    ),
    Hypothesis(
        name="ABL1 / imatinib / CML",
        gene_symbol="ABL1",
        ensembl_target_id="ENSG00000097007",
        chembl_target_id="CHEMBL1862",
        drug_aliases=["imatinib", "gleevec", "glivec", "sti571"],
        ctgov_condition_query="chronic myeloid leukemia",
    ),
]

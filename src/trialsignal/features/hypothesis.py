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
#
# Deliberately diverse in mechanism/modality, not just headcount: the first
# two hypotheses alone (EGFR/ABL1, both kinases, both small molecules) let
# the trained model separate "success" from "failure" almost entirely by
# learning "which hypothesis is this" rather than a real risk signal — see
# docs/MODEL_CARD.md. The five below add a monoclonal antibody (trastuzumab),
# a PARP inhibitor (olaparib), and a PD-1 immune checkpoint inhibitor
# (pembrolizumab) alongside two more kinase inhibitors, across four
# additional diseases, specifically to break that confound: no single
# feature (e.g. "is this drug antibody-tractable") should be able to
# perfectly separate hypotheses once there's this much mechanistic spread.
# Every ID below was verified against the live ChEMBL and Open Targets APIs
# before being hardcoded (same discipline as EGFR/ABL1), not recalled from
# memory — see the project's commit history for the verification commands.
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
    Hypothesis(
        name="BRAF / vemurafenib / melanoma",
        gene_symbol="BRAF",
        ensembl_target_id="ENSG00000157764",
        chembl_target_id="CHEMBL5145",
        drug_aliases=["vemurafenib", "zelboraf", "plx4032", "rg7204"],
        ctgov_condition_query="melanoma",
    ),
    Hypothesis(
        # The one antibody in the set (vs. everything else here being a
        # small molecule) — this is exactly the kind of mechanistic
        # diversity the confound fix above depends on.
        name="ERBB2 / trastuzumab / breast cancer",
        gene_symbol="ERBB2",
        ensembl_target_id="ENSG00000141736",
        chembl_target_id="CHEMBL1824",
        drug_aliases=["trastuzumab", "herceptin"],
        ctgov_condition_query="breast cancer",
    ),
    Hypothesis(
        name="KDR / sunitinib / renal cell carcinoma",
        gene_symbol="KDR",
        ensembl_target_id="ENSG00000128052",
        chembl_target_id="CHEMBL279",
        drug_aliases=["sunitinib", "sutent", "su11248"],
        ctgov_condition_query="renal cell carcinoma",
    ),
    Hypothesis(
        name="PARP1 / olaparib / ovarian cancer",
        gene_symbol="PARP1",
        ensembl_target_id="ENSG00000143799",
        chembl_target_id="CHEMBL3105",
        drug_aliases=["olaparib", "lynparza", "azd2281"],
        ctgov_condition_query="ovarian cancer",
    ),
    Hypothesis(
        # Also an antibody, and a fundamentally different mechanism (immune
        # checkpoint blockade, not direct target inhibition) from every
        # other hypothesis here. ChEMBL bioactivity coverage for PD-1 is
        # expected to be sparse-to-absent (ChEMBL's assays are overwhelmingly
        # small-molecule potency data; an antibody-target's chembl_* features
        # will likely fall back to target-level or come back empty) — that's
        # a real, expected limitation of this data source for this
        # hypothesis, not a pipeline bug, and build_feature_table's existing
        # chembl_matched_by_molecule_name=False fallback already handles it.
        name="PDCD1 / pembrolizumab / melanoma",
        gene_symbol="PDCD1",
        ensembl_target_id="ENSG00000188389",
        chembl_target_id="CHEMBL3307223",
        drug_aliases=["pembrolizumab", "keytruda", "mk-3475"],
        ctgov_condition_query="melanoma",
    ),
    Hypothesis(
        # New disease area (prostate) and new mechanism class (a nuclear
        # hormone receptor antagonist, not a kinase/antibody/PARP inhibitor)
        # — added specifically for disease diversity, not just target count.
        name="AR / enzalutamide / prostate cancer",
        gene_symbol="AR",
        ensembl_target_id="ENSG00000169083",
        chembl_target_id="CHEMBL1871",
        drug_aliases=["enzalutamide", "xtandi", "mdv3100"],
        ctgov_condition_query="prostate cancer",
    ),
    Hypothesis(
        # New disease area (CLL, distinct from ABL1's CML) and new
        # mechanism (BTK inhibitor) — a blood cancer outside the CML/AML
        # space this set otherwise lacks.
        name="BTK / ibrutinib / chronic lymphocytic leukemia",
        gene_symbol="BTK",
        ensembl_target_id="ENSG00000010671",
        chembl_target_id="CHEMBL5251",
        drug_aliases=["ibrutinib", "imbruvica", "pci-32765"],
        ctgov_condition_query="chronic lymphocytic leukemia",
    ),
    Hypothesis(
        # New disease area (multiple myeloma) and a third antibody in the
        # set. Open Targets names this disease "plasma cell myeloma" (see
        # entity_resolution.py's _PLASMA_CELL_MYELOMA_SYNONYM, found by
        # checking this exact hypothesis against the live API before
        # hardcoding it — same discipline as CML's myelogenous/myeloid gap).
        name="CD38 / daratumumab / multiple myeloma",
        gene_symbol="CD38",
        ensembl_target_id="ENSG00000004468",
        chembl_target_id="CHEMBL4660",
        drug_aliases=["daratumumab", "darzalex"],
        ctgov_condition_query="multiple myeloma",
    ),
    Hypothesis(
        # Same disease as KDR (renal cell carcinoma) but a different
        # mechanism (mTOR inhibitor, not VEGFR) — deliberately included to
        # give the model same-disease/different-mechanism contrast, not
        # just different-disease contrast.
        name="MTOR / everolimus / renal cell carcinoma",
        gene_symbol="MTOR",
        ensembl_target_id="ENSG00000198793",
        chembl_target_id="CHEMBL2842",
        drug_aliases=["everolimus", "afinitor", "rad001"],
        ctgov_condition_query="renal cell carcinoma",
    ),
    Hypothesis(
        # Same disease as ERBB2 (breast cancer) but a different mechanism
        # (CDK4/6, cell-cycle inhibition) — the other same-disease/
        # different-mechanism pair, alongside MTOR/KDR above.
        name="CDK4 / palbociclib / breast cancer",
        gene_symbol="CDK4",
        ensembl_target_id="ENSG00000135446",
        chembl_target_id="CHEMBL331",
        drug_aliases=["palbociclib", "ibrance", "pd-0332991", "pd0332991"],
        ctgov_condition_query="breast cancer",
    ),
    Hypothesis(
        # Same disease as EGFR (NSCLC) but a different target/mechanism —
        # gives the model a direct within-disease contrast for the very
        # first hypothesis in this set, the way MTOR/KDR and CDK4/ERBB2
        # already contrast within RCC and breast cancer.
        name="ALK / crizotinib / NSCLC",
        gene_symbol="ALK",
        ensembl_target_id="ENSG00000171094",
        chembl_target_id="CHEMBL4247",
        drug_aliases=["crizotinib", "xalkori", "pf-02341066"],
        ctgov_condition_query="non-small cell lung cancer",
    ),
    Hypothesis(
        # New disease area (acute myeloid leukemia) and new mechanism
        # (FLT3 inhibitor) — the third blood-cancer hypothesis (alongside
        # ABL1/CML and BTK/CD38's diseases) but the first in AML
        # specifically.
        name="FLT3 / midostaurin / acute myeloid leukemia",
        gene_symbol="FLT3",
        ensembl_target_id="ENSG00000122025",
        chembl_target_id="CHEMBL1974",
        drug_aliases=["midostaurin", "rydapt", "pkc412"],
        ctgov_condition_query="acute myeloid leukemia",
    ),
    Hypothesis(
        # Same disease as BTK (CLL) but a different mechanism (BCL2/
        # apoptosis-regulator inhibitor, not a kinase) — another
        # within-disease contrast pair.
        name="BCL2 / venetoclax / chronic lymphocytic leukemia",
        gene_symbol="BCL2",
        ensembl_target_id="ENSG00000171791",
        chembl_target_id="CHEMBL4860",
        drug_aliases=["venetoclax", "venclexta", "venclyxto", "abt-199", "gdc-0199"],
        ctgov_condition_query="chronic lymphocytic leukemia",
    ),
    Hypothesis(
        # New disease area (bladder cancer) and new mechanism (FGFR
        # inhibitor). Open Targets' bladder disease entries are prefixed
        # "urinary bladder ..." — checked against the live API before
        # hardcoding this hypothesis and found a fourth instance of the
        # naming-mismatch pattern (see entity_resolution.py's
        # _URINARY_BLADDER_SYNONYM).
        name="FGFR3 / erdafitinib / bladder cancer",
        gene_symbol="FGFR3",
        ensembl_target_id="ENSG00000068078",
        chembl_target_id="CHEMBL2742",
        drug_aliases=["erdafitinib", "balversa"],
        ctgov_condition_query="bladder cancer",
    ),
    Hypothesis(
        # New disease area (colorectal cancer) and a fourth antibody, but
        # against the VEGF *ligand* (VEGFA) rather than its receptor (KDR,
        # already in this set) — a real, meaningfully different target
        # despite the same pathway, not a duplicate of the KDR hypothesis.
        name="VEGFA / bevacizumab / colorectal cancer",
        gene_symbol="VEGFA",
        ensembl_target_id="ENSG00000112715",
        chembl_target_id="CHEMBL1783",
        drug_aliases=["bevacizumab", "avastin"],
        ctgov_condition_query="colorectal cancer",
    ),
]

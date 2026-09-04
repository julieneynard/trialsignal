"""Entity resolution: the unglamorous problem that determines whether this
project's joins mean anything.

Three sources, three ID systems, and none of them agree:
  - ClinicalTrials.gov conditions/interventions are free text ("NSCLC",
    "Non-Small Cell Lung Cancer", "Stage IV NSCLC" all refer to overlapping
    but not identical things).
  - Open Targets uses EFO (disease) and Ensembl gene IDs.
  - ChEMBL uses its own compound and target ChEMBL IDs.

A naive `condition_string == disease_name` join silently drops most trials
and silently mismatches the rest. This module makes the matching explicit,
scored, and reviewable instead of implicit — and refuses to guess past a
confidence threshold rather than emitting a wrong-but-confident join.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

# Curated overrides for high-volume oncology abbreviations that string
# similarity alone won't bridge (e.g. "NSCLC" vs "non-small cell lung
# carcinoma" score low on naive similarity despite being the same disease).
# This table is deliberately small and human-reviewed rather than an attempt
# at a general abbreviation expander — false expansions are worse than none.
#
# Values must already be in normalized form (lowercase, no punctuation) —
# they're substituted in as the final step of normalize_condition_text, so
# an un-normalized value (e.g. a literal hyphen) would make the same disease
# normalize differently depending on whether it arrived via alias expansion
# or via the general punctuation-stripping path, silently breaking matches.
CONDITION_ALIASES: dict[str, str] = {
    "nsclc": "non small cell lung cancer",
    "sclc": "small cell lung cancer",
    "aml": "acute myeloid leukemia",
    "cll": "chronic lymphocytic leukemia",
    "cml": "chronic myeloid leukemia",
    "hcc": "hepatocellular cancer",
    "rcc": "renal cell cancer",
    "tnbc": "triple negative breast cancer",
    "gbm": "glioblastoma",
    "mm": "multiple myeloma",
    "dlbcl": "diffuse large b cell lymphoma",
}

_STAGE_QUALIFIER = re.compile(
    r"\b(stage\s+[ivx0-9]+[ab]?|metastatic|advanced|recurrent|relapsed|refractory|unresectable)\b",
    re.IGNORECASE,
)
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")

# CT.gov condition text overwhelmingly says "cancer" (colloquial); Open
# Targets/EFO disease names overwhelmingly say "carcinoma" (formal, and
# technically epithelial-origin-specific). Measured directly against the
# live Open Targets API: "non-small cell lung cancer" vs "...carcinoma"
# scores 0.84, "renal cell cancer" vs "...carcinoma" scores 0.76 — both
# below any reasonable confidence threshold despite being the same disease
# entity in the trial-matching context this module cares about. Collapsing
# "carcinoma" -> "cancer" is a deliberate, documented simplification (not
# medically precise — carcinoma is technically a cancer subtype) made
# because it fixes a systematic false-negative across most oncology
# indications; it does NOT touch other subtype-specific terms (adenocarcinoma,
# sarcoma, lymphoma) that carry meaningfully different information.
_CARCINOMA_SYNONYM = re.compile(r"\bcarcinoma\b", re.IGNORECASE)


def normalize_condition_text(raw: str) -> str:
    """Lowercase, strip punctuation, expand known abbreviations, and drop
    stage/severity qualifiers that describe the trial population rather than
    the disease entity itself (EFO disease names don't carry stage info)."""
    text = raw.strip().lower()
    text = _STAGE_QUALIFIER.sub("", text)
    text = _CARCINOMA_SYNONYM.sub("cancer", text)
    text = _NON_ALNUM.sub(" ", text)
    text = " ".join(text.split())
    return CONDITION_ALIASES.get(text, text)


@dataclass(frozen=True)
class DiseaseMatch:
    efo_id: str
    matched_name: str
    score: float
    confident: bool


def resolve_condition_to_efo(
    condition: str,
    efo_candidates: list[tuple[str, str]],
    *,
    confidence_threshold: float = 0.85,
) -> DiseaseMatch | None:
    """Find the best EFO disease match for a free-text CT.gov condition
    string among a candidate list of (efo_id, disease_name) pairs.

    Returns None when there are no candidates at all. Otherwise always
    returns the best match found, but flags it `confident=False` below the
    threshold — callers must decide whether to keep low-confidence matches
    (e.g. for exploratory analysis) or drop them (e.g. for training labels).
    Silently accepting a low-confidence match is how a join quietly
    corrupts the dataset; this makes that choice explicit at the call site.
    """
    if not efo_candidates:
        return None

    normalized_condition = normalize_condition_text(condition)
    best_id, best_name, best_score = "", "", -1.0
    for efo_id, name in efo_candidates:
        score = SequenceMatcher(None, normalized_condition, normalize_condition_text(name)).ratio()
        if score > best_score:
            best_id, best_name, best_score = efo_id, name, score

    return DiseaseMatch(
        efo_id=best_id,
        matched_name=best_name,
        score=best_score,
        confident=best_score >= confidence_threshold,
    )


def normalize_gene_symbol(raw: str) -> str:
    """HGNC gene symbols are case-sensitive by convention but appear in every
    casing imaginable across free-text sources; uppercase + strip is the
    only normalization safe to apply without a full HGNC lookup table."""
    return raw.strip().upper()

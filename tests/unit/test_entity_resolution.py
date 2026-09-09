from trialsignal.data.entity_resolution import (
    normalize_condition_text,
    normalize_gene_symbol,
    resolve_condition_to_efo,
)


def test_normalize_expands_known_abbreviation() -> None:
    assert normalize_condition_text("NSCLC") == "non small cell lung cancer"


def test_normalize_strips_stage_qualifiers() -> None:
    assert normalize_condition_text("Stage IV NSCLC") == "non small cell lung cancer"
    assert normalize_condition_text("Metastatic Renal Cell Carcinoma") == "renal cell cancer"


def test_normalize_collapses_carcinoma_to_cancer() -> None:
    """CT.gov condition text says "cancer" (colloquial); Open Targets/EFO
    disease names say "carcinoma" (formal) — collapsing them is what makes
    the two sources' disease names comparable at all. See the module-level
    comment on _CARCINOMA_SYNONYM for the measured before/after scores."""
    assert normalize_condition_text("Non-Small Cell Lung Cancer") == normalize_condition_text(
        "non-small cell lung carcinoma"
    )
    assert normalize_condition_text("Hepatocellular Carcinoma") == "hepatocellular cancer"


def test_normalize_collapses_myelogenous_to_myeloid() -> None:
    """CT.gov trials say "chronic myeloid leukemia"; Open Targets' actual
    disease entry is "chronic myelogenous leukemia, BCR-ABL1 positive" — a
    real gap found by running the ABL1/imatinib/CML hypothesis end-to-end
    through the live /score endpoint (see _MYELOGENOUS_SYNONYM's module
    comment)."""
    assert normalize_condition_text("Chronic Myeloid Leukemia") == normalize_condition_text(
        "Chronic Myelogenous Leukemia"
    )


def test_normalize_collapses_plasma_cell_myeloma_to_multiple_myeloma() -> None:
    """CT.gov trials say "Multiple Myeloma"; Open Targets' actual disease
    entry is "plasma cell myeloma" (formal synonym) — a third instance of
    the same naming-mismatch class, found by checking the CD38/daratumumab
    hypothesis against the live API before hardcoding it (see
    _PLASMA_CELL_MYELOMA_SYNONYM's module comment)."""
    assert normalize_condition_text("Multiple Myeloma") == normalize_condition_text(
        "Plasma Cell Myeloma"
    )


def test_normalize_does_not_collapse_smoldering_myeloma() -> None:
    """Smoldering myeloma is a distinct precursor condition, not the same
    disease as active multiple myeloma — the synonym rule must not
    conflate them just because both contain "myeloma"."""
    assert normalize_condition_text("Smoldering Plasma Cell Myeloma") != normalize_condition_text(
        "Multiple Myeloma"
    )


def test_normalize_collapses_urinary_bladder_to_bladder() -> None:
    """CT.gov trials say "Bladder Cancer"; Open Targets' actual disease
    entry is "urinary bladder cancer" (anatomically formal prefix) — a
    fourth instance of the same naming-mismatch class, found by checking
    the FGFR3/erdafitinib hypothesis against the live API before
    hardcoding it (see _URINARY_BLADDER_SYNONYM's module comment)."""
    assert normalize_condition_text("Bladder Cancer") == normalize_condition_text(
        "Urinary Bladder Cancer"
    )


def test_normalize_strips_biomarker_positivity_qualifier() -> None:
    assert (
        normalize_condition_text("Chronic Myelogenous Leukemia, BCR-ABL1 Positive")
        == "chronic myeloid leukemia"
    )
    assert normalize_condition_text("Breast Cancer, HER2 Negative") == "breast cancer"


def test_normalize_strips_punctuation_and_case() -> None:
    raw = "  Multiple Myeloma, Relapsed/Refractory  "
    assert normalize_condition_text(raw) == "multiple myeloma"


def test_resolve_exact_match_after_normalization_is_confident() -> None:
    candidates = [
        ("EFO_0003060", "non-small cell lung carcinoma"),
        ("EFO_9999999", "unrelated disease"),
    ]
    match = resolve_condition_to_efo("Stage IIIB NSCLC", candidates)

    assert match is not None
    assert match.efo_id == "EFO_0003060"
    assert match.confident is True
    assert match.score == 1.0


def test_resolve_weak_match_is_not_confident() -> None:
    candidates = [("EFO_1111111", "psoriatic arthritis")]
    match = resolve_condition_to_efo("non-small cell lung carcinoma", candidates)

    assert match is not None
    assert match.confident is False


def test_resolve_returns_none_for_empty_candidate_list() -> None:
    assert resolve_condition_to_efo("nsclc", []) is None


def test_normalize_gene_symbol_uppercases_and_strips() -> None:
    assert normalize_gene_symbol(" egfr ") == "EGFR"

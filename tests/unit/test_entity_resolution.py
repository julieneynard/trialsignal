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

from trialsignal.data.entity_resolution import (
    normalize_condition_text,
    normalize_gene_symbol,
    resolve_condition_to_efo,
)


def test_normalize_expands_known_abbreviation() -> None:
    assert normalize_condition_text("NSCLC") == "non small cell lung carcinoma"


def test_normalize_strips_stage_qualifiers() -> None:
    assert normalize_condition_text("Stage IV NSCLC") == "non small cell lung carcinoma"
    assert normalize_condition_text("Metastatic Renal Cell Carcinoma") == "renal cell carcinoma"


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

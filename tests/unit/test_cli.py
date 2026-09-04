import csv
from pathlib import Path

from typer.testing import CliRunner

from trialsignal.cli import app
from trialsignal.data.schemas import (
    ChemblActivity,
    TargetDiseaseAssociation,
    TrialPhase,
    TrialRecord,
    TrialStatus,
)

runner = CliRunner()


def _write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def test_list_hypotheses_prints_curated_names() -> None:
    result = runner.invoke(app, ["list-hypotheses"])
    assert result.exit_code == 0
    assert "EGFR / osimertinib / NSCLC" in result.stdout
    assert "ABL1 / imatinib / CML" in result.stdout


def test_build_features_end_to_end_writes_csv(tmp_path: Path) -> None:
    trial = TrialRecord(
        nct_id="NCT001",
        title="t",
        status=TrialStatus.COMPLETED,
        interventions=["Osimertinib"],
        conditions=["Non-Small Cell Lung Cancer"],
        phases=[TrialPhase.PHASE2],
    )
    target_disease = TargetDiseaseAssociation(
        target_id="ENSG00000146648",
        target_symbol="EGFR",
        disease_id="MONDO_0005233",
        disease_name="non-small cell lung carcinoma",
        overall_score=0.85,
    )
    activity = ChemblActivity(
        molecule_chembl_id="CHEMBL1",
        pref_name="OSIMERTINIB",
        target_chembl_id="CHEMBL203",
        standard_type="IC50",
        pchembl_value=8.0,
    )

    trials_path = tmp_path / "trials.jsonl"
    diseases_path = tmp_path / "diseases.jsonl"
    activities_path = tmp_path / "activities.jsonl"
    output_path = tmp_path / "features.csv"
    _write_jsonl(trials_path, [trial])
    _write_jsonl(diseases_path, [target_disease])
    _write_jsonl(activities_path, [activity])

    result = runner.invoke(
        app,
        [
            "build-features",
            "EGFR / osimertinib / NSCLC",
            "--trials-path",
            str(trials_path),
            "--target-diseases-path",
            str(diseases_path),
            "--activities-path",
            str(activities_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "1/1 pulled trials matched" in result.stdout
    assert output_path.exists()

    with output_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["nct_id"] == "NCT001"
    assert rows[0]["label"] == "success"


def test_build_features_unknown_hypothesis_exits_nonzero(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")

    result = runner.invoke(
        app,
        [
            "build-features",
            "not a real hypothesis",
            "--trials-path",
            str(empty),
            "--target-diseases-path",
            str(empty),
            "--activities-path",
            str(empty),
        ],
    )

    assert result.exit_code == 1
    assert "Unknown hypothesis" in result.output

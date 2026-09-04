"""Command-line entry point: `trialsignal <command>`."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from pydantic import BaseModel

from trialsignal.data.chembl import ChemblClient
from trialsignal.data.clinicaltrials import ClinicalTrialsClient
from trialsignal.data.open_targets import OpenTargetsClient
from trialsignal.data.schemas import ChemblActivity, TargetDiseaseAssociation, TrialRecord
from trialsignal.features.build_features import build_feature_table
from trialsignal.features.hypothesis import CURATED_HYPOTHESES

app = typer.Typer(help="TrialSignal: clinical trial outcome risk modeling pipeline.")

_ModelT = TypeVar("_ModelT", bound=BaseModel)

_CONDITION_HELP = "CT.gov condition query, e.g. 'non-small cell lung cancer'"
_OUTPUT_HELP = "Where to write newline-delimited JSON."
_MAX_PAGES_HELP = "Cap pages pulled (default page size 100); omit for all pages."


def _write_jsonl(records: Iterable[BaseModel], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
            count += 1
    return count


def _read_jsonl(path: Path, model: type[_ModelT]) -> list[_ModelT]:
    with path.open(encoding="utf-8") as f:
        return [model.model_validate_json(line) for line in f if line.strip()]


@app.command()
def fetch_trials(
    condition: Annotated[str, typer.Argument(help=_CONDITION_HELP)],
    output: Annotated[Path, typer.Option(help=_OUTPUT_HELP)] = Path("data/raw/trials.jsonl"),
    max_pages: Annotated[int | None, typer.Option(help=_MAX_PAGES_HELP)] = None,
) -> None:
    """Pull every ClinicalTrials.gov study matching `condition` and write it
    as newline-delimited JSON, one TrialRecord per line."""
    client = ClinicalTrialsClient()
    count = _write_jsonl(client.iter_studies(condition, max_pages=max_pages), output)
    client.close()
    typer.echo(f"Wrote {count} trial records to {output}")


@app.command()
def fetch_target(
    ensembl_id: Annotated[str, typer.Argument(help="Ensembl gene ID, e.g. ENSG00000146648")],
    output: Annotated[
        Path, typer.Option(help=_OUTPUT_HELP)
    ] = Path("data/raw/target_diseases.jsonl"),
    max_pages: Annotated[int | None, typer.Option(help=_MAX_PAGES_HELP)] = None,
) -> None:
    """Pull every Open Targets target-disease association for `ensembl_id`."""
    client = OpenTargetsClient()
    count = _write_jsonl(client.iter_target_diseases(ensembl_id, max_pages=max_pages), output)
    client.close()
    typer.echo(f"Wrote {count} target-disease associations to {output}")


@app.command()
def fetch_activities(
    target_chembl_id: Annotated[str, typer.Argument(help="ChEMBL target ID, e.g. CHEMBL203")],
    output: Annotated[Path, typer.Option(help=_OUTPUT_HELP)] = Path("data/raw/activities.jsonl"),
    standard_type: Annotated[str, typer.Option(help="IC50 / EC50 / Ki / ...")] = "IC50",
    max_pages: Annotated[int | None, typer.Option(help=_MAX_PAGES_HELP)] = None,
) -> None:
    """Pull every ChEMBL bioactivity record for `target_chembl_id`."""
    client = ChemblClient()
    records = client.iter_activities(
        target_chembl_id, standard_type=standard_type, max_pages=max_pages
    )
    count = _write_jsonl(records, output)
    client.close()
    typer.echo(f"Wrote {count} bioactivity records to {output}")


@app.command()
def list_hypotheses() -> None:
    """List the curated target/drug/disease hypotheses build-features can use."""
    for h in CURATED_HYPOTHESES:
        typer.echo(f"{h.name}  (gene={h.gene_symbol}, condition query={h.ctgov_condition_query!r})")


@app.command()
def build_features(
    hypothesis_name: Annotated[
        str, typer.Argument(help="Exact `name` of a hypothesis from `list-hypotheses`.")
    ],
    trials_path: Annotated[Path, typer.Option(help="Output of fetch-trials.")],
    target_diseases_path: Annotated[Path, typer.Option(help="Output of fetch-target.")],
    activities_path: Annotated[Path, typer.Option(help="Output of fetch-activities.")],
    output: Annotated[Path, typer.Option(help="Where to write the feature table (CSV).")] = Path(
        "data/processed/features.csv"
    ),
) -> None:
    """Join trials + target-disease evidence + bioactivity into a training-
    ready feature table for one curated hypothesis. Reports coverage
    (how many pulled trials actually made it into the table) rather than
    hiding the drop — see build_features.py's module docstring for why
    trials get dropped."""
    hypothesis = next((h for h in CURATED_HYPOTHESES if h.name == hypothesis_name), None)
    if hypothesis is None:
        names = ", ".join(h.name for h in CURATED_HYPOTHESES)
        typer.echo(f"Unknown hypothesis {hypothesis_name!r}. Available: {names}", err=True)
        raise typer.Exit(code=1)

    trials = _read_jsonl(trials_path, TrialRecord)
    target_diseases = _read_jsonl(target_diseases_path, TargetDiseaseAssociation)
    activities = _read_jsonl(activities_path, ChemblActivity)

    rows = build_feature_table(hypothesis, trials, target_diseases, activities)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        if rows:
            fieldnames = list(rows[0].model_dump().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.model_dump())

    n_success = sum(1 for r in rows if r.label == "success")
    n_failure = len(rows) - n_success
    typer.echo(
        f"{len(rows)}/{len(trials)} pulled trials matched this hypothesis and produced a "
        f"labeled row ({n_success} success, {n_failure} failure). Wrote {output}."
    )


@app.command()
def train(
    feature_tables: Annotated[
        list[Path], typer.Argument(help="One or more CSV outputs from build-features.")
    ],
    eval_mode: Annotated[
        str, typer.Option(help="'temporal' (default, correct) or 'cv' (small-data fallback).")
    ] = "temporal",
    cutoff: Annotated[
        str, typer.Option(help="Temporal train/test split date, YYYY-MM-DD (temporal mode only).")
    ] = "2022-01-01",
    output: Annotated[Path, typer.Option(help="Where to save the trained model.")] = Path(
        "models/trialsignal_model.joblib"
    ),
    model_version: Annotated[
        str, typer.Option(help="Version tag stored with the model.")
    ] = "0.1.0",
) -> None:
    """Train baseline (logistic regression) and LightGBM models on one or
    more feature tables and save the LightGBM pipeline for serving.

    `--eval-mode temporal` (default) is the methodologically correct
    approach (see METHODS.md) but requires enough data that both classes
    appear on both sides of the cutoff — it raises
    InsufficientClassDiversityError with a clear explanation when a small,
    imbalanced dataset can't support that. `--eval-mode cv` is the
    documented small-data fallback (see train.py's cross_validate_lightgbm
    docstring for what it gives up to get there)."""
    from trialsignal.models.train import (
        InsufficientClassDiversityError,
        cross_validate_lightgbm,
        load_feature_table,
        save_model_bundle,
        train_and_evaluate,
    )

    df = load_feature_table(feature_tables)

    if eval_mode == "temporal":
        try:
            _, lightgbm_model, report = train_and_evaluate(df, cutoff=cutoff)
        except InsufficientClassDiversityError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
    elif eval_mode == "cv":
        lightgbm_model, report = cross_validate_lightgbm(df)
    else:
        typer.echo(f"Unknown --eval-mode {eval_mode!r}. Use 'temporal' or 'cv'.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"n_total={report.n_total}  n_train={report.n_train}  n_test={report.n_test}")
    results = [("baseline (logreg)", report.baseline_eval), ("lightgbm", report.lightgbm_eval)]
    for label, result in results:
        auc = f"{result.roc_auc:.3f}" if result.roc_auc is not None else "n/a"
        pr = f"{result.pr_auc:.3f}" if result.pr_auc is not None else "n/a"
        note = f"  [{result.note}]" if result.note else ""
        typer.echo(
            f"  {label}: roc_auc={auc} pr_auc={pr} brier={result.brier_score:.3f} "
            f"(n_test={result.n_test}: {result.n_test_success} success / "
            f"{result.n_test_failure} failure){note}"
        )
    shap_summary = ", ".join(f"{n}={v:.3f}" for n, v in report.top_shap_features)
    typer.echo(f"Top SHAP features (lightgbm): {shap_summary}")

    save_model_bundle(lightgbm_model, output, model_version)
    typer.echo(f"Saved model to {output}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI scoring service."""
    import uvicorn

    uvicorn.run("trialsignal.serving.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()

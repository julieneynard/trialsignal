"""Command-line entry point: `trialsignal <command>`."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from trialsignal.data.chembl import ChemblClient
from trialsignal.data.clinicaltrials import ClinicalTrialsClient
from trialsignal.data.open_targets import OpenTargetsClient

app = typer.Typer(help="TrialSignal: clinical trial outcome risk modeling pipeline.")

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
def train() -> None:
    """Train the trial-progression risk model. Pending: requires the
    Open Targets / ChEMBL feature pipeline (see README roadmap)."""
    typer.echo(
        "Training pipeline is not wired up yet — the feature-building step "
        "(entity resolution -> Open Targets / ChEMBL feature join) is the "
        "current milestone. See README.md roadmap.",
        err=True,
    )
    raise typer.Exit(code=1)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI scoring service."""
    import uvicorn

    uvicorn.run("trialsignal.serving.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()

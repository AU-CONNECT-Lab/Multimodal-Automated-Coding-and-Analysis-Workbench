from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from macaw.coding.agreement import (
    add_words,
    agreement_report,
    format_report,
    load_annotations,
    write_report,
)
from macaw.coding.describe import describe_codes, format_description
from macaw.coding.imports import import_coding
from macaw.coding.review_queue import apply_review, export_review, queue_summary, random_check_result, read_queue, write_queue
from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution import runner
from macaw.execution.runner import resolve_recordings
from macaw.interop.coding_files import SUFFIXES
from macaw.project.recording import build_recording

app = typer.Typer()


_LAYER_STEPS = {"A": "coding.layer_a", "B": "coding.judge"}


@app.command("code")
def code(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    layer: list[str] = typer.Option(None, "--layer", "-l", help="A or B; defaults to coding.layers."),
    codebook: Path = typer.Option(None, "--codebook", help="Overrides coding.codebook; required for layer B."),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Apply rule-based (A) and LLM (B) coding. Low-confidence items go to the review queue."""
    resolved = resolve_config(config, cli_overrides=set_ or ())
    if codebook is not None:
        resolved.coding.codebook = codebook
    layers = [name.upper() for name in (layer or resolved.coding.layers)]
    unknown = [name for name in layers if name not in _LAYER_STEPS]
    if unknown:
        raise MacawError(f"Layer(s) {', '.join(unknown)} cannot be coded; available: {', '.join(_LAYER_STEPS)}")
    steps = [_LAYER_STEPS[name] for name in layers]
    if "B" in layers:
        from macaw.coding.llm.judge import version_steps

        steps += version_steps(resolved)
    summary = runner.run(resolved, recordings=resolve_recordings(resolved, recordings), extractors=steps)
    typer.echo(f"tasks {summary.n_tasks}  done {summary.n_done}  cached {summary.n_cached}  failed {summary.n_failed}")
    if "B" in layers:
        pending = len(export_review(resolved, resolve_recordings(resolved, recordings)))
        typer.echo(f"{pending} unit(s) wait for review: macaw review --export queue.csv")
    if summary.n_failed:
        raise typer.Exit(code=1)


@app.command("review")
def review(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    export_to: Path = typer.Option(None, "--export", help="Write the queue to a file (.csv to edit by hand)."),
    apply_from: Path = typer.Option(None, "--apply", help="Merge back reviewed decisions."),
) -> None:
    """Export or re-import the human review queue. Fill in decision (a code) and reviewer, then --apply."""
    if (export_to is None) == (apply_from is None):
        raise MacawError("Give one of --export <file> or --apply <file>.")
    resolved = resolve_config(config)
    if export_to is not None:
        queue = export_review(resolved, resolve_recordings(resolved, recordings))
        write_queue(queue, export_to)
        summary = queue_summary(queue)
        reasons = ", ".join(f"{count} {reason}" for reason, count in summary["by_reason"].items())
        typer.echo(f"{summary['n_items']} unit(s) to review ({reasons or 'none'}) written to {export_to}")
        return
    counts = apply_review(resolved, read_queue(apply_from))
    if not counts:
        typer.echo("No decisions filled in; nothing applied.")
    for recording_id, count in counts.items():
        typer.echo(f"{recording_id}: {count}")
    checked, changed = random_check_result(resolved, resolve_recordings(resolved, recordings))
    if checked:
        typer.echo(f"random check: a person coded {changed} of {checked} units the model was sure of otherwise")


@app.command("agreement")
def agreement(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    tier: str = typer.Option(None, "--tier", help="Restrict to one annotation tier."),
    layer: str = typer.Option(None, "--layer", "-l", help="A, B, or C."),
    grid_s: float = typer.Option(None, "--grid-s", help="Compare on a fixed time grid instead of on exact intervals."),
    units: str = typer.Option(
        None, "--units", help="Compare on this coder's intervals: each takes every other coder's code covering most of it."
    ),
    out: Path = typer.Option(
        None, "--out", help="A folder for report.json and units.csv, each unit with every coder's code and its words."
    ),
    samples: bool = typer.Option(
        False, "--samples", help="Also count each answer of the language model as a coder, <coder>#1, #2 ..., to set it against itself."
    ),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Inter-annotator agreement over the annotations table: Krippendorff alpha and Cohen kappa."""
    if grid_s is not None and units is not None:
        raise MacawError("give --grid-s or --units, not both")
    if out is not None and out.is_file():
        raise MacawError(f"{out} is a file; --out takes a folder")
    resolved = resolve_config(config, cli_overrides=set_ or ())
    chosen = resolve_recordings(resolved, recordings)
    annotations = load_annotations(resolved, chosen)
    if samples:
        from macaw.coding.llm.judge import sample_annotations

        annotations = pd.concat([annotations, sample_annotations(resolved, chosen)], ignore_index=True)
    report = agreement_report(annotations, tier=tier, layer=layer, grid_s=grid_s, units_of=units)
    typer.echo(format_report(report))
    if out is not None:
        if grid_s is None:
            report["units"] = add_words(resolved, report["units"])
        write_report(report, out)
        typer.echo(f"\nreport.json and units.csv in {out}")


@app.command("import-codes")
def import_codes(
    paths: list[Path] = typer.Argument(..., help="Coding files (.eaf, .csv, .tsv, .xlsx), or folders of them."),
    coder: str = typer.Option(..., "--coder", help="Who coded. Importing again replaces this coder's codes on the same tiers."),
    recording: str = typer.Option(None, "--recording", "-r", help="The recording, when one file's name does not start with it."),
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
) -> None:
    """Bring your own coding in beside the machine's, for macaw agreement, describe-codes and the gesture measures.
    coding.imports says whose each tier is and what it is called here. Each file's name starts with its recording."""
    resolved = resolve_config(config)
    files = [
        found
        for path in paths
        for found in (sorted(item for item in path.iterdir() if item.suffix.lower() in SUFFIXES) if path.is_dir() else [path])
    ]
    if not files:
        raise MacawError(f"no coding files ({', '.join(SUFFIXES)}) there")
    if recording is not None and len(files) > 1:
        raise MacawError("-r names the recording of one file; for several, start each file's name with its recording")
    known = resolve_recordings(resolved, None)
    owners = {path: recording or _recording_of(path, known) for path in files}
    unnamed = [path.name for path, owner in owners.items() if owner is None]
    if unnamed:
        raise MacawError(f"{', '.join(unnamed)}: the name does not start with a recording; rename the file or give -r")
    for path, owner in owners.items():
        own = resolve_config(config, recording_id=owner)
        report = import_coding(own, build_recording(owner, own), path, coder)
        kept = ", ".join(f"{person or 'everyone'} {tier} {n}" for (person, tier), n in sorted(report.kept.items(), key=str))
        typer.echo(f"{owner}  {path.name}  {coder}: {kept or 'nothing brought in'}")
        for line in report.left_out:
            typer.echo(f"  left out  {line}")
        for line in report.notes:
            typer.echo(f"  note  {line}")


@app.command("draw-examples")
def draw_examples_command(
    out: Path = typer.Argument(..., help="The table to write (.csv), for coding.llm.examples."),
    coder: str = typer.Option(..., "--coder", help="Whose codes to draw from, as macaw agreement names them."),
    per_code: int = typer.Option(3, "--per-code", min=1, help="Units drawn for each code."),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Draw only from these (comma-separated ids or a txt file)."),
    seed: int = typer.Option(0, "--seed", help="Fixes the draw, so running it again picks the same units."),
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
) -> None:
    """Draw units a person coded into a table of examples the language model is shown. Open it, change what you like,
    fill in why if you want the model to read a reason, and set coding.llm.examples (or a version's) to it."""
    from macaw.coding.codebook import read_codebook
    from macaw.coding.examples import COLUMNS, draw_examples
    from macaw.coding.llm.judge import coding_units

    resolved = resolve_config(config)
    if resolved.coding.codebook is None:
        raise MacawError("set coding.codebook first: the examples are drawn for its codes")
    codebook = read_codebook(resolved.coding.codebook)
    annotations = load_annotations(resolved, resolve_recordings(resolved, recordings))
    coded = annotations[(annotations["annotator"].astype(str) == coder) & (annotations["tier"].astype(str) == codebook.name)]
    if coded.empty:
        raise MacawError(f"no codes by {coder} on the tier {codebook.name} (the codebook's name) in these recordings")
    lines = {}
    for rid in sorted(coded["recording_id"].astype(str).unique()):
        own = resolve_config(config, recording_id=rid)
        try:
            lines[rid] = coding_units(own, build_recording(rid, own), codebook.unit)
        except (MacawError, OSError) as error:
            typer.echo(f"{rid}: no transcript to take the words from, left out ({error})")
    table, report = draw_examples(coded, lines, codebook, per_code=per_code, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    table[list(COLUMNS)].to_csv(out, index=False)
    for code, (drawn, available) in report.counts.items():
        typer.echo(f"{code}: {drawn} of {available}")
    if report.no_words:
        typer.echo(f"{report.no_words} unit(s) left out: no words of that person inside them")
    for label, count in sorted(report.not_in_codebook.items()):
        typer.echo(f"{count} unit(s) coded {label}, which the codebook does not have, left out")
    typer.echo(f"{len(table)} examples written to {out}")


def _recording_of(path: Path, recordings: list[str]) -> str | None:
    """The longest recording id the file's name starts with, followed by nothing or a separator."""
    stem = path.stem
    found = [rid for rid in recordings if stem == rid or (stem.startswith(rid) and stem[len(rid)] in "_-. ")]
    return max(found, key=len) if found else None


@app.command("describe-codes")
def describe_codes_command(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    tier: str = typer.Option(None, "--tier", help="Restrict to one annotation tier."),
    layer: str = typer.Option(None, "--layer", "-l", help="A, B, or C."),
    out: Path = typer.Option(None, "--out", help="Write the table as csv."),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """How each code was used in each recording: units, time, runs of one code, and how many runs last one unit."""
    resolved = resolve_config(config, cli_overrides=set_ or ())
    annotations = load_annotations(resolved, resolve_recordings(resolved, recordings))
    if tier is not None:
        annotations = annotations[annotations["tier"] == tier]
    if layer is not None:
        annotations = annotations[annotations["layer"] == layer]
    table = describe_codes(annotations)
    typer.echo(format_description(table))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)

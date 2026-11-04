from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from macaw.coding.codebook import UNITS, read_codebook, write_codebook
from macaw.coding.llm.backends import resolve_llm_backend
from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.project.recording import build_recording

app = typer.Typer(help="Have the language model propose a clearer codebook, or draft a first one, for you to read and edit.")

OWN_WORDS = "The examples are participants' own words from the transcripts; check them before sharing the codebook."


def _backend(resolved):
    backend = resolve_llm_backend(resolved.coding.llm)
    backend.check_available()
    if backend.reporting:
        typer.echo(backend.reporting)
    return backend


def _not_the_one_in_use(resolved, out: Path) -> None:
    if resolved.coding.codebook is not None and out.resolve() == Path(resolved.coding.codebook).resolve():
        raise MacawError("give the new codebook a name of its own; the codebook in use is never written over")


@app.command("refine")
def refine(
    out: Path = typer.Argument(..., help="The proposed codebook (.yaml), written beside a list of the changes."),
    coder: str = typer.Option(None, "--coder", help="Whose codes to go by, when several people coded the codebook's tier."),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    per_kind: int = typer.Option(12, "--per-kind", min=1, help="Units of each kind the model is shown for each code."),
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
) -> None:
    """Propose clearer definitions, criteria, boundary cases and examples for the codebook in use, from the units where a
    person's codes and the model's differ, or, where nobody coded, the units the model was split on."""
    from macaw.coding.llm.refine import changes_path, changes_text, gather_evidence, refine_codebook

    resolved = resolve_config(config)
    if resolved.coding.codebook is None:
        raise MacawError("set coding.codebook first: it is the codebook to refine")
    _not_the_one_in_use(resolved, out)
    source = Path(resolved.coding.codebook)
    codebook = read_codebook(source)
    evidence = gather_evidence(resolved, codebook, resolve_recordings(resolved, recordings), coder=coder)
    typer.echo(f"going by {f'the codes of {evidence.coder}' if evidence.coder else 'the model alone; nobody coded the tier'}, {len(evidence.units)} units")
    backend = _backend(resolved)
    cfg = resolved.coding.llm
    refined, revisions = refine_codebook(
        backend, codebook, evidence, seed=cfg.seed, per_kind=per_kind, notes=cfg.notes, takes_seed=backend.takes_seed,
    )
    write_codebook(refined, out)
    changes = changes_path(out)
    changes.write_text(changes_text(source, evidence, revisions, backend.model_version()), encoding="utf-8")
    for revision in revisions:
        shown = int(revision.shown["kind"].isin(["missed", "taken", "split"]).sum()) if not revision.shown.empty else 0
        typer.echo(f"{revision.code}: {'revised' if revision.changed else 'unchanged'}, {shown} unit(s) in doubt")
    typer.echo(f"\n{out}\n{changes}\n{OWN_WORDS}")
    typer.echo(f"To code with it beside the codebook in use, add it as a version: coding.judges: {{refined: {{codebook: {out.as_posix()}}}}}")


@app.command("draft")
def draft(
    out: Path = typer.Argument(..., help="The draft codebook (.yaml), written beside a note of how each code came about."),
    aim: str = typer.Option(..., "--aim", help="What the codes are for, in a sentence; the model describes each unit with it in mind."),
    name: str = typer.Option(None, "--name", help="The codebook's name, which becomes its tier. Default: the file's name."),
    unit: str = typer.Option("utterance", "--unit", help="utterance or turn."),
    n_units: int = typer.Option(200, "--units", min=1, help="Units drawn at random across the recordings."),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    seed: int = typer.Option(0, "--seed", help="Fixes the draw, so running it again draws the same units."),
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
) -> None:
    """Draft a first codebook: the model says in a few words what each unit drawn does, then gathers that into codes,
    each with a description, criteria, boundary cases and examples. Choosing the codes in the end is yours."""
    from macaw.coding.llm.draft import draft_codebook, draft_text, sample_units
    from macaw.coding.llm.judge import coding_units

    if unit not in UNITS[:2]:
        raise MacawError(f"--unit is utterance or turn, not {unit}")
    resolved = resolve_config(config)
    _not_the_one_in_use(resolved, out)
    cfg = resolved.coding.llm
    lines = {}
    for rid in resolve_recordings(resolved, recordings):
        own = resolve_config(config, recording_id=rid)
        try:
            lines[rid] = coding_units(own, build_recording(rid, own), unit, cfg.coded_roles)
        except (MacawError, OSError) as error:
            typer.echo(f"{rid}: no transcript to draw from, left out ({error})")
    units = sample_units(lines, n_units, np.random.default_rng(seed))
    if units.empty:
        raise MacawError("no units to draw from: run text.asr first")
    backend = _backend(resolved)
    made = draft_codebook(
        backend, aim, units, name=name or out.stem, unit=unit, batch_size=cfg.batch_size, seed=seed, takes_seed=backend.takes_seed,
    )
    if not made.codebook.codes:
        raise MacawError(f"{backend.model_version()} gave no codes; try again with another --seed or a larger model")
    write_codebook(made.codebook, out)
    note = out.with_name(f"{out.stem}.draft.md")
    note.write_text(draft_text(made, aim, backend.model_version(), len(lines)), encoding="utf-8")
    counts = made.units["code"].value_counts()
    for definition in made.codebook.codes:
        typer.echo(f"{definition.code}: {int(counts.get(definition.code, 0))} of {len(made.units)} units")
    typer.echo(f"{sum(made.left_out.values())} unit(s) in no code\n\n{out}\n{note}\n{OWN_WORDS}")

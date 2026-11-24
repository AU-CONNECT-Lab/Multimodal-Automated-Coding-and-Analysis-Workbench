from __future__ import annotations

import copy
import getpass
from datetime import datetime
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.media.align import MANUAL, UNALIGNED, AlignResult, replacing, write_offset_table
from macaw.project.layout import layout_for
from macaw.signals.align import FOLLOWS, RECORDING_CLOCK, decided_offsets, describe, needs_person, signal_offsets
from macaw.signals.table import EVERYONE, signal_files, stream_mates

app = typer.Typer()
PANEL = "Other signals"


@app.command("signal-offsets", rich_help_panel=PANEL)
def show(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c", help="The project's settings file."),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
) -> None:
    """Show where each file of the signals table starts on its recording's clock, and which ones wait for a person."""
    resolved = resolve_config(config)
    cfg = resolved.signals
    for recording_id in resolve_recordings(resolved, recordings):
        files = signal_files(resolved, recording_id)
        if not files:
            continue
        typer.echo(recording_id)
        current = signal_offsets(resolved, recording_id)
        for item in files:
            result = current.get(item.key)
            if result is None:
                typer.echo(f"  {item.key}: not aligned yet, run `macaw run -r {recording_id}`")
                continue
            reason = needs_person(result, cfg) if cfg.review != "skip" else None
            typer.echo(f"  {item.key}: {describe(result)}" + (f"  [waiting: {reason}]" if reason else ""))


@app.command("accept-signal-offset", rich_help_panel=PANEL)
def accept(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c", help="The project's settings file."),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    person: list[str] = typer.Option(None, "--person", help=f"Whose files to keep, repeatable; {EVERYONE} for a file of the whole recording. Default: every one waiting."),
    file: list[str] = typer.Option(None, "--file", help="A file to keep, as signal-offsets names it, repeatable. Default: every one waiting."),
    by: str = typer.Option(None, "--by", help="Who looked at them. Default: the login name."),
) -> None:
    """Keep signal offsets waiting for review, after looking at them, so the run goes on; the other streams of an XDF file come along."""
    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    wanted, named = set(person or ()), set(file or ())
    stamp = f"{by or getpass.getuser()} {datetime.now().isoformat(timespec='seconds')}"
    for recording_id in resolve_recordings(resolved, recordings):
        listed = signal_files(resolved, recording_id)
        files = stream_mates(listed, [
            item for item in listed
            if (not wanted or (item.person or EVERYONE) in wanted) and (not named or item.key in named)
        ])
        current = signal_offsets(resolved, recording_id)
        kept, cannot = [], []
        for item in files:
            result = current.get(item.key)
            if result is None or needs_person(result, resolved.signals) is None:
                continue
            if result.method == FOLLOWS:
                typer.echo(f"{recording_id}: {result.target} follows {result.reference}; it settles when that one does")
                continue
            (cannot if result.method == UNALIGNED else kept).append(result)
        for result in cannot:
            typer.echo(f"{recording_id}: {result.target} has no offset to keep; set one with `macaw set-signal-offset`")
        if not kept:
            typer.echo(f"{recording_id}: nothing waiting that can be kept")
            continue
        decisions = {decision.target: decision for decision in decided_offsets(resolved, recording_id)}
        for result in kept:
            entry = copy.deepcopy(result)
            entry.accepted = stamp
            decisions[result.target] = replacing(entry, decisions.get(result.target))
        write_offset_table(list(decisions.values()), layout.signal_offsets_path(recording_id))
        typer.echo(f"{recording_id}: kept {', '.join(result.target for result in kept)}")


@app.command("set-signal-offset", rich_help_panel=PANEL)
def set_offset(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c", help="The project's settings file."),
    recording: str = typer.Option(..., "--recording", "-r", help="The recording's id."),
    person: str = typer.Option(None, "--person", help=f"Whose files, as project.roles names them; {EVERYONE} for a file of the whole recording."),
    offset: float = typer.Option(..., "--offset", help="Seconds: recording time = the file's own time + this."),
    stage: str = typer.Option(None, "--stage", help="Only this stage of theirs. Default: every file of theirs, which share one clock."),
    file: list[str] = typer.Option(None, "--file", help="Instead of --person: a file as signal-offsets names it, repeatable."),
    by: str = typer.Option(None, "--by", help="Who set it. Default: the login name."),
) -> None:
    """Set by hand where signal files start on the recording's clock; the other streams of an XDF file come along, and a later run keeps it."""
    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    if (person is None) == (not file):
        raise MacawError("name either --person or --file")
    listed = signal_files(resolved, recording)
    files = stream_mates(listed, [
        item for item in listed
        if (item.key in file if file else (item.person or EVERYONE) == person and (stage is None or item.stage == stage))
    ])
    if not files:
        said = ", ".join(file) if file else f"a file of {person}{f' at stage {stage}' if stage else ''}"
        raise MacawError(f"the signals table has no {said} in {recording}")
    stamp = datetime.now().isoformat(timespec="seconds")
    decisions = {decision.target: decision for decision in decided_offsets(resolved, recording)}
    for item in files:
        entry = AlignResult(
            reference=RECORDING_CLOCK, target=item.key, offset_s=offset, confidence=None, method=MANUAL,
            note=f"set by hand by {by or getpass.getuser()} {stamp}",
        )
        decisions[item.key] = replacing(entry, decisions.get(item.key), "set by hand")
    write_offset_table(list(decisions.values()), layout.signal_offsets_path(recording))
    typer.echo(f"{recording}: {', '.join(item.key for item in files)} set to {offset:+.3f} s")

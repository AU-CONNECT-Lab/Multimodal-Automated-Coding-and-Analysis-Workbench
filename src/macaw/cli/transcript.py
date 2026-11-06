from __future__ import annotations

import getpass
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import InputNotFoundError, MacawError
from macaw.extractors.audio.content.asr import DROPPED_NAME, DROPPED_SCHEMA, TRANSCRIPT_NAME, ASRExtractor, read_transcript
from macaw.extractors.audio.content.diarization import turn_takers
from macaw.interop.transcript_eaf import export_transcript_eaf, read_transcript_eaf
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.qc.transcript_check import KINDS, read_check, still_there
from macaw.store import _io

app = typer.Typer(help="Correct a transcript by hand in ELAN.")


@app.command("export")
def export(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    out: Path = typer.Option(None, "--out", help="Default: <output_dir>/exports/<recording>_transcript.eaf"),
) -> None:
    """Write the transcript as an ELAN file, one tier per person, with what was left out on tiers of its own."""
    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    directory = layout.artifacts_dir(recording, ASRExtractor.name)
    if not (directory / TRANSCRIPT_NAME).is_file():
        raise InputNotFoundError(f"{recording}: no transcript yet; run text.asr first")
    transcript = read_transcript(directory / TRANSCRIPT_NAME)
    dropped_path = directory / DROPPED_NAME
    dropped = (
        _io.read_file(dropped_path, DROPPED_SCHEMA, "dropped utterances") if dropped_path.is_file()
        else pd.DataFrame(columns=list(DROPPED_SCHEMA))
    )
    out = out or layout.output_dir / "exports" / f"{recording}_transcript.eaf"
    doubts, _ = still_there(read_check(layout.transcript_check_path(recording)), transcript)
    export_transcript_eaf(transcript, dropped, build_recording(recording, resolved), out, doubts=doubts)
    typer.echo(
        f"{out}\ncorrect the transcript tiers in ELAN (move an annotation from a dropped tier to put it back), "
        f"save, then `macaw transcript import -r {recording} {out.name}`"
    )
    if not doubts.empty:
        typer.echo(f"{len(doubts)} line(s) the model marked as possibly misheard or cut off are on the doubtful tiers, with why")


@app.command("import")
def import_(
    eaf: Path = typer.Argument(..., help="The corrected .eaf"),
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    by: str = typer.Option(None, "--by", help="Who corrected it. Default: the login name."),
) -> None:
    """Keep a corrected transcript; text.asr and everything after it use it from the next run."""
    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    corrected = read_transcript_eaf(eaf, turn_takers(resolved, build_recording(recording, resolved)))
    target = layout.corrected_transcript_path(recording)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(eaf, target)
    stamp = {"by": by or getpass.getuser(), "at": datetime.now().isoformat(timespec="seconds"), "from": eaf.name}
    layout.corrected_transcript_stamp_path(recording).write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    counts = corrected.groupby("subject_id").size()
    typer.echo(
        f"{recording}: kept {len(corrected)} utterances ({', '.join(f'{who} {n}' for who, n in counts.items())}); "
        f"apply it with `macaw run -r {recording}`"
    )


@app.command("check")
def check(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
) -> None:
    """Have the model in coding.llm mark lines that look misheard, made up or cut off, to check against the recording.
    The transcript is not changed; the lines go on tiers of their own in the ELAN export."""
    from macaw.coding.llm.backends import resolve_llm_backend
    from macaw.qc.transcript_check import check_transcript, describe

    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    path = layout.artifacts_dir(recording, ASRExtractor.name) / TRANSCRIPT_NAME
    if not path.is_file():
        raise InputNotFoundError(f"{recording}: no transcript yet; run text.asr first")
    cfg = resolved.coding.llm
    backend = resolve_llm_backend(cfg)
    backend.check_available()
    if backend.reporting:
        typer.echo(backend.reporting)
    subjects = [subject.subject_id for subject in build_recording(recording, resolved).subjects]
    found = check_transcript(
        backend, read_transcript(path), subjects, layout.transcript_check_path(recording), provider=cfg.provider,
        batch_size=cfg.batch_size, seed=cfg.seed if backend.takes_seed else None,
    )
    typer.echo(f"{recording}: {describe(found)}")
    for line in found["lines"]:
        typer.echo(f"  {line['t_start']:.1f} s {line['subject_id']}: {line['text']}  ({KINDS[line['kind']]}: {line['why']})")
    if found["lines"]:
        typer.echo(f"to correct them in ELAN: macaw transcript export -r {recording}")


@app.command("reset")
def reset(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
) -> None:
    """Drop the corrected transcript, so the next run recognises speech again."""
    layout = layout_for(resolve_config(config), create=False)
    paths = [layout.corrected_transcript_path(recording), layout.corrected_transcript_stamp_path(recording)]
    if not any(path.exists() for path in paths):
        raise MacawError(f"{recording}: no corrected transcript")
    for path in paths:
        path.unlink(missing_ok=True)
    typer.echo(f"{recording}: back to the recogniser's transcript from the next `macaw run -r {recording}`")

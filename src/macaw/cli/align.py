from __future__ import annotations

import getpass
from datetime import datetime
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import AlignmentFailedError, BackendNotAvailableError, MacawError
from macaw.execution.runner import resolve_recordings
from macaw.logging_setup import get_logger
from macaw.media.align import (
    MANUAL,
    SAMPLE_RATE,
    UNALIGNED,
    AlignResult,
    align_recording,
    describe_agreement,
    describe_drift,
    drift_flagged,
    keep_accepted,
    keep_manual,
    low_confidence,
    needs_person,
    read_offset_table,
    replacing,
    write_offset_table,
)
from macaw.media.probe import probe
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.qc.align_clip import alignment_clip, clip_path

app = typer.Typer()
logger = get_logger("align")


@app.command("align", rich_help_panel="Preparing the recordings")
def align(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c", help="The project's settings file."),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    drift_check: bool = typer.Option(True, "--drift-check/--no-drift-check", help="Also check whether the clocks drift apart over the recording."),
    set_: list[str] = typer.Option(None, "--set", help="section.key=value: a setting for this command only. Repeatable."),
) -> None:
    """Estimate from their sound how far each recording's files are offset from one another, and draw each pair for review."""
    overrides = list(set_ or ())
    if not drift_check:
        overrides.append("align.drift_check=false")
    resolved = resolve_config(config, cli_overrides=overrides)
    cfg = resolved.align
    if cfg.method == "none":
        typer.echo("align.method is none: every file is taken to start together, nothing to estimate")
        return
    if cfg.method == "manual":
        typer.echo("align.method is manual: set each offset with `macaw set-offset`")
        return

    layout = layout_for(resolved)
    failed = []
    for recording_id in resolve_recordings(resolved, recordings):
        # the recording's own settings, which may choose another setup and its devices
        own = resolve_config(config, recording_id=recording_id, cli_overrides=overrides)
        recording, cfg = build_recording(recording_id, own), own.align
        infos = _probe_all(recording)
        figures = layout.renders_dir(recording_id) / "align"
        try:
            results = align_recording(recording, cfg, infos, on_pair=_drawer(figures, cfg))
        except AlignmentFailedError as error:
            # one recording without sound to align by leaves the others to be aligned
            typer.echo(f"{recording_id}: not aligned: {error}")
            failed.append(recording_id)
            continue

        path = layout.offset_table_path(recording_id)
        if path.is_file():
            previous = read_offset_table(path)
            results, stale = keep_manual(results, previous)
            for target in stale:
                typer.echo(f"  {target}: its manual offset was set against another reference and was replaced")
            results, dropped = keep_accepted(results, previous)
            for target in dropped:
                typer.echo(f"  {target}: the result changed since it was accepted, look at it again")
        write_offset_table(results, path)
        _summarise(recording_id, results, cfg)
        typer.echo(f"  table {path}")
        if any(figures.glob("*.png")):
            typer.echo(f"  figures {figures}")
        _clip(recording, results, infos, clip_path(layout, recording_id), own.video.encoder)
        if cfg.review != "skip" and any(needs_person(result, cfg) for result in results):
            typer.echo(
                f"  files waiting for a person hold the run back: look at them, then `macaw accept-offset -r {recording_id}` "
                "to keep them or `macaw set-offset` to change one"
            )
    if failed:
        raise MacawError(f"not aligned: {', '.join(failed)}; set their offsets with `macaw set-offset` or give them a file with sound")


@app.command("set-offset", rich_help_panel="Preparing the recordings")
def set_offset(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c", help="The project's settings file."),
    recording: str = typer.Option(..., "--recording", "-r", help="The recording's id."),
    media: str = typer.Option(..., "--device", help="Device to adjust, as its folder is named."),
    offset: float = typer.Option(
        ..., "--offset", help="Seconds added to the device's own time to give the recording's: positive when it started after the reference.",
    ),
    by: str = typer.Option(None, "--by", help="Who set it. Default: the login name."),
) -> None:
    """Set one offset by hand after inspecting it; a later `macaw align` keeps it, and the entry it replaces is kept in its history."""
    resolved = resolve_config(config, recording_id=recording)
    layout = layout_for(resolved)
    built = build_recording(recording, resolved)
    names = [item.name for item in built.media]
    if media not in names:
        raise MacawError(f"{media} is not a device of {recording}; its devices are {', '.join(names)}")

    path = layout.offset_table_path(recording)
    results = read_offset_table(path) if path.is_file() else []
    if results:
        reference = results[0].reference
    else:
        _probe_all(built)
        reference = built.reference_media().name
        results = [AlignResult(reference=reference, target=reference, offset_s=0.0, confidence=None, method=MANUAL)]
    if media == reference:
        raise MacawError(f"{media} is the reference; every other offset is measured against it")

    stamp = datetime.now().isoformat(timespec="seconds")
    entry = AlignResult(
        reference=reference, target=media, offset_s=offset, confidence=None, method=MANUAL,
        note=f"set by hand by {by or getpass.getuser()} {stamp}",
    )
    old = next((result for result in results if result.target == media), None)
    replacing(entry, old, "set by hand")
    results = [result for result in results if result.target != media] + [entry]
    write_offset_table(results, path)
    typer.echo(f"{recording}: {media} set to {offset:+.3f} s against {reference}")
    _clip(built, results, None, clip_path(layout, recording), resolved.video.encoder)


@app.command("accept-offset", rich_help_panel="Preparing the recordings")
def accept_offset(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c", help="The project's settings file."),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    media: list[str] = typer.Option(None, "--device", help="Device to keep, repeatable. Default: every one waiting for a person."),
    by: str = typer.Option(None, "--by", help="Who looked at them. Default: the login name."),
) -> None:
    """Keep offsets waiting for review, after looking at their figures, so the run goes on."""
    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    wanted = set(media or ())
    stamp = f"{by or getpass.getuser()} {datetime.now().isoformat(timespec='seconds')}"
    matched = set()
    for recording_id in resolve_recordings(resolved, recordings):
        path = layout.offset_table_path(recording_id)
        if not path.is_file():
            typer.echo(f"{recording_id}: no offset table, run `macaw align` first")
            continue
        results = read_offset_table(path)
        kept = [
            result for result in results
            if (result.target in wanted if wanted else needs_person(result, resolved.align) is not None)
        ]
        matched.update(result.target for result in kept)
        if not kept:
            typer.echo(f"{recording_id}: nothing waiting")
            continue
        for result in kept:
            result.accepted = stamp
        write_offset_table(results, path)
        typer.echo(f"{recording_id}: kept {', '.join(result.target for result in kept)}")
    unknown = wanted - matched
    if unknown:
        raise MacawError(f"not in any offset table: {', '.join(sorted(unknown))}")


def _clip(recording, results, infos: dict | None, dest: Path, encoder: str) -> None:
    """A clip that cannot be made is said and never stops the command."""
    try:
        made = alignment_clip(recording, results, infos if infos is not None else _probe_all(recording), dest, encoder)
    except MacawError as error:
        typer.echo(f"  no clip: {error}")
        return
    if made is not None:
        typer.echo(f"  clip {made} (start, middle and end, all sound mixed: a file out of step sounds doubled)")


def _probe_all(recording) -> dict:
    """Durations decide the reference, and the streams decide what can be correlated."""
    infos = {}
    for item in recording.media:
        info = probe(item.path)
        item.duration_s = info.duration_s
        infos[item.name] = info
    return infos


def _drawer(out_dir: Path, cfg):
    try:
        from macaw.qc.figures import alignment_figure, require_matplotlib

        require_matplotlib()
    except BackendNotAvailableError as error:
        logger.warning("%s; no review figures drawn", error)
        return None

    def draw(reference, target, result):
        name = f"{Path(result.target).stem}.png"
        alignment_figure(reference, target, SAMPLE_RATE, result, cfg, out_dir / name)

    return draw


def _summarise(recording_id: str, results: list[AlignResult], cfg) -> None:
    typer.echo(f"{recording_id}  reference {results[0].reference if results else '-'}")
    for result in results:
        if result.target == result.reference:
            continue
        flags = []
        if result.method == UNALIGNED:
            hint = "" if result.metadata_offset_s is None else f", metadata says {result.metadata_offset_s:+.1f} s"
            line = f"not aligned ({result.note}{hint})"
            flags.append("CHECK")
        elif result.method == MANUAL:
            line = f"{result.offset_s:+.3f} s  set by hand"
        else:
            line = f"{result.offset_s:+.3f} s  confidence {result.confidence:.2f}, {describe_agreement(result)}"
            if result.drift_ppm is not None:
                line += f"  {describe_drift(result)}"
            if low_confidence(result, cfg):
                flags.append("LOW CONFIDENCE")
        if drift_flagged(result):
            flags.append("DRIFT")
        if flags and result.accepted:
            flags.append(f"(kept by {result.accepted})")
        typer.echo(f"  {result.target:<32} {line}  {' '.join(flags)}".rstrip())

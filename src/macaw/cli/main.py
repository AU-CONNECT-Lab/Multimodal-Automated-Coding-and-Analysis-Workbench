from __future__ import annotations

import sys

import typer

from macaw.cli import align, cameras, code, codebook, config_cmd, export, gaze, gui, hpc, init, methods, plugin, qc, run, recordings, scale, segments, transcript
from macaw.exceptions import MacawError
from macaw.logging_setup import setup_logging

app = typer.Typer(
    name="macaw",
    help="Multimodal Automated Coding and Analysis Workbench.",
    no_args_is_help=True,
)

app.add_typer(init.app)
app.add_typer(recordings.app)
app.add_typer(align.app)
app.add_typer(scale.app)
app.add_typer(segments.app)
app.add_typer(gaze.app)
app.add_typer(cameras.app)
app.add_typer(run.app)
app.add_typer(qc.app)
app.add_typer(code.app)
app.add_typer(export.app)
app.add_typer(methods.app)
app.add_typer(plugin.app)
app.add_typer(hpc.app, name="hpc")
app.add_typer(gui.app)
app.add_typer(config_cmd.app, name="config")
app.add_typer(transcript.app, name="transcript")
app.add_typer(codebook.app, name="codebook")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    setup_logging(verbose=verbose, quiet=quiet)


def entrypoint() -> None:
    """Wraps app() so MacawError prints as a clean message instead of a traceback."""
    try:
        app()
    except MacawError as error:
        typer.secho(f"error: {error}", fg=typer.colors.RED, err=True)
        sys.exit(1)

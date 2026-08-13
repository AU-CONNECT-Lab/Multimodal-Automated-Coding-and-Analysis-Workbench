from __future__ import annotations

import typer

from macaw.cli import align, code, config_cmd, export, gui, hpc, init, qc, run, sessions

app = typer.Typer(
    name="macaw",
    help="Multimodal Automated Coding and Analysis Workbench.",
    no_args_is_help=True,
)

app.add_typer(init.app)
app.add_typer(sessions.app)
app.add_typer(align.app)
app.add_typer(run.app)
app.add_typer(qc.app)
app.add_typer(code.app)
app.add_typer(export.app)
app.add_typer(hpc.app, name="hpc")
app.add_typer(gui.app)
app.add_typer(config_cmd.app, name="config")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    raise NotImplementedError


def entrypoint() -> None:
    """Wraps app() so MacawError prints as a clean message instead of a traceback."""
    raise NotImplementedError

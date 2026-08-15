from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml

from macaw.config.loader import resolve_config
from macaw.exceptions import ConfigError, InputNotFoundError
from macaw.extractors import get_extractor, list_extractors
from macaw.logging_setup import get_logger

app = typer.Typer(help="Inspect and validate configuration.")
logger = get_logger("config")


@app.command("show")
def show(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    section: str = typer.Option(None, "--section"),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Print the fully merged configuration; what is actually in effect."""
    data = resolve_config(config, cli_overrides=set_ or ()).model_dump(mode="json")
    if section is not None:
        if section not in data:
            raise ConfigError(f"Unknown section '{section}'. Known: {', '.join(sorted(data))}")
        data = {section: data[section]}
    typer.echo(yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip())


@app.command("validate")
def validate(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
) -> None:
    """Check the schema and reject unknown keys."""
    resolved = resolve_config(config)
    typer.echo(f"ok  project '{resolved.project.name}'  output {resolved.project.output_dir}")


def _load(path: Path) -> dict:
    if not path.is_file():
        raise InputNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _differing_sections(left: dict, right: dict) -> list[str]:
    return sorted(
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    )


def _extractors_using(sections: list[str]) -> dict[str, list[str]]:
    """A changed section only matters through the extractors that hash it."""
    affected: dict[str, list[str]] = {}
    for name in list_extractors():
        # a declared section may be dotted, and a change to the parent still reaches it
        used = sorted(
            {
                declared
                for declared in get_extractor(name).config_sections
                for section in sections
                if declared == section or declared.startswith(f"{section}.")
            }
        )
        if used:
            affected[name] = used
    return affected


def _describe(value: Any) -> str:
    text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).strip()
    return text.replace("\n", "; ")


@app.command("diff")
def diff(
    a: Path = typer.Argument(..., help="First resolved config."),
    b: Path = typer.Argument(..., help="Second resolved config."),
) -> None:
    """Show which sections differ, and therefore what a rerun would recompute."""
    left, right = _load(a), _load(b)
    sections = _differing_sections(left, right)
    if not sections:
        typer.echo("identical")
        return

    for section in sections:
        typer.echo(f"{section}")
        typer.echo(f"  {a.name}: {_describe(left.get(section))}")
        typer.echo(f"  {b.name}: {_describe(right.get(section))}")

    affected = _extractors_using(sections)
    typer.echo("")
    if not affected:
        typer.echo("no registered extractor hashes these sections; nothing would recompute")
        return
    typer.echo("would recompute:")
    for name, used in affected.items():
        typer.echo(f"  {name}\t({', '.join(used)})")

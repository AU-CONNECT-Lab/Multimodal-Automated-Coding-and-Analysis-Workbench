"""Writes commands.md from the command line's own help text before Sphinx reads the sources."""

from __future__ import annotations

import re
from pathlib import Path

INTRO = (
    "Every command reads the project from the settings file given with `-c` (by default *project.yaml* in the current folder). "
    "Most steps can also be taken in the workbench; the commands are needed on a server without a screen, on a cluster, and in scripts. "
    "`macaw <command> --help` prints the same text in the terminal."
)
SKIPPED = {"help", "install_completion", "show_completion"}


def _escape(text: str) -> str:
    # backtick spans are kept; elsewhere characters Markdown would act on are escaped
    parts = re.split(r"(`[^`]*`)", text)
    return "".join(part if part.startswith("`") else re.sub(r"([*<>\[\]_|])", r"\\\1", part) for part in parts)


def _plain(text: str | None) -> str:
    return " ".join((text or "").split())


def _metavar(param) -> str:
    if getattr(param, "is_flag", False) or getattr(param, "count", False):
        return ""
    if hasattr(param.type, "types"):
        return " ".join(part.name.upper() for part in param.type.types)
    name = param.type.name.upper() if param.type.name else "VALUE"
    name = {"BOOLEAN": "BOOL", "STR": "TEXT", "INT": "INTEGER"}.get(name, name)
    return " ".join([name] * param.nargs) if param.nargs > 1 else name


def _default(param) -> str:
    value = param.default
    if param.required or value is None or callable(value) or value == () or value == []:
        return ""
    if getattr(param, "is_flag", False):
        if param.secondary_opts:
            return param.opts[-1] if value else param.secondary_opts[0]
        return ""
    return str(value)


def _param_lines(param) -> list[str]:
    if param.param_type_name == "argument":
        head = f"`{param.human_readable_name.upper()}`"
    else:
        names = ", ".join(f"`{name}`" for name in param.opts + param.secondary_opts)
        metavar = _metavar(param)
        head = f"{names} {metavar}".rstrip() if metavar else names
    notes = []
    if param.required:
        notes.append("Required.")
    default = _default(param)
    if default:
        notes.append(f"Default: `{default}`.")
    help_text = _plain(getattr(param, "help", None))
    if getattr(param, "multiple", False) and "repeatable" not in help_text.lower():
        notes.append("Repeatable.")
    body = " ".join(filter(None, [_escape(help_text), *notes])) or " "
    return [head, f": {body}", ""]


def _usage(path: list[str], command) -> str:
    args = " ".join(
        p.human_readable_name.upper() if p.required else f"[{p.human_readable_name.upper()}]"
        for p in command.params
        if p.param_type_name == "argument"
    )
    return " ".join(filter(None, ["macaw", *path, "[OPTIONS]", args]))


def _command(path: list[str], command, level: int) -> list[str]:
    if command.hidden:
        return []
    lines = [f"{'#' * level} `macaw {' '.join(path)}`", "", _escape(_plain(command.help)), ""]
    subcommands = getattr(command, "commands", None)
    if subcommands:
        for name, sub in subcommands.items():
            lines += _command([*path, name], sub, level + 1)
        return lines
    lines += ["```text", _usage(path, command), "```", ""]
    for param in command.params:
        if param.name not in SKIPPED and not getattr(param, "hidden", False):
            lines += _param_lines(param)
    return lines


def render() -> str:
    import typer.main

    from macaw.cli.main import app

    root = typer.main.get_command(app)
    lines = ["# Command reference", "", INTRO, "", "## Options before the command", "", "These go between `macaw` and the command, as in `macaw -v run`.", ""]
    for param in root.params:
        if param.name not in SKIPPED:
            lines += _param_lines(param)
    panels: dict[str, list[tuple[str, object]]] = {}
    for name, command in root.commands.items():
        panels.setdefault(command.rich_help_panel or "Other", []).append((name, command))
    for panel, commands in panels.items():
        section = [line for name, command in commands for line in _command([name], command, 3)]
        if section:
            lines += [f"## {panel}", "", *section]
    return "\n".join(lines).rstrip() + "\n"


def _write(app) -> None:
    target = Path(app.srcdir) / "commands.md"
    text = render()
    if not target.exists() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")


def setup(app):
    app.connect("builder-inited", _write)
    return {"parallel_read_safe": True}

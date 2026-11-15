"""Writes commands.md from the command line's own help text before Sphinx reads the sources."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

INTRO = (
    "Every command reads the project from the settings file given with `-c` (by default *project.yaml* in the current folder). "
    "Most steps can also be taken in the workbench; the commands are needed on a server without a screen, on a cluster, and in scripts. "
    "`macaw <command> --help` prints the same text in the terminal."
)
SKIPPED = {"help", "install_completion", "show_completion"}
TYPE_NAMES = {Path: "PATH", str: "TEXT", int: "INTEGER", float: "FLOAT", bool: "BOOL"}


def _escape(text: str) -> str:
    # backtick spans are kept; elsewhere characters Markdown would act on are escaped
    parts = re.split(r"(`[^`]*`)", text)
    return "".join(part if part.startswith("`") else re.sub(r"([*<>\[\]_|])", r"\\\1", part) for part in parts)


def _plain(text: str | None) -> str:
    return " ".join((text or "").split())


def _hints(command) -> dict:
    # types come from the command function's own annotations; typer's parameter internals change between releases
    import typing

    function = inspect.unwrap(command.callback) if command.callback else None
    try:
        return typing.get_type_hints(function) if function else {}
    except Exception:
        return {}


def _type_name(annotation) -> str:
    import typing

    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is tuple:
        return " ".join(_type_name(arg) for arg in args)
    if origin in (list, typing.Union) or str(origin) == "types.UnionType":
        return _type_name(next(arg for arg in args if arg is not type(None)))
    return TYPE_NAMES.get(annotation, "VALUE")


def _metavar(param, hints: dict) -> str:
    if getattr(param, "is_flag", False) or getattr(param, "count", False):
        return ""
    return _type_name(hints.get(param.name, str))


def _default(param) -> str:
    value = param.default
    if param.required or value is None or callable(value) or value == () or value == []:
        return ""
    if getattr(param, "is_flag", False):
        if param.secondary_opts:
            return param.opts[-1] if value else param.secondary_opts[0]
        return ""
    return str(value)


def _param_lines(param, hints: dict) -> list[str]:
    if param.param_type_name == "argument":
        head = f"`{param.name.upper()}`"
    else:
        names = ", ".join(f"`{name}`" for name in param.opts + param.secondary_opts)
        metavar = _metavar(param, hints)
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
        p.name.upper() if p.required else f"[{p.name.upper()}]"
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
    hints = _hints(command)
    for param in command.params:
        if param.name not in SKIPPED and not getattr(param, "hidden", False):
            lines += _param_lines(param, hints)
    return lines


def render() -> str:
    import typer.main

    from macaw.cli.main import app

    root = typer.main.get_command(app)
    lines = ["# Command reference", "", INTRO, "", "## Options before the command", "", "These go between `macaw` and the command, as in `macaw -v run`.", ""]
    hints = _hints(root)
    for param in root.params:
        if param.name not in SKIPPED:
            lines += _param_lines(param, hints)
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

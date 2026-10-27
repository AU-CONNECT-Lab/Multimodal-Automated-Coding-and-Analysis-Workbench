"""Small building blocks in the QC report's look: flat tables, status pills, numbered sections."""

from __future__ import annotations

from dash import html

PILL_CLASS = {"ok": "ok", "warn": "warn", "bad": "bad", "info": "info"}


def pill(text: str, kind: str = "info"):
    return html.Span(text, className=PILL_CLASS.get(kind, "info"))


def section(title: str, *children, desc: str | None = None):
    return html.Div([html.H2(title), *([html.P(desc, className="desc")] if desc else []), *children], className="section")


def table(columns: list[str], rows: list[list], *, numeric: set[str] = frozenset()):
    head = html.Thead(html.Tr([html.Th(name, className="num" if name in numeric else None) for name in columns]))
    body = html.Tbody(
        [
            html.Tr([html.Td(cell, className="num" if name in numeric else None) for name, cell in zip(columns, row)])
            for row in rows
        ]
    )
    return html.Table([head, body], className="flat")


def command(argv: list[str]):
    """The terminal command a button stands for, so the same step can be run without the GUI."""
    return html.Pre(" ".join(f'"{part}"' if " " in part else part for part in argv), className="command")


def note(text: str, kind: str = "info"):
    return html.Div(text, className=f"note {kind}-note")


def output(code: int, text: str):
    return html.Div([pill("done" if code == 0 else "failed", "ok" if code == 0 else "bad"), html.Pre(text or "(no output)")], className="result")

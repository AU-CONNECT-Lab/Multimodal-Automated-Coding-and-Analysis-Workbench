"""Small building blocks in the workbench's look: cards, pill buttons, labelled fields, status pills."""

from __future__ import annotations

from dash import html

PILL_CLASS = {"ok": "ok", "warn": "warn", "bad": "bad", "info": "info"}


def icon(name: str):
    return html.Span(className=f"icon icon-{name}")


def button(label: str, id=None, *, icon_name: str | None = None, primary: bool = False, small: bool = False, **kwargs):
    classes = " ".join(["button", *(["primary"] if primary else []), *(["small"] if small else [])])
    children = [icon(icon_name), label] if icon_name else label
    return html.Button(children, className=classes, **({"id": id} if id is not None else {}), **kwargs)


def pill(text: str, kind: str = "info"):
    return html.Span(text, className=f"pill {PILL_CLASS.get(kind, 'info')}")


def card(title: str, *children, desc: str | None = None, actions=None, flush: bool = False, **kwargs):
    head = [html.Div([html.H2(title), *([html.P(desc)] if desc else [])])]
    if actions:
        head.append(html.Div(actions, className="actions"))
    body = [child for child in children if child is not None]
    return html.Section(
        [html.Div(head, className="card-head"), html.Div(body, className="card-body flush" if flush else "card-body")],
        className="card", **kwargs,
    )


def page_head(title: str, summary=None, actions=None):
    return html.Div(
        [
            html.Div([html.H1(title), html.P(summary, className="summary")]),
            *([html.Div(actions, className="actions")] if actions else []),
        ],
        className="page-head",
    )


def field(label: str, control, hint=None, *, changed: bool = False):
    return html.Div(
        [html.Label([html.Span(label, className="field-label"), control], className="field"),
         *([html.Div(hint, className="field-hint")] if hint else [])],
        className="field-wrap changed" if changed else "field-wrap",
    )


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


def note(text, kind: str = "info"):
    return html.Div(text, className=f"note {kind}-note")


def output(code: int, text: str):
    return html.Div([pill("done" if code == 0 else "failed", "ok" if code == 0 else "bad"), html.Pre(text or "(no output)")], className="result")

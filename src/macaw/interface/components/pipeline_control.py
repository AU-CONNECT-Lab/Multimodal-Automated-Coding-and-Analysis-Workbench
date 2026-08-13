from __future__ import annotations

from macaw.config.schema import MacawConfig


def layout(config: MacawConfig) -> object:
    raise NotImplementedError


def session_selector(session_ids: list[str]) -> object:
    raise NotImplementedError


def extractor_checklist(names: list[str]) -> object:
    raise NotImplementedError


def dag_graph(elements: dict) -> object:
    """Graph view of the task DAG."""
    raise NotImplementedError

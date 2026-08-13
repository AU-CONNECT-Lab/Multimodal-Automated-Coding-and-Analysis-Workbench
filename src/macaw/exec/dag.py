"""Dependency graph built from Extractor.requires, not from hard-coded pipeline stages.
This is the part of neuroimage-pipe's dag.py that had to be rewritten rather than forked."""

from __future__ import annotations

from dataclasses import dataclass, field

from macaw.config.schema import MacawConfig
from macaw.project.session import Session


@dataclass
class TaskNode:
    task_id: str
    session_id: str
    extractor: str
    depends_on: set[str] = field(default_factory=set)


class DAG:
    def __init__(self, config: MacawConfig):
        self.config = config
        self.nodes: dict[str, TaskNode] = {}

    def add(self, session_id: str, extractor: str) -> TaskNode:
        raise NotImplementedError

    def expand(self, sessions: list[Session], requested: list[str]) -> None:
        """Pull in each requested extractor plus everything it requires, transitively."""
        raise NotImplementedError

    def topological_order(self) -> list[str]:
        """Raises DependencyCycleError on a cycle."""
        raise NotImplementedError

    def ready_tasks(self, completed: set[str]) -> list[str]:
        raise NotImplementedError

    def to_cytoscape(self) -> dict:
        """Feeds the GUI's DAG view."""
        raise NotImplementedError

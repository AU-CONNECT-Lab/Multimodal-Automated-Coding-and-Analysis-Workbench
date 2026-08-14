"""Dependency graph built from Extractor.requires, not from hard-coded pipeline stages.

Every recording is independent, so the real graph is the extractor graph applied to each
recording. The extractor graph is therefore what gets scheduled; recordings become array
indices inside it."""

from __future__ import annotations

from dataclasses import dataclass, field

from macaw.config.schema import MacawConfig
from macaw.extractors.base import Phase


@dataclass
class TaskNode:
    extractor: str
    phase: Phase
    depends_on: set[str] = field(default_factory=set)


class DAG:
    def __init__(self, config: MacawConfig):
        self.config = config
        self.nodes: dict[str, TaskNode] = {}

    def add(self, extractor: str) -> TaskNode:
        raise NotImplementedError

    def expand(self, requested: list[str]) -> None:
        """Pull in each requested extractor plus everything it requires, transitively."""
        raise NotImplementedError

    def topological_order(self) -> list[str]:
        """Raises DependencyCycleError on a cycle."""
        raise NotImplementedError

    def layers(self) -> list[list[str]]:
        """Topological layers. Useful for reporting; not how jobs are grouped."""
        raise NotImplementedError

    def dependencies(self, extractor: str) -> set[str]:
        """Direct predecessors, which become the aftercorr edges between job arrays."""
        raise NotImplementedError

    def by_phase(self, phase: Phase) -> list[str]:
        raise NotImplementedError

    def interactive_gates(self) -> list[str]:
        """Nodes a human must clear. Cluster scripts stop before these and resume after."""
        raise NotImplementedError

    def batch_segments(self) -> list[list[str]]:
        """Runs of automatic work split at each interactive gate, in order."""
        raise NotImplementedError

    def ready(self, completed: set[str]) -> list[str]:
        raise NotImplementedError

    def to_cytoscape(self) -> dict:
        raise NotImplementedError


def build_dag(config: MacawConfig, requested: list[str] | None = None) -> DAG:
    raise NotImplementedError

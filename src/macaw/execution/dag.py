"""Dependency graph built from Extractor.requires, not from hard-coded pipeline stages.

Every recording is independent, so the real graph is the extractor graph applied to each
recording. The extractor graph is therefore what gets scheduled; recordings become array
indices inside it."""

from __future__ import annotations

from dataclasses import dataclass, field

from macaw.config.schema import MacawConfig
from macaw.exceptions import ConfigError, DependencyCycleError, ExtractorNotFoundError
from macaw.extractors import get_extractor, list_extractors
from macaw.extractors.base import Phase

PHASE_ORDER = (Phase.PREP, Phase.INTERACTIVE, Phase.EXTRACT, Phase.DERIVE, Phase.QC, Phase.AGGREGATE)


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
        existing = self.nodes.get(extractor)
        if existing is not None:
            return existing
        cls = get_extractor(extractor)
        node = TaskNode(extractor=extractor, phase=cls.phase, depends_on=set(cls.requires))
        self.nodes[extractor] = node
        return node

    def expand(self, requested: list[str]) -> None:
        """Pull in each requested extractor plus everything it requires, transitively."""
        queue = list(requested)
        while queue:
            name = queue.pop()
            if name in self.nodes:
                continue
            queue.extend(self.add(name).depends_on)

    def topological_order(self) -> list[str]:
        """Raises DependencyCycleError on a cycle."""
        return [name for level in self.layers() for name in level]

    def layers(self) -> list[list[str]]:
        """Topological layers. Useful for reporting; not how jobs are grouped."""
        remaining = {name: set(node.depends_on) & set(self.nodes) for name, node in self.nodes.items()}
        levels: list[list[str]] = []
        while remaining:
            ready = sorted(name for name, deps in remaining.items() if not deps)
            if not ready:
                stuck = ", ".join(sorted(remaining))
                raise DependencyCycleError(f"Extractor dependencies form a cycle among: {stuck}")
            levels.append(ready)
            for name in ready:
                remaining.pop(name)
            for deps in remaining.values():
                deps.difference_update(ready)
        return levels

    def dependencies(self, extractor: str) -> set[str]:
        """Direct predecessors, which become the aftercorr edges between job arrays."""
        return set(self.nodes[extractor].depends_on) & set(self.nodes)

    def by_phase(self, phase: Phase) -> list[str]:
        order = self.topological_order()
        return [name for name in order if self.nodes[name].phase == phase]

    def interactive_gates(self) -> list[str]:
        """Nodes a human must clear. Cluster scripts stop before these and resume after."""
        return self.by_phase(Phase.INTERACTIVE)

    def batch_segments(self) -> list[list[str]]:
        """Runs of automatic work split at each interactive gate, in order."""
        segments: list[list[str]] = []
        current: list[str] = []
        for name in self.topological_order():
            if self.nodes[name].phase == Phase.INTERACTIVE:
                if current:
                    segments.append(current)
                    current = []
                continue
            current.append(name)
        if current:
            segments.append(current)
        return segments

    def ready(self, completed: set[str]) -> list[str]:
        return sorted(
            name
            for name in self.nodes
            if name not in completed and self.dependencies(name) <= completed
        )

    def to_cytoscape(self) -> dict:
        elements = [
            {"data": {"id": name, "label": name, "phase": node.phase.value}}
            for name, node in sorted(self.nodes.items())
        ]
        elements += [
            {"data": {"source": dependency, "target": name}}
            for name in sorted(self.nodes)
            for dependency in sorted(self.dependencies(name))
        ]
        return {"elements": elements}


def resolve_selection(config: MacawConfig, names: list[str]) -> list[str]:
    """Group names expand to their members. Groups are the user's own; the package does not
    decide how extractors divide into modalities."""
    groups = config.pipeline.groups
    known = set(list_extractors())
    clashing = sorted(set(groups) & known)
    if clashing:
        raise ConfigError(f"pipeline.groups reuses extractor names: {', '.join(clashing)}")

    selected: list[str] = []
    for name in names:
        for member in groups.get(name, [name]):
            if member not in known:
                where = f" (in group '{name}')" if name in groups else ""
                available = ", ".join(sorted(known | set(groups)))
                raise ExtractorNotFoundError(
                    f"Unknown extractor or group '{member}'{where}. Available: {available}"
                )
            if member not in selected:
                selected.append(member)
    return selected


def build_dag(config: MacawConfig, requested: list[str] | None = None) -> DAG:
    """Without a request, pipeline.run; without that, every installed extractor."""
    dag = DAG(config)
    names = requested or config.pipeline.run or list_extractors()
    dag.expand(resolve_selection(config, names))
    return dag

from __future__ import annotations

import pytest

from macaw.exceptions import DependencyCycleError
from macaw.execution.dag import DAG, build_dag
from macaw.extractors.base import Phase


@pytest.fixture
def graph(monkeypatch):
    """Throwaway extractors, so the graph tests do not move every time the pipeline does."""
    import macaw.extractors as registry

    def build(spec, *, requested=None):
        table = {
            name: type(
                "FakeExtractor",
                (),
                {
                    "name": name,
                    "requires": tuple(requires),
                    "phase": phase,
                    "config_sections": (),
                },
            )
            for name, (requires, phase) in spec.items()
        }
        monkeypatch.setattr(registry, "_REGISTRY", table)
        monkeypatch.setattr(registry, "_builtins_loaded", True)
        return build_dag(None, requested)

    return build


LINE = {
    "prep": ((), Phase.PREP),
    "asr": (("prep",), Phase.EXTRACT),
    "turns": (("asr", "prep"), Phase.DERIVE),
}


class TestExpansion:
    def test_a_request_pulls_in_what_it_needs(self, graph):
        """Asking for turns alone still has to run prep and asr first."""
        dag = graph(LINE, requested=["turns"])
        assert set(dag.nodes) == {"prep", "asr", "turns"}

    def test_dependencies_come_first(self, graph):
        order = graph(LINE, requested=["turns"]).topological_order()
        assert order.index("prep") < order.index("asr") < order.index("turns")

    def test_no_request_means_everything_registered(self, graph):
        assert set(graph(LINE).nodes) == set(LINE)

    def test_asking_twice_adds_one_node(self, graph):
        assert len(graph(LINE, requested=["turns", "turns", "asr"]).nodes) == 3

    def test_layers_group_what_can_run_together(self, graph):
        spec = {
            "prep": ((), Phase.PREP),
            "asr": (("prep",), Phase.EXTRACT),
            "prosody": (("prep",), Phase.EXTRACT),
        }
        assert graph(spec).layers() == [["prep"], ["asr", "prosody"]]

    def test_a_cycle_names_the_nodes_involved(self, graph):
        spec = {"a": (("b",), Phase.EXTRACT), "b": (("a",), Phase.EXTRACT)}
        with pytest.raises(DependencyCycleError, match="a, b"):
            graph(spec).topological_order()


class TestEdges:
    def test_dependencies_outside_the_graph_are_ignored(self, graph):
        """A node added on its own must not wait forever on something nobody scheduled."""
        graph(LINE)
        dag = DAG(None)
        dag.add("turns")
        assert dag.dependencies("turns") == set()

    def test_dependencies_inside_the_graph_are_kept(self, graph):
        assert graph(LINE, requested=["turns"]).dependencies("turns") == {"asr", "prep"}

    def test_ready_only_offers_what_is_unblocked(self, graph):
        dag = graph(LINE, requested=["turns"])
        assert dag.ready(set()) == ["prep"]
        assert dag.ready({"prep"}) == ["asr"]
        assert dag.ready({"prep", "asr"}) == ["turns"]

    def test_ready_never_offers_something_already_done(self, graph):
        assert "prep" not in graph(LINE).ready({"prep"})


class TestPhases:
    def test_by_phase_keeps_topological_order(self, graph):
        spec = {
            "prep": ((), Phase.PREP),
            "asr": (("prep",), Phase.EXTRACT),
            "prosody": (("prep",), Phase.EXTRACT),
        }
        assert graph(spec).by_phase(Phase.EXTRACT) == ["asr", "prosody"]

    def test_an_interactive_node_is_a_gate(self, graph):
        spec = {
            "prep": ((), Phase.PREP),
            "align.review": (("prep",), Phase.INTERACTIVE),
            "asr": (("align.review",), Phase.EXTRACT),
        }
        assert graph(spec).interactive_gates() == ["align.review"]

    def test_batch_segments_break_at_the_gate(self, graph):
        """A cluster script stops here and prints what to run by hand; it cannot submit past it."""
        spec = {
            "prep": ((), Phase.PREP),
            "align.review": (("prep",), Phase.INTERACTIVE),
            "asr": (("align.review",), Phase.EXTRACT),
        }
        assert graph(spec).batch_segments() == [["prep"], ["asr"]]

    def test_without_a_gate_everything_is_one_segment(self, graph):
        assert graph(LINE).batch_segments() == [["prep", "asr", "turns"]]

    def test_a_gate_at_the_front_leaves_one_segment(self, graph):
        spec = {"review": ((), Phase.INTERACTIVE), "asr": (("review",), Phase.EXTRACT)}
        assert graph(spec).batch_segments() == [["asr"]]


class TestExport:
    def test_cytoscape_lists_every_node_and_edge(self, graph):
        payload = graph(LINE, requested=["turns"]).to_cytoscape()
        nodes = [item for item in payload["elements"] if "id" in item["data"]]
        edges = [item for item in payload["elements"] if "source" in item["data"]]
        assert len(nodes) == 3
        assert {(e["data"]["source"], e["data"]["target"]) for e in edges} == {
            ("prep", "asr"),
            ("prep", "turns"),
            ("asr", "turns"),
        }

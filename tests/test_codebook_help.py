from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from macaw.cli import codebook as command
from macaw.cli.main import app
from macaw.coding.codebook import Codebook, CodeDefinition, read_codebook, write_codebook
from macaw.coding.llm import judge
from macaw.coding.llm.backends import LLMResponse
from macaw.coding.llm.draft import draft_codebook, draft_text, sample_units
from macaw.coding.llm.refine import changes_text, gather_evidence, refine_codebook
from macaw.config.loader import resolve_config
from macaw.exceptions import LLMBackendError, MacawError
from macaw.extractors.audio.content import asr
from macaw.project.layout import layout_for
from macaw.store.annotations import write_annotations
from macaw.store.schema import ANNOTATION_SCHEMA, coerce_frame
from test_llm_coding import RECORDING, project_with, transcript_of

CODEBOOK = Codebook(
    name="acts",
    unit="utterance",
    codes=[
        CodeDefinition("question", "Asks for information.", criteria=["ends in a question"]),
        CodeDefinition("inform", "Gives information.", positive_examples=["it is red"]),
        CodeDefinition("greet", "Says hello.", criteria=["a greeting"]),
    ],
)
TALK = [
    ("mother", 0.0, 2.0, "where is it?"), ("child", 2.5, 3.0, "it is red"), ("mother", 4.0, 5.0, "is it red"),
    ("child", 6.0, 7.0, "hello"), ("mother", 8.0, 9.0, "you know where"),
]
MODEL = "llm:vllm/tiny"


class Model:
    """Answers each request with what answer(system, user) gives, and keeps the requests."""

    is_local, reporting, takes_seed, name = True, None, True, "fake"

    def __init__(self, answer):
        self.answer = answer
        self.requests: list[tuple[str, str]] = []

    def complete(self, system, user, *, seed=None, images=()):
        self.requests.append((system, user))
        return LLMResponse(text=json.dumps(self.answer(system, user)), model="tiny")

    def check_available(self):
        pass

    def model_version(self):
        return "tiny@abc"


def codes(annotator, layer, labels, times=None):
    times = times or [(row[0], row[1], row[2]) for row in TALK]
    rows = pd.DataFrame({
        "recording_id": "r1", "subject_id": [time[0] for time in times], "t_start": [time[1] for time in times],
        "t_end": [time[2] for time in times], "label": labels, "tier": "acts", "annotator": annotator, "layer": layer,
        "confidence": 1.0, "abstained": False, "needs_review": False, "group_id": "r1", "session_id": None, "task": None,
        "run": None, "execution_id": "e1", "config_hash": "h1",
    })
    return coerce_frame(rows, ANNOTATION_SCHEMA, "annotations")


@pytest.fixture
def project(tmp_path, monkeypatch):
    write_codebook(CODEBOOK, tmp_path / "acts.yaml")
    config = project_with(tmp_path, {"layers": ["A", "B"], "codebook": "acts.yaml", "llm": {"model": "tiny"}})
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["project"]["roles"] = ["mother", "child"]
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(asr, "load_transcript", transcript_of(TALK))
    monkeypatch.setattr(judge, "load_transcript", transcript_of(TALK))
    return config


def coded(config, model_labels, person_labels=None, *, person="ann"):
    folder = layout_for(resolve_config(config)).annotations_path("r1")
    write_annotations(codes(MODEL, "B", model_labels), folder)
    if person_labels is not None:
        write_annotations(codes(person, "C", person_labels), folder, append=True)


def revised(system, user):
    """A definition of the code asked about, citing the first unit listed and an id that does not exist."""
    code = re.search(r"Rewrite the definition of (\w+) only", system).group(1)
    first = re.search(r"^id (e\d+) ", user, re.MULTILINE).group(1)
    return {"description": f"New {code}.", "criteria": ["asks with words, not only with a rising tone"],
            "boundary_cases": ["a statement said as a question is a question"], "positive_examples": [first, "e99"],
            "negative_examples": [], "why": "the person coded rising statements as questions"}


class CutOff(Model):
    """Stops at max_tokens partway through each answer to a request whose instructions hold `where`."""

    def __init__(self, answer, where):
        super().__init__(answer)
        self.where = where

    def complete(self, system, user, *, seed=None, images=()):
        whole = super().complete(system, user, seed=seed, images=images)
        return LLMResponse(text=whole.text[: len(whole.text) // 2], model="tiny", cut_off=True) if self.where in system else whole


class TestRefine:
    def test_the_units_a_person_and_the_model_coded_differently_decide_what_changes(self, project):
        # the person codes the rising statement a question and the last line a question; the model says inform to both
        coded(project, ["question", "inform", "inform", "greet", "inform"], ["question", "inform", "question", "greet", "question"])
        config = resolve_config(project)
        evidence = gather_evidence(config, CODEBOOK, ["r1"])
        assert evidence.coder == "ann" and evidence.models == [MODEL]
        rising = evidence.units.set_index("words").loc["is it red"]
        assert (rising["speaker"], rising["before"], rising["person"], rising["model"]) == ("A", "B: it is red", "question", "inform")
        model = Model(revised)
        refined, revisions = refine_codebook(model, CODEBOOK, evidence, seed=0)
        # greet has nothing in doubt, so it is not asked about
        assert len(model.requests) == 2 and refined.name == "acts" and refined.unit == "utterance"
        system, user = model.requests[0]
        assert "Units the person coded question and the model coded otherwise:" in user
        assert "person: question, model: inform | speaker A (just before, B: it is red): is it red" in user
        assert "## greet" in system and "Rewrite the definition of question only" in system
        question = refined.get("question")
        assert question.description == "New question." and question.criteria == ["asks with words, not only with a rising tone"]
        # the example is the unit's own words; the made-up id is dropped
        assert question.positive_examples == ["is it red"] and revisions[0].dropped_ids == 1
        assert refined.get("greet") == CODEBOOK.get("greet") and not revisions[2].changed
        # inform's first unit listed is one the person coded question, so it cannot stand as an example of inform
        assert refined.get("inform").positive_examples == ["it is red"] and revisions[1].dropped_ids == 2
        text = changes_text(project.parent / "acts.yaml", evidence, revisions, "tiny@abc")
        assert "from the codes of ann" in text and "participants' own words" in text
        assert "## greet\n\nUnchanged: no unit showed a problem with it." in text
        assert "- r1 mother at 4.0 s (missed; person question, model inform): is it red" in text

    def test_without_a_person_the_units_the_model_was_split_on_are_shown(self, project):
        coded(project, ["question", "inform", "inform", "greet", "inform"])
        folder = layout_for(resolve_config(project)).artifacts_dir("r1", judge.JudgementExtractor.name)
        folder.mkdir(parents=True)
        answers = {4.0: ["inform", "inform", "question"], 8.0: ["inform", "question", "question"]}
        sure = {"where is it?": "question", "it is red": "inform", "hello": "greet"}
        rows = [(subject, start, end, label) for subject, start, end, text in TALK for label in answers.get(start, [sure.get(text)] * 3)]
        pd.DataFrame(rows, columns=["subject_id", "t_start", "t_end", "label"]).to_parquet(folder / judge.SAMPLES_NAME)
        evidence = gather_evidence(resolve_config(project), CODEBOOK, ["r1"])
        assert evidence.coder is None
        model = Model(revised)
        refined, revisions = refine_codebook(model, CODEBOOK, evidence, seed=0)
        system, user = model.requests[0]
        assert "Units where the model's answers were split between question and other codes:" in user
        assert "model's answers: inform 2, question 1" in user and "boundary between question and the codes the model hesitated" in system
        # a unit the model was split on is no example of anything
        assert refined.get("question").positive_examples == [] and revisions[0].dropped_ids == 2

    def test_whose_codes_to_go_by_is_said_when_several_people_coded(self, project):
        coded(project, ["question", "inform", "inform", "greet", "inform"], ["question"] * 5)
        write_annotations(codes("bo", "C", ["inform"] * 5), layout_for(resolve_config(project)).annotations_path("r1"), append=True)
        with pytest.raises(MacawError, match="several people coded acts: ann, bo; say whose"):
            gather_evidence(resolve_config(project), CODEBOOK, ["r1"])
        with pytest.raises(MacawError, match="no codes by cy"):
            gather_evidence(resolve_config(project), CODEBOOK, ["r1"], coder="cy")
        assert gather_evidence(resolve_config(project), CODEBOOK, ["r1"], coder="bo").coder == "bo"

    def test_nothing_to_refine_before_the_model_has_coded(self, project):
        write_annotations(codes("ann", "C", ["question"] * 5), layout_for(resolve_config(project)).annotations_path("r1"))
        with pytest.raises(MacawError, match="run coding.judge first"):
            gather_evidence(resolve_config(project), CODEBOOK, ["r1"])

    def test_the_command_writes_a_new_codebook_and_the_changes_never_over_the_one_in_use(self, project, monkeypatch):
        coded(project, ["question", "inform", "inform", "greet", "inform"], ["question", "inform", "question", "greet", "question"])
        monkeypatch.setattr(command, "resolve_llm_backend", lambda cfg: Model(revised))
        refused = CliRunner().invoke(app, ["codebook", "refine", str(project.parent / "acts.yaml"), "-c", str(project)])
        assert refused.exit_code != 0 and "never written over" in str(refused.exception)
        out = project.parent / "acts_refined.yaml"
        result = CliRunner().invoke(app, ["codebook", "refine", str(out), "-c", str(project)])
        assert result.exit_code == 0, result.output
        assert "question: revised, 2 unit(s) in doubt" in result.output and "greet: unchanged" in result.output
        assert read_codebook(out).get("question").description == "New question."
        assert (project.parent / "acts_refined.changes.md").is_file() and "coding.judges: {refined:" in result.output
        assert "Judge it on other recordings" in result.output and "in r1. Judge it" in (project.parent / "acts_refined.changes.md").read_text(encoding="utf-8")


def described(system, user):
    if "Descriptions:" in user:
        return {"codes": [
            {"code": "Asks Something", "description": "Asks.", "criteria": ["wants an answer"], "labels": ["asks where", "asks again", "made up"]},
            {"code": "tells", "description": "Tells.", "labels": ["tells colour", "asks where"]},
            {"code": "!!", "description": "no name", "labels": ["greets"]},
        ]}
    found = re.findall(r"^id (d\d+) \| .*: (.*)$", user, re.MULTILINE)
    said = {"where is it?": "Asks  where", "you know where": "asks where", "it is red": "tells colour", "hello": "greets"}
    return {"labels": [{"id": unit_id, "label": said.get(text, "")} for unit_id, text in found]}


class TestDraft:
    def test_units_are_described_then_gathered_into_codes_with_their_own_words_as_examples(self, monkeypatch):
        monkeypatch.setattr(judge, "load_transcript", transcript_of(TALK))
        lines = {"r1": judge.coding_units(None, RECORDING, "utterance")}
        units = sample_units(lines, 10, np.random.default_rng(0))
        assert units["id"].tolist() == ["d1", "d2", "d3", "d4", "d5"] and units["before"].iloc[1] == "A: where is it?"
        assert len(sample_units(lines, 3, np.random.default_rng(0))) == 3
        model = Model(described)
        made = draft_codebook(model, "what parents ask", units, name="acts", unit="utterance", batch_size=2)
        assert len(model.requests) == 4 and "What the codes are for: what parents ask" in model.requests[0][0]
        assert [code.code for code in made.codebook.codes] == ["asks_something", "tells"]
        asks = made.codebook.get("asks_something")
        assert asks.criteria == ["wants an answer"] and sorted(asks.positive_examples) == ["where is it?", "you know where"]
        # a description claimed twice stays with the first code; a code without a usable name and an invented description are dropped
        assert made.gathered == {"asks_something": ["asks where"], "tells": ["tells colour"]}
        assert made.left_out == {"greets": 1} and made.units["code"].fillna("").tolist() == ["asks_something", "tells", "", "", "asks_something"]
        text = draft_text(made, "what parents ask", "tiny@abc", 1)
        assert "## asks_something (2 units)" in text and "- greets (1)" in text and "1 unit(s) got no description" in text

    def test_an_answer_cut_off_is_refused_not_read_as_a_shorter_one(self, project, monkeypatch):
        monkeypatch.setattr(judge, "load_transcript", transcript_of(TALK))
        units = sample_units({"r1": judge.coding_units(None, RECORDING, "utterance")}, 10, np.random.default_rng(0))
        for request in ("You read units", "Gather them"):
            with pytest.raises(LLMBackendError, match="cut off at coding.llm.max_tokens"):
                draft_codebook(CutOff(described, request), "what parents ask", units, name="acts", unit="utterance", batch_size=2)
        coded(project, ["question", "inform", "inform", "greet", "inform"], ["question", "inform", "question", "greet", "question"])
        evidence = gather_evidence(resolve_config(project), CODEBOOK, ["r1"])
        with pytest.raises(LLMBackendError, match="cut off at coding.llm.max_tokens"):
            refine_codebook(CutOff(revised, "Rewrite the definition"), CODEBOOK, evidence, seed=0)

    def test_too_many_codes_are_merged_on_request_and_kept_to_the_largest_if_still_too_many(self, monkeypatch):
        from macaw.coding.llm import draft as draft_module

        units = pd.DataFrame({"id": [f"d{i}" for i in range(1, 7)], "speaker": "A", "before": "", "text": [f"line {i}" for i in range(1, 7)]})
        labels = ["asks", "asks", "wonders", "tells", "says", "hums"]

        def answer(merging):
            def respond(system, user):
                if "Units:" in user:
                    found = re.findall(r"^id d(\d+) ", user, re.MULTILINE)
                    return {"labels": [{"id": f"d{i}", "label": labels[int(i) - 1]} for i in found]}
                if "Descriptions:" in user:
                    return {"codes": [{"code": label, "description": label, "labels": [label]} for label in dict.fromkeys(labels)]
                            + [{"code": "empty", "description": "holds nothing", "labels": []}]}
                return merging
            return respond

        monkeypatch.setattr(draft_module, "MAX_CODES", 3)
        model = Model(answer({"codes": [
            {"code": "question", "description": "Asks.", "merges": ["asks", "wonders"]},
            {"code": "statement", "description": "Tells.", "merges": ["tells", "says"]},
        ]}))
        made = draft_codebook(model, "acts", units, name="acts", unit="utterance", batch_size=6)
        merge_system, merge_user = model.requests[-1]
        assert "Below are 5 codes" in merge_system and "- asks (2): asks" in merge_user and "empty" not in merge_user
        # hums, placed nowhere by the merge, stays as it was
        assert [code.code for code in made.codebook.codes] == ["question", "statement", "hums"] and made.proposed == [6, 3]
        assert "first proposed 6 codes" in draft_text(made, "acts", "tiny@abc", 1)

        stubborn = Model(answer({"codes": [{"code": label, "description": label, "merges": [label]} for label in dict.fromkeys(labels)]}))
        made = draft_codebook(stubborn, "acts", units, name="acts", unit="utterance", batch_size=6)
        assert made.proposed == [6, 5, 5] and len(stubborn.requests) == 4
        assert [code.code for code in made.codebook.codes][0] == "asks" and len(made.codebook.codes) == 3
        assert sum(made.left_out.values()) == 2 and "descriptions of the others are under Left out" in draft_text(made, "acts", "tiny@abc", 1)

    def test_the_command_writes_the_draft_and_a_note(self, project, monkeypatch):
        monkeypatch.setattr(command, "resolve_llm_backend", lambda cfg: Model(described))
        out = project.parent / "first.yaml"
        result = CliRunner().invoke(app, ["codebook", "draft", str(out), "--aim", "what parents ask", "-c", str(project)])
        assert result.exit_code == 0, result.output
        drafted = read_codebook(out)
        assert drafted.name == "first" and [code.code for code in drafted.codes] == ["asks_something", "tells"]
        assert (project.parent / "first.draft.md").is_file() and "participants' own words" in result.output

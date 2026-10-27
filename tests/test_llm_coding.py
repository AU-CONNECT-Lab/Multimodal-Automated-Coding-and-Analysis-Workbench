from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.coding.codebook import Codebook, CodeDefinition, write_codebook
from macaw.coding.llm import backends as backends_module
from macaw.coding.llm import judge as module
from macaw.coding.llm.backends import AnthropicBackend, OllamaBackend, OpenAIBackend, VLLMBackend
from macaw.coding.llm.judge import JudgementExtractor, aggregate_samples, coding_units, sample_labels
from macaw.coding.llm.parsing import parse_annotation_response, repair_json
from macaw.coding.llm.prompts import ABSTAIN, build_annotation_prompt
from macaw.coding.review_queue import REVIEWED_NAME, apply_review, export_review, read_queue
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.config.schema import CodingSection, LLMSection
from macaw.exceptions import BackendNotAvailableError, CodebookError, LLMBackendError
from macaw.execution.dag import build_dag
from macaw.project.layout import layout_for
from macaw.store.annotations import write_annotations
from macaw.store.schema import ANNOTATION_SCHEMA, LONG_TABLE_SCHEMA, coerce_frame, validate_frame

CODEBOOK = Codebook(
    name="acts",
    unit="utterance",
    codes=[
        CodeDefinition("question", "Asks for information.", criteria=["ends in a question"]),
        CodeDefinition("inform", "Gives information.", positive_examples=["it is red"]),
    ],
)


class Stub:
    """A model server on localhost: labels every unit by a rule, counts requests, can fail first."""

    def __init__(self, label=lambda text: "question" if "?" in text else "inform"):
        self.label = label
        self.requests: list[dict] = []
        self.fail_first = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.endswith("/models"):
                    self._send(200, {"data": [{"id": "tiny"}]})
                elif self.path == "/api/tags":
                    self._send(200, {"models": [{"name": "tiny", "digest": "abcdef1234567890"}]})
                else:
                    self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"path": self.path, "body": body, "headers": {key.lower(): value for key, value in self.headers.items()}})
                if stub.fail_first:
                    stub.fail_first -= 1
                    self._send(503, {"error": "busy"})
                    return
                user = body["messages"][-1]["content"]
                labels = [
                    {"id": unit_id, "label": stub.label(text)}
                    for unit_id, text in re.findall(r"^id (u\d+) \|.*?: (.*)$", user, re.MULTILINE)
                ]
                text = json.dumps({"labels": labels})
                if self.path.endswith("/messages"):
                    self._send(200, {"model": "claude-x", "content": [{"type": "text", "text": text}], "usage": {"input_tokens": 5, "output_tokens": 3}})
                else:
                    self._send(200, {"model": "tiny", "choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def stub():
    server = Stub()
    yield server
    server.server.shutdown()


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(backends_module.time, "sleep", lambda seconds: None)


def units(rows):
    frame = pd.DataFrame(rows, columns=["subject_id", "t_start", "t_end", "text"])
    frame["utterance_index"] = frame.groupby("subject_id").cumcount()
    return frame


RECORDING = SimpleNamespace(
    recording_id="r1",
    subjects=[SimpleNamespace(subject_id="mother"), SimpleNamespace(subject_id="child")],
    entities=lambda: {"group_id": "1", "session_id": None, "task": None, "run": None},
)
TALK = [("mother", 0.0, 2.0, "where is it?"), ("child", 1.5, 3.0, "it is red"), ("mother", 4.0, 5.0, "ok")]


class TestRepair:
    def test_fenced_and_wrapped_in_prose(self):
        text = 'Sure! Here:\n```json\n{"labels": [{"id": "u0", "label": "inform"}]}\n```\nHope that helps.'
        assert json.loads(repair_json(text)) == {"labels": [{"id": "u0", "label": "inform"}]}

    def test_trailing_remark_after_the_object(self):
        assert json.loads(repair_json('{"a": [1, 2]} and that is all {not json}')) == {"a": [1, 2]}

    def test_truncated_answer_keeps_the_complete_items(self):
        text = '{"labels": [{"id": "u0", "label": "inform"}, {"id": "u1", "lab'
        assert json.loads(repair_json(text)) == {"labels": [{"id": "u0", "label": "inform"}]}

    def test_thinking_before_the_answer_is_dropped(self):
        text = '<think>maybe {"labels": []} or not</think>\n{"labels": [{"id": "u0", "label": "inform"}]}'
        assert json.loads(repair_json(text))["labels"] == [{"id": "u0", "label": "inform"}]

    def test_braces_inside_strings_do_not_count(self):
        assert json.loads(repair_json('{"id": "u}0", "label": "a{b"} trailing')) == {"id": "u}0", "label": "a{b"}


class TestParse:
    def test_missing_unknown_and_unreadable_answers_are_abstentions(self):
        text = '{"labels": [{"id": "u0", "label": "Question"}, {"id": "u1", "label": "joke"}]}'
        parsed = parse_annotation_response(text, CODEBOOK, ["u0", "u1", "u2"])
        assert parsed["label"].tolist() == ["question", ABSTAIN, ABSTAIN]
        assert parse_annotation_response("no json here", CODEBOOK, ["u0"])["label"].tolist() == [ABSTAIN]

    def test_a_bare_list_is_read_too(self):
        parsed = parse_annotation_response('[{"id": "u0", "label": "inform"}]', CODEBOOK, ["u0"])
        assert parsed["label"].tolist() == ["inform"]

    def test_ids_not_asked_for_are_ignored(self):
        parsed = parse_annotation_response('{"labels": [{"id": "u9", "label": "inform"}]}', CODEBOOK, ["u0"])
        assert parsed["unit_id"].tolist() == ["u0"]


class TestUnitsAndPrompt:
    def test_units_get_letters_times_and_the_gap_before(self, monkeypatch):
        monkeypatch.setattr(module, "load_transcript", lambda config, rid: units(TALK))
        found = coding_units(None, RECORDING, "utterance")
        assert found["speaker"].tolist() == ["A", "B", "A"]
        assert found["unit_id"].tolist() == ["u0", "u1", "u2"]
        assert pd.isna(found["gap_before"].iloc[0])
        assert found["gap_before"].iloc[1:].round(2).tolist() == [-0.5, 1.0]

    def test_the_prompt_names_letters_not_subject_ids_and_marks_context(self, monkeypatch):
        monkeypatch.setattr(module, "load_transcript", lambda config, rid: units(TALK))
        found = coding_units(None, RECORDING, "utterance")
        system, user = build_annotation_prompt(CODEBOOK, found.iloc[1:], found.iloc[:1])
        assert "## question" in system and "ends in a question" in system and '"abstain"' in system
        assert "mother" not in user and "child" not in user
        assert user.index("context only") < user.index("Units to code:") < user.index("id u1 |")
        assert "id u0 |" not in user
        assert "overlaps the talk before by 0.5 s" in user


class TestAggregate:
    def samples(self, answers):
        return pd.DataFrame(
            [(unit_id, sample, label) for unit_id, labels in answers.items() for sample, label in enumerate(labels)],
            columns=["unit_id", "sample", "label"],
        )

    def test_majority_tie_and_abstention(self):
        found = pd.DataFrame({"unit_id": ["u0", "u1", "u2", "u3"]})
        samples = self.samples({
            "u0": ["inform"] * 4 + ["question"],
            "u1": ["inform"] * 3 + ["question"] * 2,
            "u2": ["inform", "inform", "question", "question", ABSTAIN],
            "u3": [ABSTAIN] * 3 + ["inform"] * 2,
        })
        out = aggregate_samples(found, samples, LLMSection(n_samples=5)).set_index("unit_id")
        assert out.loc["u0", "label"] == "inform" and out.loc["u0", "confidence"] == 0.8 and not out.loc["u0", "needs_review"]
        assert out.loc["u1", "confidence"] == 0.6 and out.loc["u1", "needs_review"]
        assert pd.isna(out.loc["u2", "label"]) and not out.loc["u2", "abstained"] and out.loc["u2", "needs_review"]
        assert pd.isna(out.loc["u3", "label"]) and out.loc["u3", "abstained"] and out.loc["u3", "needs_review"]

    def test_one_sample_measures_no_confidence(self):
        found = pd.DataFrame({"unit_id": ["u0", "u1"]})
        out = aggregate_samples(found, self.samples({"u0": ["inform"], "u1": [ABSTAIN]}), LLMSection(n_samples=1))
        assert out["confidence"].isna().all()
        assert out["needs_review"].tolist() == [False, True]


class TestBackends:
    def test_openai_compatible_request_carries_model_seed_and_json_mode(self, stub):
        backend = VLLMBackend(LLMSection(provider="vllm", model="tiny", base_url=f"{stub.url}/v1"))
        backend.check_available()
        response = backend.complete("system", "id u0 | speaker A | at 0.0 s: hi?", seed=7)
        body = stub.requests[0]["body"]
        assert body["model"] == "tiny" and body["seed"] == 7 and body["response_format"] == {"type": "json_object"}
        assert body["messages"][0] == {"role": "system", "content": "system"}
        assert json.loads(response.text) == {"labels": [{"id": "u0", "label": "question"}]}
        assert response.prompt_tokens == 5

    def test_ollama_records_the_digest_not_only_the_tag(self, stub):
        backend = OllamaBackend(LLMSection(model="tiny", base_url=f"{stub.url}/v1"))
        assert backend.model_version() == "tiny@abcdef123456"

    def test_a_busy_server_is_retried(self, stub):
        stub.fail_first = 2
        backend = VLLMBackend(LLMSection(provider="vllm", model="tiny", base_url=f"{stub.url}/v1"))
        backend.complete("s", "u")
        assert len(stub.requests) == 3

    def test_a_server_that_stays_down_is_an_error_not_an_abstention(self, stub):
        stub.fail_first = 10
        backend = VLLMBackend(LLMSection(provider="vllm", model="tiny", base_url=f"{stub.url}/v1"))
        with pytest.raises(LLMBackendError, match="503"):
            backend.complete("s", "u")

    def test_no_server_and_no_model_are_reported_before_the_run(self, stub):
        with pytest.raises(BackendNotAvailableError, match="coding.llm.model"):
            OllamaBackend(LLMSection(base_url=f"{stub.url}/v1")).check_available()
        stub.server.shutdown()
        stub.server.server_close()
        with pytest.raises(BackendNotAvailableError, match="No ollama server"):
            OllamaBackend(LLMSection(model="tiny", base_url=f"{stub.url}/v1")).check_available()

    def test_cloud_keys_come_from_the_environment_only(self, monkeypatch, stub):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(BackendNotAvailableError, match="OPENAI_API_KEY"):
            OpenAIBackend(LLMSection(provider="openai", model="x")).check_available()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        backend = OpenAIBackend(LLMSection(provider="openai", model="x", base_url=f"{stub.url}/v1"))
        backend.complete("s", "u")
        assert stub.requests[0]["headers"]["authorization"] == "Bearer sk-test"
        assert backend.model_version() == "tiny"

    def test_anthropic_messages_api(self, monkeypatch, stub):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        backend = AnthropicBackend(LLMSection(provider="anthropic", model="claude-x", base_url=f"{stub.url}/v1"))
        backend.check_available()
        response = backend.complete("system text", "id u0 | speaker A | at 0.0 s: hi", seed=3)
        request = stub.requests[0]
        assert request["path"] == "/v1/messages" and request["body"]["system"] == "system text"
        assert "seed" not in request["body"] and request["headers"]["x-api-key"] == "key"
        assert json.loads(response.text)["labels"][0]["label"] == "inform"
        assert response.prompt_tokens == 5 and backend.model_version() == "claude-x"


def run_judge(monkeypatch, tmp_path, stub, **llm):
    path = write_codebook(CODEBOOK, tmp_path / "codebook.yaml")
    monkeypatch.setattr(module, "load_transcript", lambda config, rid: units(TALK))
    cfg = LLMSection(provider="vllm", model="tiny", base_url=f"{stub.url}/v1", **llm)
    extractor = JudgementExtractor(SimpleNamespace(coding=CodingSection(codebook=path, layers=["A", "B"], llm=cfg)))
    return extractor.run(RECORDING, tmp_path)


class TestJudge:
    def test_every_unit_is_coded_with_confidence_and_samples_are_kept(self, monkeypatch, tmp_path, stub):
        result = run_judge(monkeypatch, tmp_path, stub, n_samples=3, batch_size=2)
        assert len(stub.requests) == 3 * 2
        assert [request["body"]["seed"] for request in stub.requests] == [0, 0, 1, 1, 2, 2]
        annotations = result.annotations.sort_values("t_start")
        assert annotations["label"].tolist() == ["question", "inform", "inform"]
        assert annotations["subject_id"].tolist() == ["mother", "child", "mother"]
        assert (annotations["annotator"] == "llm:vllm/tiny").all() and (annotations["tier"] == "acts").all()
        assert (annotations["confidence"] == 1.0).all() and not annotations["needs_review"].any()
        samples = pd.read_parquet(result.artifacts["samples.parquet"])
        assert len(samples) == 9 and set(samples.columns) >= {"unit_id", "sample", "label", "subject_id", "t_start"}
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        counts = result.rows.set_index(["subject_id", "feature"])["value"]
        assert counts[("mother", "n_units_judged")] == 2 and counts[("child", "share_needs_review")] == 0

    def test_disagreeing_samples_send_the_unit_to_review(self, monkeypatch, tmp_path, stub):
        answers = iter(["question", "inform", "question"] * 10)
        stub.label = lambda text: next(answers) if text == "ok" else "inform"
        result = run_judge(monkeypatch, tmp_path, stub, n_samples=3, batch_size=10)
        flagged = result.annotations[result.annotations["needs_review"]]
        assert flagged["t_start"].tolist() == [4.0]
        assert any(warning.check == "llm_review" for warning in result.warnings)

    def test_a_model_that_never_answers_in_format_abstains_and_warns(self, monkeypatch, tmp_path, stub):
        stub.label = lambda text: "something else"
        result = run_judge(monkeypatch, tmp_path, stub, n_samples=1)
        assert result.annotations["abstained"].all()
        assert {warning.check for warning in result.warnings} >= {"llm_mostly_abstained", "llm_single_sample"}

    def test_codebooks_it_cannot_code_are_refused(self, tmp_path):
        config = SimpleNamespace(coding=CodingSection(codebook=None))
        with pytest.raises(BackendNotAvailableError, match="coding.codebook"):
            JudgementExtractor(config)._codebook()
        several = Codebook(name="x", unit="utterance", mutually_exclusive=False, codes=CODEBOOK.codes)
        config = SimpleNamespace(coding=CodingSection(codebook=write_codebook(several, tmp_path / "c.yaml")))
        with pytest.raises(CodebookError, match="mutually exclusive"):
            JudgementExtractor(config)._codebook()

    def test_judge_only_runs_by_default_when_layer_b_is_asked_for(self, tmp_path):
        raw = tmp_path / "raw" / "r1"
        raw.mkdir(parents=True)
        config = tmp_path / "project.yaml"
        scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
        assert "coding.judge" not in build_dag(resolve_config(config)).nodes
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
        data.setdefault("coding", {})["layers"] = ["A", "B"]
        config.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert "coding.judge" in build_dag(resolve_config(config)).nodes


class TestReview:
    @pytest.fixture
    def project(self, tmp_path):
        (tmp_path / "raw" / "r1").mkdir(parents=True)
        config = tmp_path / "project.yaml"
        scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
        resolved = resolve_config(config)
        rows = pd.DataFrame({
            "recording_id": "r1", "subject_id": ["a", "b", "a"], "tier": "acts", "t_start": [0.0, 2.0, 4.0],
            "t_end": [1.0, 3.0, 5.0], "label": ["inform", "question", None], "annotator": "llm:vllm/tiny",
            "layer": "B", "confidence": [1.0, 0.6, 0.4], "abstained": [False, False, True],
            "needs_review": [False, True, True], "group_id": "1", "session_id": None, "task": None, "run": None,
            "execution_id": "e1", "config_hash": "h1",
        })
        write_annotations(coerce_frame(rows, ANNOTATION_SCHEMA, "annotations"), layout_for(resolved).annotations_path("r1"))
        return SimpleNamespace(config=config, resolved=resolved)

    def test_export_fill_in_apply_and_export_again(self, project, tmp_path):
        result = CliRunner().invoke(app, ["review", "-c", str(project.config), "--export", str(tmp_path / "q.csv")])
        assert result.exit_code == 0, result.output
        queue = read_queue(tmp_path / "q.csv")
        assert queue["t_start"].tolist() == [2.0, 4.0]

        queue.loc[queue["t_start"] == 2.0, ["decision", "reviewer"]] = ["inform", "ann"]
        queue.to_csv(tmp_path / "q.csv", index=False)
        result = CliRunner().invoke(app, ["review", "-c", str(project.config), "--apply", str(tmp_path / "q.csv")])
        assert result.exit_code == 0, result.output
        assert "r1: 1" in result.output

        reviewed = pd.read_parquet(layout_for(project.resolved).annotations_path("r1") / REVIEWED_NAME)
        assert reviewed[["annotator", "label", "t_start"]].values.tolist() == [["ann", "inform", 2.0]]
        assert export_review(project.resolved, ["r1"])["t_start"].tolist() == [4.0]

        # the same decision applied twice is still one row
        apply_review(project.resolved, queue)
        assert len(pd.read_parquet(layout_for(project.resolved).annotations_path("r1") / REVIEWED_NAME)) == 1

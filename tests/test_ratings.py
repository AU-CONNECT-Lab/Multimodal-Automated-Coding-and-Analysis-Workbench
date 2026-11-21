from __future__ import annotations

import json
import re
import shutil
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from llm_helpers import CountingBackend
from pydantic import ValidationError

from macaw.coding.llm import backends as backends_module
from macaw.coding.llm import rate as module
from macaw.coding.llm.parsing import parse_rating
from macaw.coding.llm.prompts import build_rating_prompt
from macaw.coding.scales import ALL, RatingScale, ScaleSet, read_scales, validate_scales
from macaw.config.hashing import hash_extractor_config
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.config.schema import CodingSection, LLMSection, ProjectSection
from macaw.exceptions import CodebookError, ConfigError, InputNotFoundError, LLMBackendError
from macaw.execution.dag import build_dag
from macaw.extractors.audio.content.asr import TRANSCRIPT_SCHEMA, WORD_SCHEMA, ASRExtractor
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.store import _io

SCALES = {
    "name": "talk",
    "scales": [
        {
            "scale": "warmth", "person": "parent", "range": [1, 5], "definition": "How warm the parent is.",
            "anchors": {1: "cold throughout", 3: "some warmth", 5: "warm throughout"}, "rules": ["Laughing together counts."],
        },
        {"scale": "on_task", "person": "child", "range": [0, 1], "anchors": {0: "mostly off task", 1: "mostly on task"}, "definition": "Keeps to the task."},
        {"scale": "flow", "person": ALL, "range": [1, 5], "definition": "How smoothly the two take turns."},
    ],
}
NAMES_ONLY = {"name": "talk", "scales": [{"scale": s["scale"], "person": s["person"], "range": s["range"]} for s in SCALES["scales"]]}
TALK = [
    ("parent", 1.0, 3.0, "How was school today?"),
    ("child", 3.5, 5.0, "It was fine."),
    ("parent", 4.8, 7.0, "Just fine?"),
    ("child", 20.0, 22.0, "We had a test."),
]


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


class Server:
    """An OpenAI-compatible model server on localhost that scores each scale by a rule and says its context."""

    def __init__(self):
        self.score = lambda scale, sample: json.dumps({"score": 3})
        self.requests: list[dict] = []
        self.context = None
        self.cut = False
        self.usage = {"prompt_tokens": 5, "completion_tokens": 3}
        self.reasoning = None
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.endswith("/models"):
                    self._send({"data": [{"id": "tiny"}]})
                elif self.path == "/api/ps":
                    self._send({"models": [{"name": "tiny", "context_length": server.context}] if server.context else []})
                elif self.path == "/api/tags":
                    self._send({"models": [{"name": "tiny", "digest": "abcdef1234567890"}]})
                else:
                    self._send({})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(body)
                user = body["messages"][-1]["content"]
                scale = re.search(r"^Scale: (\S+)$", user, re.MULTILINE).group(1)
                sample = sum(1 for request in server.requests if f"Scale: {scale}\n" in request["messages"][-1]["content"]) - 1
                text = "" if server.cut else server.score(scale, sample)
                message = {"content": text, **({"reasoning": server.reasoning} if server.reasoning else {})}
                choice = {"message": message, "finish_reason": "length" if server.cut else "stop"}
                self._send({"model": "tiny", "choices": [choice], "usage": server.usage})

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_port}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()


@pytest.fixture
def server():
    found = Server()
    yield found
    found.http.shutdown()


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(backends_module.time, "sleep", lambda seconds: None)


def project(tmp_path, ratings, *, llm=None, roles=("parent", "child"), segments=None):
    (tmp_path / "raw" / "r1").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "project.yaml"
    scaffold_project_config(path, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out", roles=("parent", "child"))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["project"]["roles"] = list(roles)
    data["coding"] = {"llm": llm or {"model": "tiny"}, "ratings": ratings}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    write_yaml(tmp_path / "scales.yaml", SCALES)
    write_yaml(tmp_path / "names.yaml", NAMES_ONLY)
    if segments:
        pd.DataFrame(segments, columns=["recording", "segment", "start_s", "end_s"]).to_csv(tmp_path / "segments.csv", index=False)
    return path


def transcript_frame(talk, rid="r1"):
    frame = pd.DataFrame(talk, columns=["subject_id", "t_start", "t_end", "text"])
    frame["recording_id"] = rid
    frame["utterance_index"] = frame.groupby("subject_id").cumcount()
    frame["n_words"] = frame["text"].str.split().str.len()
    for column in ("confidence", "avg_logprob", "no_speech_prob"):
        frame[column] = float("nan")
    frame["language"] = "en"
    return frame[list(TRANSCRIPT_SCHEMA)]


def words_frame(talk, rid="r1"):
    rows = []
    frame = transcript_frame(talk, rid)
    for row in frame.itertuples(index=False):
        tokens = row.text.split()
        step = (row.t_end - row.t_start) / len(tokens)
        rows += [
            (rid, row.subject_id, row.utterance_index, row.t_start + i * step, row.t_start + (i + 1) * step, token, 1.0)
            for i, token in enumerate(tokens)
        ]
    return pd.DataFrame(rows, columns=list(WORD_SCHEMA))


def transcribed(config_path, talk=TALK, *, where=None, rid="r1"):
    """text.asr's output for rid, or the same tables at where."""
    config = resolve_config(config_path)
    folder = layout_for(config).artifacts_dir(rid, ASRExtractor.name)
    folder.mkdir(parents=True, exist_ok=True)
    transcript, words = where or (folder / "transcript.parquet", folder / "words.parquet")
    _io.write_file(transcript_frame(talk, rid), transcript, TRANSCRIPT_SCHEMA, "transcript")
    _io.write_file(words_frame(talk, rid), words, WORD_SCHEMA, "words")
    return config


def rate(monkeypatch, config, name, window=(0.0, 60.0), rid="r1"):
    monkeypatch.setattr(module, "load_prep", lambda config, rid: SimpleNamespace(t_origin_s=window[0], duration_s=window[1] - window[0]))
    step = module._VERSIONS[name](config)
    result = step.run(build_recording(rid, config), layout_for(config).recording_dir(rid))
    # as the runner does once a task has finished
    shutil.rmtree(layout_for(config).unfinished_path(rid, step.name), ignore_errors=True)
    return result


class TestScales:
    def test_partial_anchors_and_everyone_together_are_read(self, tmp_path):
        scales = read_scales(write_yaml(tmp_path / "s.yaml", SCALES))
        warmth = scales.get("warmth")
        assert (warmth.low, warmth.high, sorted(warmth.anchors)) == (1, 5, [1, 3, 5]) and warmth.rules == ["Laughing together counts."]
        assert scales.get("flow").person == ALL
        assert validate_scales(scales, ["parent", "child"]) == []

    def test_names_alone_are_a_choice_and_half_described_is_a_draft(self, tmp_path):
        names = read_scales(write_yaml(tmp_path / "k0.yaml", NAMES_ONLY))
        assert names.names_only() and validate_scales(names, ["parent", "child"]) == []
        half = dict(NAMES_ONLY, scales=[SCALES["scales"][0], *NAMES_ONLY["scales"][1:]])
        problems = validate_scales(read_scales(write_yaml(tmp_path / "half.yaml", half)), ["parent", "child"])
        assert any("on_task" in problem and "other scales do" in problem for problem in problems)

    def test_a_person_who_is_not_a_role_and_anchors_outside_the_range_are_problems(self, tmp_path):
        wrong = {"name": "x", "scales": [{"scale": "a", "person": "kid", "range": [1, 3], "anchors": {4: "too high"}, "definition": "d"}]}
        problems = validate_scales(read_scales(write_yaml(tmp_path / "w.yaml", wrong)), ["parent", "child"])
        assert any("kid" in problem for problem in problems) and any("outside its range" in problem for problem in problems)

    @pytest.mark.parametrize("entry,message", [
        ({"scale": "a", "person": "parent", "range": [1, 2.5]}, "whole number"),
        ({"scale": "a", "person": "parent", "range": [1]}, "lowest, highest"),
        ({"scale": "a", "range": [1, 5]}, "no 'person'"),
        ({"scale": "a", "person": "parent", "range": [1, 5], "levels": {}}, "unknown key"),
    ])
    def test_a_scale_that_does_not_read_says_why(self, tmp_path, entry, message):
        with pytest.raises(CodebookError, match=message):
            read_scales(write_yaml(tmp_path / "bad.yaml", {"name": "x", "scales": [entry]}))

    def test_all_cannot_be_a_role(self):
        with pytest.raises(ValidationError, match="all cannot be a role"):
            ProjectSection(name="x", raw_dir=".", output_dir=".", roles=["parent", "all"])


class TestPrompt:
    def lines(self):
        frame = module.rating_lines(transcript_frame(TALK))
        return module.part_lines(frame, words_frame(TALK), 0.0, 60.0)

    def test_roles_name_the_lines_and_the_scale_comes_after_the_whole_transcript(self):
        scales = read_scales_from(SCALES)
        system, user = build_rating_prompt(scales.get("warmth"), self.lines(), people=["parent", "child"], minutes=1.0, show=[])
        assert "The people recorded: parent, child." in system and "Scale" not in system
        assert "parent: How was school today?\nchild: It was fine." in user
        assert user.index("We had a test.") < user.index("Scale: warmth") < user.index("Anchors:")
        assert "Who is rated: the parent" in user and "Scores: whole numbers from 1 to 5." in user
        assert "Scores without an anchor lie between the anchors around them." in user and "- Laughing together counts." in user

    def test_every_scale_of_a_conversation_starts_the_same(self):
        scales = read_scales_from(SCALES)
        asked = [build_rating_prompt(scale, self.lines(), people=["parent", "child"], minutes=1.0, show=["timing"]) for scale in scales.scales]
        assert len({system for system, _ in asked}) == 1
        heads = {user[: user.index("Rate the conversation")] for _, user in asked}
        assert len(heads) == 1

    def test_names_only_give_name_person_and_range(self):
        scales = read_scales_from(NAMES_ONLY)
        _, user = build_rating_prompt(scales.get("flow"), self.lines(), people=["parent", "child"], minutes=1.0)
        assert "Who is rated: the parent and the child together" in user
        assert "Definition" not in user and "Anchors" not in user and "Rules" not in user

    def test_a_full_set_of_anchors_needs_no_word_on_the_scores_between(self):
        _, user = build_rating_prompt(read_scales_from(SCALES).get("on_task"), self.lines(), people=["parent", "child"], minutes=1.0)
        assert "0: mostly off task\n1: mostly on task" in user and "between the anchors" not in user

    def test_timing_counts_from_the_start_of_what_is_rated(self):
        frame = module.rating_lines(transcript_frame(TALK))
        cut = module.part_lines(frame, words_frame(TALK), 2.0, 21.0)
        _, user = build_rating_prompt(read_scales_from(SCALES).get("flow"), cut, people=["parent", "child"], minutes=19 / 60, show=["timing"])
        # the opening question is cut at the edge to its words inside, and its time starts at 0
        assert "parent | at 0.0 s, lasts 1.0 s, opens the conversation: school today?" in user
        assert "overlaps the talk before by 0.2 s: Just fine?" in user
        assert "child | at 18.0 s, lasts 1.0 s, after a 13.0 s pause: We had" in user

    def test_reasons_come_after_the_score_and_notes_go_in_as_written(self):
        system, _ = build_rating_prompt(
            read_scales_from(SCALES).get("flow"), self.lines(), people=["parent", "child"], minutes=1.0, notes="They plan a party.", reasons=True,
        )
        assert '{"score": <score>, "reason": "<what it rests on>"}' in system and "Notes from the researchers: They plan a party." in system


def read_scales_from(data) -> ScaleSet:
    return ScaleSet(data["name"], [module_scale(entry) for entry in data["scales"]])


def module_scale(entry) -> RatingScale:
    return RatingScale(
        scale=entry["scale"], person=entry["person"], low=entry["range"][0], high=entry["range"][1],
        definition=entry.get("definition", ""), anchors=dict(entry.get("anchors") or {}), rules=list(entry.get("rules") or []),
    )


class TestParse:
    @pytest.mark.parametrize("text,expected", [
        ('{"score": 4}', (4, "", "")),
        ('{"score": "4", "reason": "asks often"}', (4, "asks often", "")),
        ('<think>maybe 5</think>{"score": 3.0}', (3, "", "")),
        ('```json\n{"score": 2}\n```', (2, "", "")),
        ('{"score": 3.5}', (None, "", "not a whole number")),
        ('{"score": "three"}', (None, "", "not a whole number")),
        ('{"score": 7}', (None, "", "outside the range")),
        ('{"score": true}', (None, "", "not a whole number")),
        ("I would say 4.", (None, "", "unreadable")),
    ])
    def test_a_score_is_whole_and_in_range_or_it_is_no_score(self, text, expected):
        assert parse_rating(text, 1, 5) == expected


class TestRun:
    def test_each_scale_is_asked_alone_n_times_and_the_median_kept(self, monkeypatch, tmp_path, server):
        server.score = lambda scale, sample: json.dumps({"score": {"warmth": [4, 5, 4], "on_task": [1, 1, 0], "flow": [2, 3, 5]}[scale][sample]})
        config = transcribed(project(tmp_path, {"full": {"scales": "scales.yaml", "show": []}}, llm={"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 3}))
        result = rate(monkeypatch, config, "full")
        assert len(server.requests) == 9
        asked = [re.search(r"^Scale: (\S+)$", body["messages"][-1]["content"], re.MULTILINE).group(1) for body in server.requests]
        assert asked == ["warmth"] * 3 + ["on_task"] * 3 + ["flow"] * 3
        assert [body["seed"] for body in server.requests] == [0, 1, 2] * 3
        ratings = pd.read_parquet(result.artifacts["ratings.parquet"]).set_index("scale")
        assert ratings.loc["warmth", "score"] == 4 and ratings.loc["on_task", "score"] == 1 and ratings.loc["flow", "score"] == 3
        assert ratings.loc["flow", "score_min"] == 2 and ratings.loc["flow", "score_max"] == 5 and ratings.loc["flow", "score_sd"] == pytest.approx(1.5275, abs=1e-4)
        assert set(ratings["rater"]) == {"full"} and set(ratings["segment"]) == {""} and ratings.loc["flow", "person"] == ALL
        assert (ratings["n_answers"] == 3).all() and ratings["group_id"].notna().all()
        answers = pd.read_parquet(result.artifacts["answers.parquet"])
        assert len(answers) == 9 and answers["answer"].iloc[0] == '{"score": 4}' and (answers["problem"] == "").all()
        assert result.rows["feature"].tolist() == ["share_unread_answers_full"] and result.rows["value"].tolist() == [0.0]
        assert result.specs[0].params["n_scales"] == 3 and not result.specs[0].params["names_only"]

    def test_the_text_sent_is_kept_once_per_segment_and_scale(self, monkeypatch, tmp_path, server):
        config = transcribed(project(
            tmp_path, {"parts": {"scales": "scales.yaml", "segments": ["opening", "late"]}},
            llm={"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 2},
            segments=[("r1", "opening", 0, 10), ("r1", "late", 15, 25)],
        ))
        result = rate(monkeypatch, config, "parts")
        kept = pd.read_parquet(result.artifacts["prompts.parquet"])
        assert list(kept.columns) == module.PROMPT_COLUMNS and len(kept) == 6 and len(server.requests) == 12
        assert set(kept["rater"]) == {"parts"} and set(kept["recording_id"]) == {"r1"}
        sent = {(body["messages"][0]["content"], body["messages"][1]["content"]) for body in server.requests}
        assert set(zip(kept["system"], kept["user"])) == sent
        late = kept[(kept["segment"] == "late") & (kept["scale"] == "flow")].iloc[0]
        assert "We had a test." in late["user"] and "school" not in late["user"]

    def test_each_answer_says_the_tokens_and_the_time_it_took(self, monkeypatch, tmp_path, server):
        llm = {"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 1}
        config = transcribed(project(tmp_path, {"full": {"scales": "names.yaml"}}, llm=llm))
        server.usage = {"prompt_tokens": 900, "completion_tokens": 400, "completion_tokens_details": {"reasoning_tokens": 380}}
        answers = pd.read_parquet(rate(monkeypatch, config, "full").artifacts["answers.parquet"])
        assert list(answers.columns) == module.ANSWER_COLUMNS
        assert answers[["prompt_tokens", "completion_tokens", "reasoning_tokens"]].values.tolist() == [[900, 400, 380]] * 3
        assert (answers["seconds"] >= 0).all() and answers["seconds"].notna().all()
        # a server that sends the reasoning back without counting it: the generated tokens are shared out by length
        server.usage = {"prompt_tokens": 900, "completion_tokens": 400}
        server.reasoning = "x" * 30
        answers = pd.read_parquet(rate(monkeypatch, config, "full").artifacts["answers.parquet"])
        assert answers["reasoning_tokens"].tolist() == [round(400 * 30 / (30 + len('{"score": 3}')))] * 3
        server.usage, server.reasoning = {}, None
        answers = pd.read_parquet(rate(monkeypatch, config, "full").artifacts["answers.parquet"])
        assert answers[["prompt_tokens", "completion_tokens", "reasoning_tokens"]].isna().all().all()

    def test_answers_without_a_score_are_counted_and_left_out(self, monkeypatch, tmp_path, server):
        server.score = lambda scale, sample: {"warmth": ['{"score": 9}', '{"score": 2}', "no idea"], "on_task": ['{"score": 1}'] * 3, "flow": ["?"] * 3}[scale][sample]
        config = transcribed(project(tmp_path, {"full": {"scales": "scales.yaml"}}, llm={"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 3}))
        result = rate(monkeypatch, config, "full")
        ratings = pd.read_parquet(result.artifacts["ratings.parquet"]).set_index("scale")
        assert ratings.loc["warmth", "score"] == 2 and ratings.loc["warmth", "n_answers"] == 1 and pd.isna(ratings.loc["flow", "score"])
        checks = {warning.check: warning.message for warning in result.warnings}
        assert checks["rating_unread"].startswith("5 of 9 answers") and "outside the range 1" in checks["rating_unread"]
        assert "flow" in checks["rating_no_score"]
        assert result.rows["value"].tolist() == [pytest.approx(5 / 9)]

    def test_answers_all_cut_off_stop_the_step(self, monkeypatch, tmp_path, server):
        server.cut = True
        config = transcribed(project(tmp_path, {"full": {"scales": "scales.yaml"}}, llm={"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 1}))
        with pytest.raises(LLMBackendError, match="raise coding.llm.max_tokens"):
            rate(monkeypatch, config, "full")

    def test_a_context_too_short_for_the_prompt_stops_the_step(self, monkeypatch, tmp_path, server):
        server.context = 256
        config = transcribed(project(tmp_path, {"full": {"scales": "scales.yaml"}}, llm={"model": "tiny", "provider": "ollama", "base_url": f"{server.url}/v1", "n_samples": 1, "think": False}))
        with pytest.raises(LLMBackendError, match="OLLAMA_CONTEXT_LENGTH=4096"):
            rate(monkeypatch, config, "full")
        assert len(server.requests) == 1
        server.context = 8192
        server.requests.clear()
        rate(monkeypatch, config, "full")
        assert len(server.requests) == 3

    def test_numbers_count_for_more_tokens_than_their_characters_suggest(self):
        assert module.estimated_tokens("at 312.5 s") == 4 + int(1.3 * 2) + 1
        assert module.estimated_tokens("你好吗") == 3

    def test_named_segments_are_rated_one_by_one_and_a_missing_one_is_said(self, monkeypatch, tmp_path, server):
        config = transcribed(project(
            tmp_path, {"parts": {"scales": "names.yaml", "segments": ["opening", "late", "absent"]}},
            llm={"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 1},
            segments=[("r1", "opening", 0, 10), ("r1", "late", 15, 25)],
        ))
        result = rate(monkeypatch, config, "parts")
        ratings = pd.read_parquet(result.artifacts["ratings.parquet"])
        asked = ratings[ratings["not_rated"] == ""]
        assert sorted(set(asked["segment"])) == ["late", "opening"] and len(asked) == 6
        assert asked.groupby("segment")["t_start"].first().to_dict() == {"late": 15.0, "opening": 0.0}
        # the segment the recording lacks is a row per scale without a score, saying why
        absent = ratings[ratings["segment"] == "absent"]
        assert len(absent) == 3 and absent["score"].isna().all() and (absent["n_asked"] == 0).all()
        assert set(absent["not_rated"]) == {"the recording has no such segment"}
        late = next(body["messages"][-1]["content"] for body in server.requests if "We had a test." in body["messages"][-1]["content"])
        assert "school" not in late and "child | at 5.0 s, lasts 2.0 s, opens the conversation: We had a test." in late
        assert any("no segment absent" in warning.message for warning in result.warnings)

    def test_the_recogniser_s_transcript_is_rated_only_where_a_person_corrected_it(self, monkeypatch, tmp_path, server):
        config = transcribed(project(
            tmp_path, {"asr": {"scales": "names.yaml", "transcript": "recognised"}},
            llm={"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 1},
        ))
        result = rate(monkeypatch, config, "asr")
        assert server.requests == [] and any(warning.check == "rating_skipped" for warning in result.warnings)
        layout = layout_for(config)
        layout.corrected_transcript_path("r1").write_text("<eaf/>", encoding="utf-8")
        with pytest.raises(InputNotFoundError, match="macaw transcript reset -r r1"):
            rate(monkeypatch, config, "asr")
        heard = [("parent", 1.0, 3.0, "How was cool today?"), ("parent", 3.5, 5.0, "It was fine.")]
        transcribed(config_path(tmp_path), heard, where=(layout.recognised_transcript_path("r1"), layout.recognised_words_path("r1")))
        rate(monkeypatch, config, "asr")
        sent = server.requests[0]["messages"][-1]["content"]
        assert "How was cool today?" in sent and "parent | at 3.5 s, lasts 1.5 s, after a 0.5 s pause: It was fine." in sent

    def test_the_recogniser_s_transcript_is_kept_once_when_a_correction_first_comes_in(self, tmp_path):
        config = transcribed(project(tmp_path, {}))
        layout = layout_for(config)
        assert module.keep_recognised(config, "r1")
        kept = _io.read_file(layout.recognised_transcript_path("r1"), TRANSCRIPT_SCHEMA, "transcript")
        assert kept["text"].tolist() == [text for *_, text in TALK] and layout.recognised_words_path("r1").is_file()
        layout.corrected_transcript_path("r1").write_text("<eaf/>", encoding="utf-8")
        layout.recognised_transcript_path("r1").unlink()
        assert not module.keep_recognised(config, "r1")
        layout.corrected_transcript_path("r1").unlink()
        layout.warnings_path("r1", ASRExtractor.name).parent.mkdir(parents=True, exist_ok=True)
        layout.warnings_path("r1", ASRExtractor.name).write_text(json.dumps([{"check": "asr_corrected"}]), encoding="utf-8")
        assert not module.keep_recognised(config, "r1")


def config_path(tmp_path):
    return tmp_path / "project.yaml"


class TestVersions:
    def test_a_version_takes_coding_llm_and_changes_what_it_names(self, tmp_path):
        coding = CodingSection(llm=LLMSection(model="qwen3:14b", n_samples=5), ratings={"k0": {"scales": tmp_path / "k0.yaml", "show": [], "n_samples": 3}})
        own, llm = coding.rating_settings("k0")
        assert own.scales == tmp_path / "k0.yaml" and own.transcript == "in_use" and own.segments == []
        assert llm.model == "qwen3:14b" and llm.n_samples == 3 and llm.show == [] and coding.llm.n_samples == 5

    @pytest.mark.parametrize("version,message", [
        ({"scales": "s.yaml", "batch_size": 4}, "batch_size"),
        ({"scales": "s.yaml", "show": ["timing", "frames"]}, "not shown frames"),
        ({"show": []}, "scales"),
        ({"scales": "s.yaml", "segments": ["a", "a"]}, "twice"),
    ])
    def test_what_a_rating_cannot_take_is_refused(self, version, message):
        with pytest.raises(ValidationError, match=message):
            CodingSection(ratings={"r": version})

    def test_each_version_is_its_own_step_keyed_on_its_own_settings(self, tmp_path):
        path = project(tmp_path, {"k0": {"scales": "names.yaml", "show": []}, "k1": {"scales": "scales.yaml", "show": ["pitch"]}})
        resolved = resolve_config(path)
        nodes = build_dag(resolved).nodes
        assert {"coding.rate.k0", "coding.rate.k1"} <= set(nodes) and "prosody.frame" in nodes["coding.rate.k1"].depends_on
        assert "prosody.frame" not in nodes["coding.rate.k0"].depends_on
        before = {name: hash_extractor_config(resolved, name) for name in ("coding.rate.k0", "coding.rate.k1")}
        write_yaml(tmp_path / "scales.yaml", dict(SCALES, scales=SCALES["scales"][:2]))
        after = {name: hash_extractor_config(resolve_config(path), name) for name in before}
        assert before["coding.rate.k0"] == after["coding.rate.k0"] and before["coding.rate.k1"] != after["coding.rate.k1"]
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["coding"]["llm"]["batch_size"] = 3
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert hash_extractor_config(resolve_config(path), "coding.rate.k0") == before["coding.rate.k0"]

    def test_a_scale_rating_someone_who_is_not_a_role_is_refused_before_the_run(self, tmp_path, server):
        path = project(tmp_path, {"k1": {"scales": "scales.yaml"}}, roles=("mother", "child"), llm={"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1"})
        with pytest.raises(CodebookError, match="rates parent"):
            module._VERSIONS["k1"](resolve_config(path)).check_available()

    def test_the_run_plan_says_how_many_requests_each_version_sends(self, tmp_path, capsys):
        from macaw.cli.run import _echo_reporting

        path = project(tmp_path, {"k1": {"scales": "scales.yaml"}, "cloud": {"scales": "names.yaml", "provider": "openai", "model": "gpt-x", "segments": ["a", "b"]}})
        resolved = resolve_config(path)
        _echo_reporting(resolved, build_dag(resolved), [])
        shown = capsys.readouterr().out
        assert "coding.rate.k1 sends 5 x 3 scales requests for each recording" in shown
        assert "coding.rate.cloud sends 5 x 3 scales requests for each of 2 segments of each recording" in shown
        assert "whole transcripts, with each line's role and times, are sent to openai's servers" in shown

    def test_the_methods_say_what_each_version_was_given(self, tmp_path):
        from macaw.project.manifest import FeatureManifest
        from macaw.qc.boilerplate import describe

        path = project(tmp_path, {"k0": {"scales": "names.yaml", "transcript": "recognised", "segments": ["opening"]}}, llm={"model": "tiny", "reasons": True, "show": ["timing"]})
        resolved = resolve_config(path)
        step = module._VERSIONS["k0"](resolved)
        manifest = FeatureManifest()
        manifest.extend(step._specs(step.scale_set(), build_recording("r1", resolved)))
        text = " ".join(describe(manifest, resolved)["Ratings"])
        assert "In a rating named k0, the language model tiny (ollama) rated the segments opening of each recording one by one" in text
        assert "given only each scale's name, the person it rates and its range" in text
        assert "the role of the person speaking, the words, its start and length" in text
        assert "asked five times at temperature 0.3; the score is the median" in text
        assert "before a person corrected it" in text and "what each score rested on, after the score" in text


def scale_prompts():
    return [("", 0.0, 60.0, module_scale(entry), "system", f"The talk.\n\nScale: {entry['scale']}\n") for entry in SCALES["scales"]]


def score_by_prompt_and_seed(user, seed):
    return json.dumps({"score": 1 + zlib.crc32(f"{user}|{seed}".encode()) % 5})


class TestAtOnce:
    def cfg(self, concurrency):
        return LLMSection(model="tiny", n_samples=5, seed=7, concurrency=concurrency)

    @pytest.mark.parametrize("concurrency", [1, 4])
    def test_as_many_requests_are_under_way_as_concurrency_says(self, concurrency):
        backend = CountingBackend(score_by_prompt_and_seed, gate=concurrency)
        answers = module.ask(scale_prompts(), backend, self.cfg(concurrency))
        assert backend.most == concurrency and len(backend.calls) == len(answers) == 15

    def test_the_answers_are_the_same_table_in_the_same_order_whichever_comes_back_first(self):
        one = CountingBackend(score_by_prompt_and_seed)
        four = CountingBackend(score_by_prompt_and_seed, gate=4, delay=lambda user, seed: 0.03 if seed == 7 else 0.0)
        alone = module.ask(scale_prompts(), one, self.cfg(1))
        together = module.ask(scale_prompts(), four, self.cfg(4))
        pd.testing.assert_frame_equal(alone.drop(columns="seconds"), together.drop(columns="seconds"))
        assert list(zip(alone["scale"], alone["sample"])) == [(scale, sample) for scale in ("warmth", "on_task", "flow") for sample in range(5)]
        assert [seed for _, seed in one.calls] == [7, 8, 9, 10, 11] * 3
        assert sorted(four.calls) == sorted(one.calls) and four.finished != one.finished

    @pytest.mark.parametrize("concurrency", [1, 3])
    def test_a_request_that_fails_keeps_the_rest_from_going_out_and_its_error_is_raised(self, concurrency):
        backend = CountingBackend(score_by_prompt_and_seed, gate=concurrency, fail_at=4, delay=lambda user, seed: 0.05)
        with pytest.raises(RuntimeError) as raised:
            module.ask(scale_prompts(), backend, self.cfg(concurrency))
        assert raised.value is backend.error
        # besides the one that failed, only those already under way went out
        assert 4 <= len(backend.calls) <= 4 + concurrency - 1

    def test_a_context_too_short_costs_one_request_however_many_go_at_once(self):
        backend = CountingBackend(score_by_prompt_and_seed, gate=4, context=16)
        with pytest.raises(LLMBackendError, match="reads 16 tokens at most"):
            module.ask(scale_prompts(), backend, self.cfg(4))
        assert len(backend.calls) == 1

    def test_answers_all_cut_off_stop_the_step_once_all_are_in(self):
        backend = CountingBackend(lambda user, seed: "", gate=4, cut_off=True)
        with pytest.raises(LLMBackendError, match="every answer was cut off"):
            module.ask(scale_prompts(), backend, self.cfg(4))
        assert len(backend.calls) == 15

    @pytest.mark.parametrize("value", [0, -2])
    def test_concurrency_below_one_is_refused(self, tmp_path, value):
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            LLMSection(concurrency=value)
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            CodingSection(ratings={"r": {"scales": "s.yaml", "concurrency": value}})
        with pytest.raises(ConfigError, match="coding.llm.concurrency"):
            resolve_config(project(tmp_path, {}), cli_overrides=[f"coding.llm.concurrency={value}"])

    def test_concurrency_changes_how_a_rating_runs_and_not_its_key(self, tmp_path):
        path = project(tmp_path, {"k0": {"scales": "names.yaml"}})
        before = hash_extractor_config(resolve_config(path), "coding.rate.k0")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["coding"]["llm"]["concurrency"] = 4
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert hash_extractor_config(resolve_config(path), "coding.rate.k0") == before
        data["coding"]["ratings"]["k0"]["concurrency"] = 8
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        resolved = resolve_config(path)
        assert hash_extractor_config(resolved, "coding.rate.k0") == before
        assert resolved.coding.rating_settings("k0")[1].concurrency == 8 and resolved.coding.llm.concurrency == 4

    def test_requests_sent_at_once_are_kept_with_the_results_and_said_in_the_plan_and_the_report(self, monkeypatch, tmp_path, server, capsys):
        from macaw.cli.run import _echo_reporting

        llm = {"model": "tiny", "provider": "ollama", "base_url": f"{server.url}/v1", "n_samples": 2, "concurrency": 4}
        server.score = lambda scale, sample: json.dumps({"score": 1})
        config = transcribed(project(tmp_path, {"full": {"scales": "scales.yaml"}, "single": {"scales": "names.yaml", "concurrency": 1}}, llm=llm))
        result = rate(monkeypatch, config, "full")
        assert result.specs[0].params["concurrency"] == 4 and len(server.requests) == 6
        assert (pd.read_parquet(result.artifacts["ratings.parquet"])["n_answers"] == 2).all()
        assert any(warning.check == "rating_at_once" and "same seed" in warning.message for warning in result.warnings)
        single = rate(monkeypatch, config, "single")
        assert "concurrency" not in single.specs[0].params and not any(warning.check == "rating_at_once" for warning in single.warnings)
        _echo_reporting(config, build_dag(config), [])
        shown = capsys.readouterr().out
        assert "coding.rate.full keeps 4 requests under way per recording, up to 16 in all with 4 tasks side by side (run -j). Ollama answers" in shown
        assert "OLLAMA_NUM_PARALLEL" in shown and "coding.rate.single keeps" not in shown

    def test_the_methods_say_that_requests_went_several_at_a_time(self, tmp_path):
        from macaw.project.manifest import FeatureManifest
        from macaw.qc.boilerplate import describe

        path = project(tmp_path, {"fast": {"scales": "names.yaml"}, "single": {"scales": "names.yaml", "concurrency": 1}}, llm={"model": "tiny", "concurrency": 4})
        resolved = resolve_config(path)
        said = {}
        for name in ("fast", "single"):
            step = module._VERSIONS[name](resolved)
            manifest = FeatureManifest()
            manifest.extend(step._specs(step.scale_set(), build_recording("r1", resolved)))
            said[name] = " ".join(describe(manifest, resolved)["Ratings"])
        assert "Requests were sent four at a time, so the server may have answered them in batches" in said["fast"]
        assert "at a time" not in said["single"]


TASKS = ("f1_task_a", "f1_task_b", "f2_task_a")
NOTES = {"notes": "A is the parent.", "notes_by": {"task": {"task_a": "They build a tower.", "task_b": ""}}}


def task_project(tmp_path, ratings, *, llm=None):
    """Three recordings named by family and task: two of task_a, one of task_b."""
    for rid in TASKS:
        (tmp_path / "raw" / rid.replace("_task", "/task", 1)).mkdir(parents=True, exist_ok=True)
    path = tmp_path / "project.yaml"
    scaffold_project_config(path, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out", roles=("parent", "child"))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["project"]["recording_pattern"] = "{group_id}/{task}"
    data["coding"] = {"llm": llm or {"model": "tiny"}, "ratings": ratings}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    write_yaml(tmp_path / "scales.yaml", SCALES)
    return path


def edit(path, change):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


class TestKeptAnswers:
    """A rating that stops part way keeps the answers it has, and its next try asks only for the rest."""

    CFG = LLMSection(model="tiny", n_samples=5, seed=7, concurrency=1)

    def test_a_try_after_a_failure_asks_only_for_the_answers_still_missing(self, tmp_path):
        kept = tmp_path / "unfinished" / "answers.jsonl"
        failing = CountingBackend(score_by_prompt_and_seed, fail_at=7)
        with pytest.raises(RuntimeError, match="fell over"):
            module.ask(scale_prompts(), failing, self.CFG, kept)
        assert len(kept.read_text(encoding="utf-8").splitlines()) == 6
        again = CountingBackend(score_by_prompt_and_seed)
        answers = module.ask(scale_prompts(), again, self.CFG, kept)
        assert len(again.calls) == 9 and set(again.calls).isdisjoint(failing.finished[:6])
        assert answers["kept"].tolist() == [True] * 6 + [False] * 9
        whole = module.ask(scale_prompts(), CountingBackend(score_by_prompt_and_seed), self.CFG)
        assert answers.drop(columns=["seconds", "kept"]).equals(whole.drop(columns=["seconds", "kept"])) and not whole["kept"].any()

    def test_several_at_once_keep_every_answer_that_came_back(self, tmp_path):
        kept = tmp_path / "answers.jsonl"
        cfg = self.CFG.model_copy(update={"concurrency": 4})
        failing = CountingBackend(score_by_prompt_and_seed, gate=4, fail_at=9)
        with pytest.raises(RuntimeError):
            module.ask(scale_prompts(), failing, cfg, kept)
        written = len(kept.read_text(encoding="utf-8").splitlines())
        assert written == len(failing.finished) - 1
        again = CountingBackend(score_by_prompt_and_seed, gate=4)
        module.ask(scale_prompts(), again, cfg, kept)
        assert len(again.calls) == 15 - written

    def test_an_answer_is_kept_only_for_the_same_question_to_the_same_model(self, tmp_path):
        kept = tmp_path / "answers.jsonl"
        module.ask(scale_prompts(), CountingBackend(score_by_prompt_and_seed), self.CFG, kept)
        for change in ({"temperature": 0.9}, {"max_tokens": 100}, {"seed": 8}, {"think": True}):
            other = CountingBackend(score_by_prompt_and_seed)
            module.ask(scale_prompts(), other, self.CFG.model_copy(update=change), tmp_path / "answers.jsonl")
            assert len(other.calls) == 15, change
        changed = [(segment, start, end, scale, system, user + "One more line.") for segment, start, end, scale, system, user in scale_prompts()]
        other = CountingBackend(score_by_prompt_and_seed)
        module.ask(changed, other, self.CFG, kept)
        assert len(other.calls) == 15
        same = CountingBackend(score_by_prompt_and_seed)
        module.ask(scale_prompts(), same, self.CFG, kept)
        assert same.calls == []

    def test_a_line_cut_short_when_the_run_stopped_is_passed_over(self, tmp_path):
        kept = tmp_path / "answers.jsonl"
        module.ask(scale_prompts(), CountingBackend(score_by_prompt_and_seed), self.CFG, kept)
        lines = kept.read_text(encoding="utf-8").splitlines()
        kept.write_text("\n".join(lines[:-1] + [lines[-1][:20]]), encoding="utf-8")
        again = CountingBackend(score_by_prompt_and_seed)
        module.ask(scale_prompts(), again, self.CFG, kept)
        assert len(again.calls) == 1

    def test_the_first_answer_is_not_kept_when_the_context_was_too_short_for_it(self, tmp_path):
        kept = tmp_path / "answers.jsonl"
        with pytest.raises(LLMBackendError, match="reads 64 tokens at most"):
            module.ask(scale_prompts(), CountingBackend(score_by_prompt_and_seed, context=64), self.CFG, kept)
        assert not kept.exists()

    def test_a_rating_says_how_many_answers_it_kept_and_a_finished_one_keeps_none(self, monkeypatch, tmp_path, server):
        llm = {"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 2}
        config = transcribed(project(tmp_path, {"full": {"scales": "names.yaml"}}, llm=llm))
        layout = layout_for(config)
        kept = layout.unfinished_path("r1", "coding.rate.full") / "answers.jsonl"
        assert layout.work_dir in kept.parents and layout.recording_dir("r1") not in kept.parents

        def falls_over(scale, sample):
            if len(server.requests) >= 4:
                raise RuntimeError("the server fell over")
            return json.dumps({"score": 3})

        server.score = falls_over
        with pytest.raises(LLMBackendError, match="did not answer after 4 tries"):
            rate(monkeypatch, config, "full")
        assert len(kept.read_text(encoding="utf-8").splitlines()) == 3
        server.score = lambda scale, sample: json.dumps({"score": 3})
        server.requests.clear()
        result = rate(monkeypatch, config, "full")
        assert len(server.requests) == 3
        said = {warning.check: warning.message for warning in result.warnings}
        assert said["rating_resumed"].startswith("3 of 6 answers were kept")
        assert "rating_resumed" not in {warning.check for warning in rate(monkeypatch, config, "full").warnings}

    def test_the_runner_drops_what_a_task_kept_once_it_has_finished(self, tmp_path):
        from macaw.execution.runner import _finished

        layout = layout_for(resolve_config(project(tmp_path, {})))
        kept = layout.unfinished_path("r1", "coding.rate.full") / "answers.jsonl"
        kept.parent.mkdir(parents=True)
        kept.write_text("{}", encoding="utf-8")
        _finished(layout, layout.completion_path("r1", "coding.rate.full"), SimpleNamespace(as_dirname=lambda: "key"), "r1", "coding.rate.full")
        assert not kept.exists() and layout.completion_path("r1", "coding.rate.full").read_text(encoding="utf-8") == "key"


class TestTimeout:
    def backend(self):
        return backends_module.resolve_llm_backend(LLMSection(model="tiny", provider="vllm", timeout_s=20))

    @pytest.mark.parametrize("error", [TimeoutError("timed out"), backends_module.urllib.error.URLError(TimeoutError("timed out"))])
    def test_a_request_out_of_time_is_sent_once_more_and_then_says_to_raise_the_timeout(self, monkeypatch, error):
        calls = []

        def out_of_time(request, timeout):
            calls.append(timeout)
            raise error

        monkeypatch.setattr(backends_module.urllib.request, "urlopen", out_of_time)
        with pytest.raises(LLMBackendError, match=r"no whole answer within 20 s, tried 2 times.*raise coding\.llm\.timeout_s"):
            self.backend().complete("system", "user")
        assert calls == [20, 20]

    def test_a_server_that_is_not_there_is_still_tried_four_times(self, monkeypatch):
        calls = []

        def refused(request, timeout):
            calls.append(timeout)
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(backends_module.urllib.request, "urlopen", refused)
        with pytest.raises(LLMBackendError, match="did not answer after 4 tries"):
            self.backend().complete("system", "user")
        assert len(calls) == 4

    def test_the_timeout_is_half_an_hour_unless_set_and_changes_no_cache_key(self, tmp_path):
        assert LLMSection().timeout_s == 1800
        path = project(tmp_path, {"full": {"scales": "scales.yaml"}})
        module.register_rating_versions(resolve_config(path))
        before = hash_extractor_config(resolve_config(path), "coding.rate.full")
        assert hash_extractor_config(resolve_config(path, cli_overrides=["coding.llm.timeout_s=60"]), "coding.rate.full") == before


class TestNotesBy:
    def test_the_prompt_holds_the_project_s_note_then_the_one_for_the_recording_s_task(self, monkeypatch, tmp_path, server):
        llm = {"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 1, **NOTES}
        path = task_project(tmp_path, {"k1": {"scales": "scales.yaml"}}, llm=llm)
        for rid in ("f1_task_a", "f1_task_b"):
            rate(monkeypatch, transcribed(path, rid=rid), "k1", rid=rid)
        systems = [request["messages"][0]["content"] for request in server.requests]
        assert "Notes from the researchers: A is the parent.\nThey build a tower.\nAnswer with" in systems[0]
        assert "Notes from the researchers: A is the parent.\nAnswer with" in systems[-1]

    def test_a_changed_note_for_one_task_asks_again_only_for_that_task(self, tmp_path):
        from macaw.execution.runner import task_key

        path = task_project(tmp_path, {"k1": {"scales": "scales.yaml"}}, llm={"model": "tiny", **NOTES})
        for rid in TASKS:
            config = transcribed(path, rid=rid)
            prep = layout_for(config).artifacts_dir(rid, "prep.media")
            prep.mkdir(parents=True)
            (prep / "prep.json").write_text("{}", encoding="utf-8")

        def keys():
            config = resolve_config(path)
            return {rid: task_key(config, "coding.rate.k1", build_recording(rid, config)) for rid in TASKS}

        before = keys()
        edit(path, lambda data: data["coding"]["llm"]["notes_by"]["task"].update(task_b="They draw."))
        after = keys()
        assert [before[rid] == after[rid] for rid in TASKS] == [True, False, True]
        edit(path, lambda data: data["coding"]["llm"].update(notes="B is the child."))
        assert not any(keys()[rid] == after[rid] for rid in TASKS)

    def test_a_task_without_a_note_stops_the_run_before_anything_is_asked(self, tmp_path):
        from macaw.exceptions import ConfigError
        from macaw.execution.runner import plan

        path = task_project(tmp_path, {"k1": {"scales": "scales.yaml"}}, llm={"model": "tiny", "notes_by": {"task": {"task_b": "They draw."}}})
        config = resolve_config(path)
        with pytest.raises(ConfigError, match=r"coding.llm.notes_by has no note for task task_a \(f1_task_a, f2_task_a\); add each"):
            plan(config, list(TASKS), ["coding.rate.k1"])
        with pytest.raises(ConfigError, match="no note for task task_a"):
            module._VERSIONS["k1"](config).prompt_note(build_recording("f2_task_a", config))
        edit(path, lambda data: data["coding"]["llm"]["notes_by"]["task"].update(task_a=""))
        assert plan(resolve_config(path), list(TASKS), ["coding.rate.k1"])

    def test_a_version_s_own_table_replaces_coding_llm_s_and_a_recording_s_file_changes_its_part(self, tmp_path):
        from macaw.coding.llm.notes import notes_by_key

        versions = {
            "own": {"scales": "scales.yaml", "notes_by": {"task": {"task_a": "Own a.", "task_b": "Own b."}}},
            "bare": {"scales": "scales.yaml", "notes_by": {}}, "same": {"scales": "scales.yaml"},
        }
        path = task_project(tmp_path, versions, llm={"model": "tiny", **NOTES})
        (tmp_path / "recordings").mkdir()
        write_yaml(tmp_path / "recordings" / "f2_task_a.yaml", {"coding": {"llm": {"notes": "Here B is the child."}}})
        config = resolve_config(path)

        def told(name, rid, own=config):
            return module._VERSIONS[name](own).prompt_note(build_recording(rid, own))

        assert told("own", "f1_task_a") == "A is the parent.\nOwn a." and told("bare", "f1_task_a") == "A is the parent."
        assert told("same", "f1_task_b") == "A is the parent."
        assert told("same", "f2_task_a", resolve_config(path, recording_id="f2_task_a")) == "Here B is the child.\nThey build a tower."
        assert notes_by_key(config, "ratings", "own") == "coding.ratings.own.notes_by"
        assert notes_by_key(config, "ratings", "same") == "coding.llm.notes_by"

    def test_notes_are_given_by_an_entity_the_recording_pattern_has(self, tmp_path):
        from macaw.exceptions import ConfigError

        with pytest.raises(ValidationError, match="keyed by group_id, session_id, task, run, not family"):
            LLMSection(notes_by={"family": {"x": "y"}})
        assert LLMSection(notes_by={"run": {1: "first"}}).notes_by == {"run": {"1": "first"}}
        path = project(tmp_path, {"k1": {"scales": "scales.yaml", "notes_by": {"task": {"a": "x"}}}})
        with pytest.raises(ConfigError, match=r"coding.ratings.k1.notes_by gives notes by task, which project.recording_pattern \(\{group_id\}\)"):
            resolve_config(path)

    def test_the_run_plan_says_each_task_s_note_and_how_many_recordings_take_it(self, tmp_path, capsys):
        from macaw.cli.run import _echo_reporting

        path = task_project(tmp_path, {"k1": {"scales": "scales.yaml"}}, llm={"model": "tiny", **NOTES})
        resolved = resolve_config(path)
        _echo_reporting(resolved, build_dag(resolved), list(TASKS))
        assert (
            "coding.rate.k1 adds to the researchers' note one by task (coding.llm.notes_by): "
            'task_a (2 recording(s)): "They build a tower."; task_b (1 recording(s)): none'
        ) in capsys.readouterr().out

    def test_the_dictionary_and_the_methods_give_one_note_per_task_not_one_per_recording(self, tmp_path):
        from macaw.coding.ratings import rater_settings
        from macaw.project.dictionary import merged_notes
        from macaw.project.manifest import FeatureManifest, recording_specs, write_manifest
        from macaw.qc.boilerplate import describe, settings_by_recording

        path = task_project(tmp_path, {"k1": {"scales": "scales.yaml"}}, llm={"model": "tiny", **NOTES})
        config = resolve_config(path)
        layout = layout_for(config)
        step = module._VERSIONS["k1"](config)
        for rid in TASKS:
            manifest = FeatureManifest()
            manifest.extend(step._specs(step.scale_set(), build_recording(rid, config)))
            write_manifest(manifest, layout.manifest_fragment_path(rid, step.name))
        table = pd.DataFrame({"recording_id": list(TASKS), "rater": "k1", "rater_kind": "model"})
        settings = rater_settings(config, table, tmp_path)["k1"]
        assert settings["notes_by"] == {"task": {"task_a": "They build a tower.", "task_b": ""}} and "by_recording" not in settings

        specs = merged_notes({rid: recording_specs(layout, rid) for rid in TASKS})
        assert settings_by_recording(specs) == ""
        manifest = FeatureManifest()
        manifest.extend(list(specs["f1_task_a"].values()))
        text = " ".join(describe(manifest, config)["Ratings"])
        assert (
            'The prompt also held this note from the researchers: "A is the parent." After it came a note from the researchers '
            'for the recording\'s task: "They build a tower." for task_a; none for task_b.'
        ) in text

    def test_two_notes_for_one_task_show_as_variants_by_recording(self):
        from macaw.project.dictionary import varying_settings

        params = {
            "f1_task_a": {"notes_by": {"task": {"task_a": "Old."}}}, "f2_task_a": {"notes_by": {"task": {"task_a": "New."}}},
            "f1_task_b": {"notes_by": {"task": {"task_b": "B."}}},
        }
        found = varying_settings(params)
        assert isinstance(found, list) and len(found) == 3


class TestEvidence:
    def llm(self, server):
        return {"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 1}

    def test_lines_are_numbered_and_the_answer_lists_its_evidence_before_the_score(self):
        frame = transcript_frame(TALK).assign(speaker=lambda f: f["subject_id"])
        lines = module.part_lines(frame, words_frame(TALK), 0.0, 60.0)
        scales = read_scales_from(SCALES)
        system, user = build_rating_prompt(scales.get("warmth"), lines, people=["parent", "child"], minutes=1.0, evidence=True, reasons=True)
        assert "parent: How was school today? [line 1]" in user and "child: We had a test. [line 4]" in user
        assert "cite only the numbered lines given" in system and '{"evidence": [{"line": <number>' in system and '"reason"' in system
        assert user.rstrip().endswith("Cite lines said by the parent.\nAnswer with the JSON object.")
        _, flow = build_rating_prompt(scales.get("flow"), lines, people=["parent", "child"], minutes=1.0, evidence=True)
        assert "Cite lines said by" not in flow
        other, _ = build_rating_prompt(scales.get("flow"), lines, people=["parent", "child"], minutes=1.0, evidence=True)
        assert other == system.replace(', "reason": "<what it rests on>"', "").replace(
            "\nThe reason names in a few words what the score rests on: what was said, the timing or the voice.", ""
        )

    def test_each_cited_line_is_checked_three_ways(self):
        from macaw.coding.llm.evidence import NO_LINE, NO_QUOTE, NOT_IN_LINE, OTHER_SPEAKER, check_cited

        lines = pd.DataFrame({"line": [1, 2], "speaker": ["parent", "child"], "t_start": [1.0, 3.5], "t_end": [3.0, 5.0], "text": ["How was school today?", "It's fine."]})
        warmth = module_scale(SCALES["scales"][0])
        rows = check_cited(
            [{"line": 1, "quote": "how was SCHOOL"}, {"line": "2", "quote": "its fine"}, {"line": 9, "quote": "x"}, {"line": 1, "quote": ""},
             {"line": 1, "quote": "was school yesterday"}, {"line": 1.5}],
            lines, warmth, 10.0,
        )
        assert [row["problem"] for row in rows] == ["", f"{OTHER_SPEAKER}", NO_LINE, NO_QUOTE, NOT_IN_LINE, NO_LINE]
        assert (rows[0]["t_start"], rows[0]["t_end"], rows[0]["speaker"]) == (11.0, 13.0, "parent")
        flow = module_scale(SCALES["scales"][2])
        assert check_cited([{"line": 2, "quote": "It's fine"}], lines, flow, 0.0)[0]["problem"] == ""

    def test_words_running_into_a_neighbouring_line_are_told_apart(self):
        from macaw.coding.llm.evidence import NOT_IN_LINE, RUNS_ON, check_cited

        lines = pd.DataFrame({
            "line": [1, 2, 3], "speaker": ["parent", "parent", "child"], "t_start": [0.0, 2.0, 4.0], "t_end": [2.0, 4.0, 5.0],
            "text": ["Shall we put the", "blocks away now?", "No."],
        })
        flow = module_scale(SCALES["scales"][2])
        rows = check_cited(
            [{"line": 1, "quote": "put the blocks away"}, {"line": 2, "quote": "put the blocks away"}, {"line": 3, "quote": "blocks away"},
             {"line": 2, "quote": "away now no"}, {"line": 1, "quote": "now no"}],
            lines, flow, 0.0,
        )
        assert [row["problem"] for row in rows] == [RUNS_ON, RUNS_ON, NOT_IN_LINE, RUNS_ON, NOT_IN_LINE]

    def test_evidence_is_kept_with_recording_times_failures_counted_and_scores_kept(self, monkeypatch, tmp_path, server):
        def answer(scale, sample):
            if scale == "warmth":
                return json.dumps({"evidence": [{"line": 1, "quote": "we had a test"}], "score": 4})
            if scale == "on_task":
                return json.dumps({"score": 1})
            return json.dumps({"evidence": [{"line": 1, "quote": "We had"}, {"line": 7, "quote": "x"}], "score": 2})

        server.score = answer
        config = transcribed(project(
            tmp_path, {"cited": {"scales": "scales.yaml", "segments": ["late"], "evidence": True}}, llm=self.llm(server),
            segments=[("r1", "late", 15, 25)],
        ))
        result = rate(monkeypatch, config, "cited")
        ratings = pd.read_parquet(result.artifacts["ratings.parquet"]).set_index("scale")
        assert ratings.loc["warmth", "score"] == 4 and ratings.loc["flow", "score"] == 2 and ratings.loc["on_task", "score"] == 1
        answers = pd.read_parquet(result.artifacts["answers.parquet"]).set_index("scale")
        assert answers.loc["warmth", "evidence_problem"] == "said by another person 1" and answers.loc["warmth", "n_evidence"] == 1
        assert answers.loc["on_task", "evidence_problem"] == "no list of evidence" and pd.isna(answers.loc["on_task", "n_evidence"])
        assert answers.loc["flow", "evidence_problem"] == "no such line 1" and answers.loc["flow", "problem"] == ""
        evidence = pd.read_parquet(result.artifacts["evidence.parquet"])
        cited = evidence[(evidence["scale"] == "flow") & (evidence["line"] == 1)].iloc[0]
        assert (cited["speaker"], cited["t_start"], cited["t_end"], cited["said"]) == ("child", 20.0, 22.0, "We had a test.")
        [warned] = [warning for warning in result.warnings if warning.check == "rating_evidence_unchecked"]
        assert "2 of the 3 lines the model cited failed the check" in warned.message and "1 of 3 answers gave no list" in warned.message
        assert result.specs[0].params["evidence"] is True

    def test_without_evidence_the_answers_say_nothing_of_it_and_the_key_is_as_before(self, monkeypatch, tmp_path, server):
        path = project(tmp_path, {"plain": {"scales": "scales.yaml"}}, llm=self.llm(server))
        before = hash_extractor_config(resolve_config(path), "coding.rate.plain")
        result = rate(monkeypatch, transcribed(path), "plain")
        answers = pd.read_parquet(result.artifacts["answers.parquet"])
        assert "evidence.parquet" not in result.artifacts and answers["n_evidence"].isna().all() and answers["evidence_problem"].isna().all()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["coding"]["ratings"]["plain"]["evidence"] = False
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert hash_extractor_config(resolve_config(path), "coding.rate.plain") == before
        data["coding"]["ratings"]["plain"]["evidence"] = True
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert hash_extractor_config(resolve_config(path), "coding.rate.plain") != before

    def test_the_export_writes_the_cited_lines_with_a_dictionary_and_the_methods_say_how_they_were_checked(self, monkeypatch, tmp_path, server):
        from typer.testing import CliRunner

        from macaw.cli.main import app
        from macaw.project.manifest import FeatureManifest, write_manifest
        from macaw.qc.boilerplate import describe

        server.score = lambda scale, sample: json.dumps({"evidence": [{"line": 2, "quote": "fine"}], "score": 1})
        path = project(tmp_path, {"cited": {"scales": "scales.yaml", "evidence": True}}, llm=self.llm(server))
        config = transcribed(path)
        result = rate(monkeypatch, config, "cited")
        # as the runner leaves a finished task: its files among the step's artifacts, its specs in the manifest
        layout = layout_for(config)
        folder = layout.artifacts_dir("r1", "coding.rate.cited")
        folder.mkdir(parents=True, exist_ok=True)
        for name, made in result.artifacts.items():
            shutil.copy(made, folder / name)
        fragment = FeatureManifest()
        fragment.extend(result.specs)
        write_manifest(fragment, layout.manifest_fragment_path("r1", "coding.rate.cited"))
        out = tmp_path / "exported"
        shown = CliRunner().invoke(app, ["ratings", "export", str(out), "-c", str(path)])
        assert shown.exit_code == 0, shown.output
        assert "3 lines they cited" in shown.output
        evidence = pd.read_csv(out / "evidence.csv")
        assert len(evidence) == 3 and set(evidence["speaker"]) == {"child"} and "group_id" in evidence.columns
        assert (out / "evidence_dictionary.csv").is_file()
        assert "before each score the model was asked to list the lines" in " ".join(describe(fragment, config)["Ratings"])


EXAMPLES = {
    "examples": {
        "warmth": [
            {"lines": [{"parent": "Did you have fun?"}, {"child": "Yes."}], "score": 4, "why": "Warm interest."},
            {"description": "The parent answers briefly and turns back to the task.", "score": 2},
        ],
    },
}


class TestExamples:
    def test_examples_are_read_and_checked(self, tmp_path):
        from macaw.coding.scales import read_examples

        scales = read_scales_from(SCALES)
        found = read_examples(write_yaml(tmp_path / "e.yaml", EXAMPLES), scales, ["parent", "child", "other"])
        assert [example.score for example in found["warmth"]] == [4, 2] and found["warmth"][0].lines[1] == ("child", "Yes.")
        for entry, message in (
            ({"cheer": []}, "is not a scale"),
            ({"warmth": [{"lines": [{"parent": "a"}], "description": "b", "score": 3}]}, "either lines or a description"),
            ({"warmth": [{"description": "b", "score": 7}]}, "outside the scale's range"),
            ({"warmth": [{"lines": [{"teacher": "a"}], "score": 3}]}, "teacher is not one of the speakers"),
            ({"warmth": [{"description": "b"}]}, "has no score"),
        ):
            with pytest.raises(CodebookError, match=message):
                read_examples(write_yaml(tmp_path / "bad.yaml", {"examples": entry}), scales, ["parent", "child", "other"])

    def test_a_line_of_yes_or_no_stays_a_word(self, tmp_path):
        from macaw.coding.scales import read_examples

        path = tmp_path / "e.yaml"
        lines = ["examples:", "  warmth:", "    - score: 4", "      lines:", "        - parent: Ready?", "        - child: yes", "        - parent: no"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        found = read_examples(path, read_scales_from(SCALES), ["parent", "child"])
        assert found["warmth"][0].lines == [("parent", "Ready?"), ("child", "yes"), ("parent", "no")]

    def test_examples_follow_the_scale_and_the_transcript_still_comes_first(self, tmp_path):
        from macaw.coding.scales import read_examples

        scales = read_scales_from(SCALES)
        examples = read_examples(write_yaml(tmp_path / "e.yaml", EXAMPLES), scales, ["parent", "child"])
        lines = TestPrompt().lines()
        _, warmth = build_rating_prompt(scales.get("warmth"), lines, people=["parent", "child"], minutes=1.0, examples=examples["warmth"])
        _, flow = build_rating_prompt(scales.get("flow"), lines, people=["parent", "child"], minutes=1.0)
        assert warmth.index("Rules:") < warmth.index("Examples people rated in other conversations, not this one:") < warmth.index("Answer with")
        assert "Example 1, scored 4:\nparent: Did you have fun?\nchild: Yes.\nWhy: Warm interest.\nExample 2, scored 2:\nThe parent answers" in warmth
        transcript = warmth.split("\n\nRate the conversation")[0]
        assert flow.startswith(transcript)

    def test_a_version_shows_its_examples_says_so_and_is_keyed_on_them(self, monkeypatch, tmp_path, server, capsys):
        from macaw.cli.run import _echo_reporting
        from macaw.project.manifest import FeatureManifest
        from macaw.qc.boilerplate import describe

        write_yaml(tmp_path / "examples.yaml", EXAMPLES)
        llm = {"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 1}
        path = project(tmp_path, {"plain": {"scales": "scales.yaml"}, "shown": {"scales": "scales.yaml", "examples": "examples.yaml"}}, llm=llm)
        config = transcribed(path)
        before = {name: hash_extractor_config(config, f"coding.rate.{name}") for name in ("plain", "shown")}
        result = rate(monkeypatch, config, "shown")
        scale_of = re.compile(r"^Scale: (\S+)$", re.MULTILINE)
        sent = {scale_of.search(body["messages"][-1]["content"]).group(1): body["messages"][-1]["content"] for body in server.requests}
        assert "Example 2, scored 2" in sent["warmth"] and "Example" not in sent["flow"]
        assert result.specs[0].params["n_examples"] == 2 and result.specs[0].params["n_examples_with_reasons"] == 1
        manifest = FeatureManifest()
        manifest.extend(result.specs)
        assert 'shown the examples of that scale in "examples.yaml", 2 in all' in " ".join(describe(manifest, config)["Ratings"])
        _echo_reporting(config, build_dag(config), [])
        assert "coding.rate.shown shows 2 examples people rated; they add up to about" in capsys.readouterr().out
        write_yaml(tmp_path / "examples.yaml", {"examples": {"warmth": EXAMPLES["examples"]["warmth"][:1]}})
        after = {name: hash_extractor_config(resolve_config(path), f"coding.rate.{name}") for name in before}
        assert after["plain"] == before["plain"] and after["shown"] != before["shown"]


class TestExcerpts:
    def shown(self):
        frame = transcript_frame(TALK).assign(speaker=lambda f: f["subject_id"])
        return module.part_lines(frame, words_frame(TALK), 0.0, 60.0)

    def test_the_lines_overlapping_a_stretch_are_shown_whole_and_numbered_again(self):
        found = module.excerpt_lines(self.shown(), [(4.0, 4.9), (21.0, 21.0)], 0.0)
        assert found["text"].tolist() == ["It was fine.", "Just fine?", "We had a test."]
        assert found["line"].tolist() == [1, 2, 3] and found["opens"].tolist() == [True, False, False]
        assert module.excerpt_lines(self.shown(), [(14.0, 15.0)], 0.0).empty

    def test_each_scale_is_rated_from_its_own_stretches_and_one_without_any_is_not_rated(self, monkeypatch, tmp_path, server):
        pd.DataFrame(
            [("r1", "warmth", 1.0, 2.0), ("r1", "warmth", 21.0, 21.0), ("r1", "flow", 14.0, 15.0), ("r2", "on_task", 0.0, 60.0)],
            columns=["recording", "scale", "start_s", "end_s"],
        ).to_csv(tmp_path / "excerpts.csv", index=False)
        llm = {"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 1, "show": ["timing"]}
        config = transcribed(project(tmp_path, {"marked": {"scales": "scales.yaml", "excerpts": "excerpts.csv"}}, llm=llm))
        result = rate(monkeypatch, config, "marked")
        [user] = [body["messages"][-1]["content"] for body in server.requests]
        assert "parent | at 1.0 s, lasts 2.0 s, opens an excerpt: How was school today?\n...\nchild | at 20.0 s" in user
        assert "It was fine." not in user and "Excerpts of the transcript of a 1.0-minute conversation" in user
        ratings = pd.read_parquet(result.artifacts["ratings.parquet"])
        assert ratings["scale"].tolist() == ["warmth", "on_task", "flow"] and ratings["segment"].tolist() == ["", "", ""]
        assert ratings["not_rated"].tolist()[0] == "" and ratings["score"].iloc[1:].isna().all()
        assert set(ratings["not_rated"].iloc[1:]) == {"the excerpts table marks no stretch with a line in it"}
        [skipped] = [warning for warning in result.warnings if warning.check == "rating_skipped"]
        assert "on_task, flow" in skipped.message and "those scales were not rated" in skipped.message
        assert result.specs[0].params["excerpts"] == "excerpts.csv"

    def test_only_a_recording_s_own_stretches_are_part_of_its_key(self, tmp_path):
        def key(rows):
            pd.DataFrame(rows, columns=["recording", "scale", "start_s", "end_s"]).to_csv(tmp_path / "excerpts.csv", index=False)
            config = resolve_config(path)
            return module._VERSIONS["marked"](config).extra_key(build_recording("r1", config))

        path = project(tmp_path, {"marked": {"scales": "scales.yaml", "excerpts": "excerpts.csv"}})
        first = key([("r1", "warmth", 1.0, 2.0)])
        assert key([("r1", "warmth", 1.0, 2.0), ("r2", "flow", 0.0, 9.0)]) == first
        assert key([("r1", "warmth", 1.0, 3.0)]) != first

    @pytest.mark.parametrize("rows,message", [
        ([("r1", "cheer", 1.0, 2.0)], "cheer is not a scale"),
        ([("r1", "warmth", 3.0, 2.0)], "start_s and end_s must be seconds, the start not after the end"),
    ])
    def test_a_table_that_does_not_fit_says_where(self, tmp_path, rows, message):
        pd.DataFrame(rows, columns=["recording", "scale", "start_s", "end_s"]).to_csv(tmp_path / "excerpts.csv", index=False)
        with pytest.raises(ConfigError, match=f"line 2: {message}"):
            module.read_excerpts(tmp_path / "excerpts.csv", read_scales_from(SCALES))

    def test_excerpts_and_segments_do_not_go_together(self):
        with pytest.raises(ValidationError, match="excerpts or segments"):
            CodingSection.model_validate({"ratings": {"both": {"scales": "s.yaml", "excerpts": "e.csv", "segments": ["a"]}}})

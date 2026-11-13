from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from macaw.coding.llm import backends as backends_module
from macaw.coding.llm import rate as module
from macaw.coding.llm.parsing import parse_rating
from macaw.coding.llm.prompts import build_rating_prompt
from macaw.coding.scales import ALL, RatingScale, ScaleSet, read_scales, validate_scales
from macaw.config.hashing import hash_extractor_config
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.config.schema import CodingSection, LLMSection, ProjectSection
from macaw.exceptions import CodebookError, InputNotFoundError, LLMBackendError
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
                choice = {"message": {"content": text}, "finish_reason": "length" if server.cut else "stop"}
                self._send({"model": "tiny", "choices": [choice], "usage": {"prompt_tokens": 5, "completion_tokens": 3}})

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
    scaffold_project_config(path, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["project"]["roles"] = list(roles)
    data["coding"] = {"llm": llm or {"model": "tiny"}, "ratings": ratings}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    write_yaml(tmp_path / "scales.yaml", SCALES)
    write_yaml(tmp_path / "names.yaml", NAMES_ONLY)
    if segments:
        pd.DataFrame(segments, columns=["recording", "segment", "start_s", "end_s"]).to_csv(tmp_path / "segments.csv", index=False)
    return path


def transcript_frame(talk):
    frame = pd.DataFrame(talk, columns=["subject_id", "t_start", "t_end", "text"])
    frame["recording_id"] = "r1"
    frame["utterance_index"] = frame.groupby("subject_id").cumcount()
    frame["n_words"] = frame["text"].str.split().str.len()
    for column in ("confidence", "avg_logprob", "no_speech_prob"):
        frame[column] = float("nan")
    frame["language"] = "en"
    return frame[list(TRANSCRIPT_SCHEMA)]


def words_frame(talk):
    rows = []
    frame = transcript_frame(talk)
    for row in frame.itertuples(index=False):
        tokens = row.text.split()
        step = (row.t_end - row.t_start) / len(tokens)
        rows += [
            ("r1", row.subject_id, row.utterance_index, row.t_start + i * step, row.t_start + (i + 1) * step, token, 1.0)
            for i, token in enumerate(tokens)
        ]
    return pd.DataFrame(rows, columns=list(WORD_SCHEMA))


def transcribed(config_path, talk=TALK, *, where=None):
    """text.asr's output for r1, or the same tables at where."""
    config = resolve_config(config_path)
    folder = layout_for(config).artifacts_dir("r1", ASRExtractor.name)
    folder.mkdir(parents=True, exist_ok=True)
    transcript, words = where or (folder / "transcript.parquet", folder / "words.parquet")
    _io.write_file(transcript_frame(talk), transcript, TRANSCRIPT_SCHEMA, "transcript")
    _io.write_file(words_frame(talk), words, WORD_SCHEMA, "words")
    return config


def rate(monkeypatch, config, name, window=(0.0, 60.0)):
    monkeypatch.setattr(module, "load_prep", lambda config, rid: SimpleNamespace(t_origin_s=window[0], duration_s=window[1] - window[0]))
    step = module._VERSIONS[name](config)
    return step.run(build_recording("r1", config), layout_for(config).recording_dir("r1"))


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

    def test_named_segments_are_rated_one_by_one_and_a_missing_one_is_said(self, monkeypatch, tmp_path, server):
        config = transcribed(project(
            tmp_path, {"parts": {"scales": "names.yaml", "segments": ["opening", "late", "absent"]}},
            llm={"model": "tiny", "provider": "vllm", "base_url": f"{server.url}/v1", "n_samples": 1},
            segments=[("r1", "opening", 0, 10), ("r1", "late", 15, 25)],
        ))
        result = rate(monkeypatch, config, "parts")
        ratings = pd.read_parquet(result.artifacts["ratings.parquet"])
        assert sorted(set(ratings["segment"])) == ["late", "opening"] and len(ratings) == 6
        assert ratings.groupby("segment")["t_start"].first().to_dict() == {"late": 15.0, "opening": 0.0}
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
        _echo_reporting(resolved, build_dag(resolved))
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
        manifest.extend(step._specs(step.scale_set()))
        text = " ".join(describe(manifest, resolved)["Ratings"])
        assert "In a rating named k0, the language model tiny (ollama) rated the segments opening of each recording one by one" in text
        assert "given only each scale's name, the person it rates and its range" in text
        assert "the role of the person speaking, the words, its start and length" in text
        assert "asked five times at temperature 0.3; the score is the median" in text
        assert "before a person corrected it" in text and "what each score rested on, after the score" in text

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from test_ratings import SCALES, TALK, project, rate, read_scales_from, server, transcribed, transcript_frame, words_frame  # noqa: F401

from macaw.coding.llm import backends as backends_module
from macaw.coding.llm import rate as module
from macaw.coding.llm import rating_frames
from macaw.coding.llm.backends import OllamaBackend, VLLMBackend, in_order
from macaw.coding.llm.prompts import PICTURE_MARK, build_rating_prompt
from macaw.coding.scales import ScaleExample
from macaw.config.loader import resolve_config
from macaw.config.schema import LLMSection, RatingFrames
from macaw.exceptions import ConfigError
from macaw.media.ffmpeg import run_ffmpeg
from macaw.project.recording import MediaFile


class TestTimes:
    def test_frames_start_at_the_part_and_stay_inside_it(self):
        assert rating_frames.frame_times(300.0, 306.0, 1.0) == [300.0, 301.0, 302.0, 303.0, 304.0, 305.0]
        assert rating_frames.frame_times(0.0, 2.5, 1.0) == [0.0, 1.0, 2.0]

    def test_marked_stretches_each_start_their_own_frames(self):
        found = rating_frames.frame_times(0.0, 8.0, 1.0, [(1.0, 1.2), (5.0, 7.0), (6.5, 9.0)])
        # a stretch shorter than every_s still shows one; one reaching past the part is cut at its end
        assert found == [1.0, 5.0, 6.0, 6.5, 7.5]


class TestCrop:
    def test_the_box_is_the_bodies_union_without_strays_and_with_margin(self):
        boxes = pd.DataFrame({
            "subject_id": ["parent"] * 50 + ["child"] * 50 + ["parent"],
            "frame": list(range(50)) * 2 + [50],
            "x0": [100.0] * 50 + [300.0] * 50 + [0.0], "y0": [50.0] * 100 + [0.0],
            "x1": [200.0] * 50 + [400.0] * 50 + [640.0], "y1": [300.0] * 100 + [480.0],
        })
        # the single stray box covering the picture is left out at the 2% edges; 10% of 300 by 250 is added on each side
        assert rating_frames.people_box(boxes, 0.1, 640, 480) == (70, 25, 430, 325)

    def test_a_box_given_or_the_whole_picture_needs_no_bodies(self, monkeypatch):
        given = rating_frames.crop_for(None, None, None, RatingFrames(crop=(10, 20, 111, 221)), 0.0, 1.0, (640, 480))
        assert (given.box, given.source) == ((10, 20, 110, 220), rating_frames.GIVEN)
        whole = rating_frames.crop_for(None, None, None, RatingFrames(crop="whole"), 0.0, 1.0, (640, 480))
        assert (whole.box, whole.source) == ((0, 0, 640, 480), rating_frames.WHOLE_PICTURE)
        monkeypatch.setattr(rating_frames, "body_boxes", lambda *args: pd.DataFrame(columns=["subject_id", "frame", "x0", "y0", "x1", "y1"]))
        unnamed = rating_frames.crop_for(None, None, None, RatingFrames(), 0.0, 1.0, (640, 480))
        assert (unnamed.box, unnamed.source) == ((0, 0, 640, 480), rating_frames.NO_POSE)

    def test_a_box_must_run_from_its_top_left_to_its_bottom_right(self):
        with pytest.raises(ValueError, match="x0 below x1"):
            RatingFrames(crop=(100, 0, 50, 10))


def shown_lines():
    frame = module.rating_lines(transcript_frame(TALK))
    return module.part_lines(frame, words_frame(TALK), 0.0, 60.0)


TEXT_ONLY = Path(__file__).with_name("rating_prompts_text_only.json")


def text_only_prompts() -> dict[str, list[str]]:
    """Prompts of versions shown no frames, in every combination of the options that change their wording."""
    lines, scales, people = shown_lines(), read_scales_from(SCALES), ["parent", "child"]
    cases = {}
    for scale in ("warmth", "flow"):
        for evidence in (False, True):
            for reasons in (False, True):
                for show in ((), ("timing",)):
                    name = f"{scale}-{'evidence' if evidence else 'plain'}-{'reasons' if reasons else 'bare'}-{'+'.join(show) or 'none'}"
                    cases[name] = build_rating_prompt(
                        scales.get(scale), lines, people=people, minutes=1.0, show=show, reasons=reasons, evidence=evidence, start=0.0, end=60.0,
                    )
    spans = [(1.0, 4.0), (20.0, 21.0)]
    cases["excerpts"] = build_rating_prompt(
        scales.get("warmth"), module.excerpt_lines(lines, spans), people=people, minutes=1.0, show=("timing",), evidence=True, reasons=True,
        excerpts=True, start=0.0, end=60.0, spans=spans,
    )
    example = ScaleExample(score=4, lines=[("parent", "Well done!"), ("child", "Thanks.")], why="Praise.")
    cases["examples-notes"] = build_rating_prompt(
        scales.get("flow"), lines, people=people, minutes=1.0, notes="Mind the toys.", examples=[example], evidence=True,
    )
    return {name: list(pair) for name, pair in cases.items()}


class TestPrompt:
    def test_frames_and_lines_alternate_in_time_and_every_scale_starts_the_same(self):
        frames = [(0.0, "f00001.jpg"), (2.0, "f00002.jpg"), (4.0, "f00003.jpg")]
        asked = [
            build_rating_prompt(scale, shown_lines(), people=["parent", "child"], minutes=1.0, show=["timing"], start=0.0, end=60.0,
                                frames=frames, every_s=2.0, cropped=True)
            for scale in read_scales_from(SCALES).scales
        ]
        system, user = asked[0]
        assert "one every 2 s, cropped to the people" in system
        body = user[: user.index("Rate the conversation")]
        order = [body.index(text) for text in (
            "[t=0.0 s]\n<frame f00001.jpg>", "How was school today?", "[t=2.0 s]\n<frame f00002.jpg>", "It was fine.",
            "[t=4.0 s]\n<frame f00003.jpg>", "Just fine?", "We had a test.",
        )]
        assert order == sorted(order)
        assert len({system for system, _ in asked}) == 1 and len({user[: user.index("Rate the conversation")] for _, user in asked}) == 1

    def test_with_excerpts_each_stretch_keeps_its_frames_and_lines(self):
        spans = [(1.0, 4.0), (20.0, 21.0)]
        lines = module.excerpt_lines(shown_lines(), spans)
        _, user = build_rating_prompt(
            read_scales_from(SCALES).get("flow"), lines, people=["parent", "child"], minutes=1.0, excerpts=True, start=0.0, end=60.0,
            frames=[(1.0, "a.jpg"), (3.0, "b.jpg"), (20.0, "c.jpg")], every_s=2.0, spans=spans,
        )
        body = user[: user.index("Rate the conversation")]
        # the header says what ... means; one stands between the two stretches
        assert body.count("\n...\n") == 1
        assert body.index("<frame b.jpg>") < body.index("\n...\n") < body.index("<frame c.jpg>") < body.index("We had a test.")

    def test_versions_without_frames_are_asked_word_for_word_as_before(self):
        assert text_only_prompts() == json.loads(TEXT_ONLY.read_text(encoding="utf-8"))

    def test_with_evidence_frames_are_numbered_in_order_and_may_be_cited(self):
        frames = [(0.0, "f1.jpg"), (2.0, "f2.jpg"), (4.0, "f3.jpg")]
        scales = read_scales_from(SCALES)
        system, user = build_rating_prompt(
            scales.get("warmth"), shown_lines(), people=["parent", "child"], minutes=1.0, start=0.0, end=60.0, frames=frames, every_s=2.0,
            evidence=True, reasons=True,
        )
        assert [(int(number), float(moment)) for number, moment in re.findall(r"\[frame (\d+), t=([\d.]+) s\]", user)] == [(1, 0.0), (2, 2.0), (3, 4.0)]
        assert "a line giving its number and its time" in system
        assert "Rate only from the transcript and the frames, and cite only the numbered lines and frames given." in system
        assert '{"frame": <number>, "seen": "<a few words on what the frame shows>"}' in system
        assert "what was said, the timing, the voice or what was seen." in system
        assert "Cite lines said by the parent or frames showing the parent." in user
        _, flow = build_rating_prompt(
            scales.get("flow"), shown_lines(), people=["parent", "child"], minutes=1.0, frames=frames, every_s=2.0, evidence=True,
        )
        assert "Cite lines said by" not in flow
        # without evidence a frame keeps the label it had
        _, plain = build_rating_prompt(scales.get("warmth"), shown_lines(), people=["parent", "child"], minutes=1.0, frames=frames, every_s=2.0)
        assert "[t=2.0 s]\n<frame f2.jpg>" in plain and "[frame" not in plain

    def test_numbers_run_on_across_marked_stretches(self):
        spans = [(1.0, 4.0), (20.0, 21.0)]
        _, user = build_rating_prompt(
            read_scales_from(SCALES).get("flow"), module.excerpt_lines(shown_lines(), spans), people=["parent", "child"], minutes=1.0,
            excerpts=True, frames=[(1.0, "a.jpg"), (3.0, "b.jpg"), (20.0, "c.jpg")], every_s=2.0, spans=spans, evidence=True,
        )
        assert user.index("[frame 2, t=3.0 s]") < user.index("\n...\n") < user.index("[frame 3, t=20.0 s]\n<frame c.jpg>")

    def test_pictures_go_where_the_text_marks_them(self):
        assert in_order("a\n<frame f1.jpg>\nb", [b"1"]) == ["a\n", b"1", "\nb"]
        # a judge's text has no marks: its pictures follow it, as before
        assert in_order("plain", [b"1", b"2"]) == ["plain", b"1", b"2"]


class TestFrameEvidence:
    def test_a_cited_frame_must_have_been_shown_and_say_what_it_shows(self):
        from macaw.coding.llm.evidence import FRAME, LINE, NO_FRAME, NOTHING_SEEN, OTHER_SPEAKER, check_cited

        shown = pd.DataFrame({"frame": [1, 2], "t": [0.0, 5.0], "from_recording": ["r1", "r1"], "from_t": [0.0, 5.0]})
        lines = pd.DataFrame({"line": [1], "speaker": ["child"], "t_start": [1.0], "t_end": [2.0], "text": ["It was fine."]})
        warmth = read_scales_from(SCALES).get("warmth")
        rows = check_cited(
            [{"frame": 2, "seen": "child turns away"}, {"frame": "[frame 1, t=0.0 s]", "seen": "parent smiles"}, {"frame": 7, "seen": "x"},
             {"frame": 1, "seen": "  "}, {"frame": 1}, {"line": 1, "quote": "fine"}],
            lines, warmth, shown,
        )
        assert [row["kind"] for row in rows] == [FRAME] * 5 + [LINE]
        # a frame of the child passes for a scale that rates the parent: who is in a frame is not checked
        assert [row["problem"] for row in rows] == ["", "", NO_FRAME, NOTHING_SEEN, NOTHING_SEEN, OTHER_SPEAKER]
        assert (rows[0]["frame"], rows[0]["frame_t"], rows[0]["frame_from"], rows[0]["frame_from_t"]) == (2, 5.0, "r1", 5.0)
        assert rows[1]["frame"] == 1 and pd.isna(rows[0]["t_start"]) and rows[5]["frame"] is None
        # with no frames shown, any frame cited is one that was not
        assert check_cited([{"frame": 1, "seen": "x"}], lines, warmth)[0]["problem"] == NO_FRAME

    def test_frames_given_the_same_words_are_counted_and_reported(self):
        from macaw.coding.llm.evidence import evidence_table, evidence_warnings

        shown = pd.DataFrame({"frame": [1, 2, 3], "t": [0.0, 5.0, 10.0], "from_recording": ["r1"] * 3, "from_t": [0.0, 5.0, 10.0]})
        lines = pd.DataFrame({"line": [1], "speaker": ["child"], "t_start": [1.0], "t_end": [2.0], "text": ["It was fine."]})
        flow = read_scales_from(SCALES).get("flow")
        items = [
            {"frame": 1, "seen": "People sitting at a table."}, {"frame": 2, "seen": "people sitting at a table"},
            {"frame": 3, "seen": "child turns away"}, {"frame": 9, "seen": "people sitting at a table"},
            {"frame": 3, "seen": "child turns away"}, {"line": 1, "quote": "fine"},
        ]
        answers = pd.DataFrame({
            "recording_id": ["r1", "r1"], "segment": ["", ""], "scale": ["flow", "flow"], "rater": ["seen", "seen"], "sample": [0, 1],
            "answer": [json.dumps({"evidence": items, "score": 3}), json.dumps({"evidence": [{"frame": 3, "seen": "x"}], "score": 3})],
        })
        table, counted = evidence_table(answers, [module.Prompt("", 0.0, 60.0, flow, "", "", lines, (), shown)])
        # the same frame cited twice, or a frame not shown, shares its words with no other frame
        assert table[table["sample"] == 0]["seen_elsewhere"].tolist() == [True, True, False, False, False, pd.NA]
        assert counted["n_evidence_frames_same_words"].tolist() == [2, 0] and table["problem"].tolist()[:3] == ["", "", ""]
        [warned] = [warning for warning in evidence_warnings(table, counted, "r1") if warning.check == "rating_frames_same_words"]
        assert "In 1 of the 2 answers that cited frames" in warned.message and "(2 of the 5 cited frames that were shown)" in warned.message


class TestBorrowed:
    def test_a_moment_past_the_shorter_part_goes_round_to_its_start(self):
        assert rating_frames.borrowed_moment(15.0, 0.0, 100.0, 118.0) == (115.0, False)
        assert rating_frames.borrowed_moment(20.0, 0.0, 100.0, 118.0) == (102.0, True)
        assert rating_frames.borrowed_moment(318.0, 300.0, 100.0, 118.0) == (100.0, True)

    @pytest.mark.parametrize("rows,message", [
        ({"recording": ["r1"], "from_recording": ["r1"]}, "its own frames"),
        ({"recording": ["r1", "r1"], "from_recording": ["r2", "r3"]}, "listed twice"),
        ({"recording": ["r1"], "from_recording": [""]}, "gives both"),
        ({"recording": ["r1"], "other": ["r2"]}, "no column from_recording"),
    ])
    def test_the_table_is_checked_line_by_line(self, tmp_path, rows, message):
        pd.DataFrame(rows).to_csv(tmp_path / "borrowed.csv", index=False)
        with pytest.raises(ConfigError, match=message):
            rating_frames.read_borrowed(tmp_path / "borrowed.csv")


class TestServer:
    def test_the_request_holds_text_and_pictures_in_order(self, monkeypatch):
        sent = {}

        def post(self, url, body, headers):
            sent.update(body)
            return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}}

        monkeypatch.setattr(VLLMBackend, "_post", post)
        VLLMBackend(LLMSection(provider="vllm", model="m")).complete("s", "a <frame x.jpg> b <frame y.jpg>", images=[b"x", b"y"])
        content = sent["messages"][1]["content"]
        assert [part["type"] for part in content] == ["text", "image_url", "text", "image_url"]
        assert content[1]["image_url"]["url"].endswith(backends_module._base64(b"x"))

    def test_ollama_says_whether_the_model_reads_pictures_and_what_one_takes(self, monkeypatch):
        backend = OllamaBackend(LLMSection(provider="ollama", model="qwen3-vl:8b"))
        monkeypatch.setattr(OllamaBackend, "_post", lambda self, url, body, headers: {"capabilities": ["completion", "vision"]})
        assert backend.reads_pictures() is True
        monkeypatch.setattr(OllamaBackend, "_post", lambda self, url, body, headers: {"capabilities": ["completion"]})
        assert backend.reads_pictures() is False
        monkeypatch.setattr(
            OllamaBackend, "_post",
            lambda self, url, body, headers: {"usage": {"prompt_tokens": 10 if isinstance(body["messages"][0]["content"], str) else 1044}},
        )
        assert backend.picture_tokens(b"jpg") == 1034


def tiny_video(path, source="testsrc"):
    joined = f"{source}{':' if '=' in source else '='}size=320x240:rate=10:duration=30"
    run_ffmpeg(["-f", "lavfi", "-i", joined, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "10", str(path)])
    return path


def sent_pictures(request):
    from PIL import Image

    parts = [part for part in request["messages"][-1]["content"] if part["type"] == "image_url"]
    return [Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))).convert("RGB") for part in parts]


class TestRun:
    def framed(self, tmp_path, server, frames=None, provider="vllm", **own):
        llm = {"model": "tiny", "provider": provider, "base_url": f"{server.url}/v1", "n_samples": 1, "show": ["timing"]}
        version = {"scales": "names.yaml", "frames": frames or {"every_s": 5.0, "long_side": 160}, **own}
        return transcribed(project(tmp_path, {"seen": version}, llm=llm))

    def test_cited_frames_are_kept_with_their_times_and_counted_apart_from_lines(self, monkeypatch, tmp_path, server):
        from macaw.coding.llm.evidence import FRAME, LINE, NO_FRAME, NOTHING_SEEN, OTHER_SPEAKER

        config = self.framed(tmp_path, server, evidence=True)
        camera = MediaFile(tiny_video(tmp_path / "room.mp4"), "video", device="room")
        monkeypatch.setattr(module, "apply_prep_timing", lambda config, recording: None)
        monkeypatch.setattr(module, "frame_camera", lambda recording: camera)
        server.score = lambda scale, sample: json.dumps({
            "evidence": [{"line": 1, "quote": "How was school"}, {"frame": 2, "seen": "child looks away"}, {"frame": 9, "seen": "x"},
                         {"frame": 1, "seen": ""}],
            "score": 1,
        })
        result = rate(monkeypatch, config, "seen", window=(0.0, 24.0))
        table = pd.read_parquet(result.artifacts["frames.parquet"])
        # numbered 1 to 5 in time order, as the request labels them
        assert table["number"].tolist() == [1, 2, 3, 4, 5] and table["t"].tolist() == [0.0, 5.0, 10.0, 15.0, 20.0]
        user = "".join(part.get("text", "") for part in server.requests[0]["messages"][-1]["content"])
        labels = [(int(number), float(moment)) for number, moment in re.findall(r"\[frame (\d+), t=([\d.]+) s\]", user)]
        assert labels == list(zip(table["number"], table["t"]))
        evidence = pd.read_parquet(result.artifacts["evidence.parquet"])
        warmth = evidence[evidence["scale"] == "warmth"]
        assert warmth["kind"].tolist() == [LINE, FRAME, FRAME, FRAME]
        assert warmth["problem"].tolist() == ["", "", NO_FRAME, NOTHING_SEEN]
        seen = warmth.iloc[1]
        assert (seen["frame"], seen["frame_t"], seen["frame_from"], seen["frame_from_t"], seen["seen"]) == (2, 5.0, "r1", 5.0, "child looks away")
        assert evidence[evidence["scale"] == "on_task"]["problem"].iloc[0] == OTHER_SPEAKER
        answers = pd.read_parquet(result.artifacts["answers.parquet"]).set_index("scale")
        assert answers.loc["warmth", ["n_evidence", "n_evidence_lines", "n_evidence_frames", "n_evidence_frames_same_words"]].tolist() == [4, 1, 3, 0]
        assert answers.loc["warmth", "evidence_problem"] == "no such frame 1; nothing said of what was seen 1"
        [warned] = [warning for warning in result.warnings if warning.check == "rating_evidence_unchecked"]
        assert "lines and frames the model cited" in warned.message

    def test_borrowed_frames_come_from_the_paired_recording_as_many_and_as_far_apart(self, monkeypatch, tmp_path, server):
        from types import SimpleNamespace

        from macaw.project.layout import layout_for
        from macaw.project.recording import build_recording

        (tmp_path / "raw" / "r2").mkdir(parents=True)
        table_path = tmp_path / "borrowed.csv"
        pd.DataFrame({"recording": ["r1", "r3"], "from_recording": ["r2", "r1"]}).to_csv(table_path, index=False)
        config = self.framed(tmp_path, server, {"every_s": 5.0, "crop": "whole", "from": "borrowed.csv"})
        cameras = {
            "r1": MediaFile(tiny_video(tmp_path / "own.mp4", "color=c=red"), "video", device="room"),
            # the other family's part is 18 s long, on a camera that starts 100 s into its recording
            "r2": MediaFile(tiny_video(tmp_path / "lent.mp4", "color=c=blue"), "video", device="room", offset_s=100.0),
        }
        windows = {"r1": (0.0, 24.0), "r2": (100.0, 118.0)}
        monkeypatch.setattr(module, "load_prep", lambda config, rid: SimpleNamespace(t_origin_s=windows[rid][0], duration_s=windows[rid][1] - windows[rid][0]))
        monkeypatch.setattr(module, "apply_prep_timing", lambda config, recording: None)
        monkeypatch.setattr(module, "frame_camera", lambda recording: cameras[recording.recording_id])
        monkeypatch.setattr(module, "load_video_prep", lambda config, rid: {"room": {}})
        step = module._VERSIONS["seen"](config)
        result = step.run(build_recording("r1", config), layout_for(config).recording_dir("r1"))
        table = pd.read_parquet(result.artifacts["frames.parquet"])
        # as many frames as r1's own, 5 s apart, labelled with r1's times; the last goes round to the start of r2's part
        assert table["t"].tolist() == [0.0, 5.0, 10.0, 15.0, 20.0] and table["number"].tolist() == [1, 2, 3, 4, 5]
        assert set(table["from_recording"]) == {"r2"} and table["from_t"].tolist() == [100.0, 105.0, 110.0, 115.0, 102.0]
        assert table["wrapped"].tolist() == [False, False, False, False, True] and table["video_t"].tolist() == [0.0, 5.0, 10.0, 15.0, 2.0]
        user = "".join(part.get("text", "") for part in server.requests[0]["messages"][-1]["content"])
        assert re.findall(r"\[t=([\d.]+) s\]", user) == ["0.0", "5.0", "10.0", "15.0", "20.0"]
        pictures = sent_pictures(server.requests[0])
        assert len(pictures) == 5 and all(picture.getpixel((160, 120))[2] > 200 > picture.getpixel((160, 120))[0] for picture in pictures)
        crop = json.loads(result.artifacts["frames_crop.json"].read_text(encoding="utf-8"))
        assert crop["from_recording"] == "r2" and crop["box"] == [0, 0, 320, 240]
        checks = {warning.check: warning.message for warning in result.warnings}
        assert "borrowed.csv" in checks["rating_frames_borrowed"] and "1 of them were taken again" in checks["rating_frames_borrowed"]
        assert result.specs[0].params["frames"]["from"] == "borrowed.csv" and "from_" not in result.specs[0].params["frames"]
        # a recording the table does not list is shown its own frames
        assert step.lender("r3") == "r1" and step.lender("r2") is None
        # only a change to a recording's own row reruns it
        before = step.extra_key(build_recording("r1", config))
        pd.DataFrame({"recording": ["r1", "r3"], "from_recording": ["r2", "r2"]}).to_csv(table_path, index=False)
        assert step.extra_key(build_recording("r1", config)) == before
        pd.DataFrame({"recording": ["r1"], "from_recording": ["r3"]}).to_csv(table_path, index=False)
        assert step.extra_key(build_recording("r1", config)) != before

    def test_the_paired_recording_s_frames_are_cut_to_its_own_box(self, tmp_path, server):
        from macaw.coding.llm.rating_frames import PEOPLE

        config = self.framed(tmp_path, server, {"every_s": 5.0, "crop": "whole", "from": "borrowed.csv"})
        own = Path(config.project.recording_config_dir)
        own.mkdir(parents=True, exist_ok=True)
        step = module._VERSIONS["seen"](config)
        assert step._lender_cut("r2", "r1") == "whole"
        # a box r1's own settings give belongs to r1's camera
        (own / "r1.yaml").write_text("coding: {ratings: {seen: {frames: {crop: [0, 0, 100, 100]}}}}\n", encoding="utf-8")
        assert step._lender_cut("r2", "r1") == PEOPLE
        (own / "r2.yaml").write_text("coding: {ratings: {seen: {frames: {crop: [10, 20, 110, 220]}}}}\n", encoding="utf-8")
        assert step._lender_cut("r2", "r1") == (10.0, 20.0, 110.0, 220.0)

    def test_frames_from_a_recording_without_a_prepared_video_stop_with_what_to_run(self, monkeypatch, tmp_path, server):
        from macaw.exceptions import InputNotFoundError

        (tmp_path / "raw" / "r2").mkdir(parents=True)
        pd.DataFrame({"recording": ["r1"], "from_recording": ["r2"]}).to_csv(tmp_path / "borrowed.csv", index=False)
        config = self.framed(tmp_path, server, {"every_s": 5.0, "crop": "whole", "from": "borrowed.csv"})
        monkeypatch.setattr(module, "apply_prep_timing", lambda config, recording: None)
        with pytest.raises(InputNotFoundError, match="run prep.video for r2"):
            rate(monkeypatch, config, "seen", window=(0.0, 24.0))

    def test_frames_go_between_the_lines_and_what_was_shown_is_kept(self, monkeypatch, tmp_path, server):
        config = self.framed(tmp_path, server)
        camera = MediaFile(tiny_video(tmp_path / "room.mp4"), "video", device="room")
        monkeypatch.setattr(module, "apply_prep_timing", lambda config, recording: None)
        monkeypatch.setattr(module, "frame_camera", lambda recording: camera)
        result = rate(monkeypatch, config, "seen", window=(0.0, 24.0))
        body = server.requests[0]["messages"][-1]["content"]
        kinds = [part["type"] for part in body]
        # five frames, at 0, 5, 10, 15 and 20 s, each after its time and before the lines that follow
        assert kinds.count("image_url") == 5 and kinds[0] == "text" and kinds[1] == "image_url"
        texts = [part["text"] for part in body if part["type"] == "text"]
        assert texts[0].rstrip().endswith("[t=0.0 s]") and "How was school today?" in texts[1] and "[t=20.0 s]" in texts[4]
        prompts = pd.read_parquet(result.artifacts["prompts.parquet"])
        assert PICTURE_MARK.findall(prompts["user"].iloc[0]) == [f"f0000{index}.jpg" for index in range(1, 6)]
        table = pd.read_parquet(result.artifacts["frames.parquet"])
        assert table["t"].tolist() == [0.0, 5.0, 10.0, 15.0, 20.0] and table["shown"].all() and set(table["video"]) == {"room"}
        crop = json.loads(result.artifacts["frames_crop.json"].read_text(encoding="utf-8"))
        # no one was named on the camera, so the whole picture is shown and the QC says so
        assert crop["source"] == rating_frames.NO_POSE and crop["box"] == [0, 0, 320, 240]
        assert result.artifacts["frames_crop.jpg"].is_file()
        checks = {warning.check: warning.message for warning in result.warnings}
        assert "rating_frames_whole" in checks and "700 tokens" in checks["rating_frame_tokens"]

    def test_a_moment_the_video_does_not_reach_shows_no_frame(self, monkeypatch, tmp_path, server):
        config = self.framed(tmp_path, server, {"every_s": 10.0, "crop": "whole"})
        camera = MediaFile(tiny_video(tmp_path / "room.mp4"), "video", device="room", offset_s=15.0)
        monkeypatch.setattr(module, "apply_prep_timing", lambda config, recording: None)
        monkeypatch.setattr(module, "frame_camera", lambda recording: camera)
        result = rate(monkeypatch, config, "seen", window=(0.0, 60.0))
        table = pd.read_parquet(result.artifacts["frames.parquet"])
        # the camera starts 15 s into the recording and lasts 30 s
        assert table["shown"].tolist() == [False, False, True, True, True, False]
        assert any(warning.check == "rating_frames_unseen" for warning in result.warnings)
        # without long_side a frame goes as the camera filmed it
        import base64
        import io

        from PIL import Image

        sent = next(part for part in server.requests[0]["messages"][-1]["content"] if part["type"] == "image_url")
        assert Image.open(io.BytesIO(base64.b64decode(sent["image_url"]["url"].split(",", 1)[1]))).size == (320, 240)

    def test_frames_go_to_no_company_s_servers(self, tmp_path, server):
        config = self.framed(tmp_path, server, provider="openai")
        with pytest.raises(ConfigError, match="only to a model on this machine"):
            module._VERSIONS["seen"](config).check_available()

    def test_frames_need_the_prepared_video_and_come_after_the_people_are_named(self, tmp_path, server):
        config = self.framed(tmp_path, server)
        step = module._VERSIONS["seen"]
        assert "prep.video" in step.dependencies(config) and "pose.keypoints" in step.ordered_after(config)
        plain = resolve_config(project(tmp_path / "plain", {"text_only": {"scales": "names.yaml"}}))
        assert "prep.video" not in module._VERSIONS["text_only"].dependencies(plain)
        assert "pose.keypoints" not in module._VERSIONS["text_only"].ordered_after(plain)


class TestReports:
    def test_the_methods_say_how_the_frames_were_taken_and_cut(self):
        from types import SimpleNamespace

        from macaw.qc.boilerplate import _rating

        params = {
            "rating": "seen", "scales": "talk", "n_scales": 3, "provider": "ollama", "model": "m", "show": ["timing"], "n_samples": 1,
            "temperature": 0.0, "frames": {"every_s": 2.0, "long_side": 448, "crop": "people", "margin": 0.1},
        }
        text = _rating(None, SimpleNamespace(params=params, backend_version="m"), {})
        assert "one every 2 s from the start of each part rated" in text and "widened by 10% on each side" in text
        assert "scaled to 448 pixels on the longer side" in text
        params["frames"].pop("long_side")
        text = _rating(None, SimpleNamespace(params=params, backend_version="m"), {})
        assert "at the camera's own resolution" in text and "control" not in text
        params["frames"]["from"] = "borrowed.csv"
        params["evidence"] = True
        text = _rating(None, SimpleNamespace(params=params, backend_version="m"), {})
        assert 'each recording listed in "borrowed.csv" was shown instead the frames of the other recording' in text
        assert "lines and frames the score rested on" in text and "each cited frame was checked to have been shown" in text

    def test_the_quality_report_gives_the_box_and_where_to_change_it(self, tmp_path, server):
        from macaw.project.layout import layout_for
        from macaw.qc.report import _rating_frames

        config = TestRun().framed(tmp_path, server)
        layout = layout_for(config)
        folder = layout.artifacts_dir("r1", "coding.rate.seen")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "frames_crop.json").write_text(json.dumps(
            {"camera": "room", "box": [10, 20, 300, 220], "source": "people", "width": 320, "height": 240, "people": ["child", "parent"]}
        ), encoding="utf-8")
        row = _rating_frames(config, layout, "r1")[0]
        assert row["by"] == "software" and "cut to the bodies of child, parent, box [10, 20, 300, 220]" in row["decided"]
        assert "coding.ratings.seen.frames.crop" in row["decided"] and "r1.yaml" in row["decided"]
        (folder / "frames_crop.json").write_text(json.dumps(
            {"camera": "room", "box": [0, 0, 320, 240], "source": "whole", "width": 320, "height": 240, "people": [], "from_recording": "r2"}
        ), encoding="utf-8")
        borrowed = _rating_frames(config, layout, "r1")[0]["decided"]
        # the box to change is the other recording's
        assert borrowed.startswith("borrowed from r2, as frames.from gives, from its room") and "r2.yaml" in borrowed


class TestPlan:
    def test_the_plan_gives_what_a_frame_takes_when_the_model_is_loaded(self, monkeypatch, tmp_path, server):
        config = TestRun().framed(tmp_path, server, {"every_s": 5.0, "long_side": 320})
        monkeypatch.setattr(module, "build_recording", lambda rid, config: None)
        monkeypatch.setattr(module, "frame_camera", lambda recording: MediaFile(tmp_path / "room.mp4", "video"))
        monkeypatch.setattr(module, "picture_size", lambda path: (640, 480))
        asked = []
        monkeypatch.setattr(VLLMBackend, "picture_tokens", lambda self, picture: asked.append(picture) or 1000)
        notice = module.frame_cost_notice(config, "coding.rate.seen", ["r1"])
        # the test server serves the model, so it counts as loaded; the frame is scaled to long_side as the step would
        assert "about 1000 tokens" in notice and "grey 320x240 frame" in notice and "12000 tokens a minute" in notice
        monkeypatch.setattr(VLLMBackend, "loaded", lambda self: False)
        assert module.frame_cost_notice(config, "coding.rate.seen", ["r1"]) is None and len(asked) == 1

    def test_ollama_counts_a_model_as_loaded_only_when_it_is_in_memory(self, monkeypatch):
        backend = OllamaBackend(LLMSection(provider="ollama", model="qwen3-vl:8b"))
        monkeypatch.setattr(backends_module, "_get", lambda url, *args, **kwargs: {"models": [{"name": "qwen3-vl:8b"}]})
        assert backend.loaded()
        monkeypatch.setattr(backends_module, "_get", lambda url, *args, **kwargs: {"models": []})
        assert not backend.loaded()

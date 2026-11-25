from __future__ import annotations

import json

import pandas as pd
import pytest
from test_ratings import SCALES, TALK, project, rate, read_scales_from, server, transcribed, transcript_frame, words_frame  # noqa: F401

from macaw.coding.llm import backends as backends_module
from macaw.coding.llm import rate as module
from macaw.coding.llm import rating_frames
from macaw.coding.llm.backends import OllamaBackend, VLLMBackend, in_order
from macaw.coding.llm.prompts import PICTURE_MARK, build_rating_prompt
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

    def test_pictures_go_where_the_text_marks_them(self):
        assert in_order("a\n<frame f1.jpg>\nb", [b"1"]) == ["a\n", b"1", "\nb"]
        # a judge's text has no marks: its pictures follow it, as before
        assert in_order("plain", [b"1", b"2"]) == ["plain", b"1", b"2"]


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


def tiny_video(path):
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "10", str(path)])
    return path


class TestRun:
    def framed(self, tmp_path, server, frames=None, provider="vllm"):
        llm = {"model": "tiny", "provider": provider, "base_url": f"{server.url}/v1", "n_samples": 1, "show": ["timing"]}
        return transcribed(project(tmp_path, {"seen": {"scales": "names.yaml", "frames": frames or {"every_s": 5.0, "long_side": 160}}}, llm=llm))

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
        assert "at the camera's own resolution" in _rating(None, SimpleNamespace(params=params, backend_version="m"), {})

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

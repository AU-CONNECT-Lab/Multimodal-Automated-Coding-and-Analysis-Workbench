from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macaw.qc.checks import CANDIDATE_SCHEMA
from macaw.qc.figures import motion_figure, overview_figure, pitch_figure, present_spans, timeline_figure, tracks_figure
from macaw.store.schema import coerce_frame

pytest.importorskip("matplotlib")

# the column types candidates come back from disk with, where a span of nobody's is NA
CANDIDATES = coerce_frame(
    pd.DataFrame(
        {"recording_id": "r", "subject_id": ["parent", None], "check": "c", "reason": ["octave_error", "track_swap"],
         "t_start": [2.0, 5.0], "t_end": [3.0, 5.0], "t": [2.5, 5.0], "score": 1.0}
    ),
    CANDIDATE_SCHEMA,
    "candidates",
)


class TestPresentSpans:
    def test_nearby_times_make_one_run(self):
        assert present_spans(np.array([0.1, 0.3, 0.5]), step=0.5) == [(0.0, 1.0)]

    def test_a_gap_of_a_bin_splits_the_run(self):
        assert present_spans(np.array([0.1, 1.2]), step=0.5) == [(0.0, 0.5), (1.0, 0.5)]

    def test_nothing_seen_draws_nothing(self):
        assert present_spans(np.empty(0)) == []


class TestTracks:
    """Where each tracked body was on each camera's picture."""

    def view(self, camera, picture=None):
        steady = np.full(400, 200.0)
        return {
            "camera": camera, "size": (640, 360), "picture": picture, "areas": [(480.0, 0.0, 640.0, 180.0)],
            "tracks": [
                {"who": "parent", "ignored": False, "label": "parent", "x": steady, "y": steady},
                {"who": None, "ignored": False, "label": "track 7", "x": steady + 150, "y": steady},
                {"who": None, "ignored": False, "label": "", "x": steady[:3] + 50, "y": steady[:3]},
                {"who": None, "ignored": True, "label": "track 2", "x": steady + 350, "y": steady - 120},
            ],
        }

    def size(self, path):
        from matplotlib.image import imread

        height, width = imread(path).shape[:2]
        return width, height

    def test_it_is_drawn_without_a_picture_behind_it(self, tmp_path):
        path = tracks_figure([self.view("room")], ["parent", "child"], tmp_path / "tracks.png")
        assert self.size(path)[0] == round((12 + 2 * 0.15) * 300)

    def test_a_picture_behind_it_is_never_drawn_larger_than_its_own_pixels(self, tmp_path):
        from macaw.qc.figures import tracks_panel_width

        small, full = np.zeros((360, 640, 3), dtype=np.uint8), np.zeros((1080, 1920, 3), dtype=np.uint8)
        assert tracks_panel_width([self.view("room", small)], 1) == 640 / 300
        assert tracks_panel_width([self.view("room", full)], 1) == 1920 / 300
        assert tracks_panel_width([self.view("a", full), self.view("b", full)], 2) == 6.0
        assert tracks_panel_width([self.view("a", full), self.view("b", small), self.view("c")], 2) == 640 / 300
        # a small picture keeps its size in a figure wide enough for the legend
        one = tracks_figure([self.view("room", small)], ["parent", "child"], tmp_path / "one.png")
        assert self.size(one)[0] == 6 * 300
        three = tracks_figure([self.view("a", small), self.view("b", small), self.view("c")], ["parent"], tmp_path / "three.png")
        assert self.size(three)[1] > 2 * 360


def test_the_report_draws_the_tracks_of_each_camera_and_lists_what_was_left_out(tmp_path):
    import json

    import yaml
    from setup_helpers import camera, put, write_setup

    from macaw.config.loader import resolve_config, scaffold_project_config
    from macaw.extractors.prep import VIDEO_INFO_NAME, VideoPrepExtractor
    from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
    from macaw.media.ffmpeg import run_ffmpeg
    from macaw.project.layout import layout_for
    from macaw.project.recording import build_recording
    from macaw.qc.checks import RecordingResults
    from macaw.qc.report import _decisions, _tracks
    from macaw.store.keypoints import write_keypoints
    from macaw.store.schema import KEYPOINT_SCHEMA

    path = tmp_path / "project.yaml"
    scaffold_project_config(path, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["project"]["roles"] = ["parent", "child"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    setup = write_setup(path, cameras={"room": camera()})
    setup.write_text(setup.read_text(encoding="utf-8") + "ignore: {room: [[0.75, 0, 1, 0.5]]}\n", encoding="utf-8")
    video = put(tmp_path / "raw", "003", "room", "a.mp4")
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc=s=640x360:r=10:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)])
    config = resolve_config(path, recording_id="003")
    layout = layout_for(config)

    rows = []
    for track, who, cx, cy in ((1, "parent", 200.0, 220.0), (2, "child", 380.0, 240.0), (3, None, 560.0, 90.0), (4, None, 60.0, 300.0)):
        for frame in range(30 if track < 4 else 6):
            for index, joint in enumerate(STANDARD_JOINTS):
                rows.append({
                    "recording_id": "003", "subject_id": who, "source": "pose.keypoints", "video": "room", "frame": frame, "t": frame / 10,
                    "joint": joint, "x": cx + 3 * (index % 5) + frame % 7, "y": cy + 5 * (index // 5), "z": float("nan"), "confidence": 0.9,
                    "track_id": track, "units": "pixel", "metric_scale_valid": None,
                })
    write_keypoints(coerce_frame(pd.DataFrame(rows), KEYPOINT_SCHEMA, "keypoints"), layout.keypoints_path("003"))
    kept = layout.artifacts_dir("003", "pose.keypoints")
    kept.mkdir(parents=True)
    (kept / "ignored.json").write_text(json.dumps({"room": {"tracks": [3], "frames": 30, "seconds": 3.0}}), encoding="utf-8")
    prepared = layout.artifacts_dir("003", VideoPrepExtractor.name)
    prepared.mkdir(parents=True)
    (prepared / VIDEO_INFO_NAME).write_text(json.dumps([{"name": "room", "file": None, "size": [640, 360], "prepared_size": [640, 360]}]), encoding="utf-8")

    results = RecordingResults(layout, "003")
    page = layout.qc_report_path("003")
    found = _tracks(config, layout, results, build_recording("003", config), page)
    assert found["ignored"] == [{"file": "room", "tracks": "3", "seconds": "3 s"}]
    figure = layout.renders_dir("003") / "report" / "tracks.png"
    assert figure.is_file() and found["href"].endswith("report/tracks.png")
    from matplotlib.image import imread

    assert imread(figure).shape[1] == 6 * 300 and (figure.parent / "tracks_room.jpg").is_file()
    rows = _decisions(config, layout, results, "003", [], [], [], [], [], found)
    left_out = next(row for row in rows if row["what"] == "Bodies left out in an area to ignore")
    assert "room: tracks 3, in view 3 s" in left_out["decided"] and left_out["by"] == "software" and left_out["href"] == "#Camera"
    assert _tracks(config, layout, RecordingResults(layout, "nobody"), build_recording("003", config), page) is None


class TestFigures:
    def test_the_timeline_is_written_even_for_a_person_never_seen(self, tmp_path):
        seen = {"parent": present_spans(np.arange(0, 10, 0.04))}
        heard = {"parent": [(1.0, 3.0)], "child": [(5.0, 1.0)]}
        path = timeline_figure(["parent", "child"], seen, heard, "speaking", CANDIDATES, tmp_path / "t.png")
        assert path.stat().st_size > 0

    def test_a_voice_that_is_no_ones_gets_a_heard_lane_only(self, tmp_path, monkeypatch):
        from macaw.qc import figures

        drawn = []
        monkeypatch.setattr(figures, "_save", lambda figure, path: drawn.append(figure) or path)
        seen = {"parent": present_spans(np.arange(0, 10, 0.04))}
        heard = {"parent": [(1.0, 3.0)], "other": [(4.0, 2.0)], "none": []}
        timeline_figure(["parent", "child"], seen, heard, "speaking", CANDIDATES, tmp_path / "t.png")
        lanes = [label.get_text() for label in drawn[0].axes[0].get_yticklabels()]
        assert lanes == ["parent  seen", "parent  heard", "child  seen", "child  heard", "other  heard"]

    def test_the_pitch_panels_cover_a_person_without_pitch(self, tmp_path):
        pitch = {"parent": (np.arange(0, 10, 0.01), np.full(1000, 220.0))}
        path = pitch_figure(["parent", "child"], pitch, CANDIDATES, tmp_path / "p.png")
        assert path.stat().st_size > 0

    def test_the_motion_panels_shade_spans_of_nobody_too(self, tmp_path):
        energy = {"parent": (np.arange(0, 10, 0.04), np.full(250, 0.02))}
        path = motion_figure(["parent", "child"], energy, CANDIDATES, tmp_path / "m.png")
        assert path.stat().st_size > 0


class TestOverview:
    def draw(self, tmp_path, span, name="o.png"):
        t = np.arange(0.0, span[1], 0.5)
        lanes = [
            ("interpersonal_distance", [("parent to child", "parent", t, 100 + np.sin(t))], False),
            ("left_wrist_speed", [("parent", "parent", t, np.cos(t)), ("child", "child", t, np.sin(t))], False),
            ("not_made", [], False),
        ]
        codes = pd.DataFrame(
            {"tier": ["backchannel", "speech act"], "subject_id": ["child", None], "t_start": [10.0, 150.0], "t_end": [10.5, 152.0]}
        )
        talk = {"parent": [(0.0, 5.0)], "child": [(120.0, 4.0)], "other": [(60.0, 2.0)]}
        return overview_figure(["parent", "child"], talk, lanes, codes, CANDIDATES, span, tmp_path / name)

    def test_it_is_drawn_at_report_resolution(self, tmp_path):
        from matplotlib.image import imread

        assert imread(self.draw(tmp_path, (0.0, 200.0))).shape[1] == 12 * 300

    def test_a_recording_longer_than_a_row_wraps_onto_more_rows(self, tmp_path):
        from matplotlib.image import imread

        one = imread(self.draw(tmp_path, (0.0, 200.0), "one.png")).shape[0]
        two = imread(self.draw(tmp_path, (0.0, 400.0), "two.png")).shape[0]
        assert two > 1.8 * one

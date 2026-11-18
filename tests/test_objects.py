from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.schema import ObjectsSection
from macaw.extractors.video.objects import track as track_module
from macaw.extractors.video.objects.decisions import (
    ABSENT,
    KEPT,
    STARTS,
    add_start,
    clear,
    keep,
    read_decisions,
    set_absent,
    tracking_key,
    write_decisions,
)
from macaw.extractors.video.objects.outline import distance, inside, mask_outline, polygons
from macaw.extractors.video.objects.track import (
    FINDER,
    NOT_FOUND,
    NOT_HERE,
    PERSON,
    ObjectTrackExtractor,
    doubts,
    followed,
    outline_rows,
    proposals,
)
from setup_helpers import recording_of, video_file

FPS = 10.0
SIZE = (60, 80)


def square(x0, y0, x1, y1, shape=SIZE):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class TestOutline:
    def test_a_mask_becomes_its_edge_and_specks_are_left_out(self):
        mask = square(10, 10, 30, 40)
        mask[55, 75] = True
        outline = mask_outline(mask)
        assert len(outline) == 1
        corners = polygons(outline)[0]
        assert corners[:, 0].min() == 10 and corners[:, 0].max() == 29 and corners[:, 1].min() == 10 and corners[:, 1].max() == 39

    def test_inside_and_distance_follow_the_outline_not_its_box(self):
        triangle = [[0, 0, 100, 0, 0, 100]]
        assert inside(triangle, 10, 10) and not inside(triangle, 90, 90)
        assert distance(triangle, 10, 10) == 0.0
        assert distance(triangle, 100, 100) == pytest.approx(50 * np.sqrt(2))
        assert distance(triangle, -3, 50) == pytest.approx(3.0)
        assert np.isnan(distance([], 1, 1)) and np.isnan(distance(triangle, np.nan, 1))

    def test_two_parts_take_the_nearer(self):
        parts = [[0, 0, 10, 0, 10, 10, 0, 10], [50, 0, 60, 0, 60, 10, 50, 10]]
        assert distance(parts, 45, 5) == pytest.approx(5.0) and inside(parts, 55, 5)


class TestStarts:
    def test_the_best_box_of_a_name_starts_it_and_others_in_its_frame_are_counted(self):
        found = pd.DataFrame([
            {"frame": 4, "label": "ball", "x0": 1, "y0": 2, "x1": 3, "y1": 4, "score": 0.5},
            {"frame": 6, "label": "ball", "x0": 5, "y0": 6, "x1": 7, "y1": 8, "score": 0.8},
            {"frame": 6, "label": "ball", "x0": 9, "y0": 9, "x1": 12, "y1": 12, "score": 0.4},
            {"frame": 6, "label": "cup", "x0": 0, "y0": 0, "x1": 2, "y1": 2, "score": 0.9},
        ])
        chosen = proposals(found, ["ball", "book"])
        assert chosen == {"ball": {"frame": 6, "box": [5.0, 6.0, 7.0, 8.0], "score": 0.8, "candidates": 2}}

    def test_time_in_view_and_the_longest_gap_count_from_the_start(self):
        assert followed(np.array([10, 11, 12, 16, 17]), 10, 20, FPS) == {"frames": 10, "found": 5, "longest_gap_s": 0.3}
        assert followed(np.array([], dtype=int), 10, 20, FPS) == {"frames": 10, "found": 0, "longest_gap_s": 1.0}

    def test_doubts_name_several_candidates_little_time_in_view_and_no_start(self):
        assert doubts({"from": NOT_FOUND}) == ["not found by name on this camera"]
        sure = {"from": FINDER, "candidates": 1, "frames": 10, "found": 9}
        assert doubts(sure) == []
        assert doubts({**sure, "candidates": 3}) == ["3 things in the start frame could be it"]
        assert doubts({**sure, "found": 2}) == ["in view in 20% of the frames after its start"]
        assert doubts({"from": PERSON, "candidates": 3, "frames": 10, "found": 9}) == []


class TestDecisions:
    def test_a_start_replaces_a_not_here_and_a_keep_and_only_starts_and_not_here_change_the_tracking(self, tmp_path):
        path = tmp_path / "objects.json"
        said = read_decisions(path)
        set_absent(said, "front", "cup", "ann")
        keep(said, "front", "ball", {"frame": 3, "box": [1, 2, 3, 4]}, "ann")
        before = tracking_key(said)
        keep(said, "front", "book", {"frame": 1, "box": [1, 1, 2, 2]}, "ann")
        assert tracking_key(said) == before
        add_start(said, "front", "cup", 2.0, 10, 20, "bob")
        add_start(said, "front", "cup", 2.0, 12, 22, "bob")
        add_start(said, "front", "cup", 1.0, 5, 5, "bob")
        assert tracking_key(said) != before
        assert "cup" not in said[ABSENT].get("front", {})
        assert [(point["t"], point["x"]) for point in said[STARTS]["front"]["cup"]] == [(1.0, 5.0), (2.0, 12.0)]
        add_start(said, "front", "ball", 4.0, 1, 1, "bob")
        assert "ball" not in said[KEPT]["front"]
        clear(said, "front", "cup")
        write_decisions(path, said)
        again = read_decisions(path)
        assert again[STARTS] == {"front": {"ball": said[STARTS]["front"]["ball"]}} and set(again[KEPT]["front"]) == {"book"}
        assert again[ABSENT] == {}


class FakeFinder:
    name, version, default_model = "fake_finder", "1", None

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def infer(self, video_path, *, frames, step, names, **kwargs):
        self.calls.append((frames, tuple(names)))
        rows = [{"frame": frame, "label": name, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "score": score}
                for frame, name, (x0, y0, x1, y1), score in self.boxes if frames[0] <= frame < frames[1] and name in names and (frame - frames[0]) % step == 0]
        return pd.DataFrame(rows, columns=["frame", "label", "x0", "y0", "x1", "y1", "score"])


class FakeTracker:
    """Each thing stays where it started; the ball is out of view in frames 25 to 29."""

    name, version, default_model = "fake_tracker", "1", None

    def __init__(self):
        self.starts = None

    def track(self, video_path, *, frames, starts, device, model):
        self.starts = starts
        if not starts:
            return
        where = {}
        for frame in range(min(start.frame for start in starts), frames[1]):
            for start in starts:
                if start.frame == frame:
                    x, y = (start.box[0], start.box[1]) if start.box is not None else start.point
                    where[start.name] = square(int(x), int(y), int(x) + 10, int(y) + 10)
            yield frame, {name: (mask, 0.1 if name == "ball" and 25 <= frame < 30 else 0.9) for name, mask in where.items()}


@pytest.fixture
def run(tmp_path, monkeypatch):
    def build(names, boxes, said=None, cameras=("front.mp4",)):
        finder, tracker = FakeFinder(boxes), FakeTracker()
        decisions = tmp_path / "objects.json"
        if said is not None:
            write_decisions(decisions, said)
        monkeypatch.setattr(track_module, "resolve_finder", lambda name: finder)
        monkeypatch.setattr(track_module, "resolve_tracker", lambda name: tracker)
        monkeypatch.setattr(track_module, "apply_prep_timing", lambda config, rec: SimpleNamespace(t_origin_s=0.0, duration_s=4.0))
        monkeypatch.setattr(track_module, "layout_for", lambda config, create=False: SimpleNamespace(objects_path=lambda rid: decisions))
        files = [video_file(name, fps=FPS, offset_s=0.0) for name in cameras]
        recording = recording_of(files, recording_id="r1")
        step = ObjectTrackExtractor(SimpleNamespace(objects=ObjectsSection(names=list(names))))
        return finder, tracker, step.run(recording, tmp_path)

    return build


class TestExtractor:
    def test_a_thing_found_by_name_is_followed_from_its_best_box_and_one_never_found_is_reported(self, run, tmp_path):
        boxes = [(4, "ball", (20, 20, 30, 30), 0.6), (6, "ball", (30, 30, 40, 40), 0.9), (6, "ball", (0, 0, 5, 5), 0.5)]
        finder, tracker, result = run(["ball", "cup"], boxes)
        rows = pd.read_parquet(result.artifacts["objects.parquet"])
        assert set(rows["label"]) == {"ball"} and rows["frame"].min() == 6 and rows["frame"].max() == 40
        assert 27 not in set(rows["frame"]) and len(rows) == 35 - 5
        first = rows.iloc[0]
        assert (first["x0"], first["y0"], first["x1"], first["y1"]) == (30.0, 30.0, 40.0, 40.0) and first["area"] == 100.0
        assert inside(first["outline"], 35, 35)
        starts = json.loads(result.artifacts["starts.json"].read_text(encoding="utf-8"))["front.mp4"]
        assert starts["ball"]["from"] == FINDER and starts["ball"]["candidates"] == 2 and starts["ball"]["t"] == 0.6
        assert starts["ball"]["found"] == 30 and starts["ball"]["longest_gap_s"] == 0.5
        assert starts["ball"]["doubts"] == ["2 things in the start frame could be it"]
        assert starts["cup"] == {"from": NOT_FOUND, "doubts": ["not found by name on this camera"]}
        assert any("cup was not found by name on front.mp4" in warning.message for warning in result.warnings)
        # looked for a stretch at a time until the end, as the cup never turned up
        assert [frames for frames, _ in finder.calls] == [(0, 41)]

    def test_a_person_start_wins_over_the_name_and_a_thing_said_not_on_a_camera_is_not_looked_for(self, run):
        said = {STARTS: {}, ABSENT: {}, KEPT: {}}
        add_start(said, "front.mp4", "ball", 1.0, 5, 5, "ann")
        add_start(said, "front.mp4", "ball", 3.0, 40, 40, "ann")
        set_absent(said, "front.mp4", "cup", "ann")
        finder, tracker, result = run(["ball", "cup"], [(0, "ball", (30, 30, 40, 40), 0.9)], said=said)
        assert finder.calls == []
        assert [(start.name, start.frame, start.point) for start in tracker.starts] == [("ball", 10, (5.0, 5.0)), ("ball", 30, (40.0, 40.0))]
        starts = json.loads(result.artifacts["starts.json"].read_text(encoding="utf-8"))["front.mp4"]
        assert starts["ball"]["from"] == PERSON and starts["ball"]["t"] == 1.0 and starts["ball"]["doubts"] == []
        assert starts["cup"] == {"from": NOT_HERE, "doubts": []}
        rows = pd.read_parquet(result.artifacts["objects.parquet"])
        assert rows.loc[rows["frame"] == 31, "x0"].tolist() == [40.0]

    def test_a_thing_out_of_view_leaves_no_row(self):
        seen = {"ball": (square(1, 1, 5, 5), 0.3), "cup": (np.zeros(SIZE, dtype=bool), 0.9), "book": (square(1, 1, 5, 5), 0.8)}
        assert [row["label"] for row in outline_rows("front.mp4", 3, seen)] == ["book"]


class TestSettings:
    def test_names_are_checked(self):
        with pytest.raises(ValueError, match="names a thing twice"):
            ObjectsSection(names=["ball", "ball"])
        with pytest.raises(ValueError, match="elsewhere"):
            ObjectsSection(names=["elsewhere"])

    def test_a_name_may_not_be_a_role(self, tmp_path):
        from macaw.config.loader import scaffold_project_config, resolve_config

        project = tmp_path / "project.yaml"
        scaffold_project_config(project, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
        roles = resolve_config(project).project.roles
        project.write_text(project.read_text(encoding="utf-8") + f"objects:\n  names: [{roles[0]}]\n", encoding="utf-8")
        with pytest.raises(Exception, match="which are roles"):
            resolve_config(project)

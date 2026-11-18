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


@pytest.fixture
def project(tmp_path):
    """One recording with a room camera read for everyone, following a ball and a cup."""
    import yaml

    from macaw.config.loader import resolve_config, scaffold_project_config
    from setup_helpers import camera, mics, put, write_setup

    raw = tmp_path / "raw"
    for device, name in (("parent_mic", "a.wav"), ("child_mic", "a.wav"), ("room", "a.mp4")):
        put(raw, "003", device, name)
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=raw, output_dir=tmp_path / "out")
    write_setup(config, cameras={"room": camera()}, microphones=mics())

    def configure(**values):
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
        data.setdefault("objects", {}).update(values)
        config.write_text(yaml.safe_dump(data), encoding="utf-8")

    def starts(entries):
        from macaw.project.layout import layout_for

        folder = layout_for(resolve_config(config)).artifacts_dir("003", "objects.track")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "starts.json").write_text(json.dumps({"room": {name: {**entry, "doubts": doubts(entry)} for name, entry in entries.items()}}), encoding="utf-8")

    configure(names=["ball", "cup"])
    return SimpleNamespace(config=config, configure=configure, starts=starts, resolved=lambda: resolve_config(config))


BALL = {"from": FINDER, "frame": 6, "t": 0.6, "box": [30.0, 30.0, 40.0, 40.0], "score": 0.9, "candidates": 1, "frames": 35, "found": 30, "longest_gap_s": 0.5}


def macaw(project, *args):
    from typer.testing import CliRunner

    from macaw.cli.main import app

    return CliRunner().invoke(app, ["objects", "-c", str(project.config), "-r", "003", *args])


class TestReview:
    def gate(self, project):
        from macaw.extractors.video.objects.review import ObjectReviewGate
        from macaw.project.recording import build_recording

        config = project.resolved()
        return ObjectReviewGate(config), build_recording("003", config)

    def test_every_start_found_by_name_waits_until_kept_and_a_thing_not_found_until_a_person_says_where(self, project):
        from macaw.extractors.base import PERSON as BY_PERSON
        from macaw.extractors.video.objects.review import waiting

        project.starts({"ball": BALL, "cup": {"from": NOT_FOUND}})
        gate, recording = self.gate(project)
        assert [(item.name, item.why) for item in waiting(project.resolved(), "003")] == [
            ("ball", "found by name, not looked at yet"), ("cup", "not found by name: click it on a frame, or say it is not on this camera"),
        ]
        assert "macaw objects -r 003" in gate.pending(recording)
        listed = macaw(project)
        assert listed.exit_code == 0 and "room ball: starts at 0.6 s, found by name (score 0.90); in view 86%" in listed.output
        assert "[waits: found by name, not looked at yet]" in listed.output
        kept = macaw(project, "--keep", "--by", "ann")
        assert kept.exit_code == 0 and "kept the starts of ball on room" in kept.output
        said = macaw(project, "--not-here", "cup", "--camera", "room")
        assert said.exit_code == 0 and "stops looking for it there" in said.output
        assert [item.why for item in waiting(project.resolved(), "003")] == ["followed again from what a person said on the next run"]
        project.starts({"ball": BALL, "cup": {"from": NOT_HERE}})
        gate, recording = self.gate(project)
        assert gate.pending(recording) is None and gate.decided_by(recording) == BY_PERSON
        assert "kept by ann" in macaw(project).output

    def test_a_kept_start_no_longer_counts_once_the_name_finds_another(self, project):
        from macaw.extractors.video.objects.review import waiting

        project.configure(names=["ball"])
        project.starts({"ball": BALL})
        macaw(project, "--keep")
        assert waiting(project.resolved(), "003") == []
        project.starts({"ball": {**BALL, "frame": 9, "box": [1.0, 1.0, 5.0, 5.0]}})
        assert [item.name for item in waiting(project.resolved(), "003")] == ["ball"]

    def test_keeping_leaves_a_start_a_person_changed_before_the_next_run(self, project):
        from macaw.extractors.video.objects.decisions import KEPT
        from macaw.project.layout import layout_for

        project.starts({"ball": BALL, "cup": {**BALL, "box": [50.0, 5.0, 60.0, 15.0]}})
        macaw(project, "--start", "cup", "--camera", "room", "--at", "1", "--point", "5,5")
        kept = macaw(project, "--keep")
        assert "kept the starts of ball on room" in kept.output and "cup" not in kept.output
        assert set(read_decisions(layout_for(project.resolved()).objects_path("003"))[KEPT]["room"]) == {"ball"}

    def test_flagged_waits_only_for_doubts_and_skip_leaves_a_note(self, project):
        from macaw.extractors.base import SOFTWARE
        from macaw.extractors.video.objects.review import waiting

        project.starts({"ball": BALL, "cup": {**BALL, "candidates": 2}})
        project.configure(review="flagged")
        assert [(item.name, item.why) for item in waiting(project.resolved(), "003")] == [("cup", "2 things in the start frame could be it")]
        project.configure(review="skip")
        project.starts({"ball": BALL, "cup": {"from": NOT_FOUND}})
        gate, recording = self.gate(project)
        assert gate.pending(recording) is None and gate.decided_by(recording) == SOFTWARE
        notes = gate.notes(recording)
        assert len(notes) == 1 and "not found by name and so not followed: cup on room" in notes[0].message

    def test_a_start_needs_a_known_name_a_camera_a_time_and_a_point(self, project):
        from macaw.extractors.video.objects.decisions import STARTS
        from macaw.project.layout import layout_for

        assert "not in objects.names" in str(macaw(project, "--start", "book", "--camera", "room", "--at", "1", "--point", "1,2").exception)
        assert "--camera takes one of room" in str(macaw(project, "--start", "ball", "--camera", "front", "--at", "1", "--point", "1,2").exception)
        assert "needs --at and --point" in str(macaw(project, "--start", "ball", "--camera", "room").exception)
        given = macaw(project, "--start", "ball", "--camera", "room", "--at", "2.5", "--point", "100,50", "--by", "ann")
        assert given.exit_code == 0 and "follows it from there" in given.output
        said = read_decisions(layout_for(project.resolved()).objects_path("003"))
        assert said[STARTS]["room"]["ball"][0]["x"] == 100.0
        cleared = macaw(project, "--clear", "ball", "--camera", "room")
        assert cleared.exit_code == 0 and "looks for it by name" in cleared.output

    def test_the_quality_page_counts_the_starts_and_who_looked(self, project):
        from macaw.project.layout import layout_for
        from macaw.qc.report import _object_starts

        project.starts({"ball": BALL, "cup": {"from": NOT_FOUND}})
        config = project.resolved()
        row = _object_starts(config, layout_for(config), "003")
        assert row["by"] == "waiting" and "1 found by name (0 kept by a person), 1 not found and not followed" in row["decided"]
        macaw(project, "--keep")
        macaw(project, "--not-here", "cup", "--camera", "room")
        project.starts({"ball": BALL, "cup": {"from": NOT_HERE}})
        row = _object_starts(config, layout_for(config), "003")
        assert row["by"] == "person" and "1 found by name (1 kept by a person), 1 said not to be on a camera" in row["decided"]


class TestReviewPage:
    def test_the_card_lists_each_start_with_its_state_and_the_buttons_stand_for_macaw_objects(self, project):
        from macaw.interface.pages import review

        project.starts({"ball": BALL, "cup": {"from": NOT_FOUND}})
        shown = json.dumps(review.body(project.config, "003")[-1].to_plotly_json(), default=lambda item: item.to_plotly_json())
        assert "Things followed" in shown and "found by name, not looked at yet" in shown and "Keep the starts found by name" in shown
        assert '"t": "0.60"' in shown and "86%" in shown
        frame = {"camera": "room", "t": 1.5}
        start = review.object_command(project.config, "003", "object-start", "", "ann", "ball", [10.0, 20.5], frame)
        assert start[1:2] == ["objects"] and start[-8:] == ["--camera", "room", "--at", "1.5", "--point", "10,20.5", "--by", "ann"]
        assert review.object_command(project.config, "003", "object-start", "", "ann", "ball", None, frame) == "Click on the thing in the frame first."
        assert review.object_command(project.config, "003", "object-not-here", "", "ann", None, None, frame) == "Pick the thing first."
        assert review.object_command(project.config, "003", "object-clear", "room|cup", "", None, None, None)[-4:] == ["--clear", "cup", "--camera", "room"]
        macaw(project, "--keep")
        macaw(project, "--not-here", "cup", "--camera", "room")
        project.starts({"ball": BALL, "cup": {"from": NOT_HERE}})
        shown = json.dumps(review.body(project.config, "003")[-1].to_plotly_json(), default=lambda item: item.to_plotly_json())
        assert "kept by" in shown and "Forget" in shown and "Keep the starts found by name" not in shown

    def test_a_click_puts_one_point_on_the_frame(self):
        from macaw.interface.pages.review import OBJECT_POINT, with_point

        figure = {"data": [{"type": "image"}], "layout": {}}
        once = with_point(figure, [1.0, 2.0])
        twice = with_point(once, [3.0, 4.0])
        assert len(twice["data"]) == 2 and twice["data"][-1]["name"] == OBJECT_POINT and twice["data"][-1]["x"] == [3.0]

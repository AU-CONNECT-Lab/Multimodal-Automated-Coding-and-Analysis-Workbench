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
        sure = {"from": FINDER, "candidates": 1, "frames": 10, "found": 9, "share_after_start": 0.9}
        assert doubts(sure) == []
        assert doubts({**sure, "candidates": 3}) == ["3 things in the start frame could be it"]
        # in view before its start says nothing about how well it was followed after it
        assert doubts({**sure, "found": 9, "share_after_start": 0.2}) == ["in view in 20% of the frames after its start"]
        assert doubts({**sure, "found": 2}) == []
        assert doubts({"from": PERSON, "candidates": 3, "frames": 10, "found": 9, "share_after_start": 0.9}) == []


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
    """Each thing stays where it started, and where it first started before that; the ball is out of view in frames 25 to 29."""

    name, version, default_model = "fake_tracker", "1", None

    def __init__(self):
        self.starts = None

    def track(self, video_path, *, frames, starts, device, model):
        self.starts = starts
        if not starts:
            return
        def at(start):
            x, y = (start.box[0], start.box[1]) if start.box is not None else start.point
            return square(int(x), int(y), int(x) + 10, int(y) + 10)

        where = {start.name: at(start) for start in sorted(starts, key=lambda start: -start.frame)}
        for frame in range(frames[0], frames[1]):
            for start in starts:
                if start.frame == frame:
                    where[start.name] = at(start)
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
        # followed back from its start in frame 6 to the beginning, and on to the end
        assert set(rows["label"]) == {"ball"} and rows["frame"].min() == 0 and rows["frame"].max() == 40
        shown = rows[rows["in_view"]]
        assert 27 not in set(shown["frame"]) and len(shown) == 41 - 5 and rows["frame"].is_monotonic_increasing
        # a frame scored below the cut keeps its row and its score
        assert len(rows) == 41 and rows.loc[rows["frame"] == 27, "score"].notna().all()
        first = rows.iloc[0]
        assert (first["x0"], first["y0"], first["x1"], first["y1"]) == (30.0, 30.0, 40.0, 40.0) and first["area"] == 100.0
        assert inside(first["outline"], 35, 35)
        starts = json.loads(result.artifacts["starts.json"].read_text(encoding="utf-8"))["front.mp4"]
        assert starts["ball"]["from"] == FINDER and starts["ball"]["candidates"] == 2 and starts["ball"]["t"] == 0.6
        assert starts["ball"]["frames"] == 41 and starts["ball"]["found"] == 36 and starts["ball"]["longest_gap_s"] == 0.5
        assert starts["ball"]["share_after_start"] == round(30 / 35, 3)
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

    def test_a_thing_out_of_view_keeps_its_score_and_any_outline(self):
        seen = {"ball": (square(1, 1, 5, 5), 0.3), "cup": (np.zeros(SIZE, dtype=bool), 0.9), "book": (square(1, 1, 5, 5), 0.8)}
        rows = {row["label"]: row for row in outline_rows("front.mp4", 3, seen)}
        assert [label for label, row in rows.items() if row["in_view"]] == ["book"]
        assert rows["ball"]["score"] == 0.3 and rows["ball"]["outline"] and rows["cup"]["outline"] == [] and "area" not in rows["cup"]


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


BALL = {"from": FINDER, "frame": 6, "t": 0.6, "box": [30.0, 30.0, 40.0, 40.0], "score": 0.9, "candidates": 1, "frames": 35, "found": 30, "longest_gap_s": 0.5, "share_after_start": 0.857}


def macaw(project, *args):
    from typer.testing import CliRunner

    from macaw.cli.main import app

    return CliRunner().invoke(app, ["objects", "-c", str(project.config), "-r", "003", *args])


class TestFinders:
    def test_every_step_th_frame_is_looked_at_and_handed_out_in_batches(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        pytest.importorskip("PIL")
        path = tmp_path / "count.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (32, 32))
        for index in range(30):
            writer.write(np.full((32, 32, 3), index * 8, dtype=np.uint8))
        writer.release()
        batches = list(track_module._batches(track_module._every_step(path, (4, 21), 5), 2))
        assert [[index for index, _ in batch] for batch in batches] == [[4, 9], [14, 19]]
        assert all(abs(int(np.asarray(image).mean()) - index * 8) <= 2 for batch in batches for index, image in batch)

    def test_both_finders_are_there_by_name(self):
        from macaw.extractors import get_backend

        assert get_backend(track_module.FINDERS, "grounding_dino") is track_module.GroundingDinoFinder
        assert get_backend(track_module.FINDERS, "sam3") is track_module.Sam3Finder


class TestFollowingBack:
    def test_frames_are_handed_out_backward_a_stretch_at_a_time(self, tmp_path, monkeypatch):
        cv2 = pytest.importorskip("cv2")
        path = tmp_path / "count.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (32, 32))
        for index in range(30):
            writer.write(np.full((32, 32, 3), index * 8, dtype=np.uint8))
        writer.release()
        # eight frames a stretch, so 4 to 21 is read in three stretches
        monkeypatch.setattr(track_module, "_BACK_BYTES", 8 * 32 * 32 * 3)
        handed = list(track_module._pictures_back(path, 21, 4))
        assert [index for index, _ in handed] == list(range(21, 3, -1))
        assert all(abs(int(image.mean()) - index * 8) <= 2 for index, image in handed)

    def test_each_thing_is_followed_back_only_before_its_own_first_start(self, monkeypatch):
        from macaw.extractors.video.objects.track import Sam2Tracker, TrackStart

        calls = []

        def follow(self, pictures, starts, device, model):
            pictures = list(pictures)
            calls.append(([index for index, _ in pictures], sorted((start.name, start.frame) for start in starts)))
            for index, _ in pictures:
                yield index, {start.name: index for start in starts}

        monkeypatch.setattr(Sam2Tracker, "_follow", follow)
        monkeypatch.setattr(track_module, "_pictures_back", lambda path, last, first: ((index, None) for index in range(last, first - 1, -1)))
        monkeypatch.setattr(track_module, "_pictures_on", lambda path, first, stop: ((index, None) for index in range(first, stop)))
        monkeypatch.setattr(track_module, "_device", lambda device: "cpu")
        starts = [TrackStart("ball", 5, point=(1, 1)), TrackStart("cup", 8, point=(2, 2)), TrackStart("ball", 9, point=(3, 3))]
        out = list(Sam2Tracker().track("v.mp4", frames=(2, 12), starts=starts, device="auto", model=None))
        back, on = calls
        assert back == (list(range(8, 1, -1)), [("ball", 5), ("cup", 8)])
        assert on == (list(range(5, 12)), [("ball", 5), ("ball", 9), ("cup", 8)])
        followed_back = {index: sorted(seen) for index, seen in out[:len(back[0]) - 1]}
        assert followed_back == {7: ["cup"], 6: ["cup"], 5: ["cup"], 4: ["ball", "cup"], 3: ["ball", "cup"], 2: ["ball", "cup"]}
        assert [index for index, _ in out[len(followed_back):]] == list(range(5, 12))
        # a thing starting on the first frame is not followed back
        calls.clear()
        list(Sam2Tracker().track("v.mp4", frames=(2, 12), starts=[TrackStart("ball", 2, point=(1, 1))], device="auto", model=None))
        assert [call[0][0] for call in calls] == [2]


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


class TestTouch:
    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        from macaw.extractors.video.objects import touch as touch_module
        from macaw.extractors.video.objects.touch import ObjectTouchExtractor
        from macaw.extractors.video.pose import kinematics as kinematics_module
        from macaw.store.keypoints import write_keypoints
        from test_kinematics import person

        def build(touch_px=10.0):
            stored = tmp_path / "keypoints"
            path = np.column_stack([np.linspace(100, 390, 30), np.full(30, 200.0)])
            write_keypoints(person("parent", n_frames=30, moves={"left_wrist": path}), stored)
            monkeypatch.setattr(kinematics_module, "_keypoints_file", lambda config, rid: stored / "pose.keypoints.parquet")
            square = [[200, 180, 240, 180, 240, 220, 200, 220]]
            things = pd.DataFrame(
                [{"video": "parent.mp4", "frame": f, "label": "ball", "outline": square} for f in range(30)]
                + [{"video": "parent.mp4", "frame": f, "label": "toy car", "outline": [[0, 0, 5, 0, 5, 5, 0, 5]]} for f in range(10)]
                + [{"video": "other.mp4", "frame": f, "label": "book", "outline": square} for f in range(30)]
            )
            monkeypatch.setattr(touch_module, "load_objects", lambda config, rid: things)
            config = SimpleNamespace(objects=ObjectsSection(names=["ball", "toy car", "book"], touch_px=touch_px))
            return ObjectTouchExtractor(config).run(recording_of([], ["parent"]), tmp_path)

        return build

    def test_a_wrist_near_the_outline_puts_the_hand_on_the_thing_frame_by_frame_and_on_a_tier(self, run):
        from macaw.store.schema import ANNOTATION_SCHEMA, LONG_TABLE_SCHEMA, validate_frame

        result = run()
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        validate_frame(result.annotations, ANNOTATION_SCHEMA, "annotations")
        assert {spec.name for spec in result.specs} == set(result.rows.feature) == {
            f"{side}_{kind}_{thing}" for side in ("left", "right") for kind in ("wrist_to", "hand_on") for thing in ("ball", "toy_car")
        }
        far = result.rows[result.rows.feature == "left_wrist_to_ball"].sort_values("t_start")["value"].to_numpy()
        on = result.rows[result.rows.feature == "left_hand_on_ball"].sort_values("t_start")["value"].to_numpy()
        x = np.linspace(100, 390, 30)
        assert far == pytest.approx(np.where(x < 200, 200 - x, np.where(x > 240, x - 240, 0.0)))
        assert on.tolist() == [float(abs(value) <= 10) for value in far]
        car = result.rows[result.rows.feature == "left_hand_on_toy_car"].sort_values("t_start")["value"]
        assert car.iloc[:10].tolist() == [0.0] * 10 and car.iloc[10:].isna().all()
        tier = result.annotations[result.annotations.tier == "on_ball"]
        assert tier.label.tolist() == ["left"] and tier.subject_id.tolist() == ["parent"]
        inside = np.flatnonzero(on == 1) / 30.0
        assert tier.t_start.iloc[0] == pytest.approx(inside[0]) and tier.t_end.iloc[0] == pytest.approx(inside[-1] + 1 / 30.0)
        assert "stretches with a hand on a thing: ball 1, toy car 0" in [w.message for w in result.warnings if w.severity == "info"][0]
        assert "not contact" in result.specs[0].description

    def test_the_step_runs_only_with_a_distance(self):
        from macaw.exceptions import ConfigError
        from macaw.extractors.video.objects.touch import ObjectTouchExtractor

        assert not ObjectTouchExtractor.wanted_by_default(SimpleNamespace(objects=ObjectsSection(names=["ball"])))
        assert ObjectTouchExtractor.wanted_by_default(SimpleNamespace(objects=ObjectsSection(names=["ball"], touch_px=5)))
        with pytest.raises(ConfigError, match="objects.touch_px"):
            ObjectTouchExtractor(SimpleNamespace(objects=ObjectsSection(names=["ball"]))).check_available()

    def test_stretches_break_at_a_jump_in_time(self):
        from macaw.extractors.video.objects.touch import on_spans, slug

        times = np.array([0.0, 0.1, 0.2, 0.3, 0.7, 0.8])
        found = on_spans(times, [1, 1, 0, 1, 1, np.nan])
        assert [(round(a, 2), round(b, 2)) for a, b in zip(found["t_start"], found["t_end"])] == [(0.0, 0.2), (0.3, 0.4), (0.7, 0.8)]
        assert slug(" Toy Car! ") == "toy_car"


class TestPosition:
    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        from macaw.extractors.video.objects import position as position_module
        from macaw.extractors.video.objects.position import ObjectPositionExtractor

        def build(starts):
            things = pd.DataFrame(
                [{"video": "front.mp4", "frame": f, "label": "ball", "cx": 10.0 + f, "cy": 20.0, "area": 100.0, "score": 0.9, "in_view": True}
                 for f in range(5, 30) if not 12 <= f < 15]
                + [{"video": "front.mp4", "frame": f, "label": "ball", "score": 0.2, "in_view": False} for f in range(12, 15)]
                + [{"video": "side.mp4", "frame": f, "label": "toy car", "cx": 1.0, "cy": 2.0, "area": 9.0, "score": 0.8, "in_view": True} for f in range(40)]
            )
            monkeypatch.setattr(position_module, "load_objects", lambda config, rid, everything=False: things)
            monkeypatch.setattr(position_module, "load_starts", lambda config, rid: starts)
            monkeypatch.setattr(position_module, "apply_prep_timing", lambda config, rec: SimpleNamespace(t_origin_s=0.0, duration_s=3.0))
            files = [video_file("front.mp4", ["parent"], fps=FPS, offset_s=0.5), video_file("side.mp4", ["child"], fps=FPS, offset_s=0.0)]
            config = SimpleNamespace(objects=ObjectsSection(names=["ball", "toy car"]))
            return ObjectPositionExtractor(config).run(recording_of(files, ["parent", "child"], recording_id="r1"), tmp_path)

        return build

    def test_each_thing_on_each_camera_has_its_place_frame_by_frame_and_belongs_to_the_recording(self, run):
        from macaw.store.schema import LONG_TABLE_SCHEMA, validate_frame

        followed = {"from": PERSON, "frame": 5}
        result = run({"front.mp4": {"ball": followed, "toy car": {"from": NOT_HERE}}, "side.mp4": {"toy car": followed, "ball": {"from": NOT_FOUND}}})
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        assert {spec.name for spec in result.specs} == set(result.rows.feature) == {
            f"{thing}_{camera}_{measure}" for thing, camera in (("ball", "front_mp4"), ("toy_car", "side_mp4"))
            for measure in ("in_view", "x", "y", "area", "score", "in_view_share", "longest_gap_s")
        }
        assert result.rows.subject_id.isna().all()
        ball = lambda measure: result.rows[result.rows.feature == f"ball_front_mp4_{measure}"].sort_values("t_start")
        seen = ball("in_view")
        # the front camera starts half a second into the 3 s recording, so its frames run from 0.5 s to 3.0 s
        assert len(seen) == 26 and seen.t_start.iloc[0] == pytest.approx(0.5) and seen.t_start.iloc[-1] == pytest.approx(3.0)
        assert seen.value.tolist() == [0.0] * 5 + [1.0] * 7 + [0.0] * 3 + [1.0] * 11
        x = ball("x").value.to_numpy()
        assert x[5] == 15.0 and np.isnan(x[:5]).all() and np.isnan(x[12:15]).all()
        specs = {spec.name: spec for spec in result.specs}
        assert specs["ball_front_mp4_area"].unit == "px^2" and "camera front.mp4" in specs["ball_front_mp4_x"].description
        score = ball("score").value.to_numpy()
        assert np.isnan(score[:5]).all() and score[12:15].tolist() == [0.2] * 3 and score[5] == 0.9
        assert ball("in_view_share").value.item() == pytest.approx(18 / 26)
        assert ball("longest_gap_s").value.item() == pytest.approx(5 / FPS)

    def test_a_segment_has_its_own_share_in_view(self, run, tmp_path, monkeypatch):
        from macaw.extractors.video.objects import position as position_module
        from macaw.extractors.video.objects.position import ObjectPositionExtractor
        from macaw.store.longtable import write_long_table

        followed = {"from": PERSON, "frame": 5}
        result = run({"front.mp4": {"ball": followed}, "side.mp4": {"toy car": followed}})
        write_long_table(result.rows, tmp_path / "features")
        monkeypatch.setattr(position_module, "layout_for", lambda config, create=False: SimpleNamespace(long_table_path=lambda rid: tmp_path / "features"))
        recording = SimpleNamespace(recording_id="r1", entities=lambda: {})
        part = ObjectPositionExtractor(SimpleNamespace(objects=ObjectsSection(names=["ball", "toy car"]))).summarise(recording, 0.9, 1.5)
        share = part[part.feature == "ball_front_mp4_in_view_share"]
        # the front camera's frames from 0.9 s to 1.5 s are 4 to 9, and the ball is in view from frame 5
        assert share.value.item() == pytest.approx(5 / 6) and share.t_start.item() == 0.9 and share.t_end.item() == 1.5

    def test_nothing_followed_gives_no_rows_and_says_so(self, run):
        result = run({"front.mp4": {"ball": {"from": NOT_FOUND}}})
        assert result.rows.empty and result.specs == [] and "no thing was followed" in result.warnings[0].message


def test_the_methods_text_tells_the_objects_once_and_says_the_touch_is_read_from_the_picture():
    from macaw.qc.boilerplate import _followed_things, _touch

    config = SimpleNamespace(objects=ObjectsSection(names=["ball"], touch_px=12))
    text = _touch(config, None, set())
    assert text.startswith(_followed_things(config)) and "within 12 px" in text and "not contact" in text and "[ravi2024]" in text


def test_the_methods_text_names_the_finder_used():
    from macaw.qc.boilerplate import _followed_things

    dino = _followed_things(SimpleNamespace(objects=ObjectsSection(names=["ball"])))
    sam3 = _followed_things(SimpleNamespace(objects=ObjectsSection(names=["ball"], finder="sam3")))
    assert "Grounding DINO [liu2024] (IDEA-Research/grounding-dino-tiny)" in dino and "SAM 3" not in dino
    assert "located by name with SAM 3 [carion2025] (facebook/sam3)" in sam3 and "Grounding DINO" not in sam3

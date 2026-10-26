from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.extractors.video.pose.keypoints import PoseExtractor, PoseReviewGate
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.extractors.video.pose.track_repair import (
    choose_camera_owner,
    owner_rows,
    several_people,
)
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording

FPS = 10.0


def person(track, frames, *, cx=200.0, cy=200.0, size=100.0):
    """A body whose standard joints spread over about size pixels around (cx, cy)."""
    rows = []
    for frame in frames:
        for index, joint in enumerate(STANDARD_JOINTS):
            rows.append(
                {
                    "frame": frame,
                    "t": frame / FPS,
                    "joint": joint,
                    "x": cx + size * ((index % 5) - 2) / 5,
                    "y": cy + size * ((index // 5) - 1.5) / 5,
                    "confidence": 0.9,
                    "track_id": track,
                }
            )
    return rows


def table(*people):
    frame = pd.DataFrame([row for rows in people for row in rows])
    frame["track_id"] = frame["track_id"].astype("Int64")
    return frame


class TestChoosingTheOwner:
    def test_tracker_fragments_are_joined_and_a_passer_by_is_not(self):
        keypoints = table(
            person(1, range(0, 52)),
            person(2, range(50, 100), cx=210),       # same person after a hand-over
            person(3, range(20, 30), cx=60, size=160),  # someone walking past, nearer the lens
        )
        choice = choose_camera_owner(keypoints)
        assert sorted(choice.tracks) == [1, 2]
        assert choice.doubt is None

    def test_one_track_per_frame_where_fragments_overlap(self):
        keypoints = table(person(1, range(0, 52)), person(2, range(50, 100), cx=210))
        owned = keypoints[owner_rows(keypoints, [1, 2])]
        assert not owned.duplicated(["frame", "joint"]).any()
        assert owned["frame"].nunique() == 100
        assert set(owned.loc[owned["frame"] == 51, "track_id"]) == {1}

    def test_someone_else_in_view_as_much_is_a_doubt(self):
        keypoints = table(person(1, range(0, 100)), person(2, range(0, 90), cx=60))
        choice = choose_camera_owner(keypoints)
        assert choice.tracks == [1]
        assert "track 2" in choice.doubt
        assert choice.spans == [(0.0, 8.9)]

    def test_an_owner_rarely_in_view_is_a_doubt(self):
        keypoints = table(person(1, range(0, 30)), person(2, range(30, 100), cx=60, size=20))
        choice = choose_camera_owner(keypoints)
        assert choice.tracks == [1]
        assert "30%" in choice.doubt

    def test_one_person_in_view_is_not_several(self):
        assert not several_people(table(person(1, range(10)), person(2, range(10, 20))))
        assert several_people(table(person(1, range(10)), person(2, range(5, 15))))


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    (raw / "003").mkdir(parents=True)
    for file in ("003_parent.mp4", "003_child.mp4"):
        (raw / "003" / file).write_bytes(b"")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=raw, output_dir=tmp_path / "out")

    def owners(entries):
        resolved = resolve_config(config)
        path = layout_for(resolved).artifacts_dir("003", PoseExtractor.name) / "owners"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries), encoding="utf-8")

    def pending():
        resolved = resolve_config(config)
        return PoseReviewGate(resolved).pending(build_recording("003", resolved))

    return SimpleNamespace(config=config, owners=owners, pending=pending, resolved=lambda: resolve_config(config))


def owner(tracks, doubt=None):
    return {"subject_id": "parent", "tracks": tracks, "chosen_by": "automatic", "doubt": doubt, "spans": []}


def invoke(*args):
    return CliRunner().invoke(app, list(args))


class TestReviewGate:
    def test_clear_without_any_choice_recorded(self, project):
        assert project.pending() is None

    def test_clear_when_nothing_is_in_doubt(self, project):
        project.owners({"003_parent.mp4": owner([1])})
        assert project.pending() is None

    def test_a_doubt_waits_and_says_what_to_run(self, project):
        project.owners({"003_parent.mp4": owner([1], "track 2 is in view nearly as much")})
        reason = project.pending()
        assert "003_parent.mp4" in reason and "macaw accept-owner -r 003" in reason

    def test_skip_lets_it_through_with_a_note(self, project):
        project.owners({"003_parent.mp4": owner([1], "doubt")})
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data.setdefault("pose", {})["review"] = "skip"
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert project.pending() is None
        resolved = project.resolved()
        notes = PoseReviewGate(resolved).notes(build_recording("003", resolved))
        assert [note.check for note in notes] == ["camera_owner_unreviewed"]


class TestAcceptOwner:
    def test_keeping_the_choice_clears_the_gate(self, project):
        project.owners({"003_parent.mp4": owner([1, 4], "doubt")})
        outcome = invoke("accept-owner", "-c", str(project.config), "-r", "003", "--by", "ann")
        assert outcome.exit_code == 0, outcome.output
        assert "kept 003_parent.mp4 as tracks [1, 4]" in outcome.output
        assert project.pending() is None

    def test_an_acceptance_lapses_when_the_choice_changes(self, project):
        project.owners({"003_parent.mp4": owner([1], "doubt")})
        invoke("accept-owner", "-c", str(project.config), "-r", "003")
        project.owners({"003_parent.mp4": owner([2], "doubt")})
        assert project.pending() is not None

    def test_naming_tracks_records_them_for_pose_keypoints(self, project):
        outcome = invoke(
            "accept-owner", "-c", str(project.config), "-r", "003",
            "--media", "003_parent.mp4", "--tracks", "2,7", "--by", "ann",
        )
        assert outcome.exit_code == 0, outcome.output
        assert "--no-resume" in outcome.output
        path = layout_for(project.resolved()).camera_owners_path("003")
        entry = json.loads(path.read_text(encoding="utf-8"))["003_parent.mp4"]
        assert entry["tracks"] == [2, 7] and entry["by"].startswith("ann ")

    def test_tracks_need_exactly_one_camera(self, project):
        outcome = invoke("accept-owner", "-c", str(project.config), "-r", "003", "--tracks", "2")
        assert outcome.exit_code != 0

    def test_a_file_without_a_choice_is_an_error(self, project):
        project.owners({"003_parent.mp4": owner([1], "doubt")})
        outcome = invoke("accept-owner", "-c", str(project.config), "-r", "003", "--media", "nope.mp4")
        assert outcome.exit_code != 0

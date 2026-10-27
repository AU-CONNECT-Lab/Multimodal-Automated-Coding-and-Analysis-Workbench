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
    choose_people,
    join_fragments,
    owner_rows,
    several_people,
    subject_column,
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
        assert "tracks [2]" in choice.doubt
        assert choice.spans == [(0.0, 8.9)]

    def test_an_owner_rarely_in_view_is_a_doubt(self):
        keypoints = table(person(1, range(0, 30)), person(2, range(30, 100), cx=60, size=20))
        choice = choose_camera_owner(keypoints)
        assert choice.tracks == [1]
        assert "30%" in choice.doubt

    def test_one_person_in_view_is_not_several(self):
        assert not several_people(table(person(1, range(10)), person(2, range(10, 20))))
        assert several_people(table(person(1, range(10)), person(2, range(5, 15))))


class TestJoiningFragments:
    def test_a_person_found_again_after_a_break_is_one_chain(self):
        chains = join_fragments(table(person(1, range(0, 40)), person(2, range(60, 100), cx=215)))
        assert [chain.tracks for chain in chains] == [[1, 2]]

    def test_two_people_side_by_side_stay_apart(self):
        chains = join_fragments(table(person(1, range(0, 100)), person(2, range(0, 100), cx=500)))
        assert sorted(chain.tracks for chain in chains) == [[1], [2]]

    def test_a_second_box_on_the_same_person_joins_them(self):
        chains = join_fragments(table(person(1, range(0, 100)), person(2, range(20, 40), cx=210, size=120)))
        assert [chain.tracks for chain in chains] == [[1, 2]]

    def test_someone_appearing_elsewhere_is_not_joined(self):
        chains = join_fragments(table(person(1, range(0, 40)), person(2, range(60, 100), cx=600)))
        assert sorted(chain.tracks for chain in chains) == [[1], [2]]


def room(*extra):
    """Two seated people, the left one lost by the tracker halfway and found as track 3."""
    return table(
        person(1, range(0, 50), cx=100),
        person(3, range(55, 100), cx=105),
        person(2, range(0, 100), cx=500),
        *extra,
    )


class TestChoosingPeople:
    def test_anchors_place_the_people_and_leave_no_doubt(self):
        choice = choose_people(room(), ["parent", "child"], identity_anchors={"child": 0.1, "parent": 0.9})
        assert choice.people == {"parent": [2], "child": [1, 3]}
        assert choice.doubt is None

    def test_without_anchors_the_configured_order_is_a_guess_to_check(self):
        choice = choose_people(room(), ["parent", "child"])
        assert choice.people == {"parent": [1, 3], "child": [2]}
        assert "not yet checked" in choice.doubt and "parent at x 1" in choice.doubt

    def test_one_person_has_no_order_to_check(self):
        choice = choose_people(table(person(1, range(0, 50)), person(2, range(55, 100), cx=205)), ["parent"])
        assert choice.people == {"parent": [1, 2]} and choice.doubt is None

    def test_fewer_people_than_subjects_is_a_doubt(self):
        choice = choose_people(table(person(1, range(100))), ["parent", "child"], identity_anchors={"parent": 0.1, "child": 0.9})
        assert choice.people == {"parent": [1]}
        assert "found 1 of the 2" in choice.doubt

    def test_a_third_person_in_view_as_much_is_a_doubt_with_where_to_look(self):
        visitor = person(7, range(30, 80), cx=900)
        choice = choose_people(room(visitor), ["parent", "child"], identity_anchors={"parent": 0.1, "child": 0.9})
        assert choice.people == {"parent": [1, 3], "child": [2]}
        assert "tracks [7]" in choice.doubt and choice.spans == [(3.0, 7.9)]

    def test_tracks_named_by_a_person_are_kept_and_the_rest_is_placed_around_them(self):
        choice = choose_people(room(), ["parent", "child"], named={"parent": [2]})
        assert choice.people == {"parent": [2], "child": [1, 3]}
        assert choice.doubt is None

    def test_each_frame_and_joint_belongs_to_one_person(self):
        keypoints = room(person(4, range(0, 100), cx=110, size=110))
        choice = choose_people(keypoints, ["parent", "child"], identity_anchors={"parent": 0.1, "child": 0.9})
        keypoints["subject_id"] = subject_column(keypoints, choice.people)
        assigned = keypoints.dropna(subset=["subject_id"])
        assert not assigned.duplicated(["subject_id", "frame", "joint"]).any()
        assert assigned.loc[assigned.subject_id == "parent", "frame"].nunique() == 100


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    (raw / "003").mkdir(parents=True)
    for file in ("003_parent.mp4", "003_child.mp4", "003_room.mp4"):
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


def owner(tracks, doubt=None, subject="parent"):
    people = tracks if isinstance(tracks, dict) else {subject: tracks}
    return {"people": people, "chosen_by": {who: "automatic" for who in people}, "doubt": doubt, "spans": []}


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

    def test_always_holds_every_camera_until_a_person_keeps_it(self, project):
        project.owners({"003_parent.mp4": owner([1]), "003_child.mp4": owner([3], subject="child")})
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data.setdefault("pose", {})["review"] = "always"
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        reason = project.pending()
        assert "003_parent.mp4" in reason and "003_child.mp4" in reason and "always" in reason
        outcome = invoke("accept-owner", "-c", str(project.config), "-r", "003", "--by", "ann")
        assert outcome.exit_code == 0, outcome.output
        assert project.pending() is None


class TestAcceptOwner:
    def test_keeping_the_choice_clears_the_gate(self, project):
        project.owners({"003_parent.mp4": owner([1, 4], "doubt")})
        outcome = invoke("accept-owner", "-c", str(project.config), "-r", "003", "--by", "ann")
        assert outcome.exit_code == 0, outcome.output
        assert "kept 003_parent.mp4 as parent tracks 1, 4" in outcome.output
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
        assert "-e pose.kinematics`" in outcome.output
        path = layout_for(project.resolved()).camera_owners_path("003")
        entry = json.loads(path.read_text(encoding="utf-8"))["003_parent.mp4"]
        assert entry["people"] == {"parent": [2, 7]} and entry["by"].startswith("ann ")

    def test_tracks_need_exactly_one_camera(self, project):
        outcome = invoke("accept-owner", "-c", str(project.config), "-r", "003", "--tracks", "2")
        assert outcome.exit_code != 0

    def test_a_file_without_a_choice_is_an_error(self, project):
        project.owners({"003_parent.mp4": owner([1], "doubt")})
        outcome = invoke("accept-owner", "-c", str(project.config), "-r", "003", "--media", "nope.mp4")
        assert outcome.exit_code != 0


class TestCorrectingAChoice:
    def test_rename_gives_the_tracks_shown_under_one_name_to_the_right_person(self, project):
        project.owners({"003_room.mp4": owner({"parent": [1, 3], "child": [2]}, "not yet checked")})
        outcome = invoke(
            "accept-owner", "-c", str(project.config), "-r", "003",
            "--media", "003_room.mp4", "--rename", "parent=child,child=parent", "--by", "ann",
        )
        assert outcome.exit_code == 0, outcome.output
        path = layout_for(project.resolved()).camera_owners_path("003")
        entry = json.loads(path.read_text(encoding="utf-8"))["003_room.mp4"]
        assert entry["people"] == {"child": [1, 3], "parent": [2]}

    def test_rename_to_someone_who_is_not_a_subject_is_refused(self, project):
        project.owners({"003_room.mp4": owner({"parent": [1], "child": [2]}, "doubt")})
        outcome = invoke(
            "accept-owner", "-c", str(project.config), "-r", "003", "--media", "003_room.mp4", "--rename", "parent=uncle",
        )
        assert outcome.exit_code != 0 and "uncle" in str(outcome.exception)

    def test_tracks_on_a_shared_camera_need_a_subject(self, project):
        outcome = invoke("accept-owner", "-c", str(project.config), "-r", "003", "--media", "003_room.mp4", "--tracks", "4")
        assert outcome.exit_code != 0 and "--subject" in str(outcome.exception)

    def test_a_later_correction_replaces_an_earlier_claim_on_the_same_tracks(self, project):
        args = ["accept-owner", "-c", str(project.config), "-r", "003", "--media", "003_room.mp4"]
        invoke(*args, "--tracks", "4", "--subject", "parent")
        invoke(*args, "--tracks", "4", "--subject", "child")
        path = layout_for(project.resolved()).camera_owners_path("003")
        assert json.loads(path.read_text(encoding="utf-8"))["003_room.mp4"]["people"] == {"child": [4]}

    def test_keeping_a_shared_choice_clears_the_gate(self, project):
        project.owners({"003_room.mp4": owner({"parent": [1, 3], "child": [2]}, "not yet checked")})
        assert "--rename" in project.pending()
        invoke("accept-owner", "-c", str(project.config), "-r", "003")
        assert project.pending() is None


class TestReportPage:
    def test_the_page_shows_who_is_on_camera_and_what_is_in_doubt(self, project):
        from macaw.qc.report import write_recording_report

        project.owners({"003_room.mp4": owner({"parent": [1, 3], "child": [2]}, "who is who is not yet checked")})
        page, summary = write_recording_report(project.resolved(), "003")
        text = page.read_text(encoding="utf-8")
        assert "003_room.mp4" in text and "1, 3" in text and "who is who is not yet checked" in text
        assert "pose.review" in summary["waiting"]

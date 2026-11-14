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
from macaw.extractors.video.pose.look_names import named_by_look
from macaw.extractors.video.pose.track_repair import (
    Click,
    join_fragments,
    most_in_view,
    named_by_clicks,
    owner_rows,
    several_people,
    subject_column,
)
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from setup_helpers import put, write_setup

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


class TestOneTrackPerFrame:
    def test_one_track_per_frame_where_fragments_overlap(self):
        keypoints = table(person(1, range(0, 52)), person(2, range(50, 100), cx=210))
        owned = keypoints[owner_rows(keypoints, [1, 2])]
        assert not owned.duplicated(["frame", "joint"]).any()
        assert owned["frame"].nunique() == 100
        assert set(owned.loc[owned["frame"] == 51, "track_id"]) == {1}

    def test_one_person_in_view_is_not_several(self):
        assert not several_people(table(person(1, range(10)), person(2, range(10, 20))))
        assert several_people(table(person(1, range(10)), person(2, range(5, 15))))


class TestMostInView:
    def test_tracker_fragments_are_joined_and_a_passer_by_is_not(self):
        keypoints = table(
            person(1, range(0, 52)),
            person(2, range(50, 100), cx=210),          # the same person after a hand-over
            person(3, range(20, 30), cx=60, size=160),  # someone walking past, nearer the lens
        )
        choice = most_in_view(keypoints, "child")
        assert choice.people == {"child": [1, 2]} and choice.doubt is None

    def test_someone_else_in_view_at_least_half_as_long_is_a_doubt_with_where_to_look(self):
        keypoints = table(person(1, range(0, 100)), person(2, range(50, 100), cx=60, size=20))
        choice = most_in_view(keypoints, "child")
        assert choice.people == {"child": [1]}
        assert "in view nearly as long" in choice.doubt and choice.spans == [(5.0, 9.9)]

    def test_one_chosen_but_rarely_in_view_is_a_doubt(self):
        others = [person(track, range(start, start + 14), cx=cx, size=20) for track, start, cx in ((2, 30, 600), (3, 50, 900), (4, 70, 1200))]
        choice = most_in_view(table(person(1, range(0, 30), size=300), *others), "child")
        assert choice.people == {"child": [1]} and "in only 42% of the frames with anyone" in choice.doubt

    def test_nobody_in_view_names_nobody(self):
        assert most_in_view(table(person(1, range(0, 10))).iloc[0:0], "child").people == {}


def answers_by_place(*names):
    """A model that names whoever is left of x 300 as the first name and the rest as the second."""
    def ask(picture):
        return {"person": True, "who": names[0] if picture < 300 else names[1]}
    return ask


def frames_with_place(keypoints):
    """Reads each frame as the x of the person it will be cut around, which the fake model turns into a name."""
    def read(frames):
        for frame in frames:
            yield frame, frame
    return read


class TestNamedByLook:
    def ask_about(self, keypoints, answer, monkeypatch):
        import macaw.extractors.video.pose.look_names as module

        monkeypatch.setattr(module, "picture_of", lambda image, rows, chain, frame: float(chain.x))
        return named_by_look(keypoints, ["child", "parent"], frames_with_place(keypoints), answer, "the model")

    def test_a_name_given_on_most_frames_names_the_person(self, monkeypatch):
        choice, record = self.ask_about(room(), answers_by_place("child", "parent"), monkeypatch)
        assert choice.people == {"child": [1, 3], "parent": [2]} and choice.doubt is None
        assert [entry["named"] for entry in record] == ["parent", "child"] and len(record[0]["answers"]) == 5

    def test_a_name_given_to_two_people_goes_to_neither(self, monkeypatch):
        choice, _ = self.ask_about(room(), answers_by_place("parent", "parent"), monkeypatch)
        assert choice.people == {} and "took two people for parent" in choice.doubt and "did not clearly tell child, parent" in choice.doubt

    def test_unsure_answers_leave_the_person_waiting(self, monkeypatch):
        choice, _ = self.ask_about(room(), answers_by_place("unsure", "parent"), monkeypatch)
        assert choice.people == {"parent": [2]}
        assert "did not clearly tell child" in choice.doubt and "unsure 5" in choice.doubt

    def test_two_names_for_one_person_name_them_nothing(self, monkeypatch):
        flip = iter(["child", "parent"] * 10)
        choice, _ = self.ask_about(table(person(1, range(0, 100))), lambda picture: {"person": True, "who": next(flip)}, monkeypatch)
        assert choice.people == {}


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


def click(person, at, x, y=200.0):
    return Click(person=person, at=at, t=at, x=x, y=y)


class TestNamingByClicks:
    def test_a_click_names_the_person_under_it_for_as_long_as_they_are_followed(self):
        choice = named_by_clicks(room(), [click("child", 1.0, 100), click("parent", 1.0, 500)], 1 / FPS)
        assert choice.people == {"child": [1, 3], "parent": [2]}
        assert choice.doubt is None

    def test_nothing_is_placed_by_position(self):
        assert named_by_clicks(room(), [], 1 / FPS).people == {}

    def test_a_click_on_nobody_or_at_a_moment_without_anyone_is_said(self):
        keypoints = table(person(1, range(0, 50), cx=100))
        nowhere = named_by_clicks(keypoints, [click("child", 1.0, 900)], 1 / FPS)
        assert nowhere.people == {} and "nobody was detected where child clicked at 1 s" in nowhere.doubt
        later = named_by_clicks(keypoints, [click("child", 8.0, 100)], 1 / FPS)
        assert "nobody was detected in the frame where child clicked at 8 s" in later.doubt

    def test_two_names_on_one_person_are_said(self):
        choice = named_by_clicks(room(), [click("child", 1.0, 100), click("parent", 2.0, 102)], 1 / FPS)
        assert choice.people == {"child": [1, 3]} and "child and parent were clicked on the same person" in choice.doubt

    def test_a_click_where_two_people_overlap_names_the_one_clearly_nearest_or_asks_again(self):
        keypoints = table(person(1, range(0, 20), cx=200), person(2, range(0, 20), cx=260))
        near = named_by_clicks(keypoints, [click("child", 1.0, 160)], 1 / FPS)
        assert near.people == {"child": [1]}
        between = named_by_clicks(keypoints, [click("child", 1.0, 230)], 1 / FPS)
        assert between.people == {} and "falls on two people" in between.doubt

    def test_a_lost_person_is_named_again_by_a_second_click(self):
        far = table(person(1, range(0, 40), cx=100), person(5, range(60, 100), cx=700), person(2, range(0, 100), cx=400))
        once = named_by_clicks(far, [click("child", 1.0, 100), click("parent", 1.0, 400)], 1 / FPS)
        assert once.people["child"] == [1] and "tracks [5]" in once.doubt and once.spans == [(6.0, 9.9)]
        twice = named_by_clicks(far, [click("child", 1.0, 100), click("parent", 1.0, 400), click("child", 7.0, 700)], 1 / FPS)
        assert twice.people == {"child": [1, 5], "parent": [2]} and twice.doubt is None

    def test_each_frame_and_joint_belongs_to_one_person(self):
        keypoints = room(person(4, range(0, 100), cx=110, size=110))
        choice = named_by_clicks(keypoints, [click("child", 1.0, 100), click("parent", 1.0, 500)], 1 / FPS)
        keypoints["subject_id"] = subject_column(keypoints, choice.people)
        assigned = keypoints.dropna(subset=["subject_id"])
        assert not assigned.duplicated(["subject_id", "frame", "joint"]).any()
        assert assigned.loc[assigned.subject_id == "parent", "frame"].nunique() == 100


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    for device in ("parent_cam", "child_cam", "room"):
        put(raw, "003", device, "a.mp4")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=raw, output_dir=tmp_path / "out")
    write_setup(config, cameras={
        "parent_cam": {"body": ["parent"], "face": ["parent"]},
        "child_cam": {"body": ["child"], "face": ["child"]},
        "room": {"body": ["parent", "child"], "gaze": ["parent", "child"]},
    })

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
        project.owners({"parent_cam": owner([1])})
        assert project.pending() is None

    def test_a_doubt_waits_and_says_what_to_run(self, project):
        project.owners({"parent_cam": owner([1], "track 2 is in view nearly as much")})
        reason = project.pending()
        assert "parent_cam" in reason and "macaw name-people -r 003" in reason and "macaw accept-owner -r 003" in reason

    def test_skip_lets_it_through_with_a_note(self, project):
        project.owners({"parent_cam": owner([1], "doubt")})
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data.setdefault("pose", {})["review"] = "skip"
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert project.pending() is None
        resolved = project.resolved()
        notes = PoseReviewGate(resolved).notes(build_recording("003", resolved))
        assert [note.check for note in notes] == ["camera_owner_unreviewed"]

    def test_always_holds_every_camera_until_a_person_keeps_it(self, project):
        project.owners({"parent_cam": owner([1]), "child_cam": owner([3], subject="child")})
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data.setdefault("pose", {})["review"] = "always"
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        reason = project.pending()
        assert "parent_cam" in reason and "child_cam" in reason and "always" in reason
        outcome = invoke("accept-owner", "-c", str(project.config), "-r", "003", "--by", "ann")
        assert outcome.exit_code == 0, outcome.output
        assert project.pending() is None


class TestAcceptOwner:
    def test_keeping_the_choice_clears_the_gate(self, project):
        project.owners({"parent_cam": owner([1, 4], "doubt")})
        outcome = invoke("accept-owner", "-c", str(project.config), "-r", "003", "--by", "ann")
        assert outcome.exit_code == 0, outcome.output
        assert "kept parent_cam as parent tracks 1, 4" in outcome.output
        assert project.pending() is None

    def test_an_acceptance_lapses_when_the_choice_changes(self, project):
        project.owners({"parent_cam": owner([1], "doubt")})
        invoke("accept-owner", "-c", str(project.config), "-r", "003")
        project.owners({"parent_cam": owner([2], "doubt")})
        assert project.pending() is not None

    def test_a_file_without_a_choice_is_an_error(self, project):
        project.owners({"parent_cam": owner([1], "doubt")})
        outcome = invoke("accept-owner", "-c", str(project.config), "-r", "003", "--camera", "nope.mp4")
        assert outcome.exit_code != 0


class TestNamePeople:
    def name(self, project, *args):
        return invoke("name-people", "-c", str(project.config), "-r", "003", *args)

    def test_clicks_are_kept_with_who_made_them_and_join_the_earlier_ones(self, project):
        outcome = self.name(project, "--camera", "room", "--at", "2.5", "--person", "parent=0.25,0.4", "--person", "child=0.7,0.45", "--by", "ann")
        assert outcome.exit_code == 0, outcome.output
        assert "on room, parent at 2.5 s, child at 2.5 s" in outcome.output and "-e pose.keypoints" in outcome.output
        self.name(project, "--camera", "room", "--at", "9", "--person", "child=0.6,0.5")
        clicks = json.loads(layout_for(project.resolved()).names_path("003").read_text(encoding="utf-8"))["room"]
        assert [(c["person"], c["at"], c["x"], c["y"]) for c in clicks] == [("parent", 2.5, 0.25, 0.4), ("child", 2.5, 0.7, 0.45), ("child", 9.0, 0.6, 0.5)]
        assert clicks[0]["by"].startswith("ann ")
        assert self.name(project, "--camera", "room", "--clear").exit_code == 0
        assert not layout_for(project.resolved()).names_path("003").read_text(encoding="utf-8").strip("{}\n ")

    def test_one_click_is_taken_back_and_the_others_stay(self, project):
        self.name(project, "--camera", "room", "--at", "2.5", "--person", "parent=0.25,0.4", "--person", "child=0.7,0.45")
        self.name(project, "--camera", "room", "--at", "9", "--person", "child=0.6,0.5")
        outcome = self.name(project, "--camera", "room", "--remove", "--at", "2.5", "--person", "child")
        assert outcome.exit_code == 0 and "took back 1 click on room at 2.5 s" in outcome.output, outcome.output
        clicks = json.loads(layout_for(project.resolved()).names_path("003").read_text(encoding="utf-8"))["room"]
        assert [(c["person"], c["at"]) for c in clicks] == [("parent", 2.5), ("child", 9.0)]
        assert "took back 1 click" in self.name(project, "--camera", "room", "--remove", "--at", "2.5").output
        missing = self.name(project, "--camera", "room", "--remove", "--at", "4")
        assert missing.exit_code != 0 and "no click on room at 4 s" in str(missing.exception)

    @pytest.mark.parametrize("args, reason", [
        (["--camera", "nope", "--at", "1", "--person", "parent=0.1,0.1"], "not a camera of 003"),
        (["--camera", "parent_cam", "--at", "1", "--person", "child=0.1,0.1"], "reads parent_cam for parent, not child"),
        (["--camera", "room", "--at", "1", "--person", "parent=1.5,0.1"], "from 0 to 1"),
        (["--camera", "room", "--at", "1", "--person", "parent=left"], "ROLE=X,Y"),
        (["--camera", "room", "--person", "parent=0.1,0.1"], "--at"),
    ])
    def test_a_click_that_cannot_be_read_one_way_is_refused(self, project, args, reason):
        outcome = self.name(project, *args)
        assert outcome.exit_code != 0 and reason in str(outcome.exception)


class TestReportPage:
    def test_the_page_shows_who_is_on_camera_and_what_is_in_doubt(self, project):
        from macaw.qc.report import write_recording_report

        project.owners({"room": owner({"parent": [1, 3], "child": [2]}, "tracks [7], which nobody was clicked on, are in view")})
        page, summary = write_recording_report(project.resolved(), "003")
        text = page.read_text(encoding="utf-8")
        assert "room" in text and "1, 3" in text and "which nobody was clicked on" in text
        assert "pose.review" in summary["waiting"]

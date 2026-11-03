from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from macaw.coding.llm.backends import LLMResponse
from macaw.config.schema import LookSection
from macaw.exceptions import MacawError
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.qc import look
from macaw.qc.look import MISSED, ON_NO_ONE, RIGHT, TWICE, WRONG, WRONG_NAME, problems
from macaw.qc.overlay import BASELINE, REVIEW_SCHEMA
from macaw.project.recording import MediaFile
from macaw.store.schema import KEYPOINT_SCHEMA, coerce_frame

RID = "dyad01"
NAMES = ["parent", "child"]
LOOKS = {"parent": "an adult", "child": "a young child"}


def box(name="parent", person=True, who="unsure"):
    return {"name": name, "track": 0, "person": person, "who": who}


class TestProblems:
    def test_nothing_is_wrong_when_the_answers_match_the_drawing(self):
        assert problems(2, [box("parent", who="parent"), box("child", who="child")], NAMES) == []

    def test_more_skeletons_on_people_than_people_in_view_is_someone_drawn_twice(self):
        found = problems(2, [box("parent"), box(None), box("child")], NAMES)
        assert [item["kind"] for item in found] == [TWICE]
        assert found[0]["says"] == "3 skeletons on 2 people in view: someone is drawn twice"

    def test_more_people_in_view_than_skeletons_on_people_is_someone_missed(self):
        found = problems(3, [box("parent"), box("child")], NAMES)
        assert [item["kind"] for item in found] == [MISSED]
        assert found[0]["says"] == "3 people in view, 2 with a skeleton on them"

    def test_a_box_on_no_one_also_leaves_its_person_missed(self):
        found = problems(2, [box("parent"), box("child", person=False)], NAMES)
        assert [(item["kind"], item["box"]) for item in found] == [(MISSED, None), (ON_NO_ONE, "child")]

    def test_a_name_on_the_wrong_person(self):
        found = problems(2, [box("parent", who="child"), box("child", who="parent")], NAMES)
        assert [(item["kind"], item["box"]) for item in found] == [(WRONG_NAME, "parent"), (WRONG_NAME, "child")]
        assert found[0]["says"] == "the box named parent looks like child"

    def test_one_name_given_to_two_people_says_nothing(self):
        """Two adults, or a child on an adult's lap, both called the same: the model cannot tell them apart."""
        assert problems(2, [box("parent", who="child"), box("child", who="child")], NAMES) == []

    def test_unsure_and_unassigned_say_nothing_about_names(self):
        assert problems(2, [box("parent", who="unsure"), box(None, who="child")], NAMES) == []

    def test_an_uncounted_frame_says_nothing_about_missed_people(self):
        assert problems(None, [box("parent")], NAMES) == []


cv2 = pytest.importorskip("cv2")
PARENT, CHILD = (60, 180, 255), (255, 140, 40)


def body(subject_id, *, frame=5, centre=(100, 120), track_id=0, video="room.avi"):
    offsets = {joint: (index % 3 * 10 - 10, index * 5 - 40) for index, joint in enumerate(STANDARD_JOINTS)}
    rows = [
        {
            "recording_id": RID, "subject_id": subject_id, "source": "pose.keypoints", "video": video, "frame": frame,
            "t": frame / 10.0, "joint": joint, "x": float(centre[0] + dx), "y": float(centre[1] + dy), "z": float("nan"),
            "confidence": 0.9, "track_id": track_id, "units": "pixel", "metric_scale_valid": None,
        }
        for joint, (dx, dy) in offsets.items()
    ]
    return coerce_frame(pd.DataFrame(rows), KEYPOINT_SCHEMA, "keypoints")


class Model:
    """Answers the count with people (or the next of a list of them) and each person with the next of persons; keeps every picture it was sent."""

    def __init__(self, people=2, persons=None):
        self.people = people
        self.persons = list(persons or [])
        self.pictures: list[tuple[str, bytes]] = []
        self.position = 0

    def model_version(self):
        return "vision@abc"

    def complete(self, system, user, *, seed=None, images=()):
        if system == look.COUNT:
            self.pictures.append(("count", images[0]))
            people = self.people.pop(0) if isinstance(self.people, list) else self.people
            text = people if isinstance(people, str) else json.dumps({"people": people})
        else:
            self.pictures.append(("person", images[0]))
            answer = self.persons[self.position % len(self.persons)] if self.persons else {"person": True, "who": "unsure"}
            self.position += 1
            text = answer if isinstance(answer, str) else json.dumps(answer)
        return LLMResponse(text=text, model="vision@abc")


def decoded(picture: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(picture, np.uint8), cv2.IMREAD_COLOR)


class TestLookAtFrame:
    def test_one_count_and_one_picture_per_person(self):
        people = pd.concat([body("parent"), body("child", centre=(220, 120), track_id=1)])
        model = Model(people=2, persons=[{"person": True, "who": "unsure"}, {"person": True, "who": "parent"}])
        found = look.look_at_frame(model, np.zeros((240, 320, 3), np.uint8), people, NAMES, LOOKS)
        assert [kind for kind, _ in model.pictures] == ["count", "person", "person"]
        assert found["people_seen"] == 2
        assert [(item["kind"], item["box"]) for item in found["problems"]] == [(WRONG_NAME, "child")]

    def test_a_person_is_cut_out_with_only_their_own_skeleton(self):
        """Another skeleton in the cut would leave the model to guess which one is asked about."""
        people = pd.concat([body("parent"), body("child", centre=(115, 120), track_id=1)])
        model = Model()
        look.look_at_frame(model, np.zeros((240, 320, 3), np.uint8), people, NAMES, {})
        parent = decoded(model.pictures[1][1])
        assert max(parent.shape[:2]) == look.CROP_SIDE

        def near(image, colour):
            return int((np.abs(image.astype(int) - np.array(colour)).sum(axis=2) < 60).sum())

        assert near(parent, PARENT) > 0 and near(parent, CHILD) == 0

    def test_an_answer_that_does_not_read_is_counted_and_raises_nothing(self):
        model = Model(people="no idea", persons=["{not json"])
        found = look.look_at_frame(model, np.zeros((240, 320, 3), np.uint8), body("parent"), NAMES, {})
        assert found["unreadable"] == 2 and found["problems"] == []

    def test_who_someone_is_is_asked_only_of_the_people_described(self):
        """A name such as p1 says nothing about a person, so without a description the model would only guess."""
        seen = []
        model = Model()
        model.complete = lambda system, user, **_: seen.append(system) or LLMResponse(text='{"people": 1, "person": true, "who": "child"}', model="m")
        found = look.look_at_frame(model, np.zeros((240, 320, 3), np.uint8), body("parent"), NAMES, {})
        assert '"who"' not in seen[1] and found["boxes"][0]["who"] == look.UNSURE and found["problems"] == []
        found = look.look_at_frame(model, np.zeros((240, 320, 3), np.uint8), body("parent"), NAMES, LOOKS)
        assert '"parent" (an adult), "child" (a young child)' in seen[3]
        assert [item["kind"] for item in found["problems"]] == [WRONG_NAME]


def write_video(path: Path, n_frames=10, fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    for _ in range(n_frames):
        writer.write(np.zeros((240, 320, 3), np.uint8))
    writer.release()
    return path


@pytest.fixture
def drawn(tmp_path):
    """Four review frames rendered with their plain copies, and a layout that keeps everything under tmp_path."""
    from macaw.qc.overlay import render_review

    video = write_video(tmp_path / "room.avi")
    recording = SimpleNamespace(
        recording_id=RID, subjects=[SimpleNamespace(subject_id=name) for name in NAMES],
        media=[MediaFile(path=video, kind="video", subject_id=None, fps=10.0, offset_s=0.0)],
    )
    keypoints = pd.concat([body("parent", frame=frame) for frame in range(10)])
    moments = coerce_frame(
        pd.DataFrame([{"recording_id": RID, "subject_id": None, "check": None, "reason": BASELINE, "t": t, "score": np.nan} for t in (0.1, 0.3, 0.5, 0.7)]),
        REVIEW_SCHEMA, "review sample",
    )
    layout = SimpleNamespace(
        renders_dir=lambda rid: tmp_path / "renders" / rid,
        look_path=lambda rid: tmp_path / "qc" / rid / "look.json",
        look_checks_path=lambda rid: tmp_path / "qc" / rid / "look_checks.json",
    )
    index = render_review(recording, keypoints, moments, layout.renders_dir(RID), plain_dir=layout.renders_dir(RID) / look.PLAIN)
    return SimpleNamespace(recording=recording, keypoints=keypoints, index=index, layout=layout)


class TestLookAtRecording:
    def run(self, drawn, model, **settings):
        cfg = LookSection(model="vision", **settings)
        return look.look_at_recording(model, cfg, drawn.layout, drawn.recording, drawn.keypoints, drawn.index, seed=0)

    def test_every_frame_is_asked_about_and_kept(self, drawn):
        found = self.run(drawn, Model(people=1))
        assert len(found["frames"]) == 4 and found["model"] == "vision@abc"
        assert json.loads(drawn.layout.look_path(RID).read_text(encoding="utf-8"))["frames"] == found["frames"]

    def test_a_frame_answered_before_is_not_asked_again(self, drawn):
        self.run(drawn, Model(people=1))
        again = Model(people=1)
        self.run(drawn, again)
        assert again.pictures == []
        described = Model(people=1)
        self.run(drawn, described, people={"parent": "an adult"})
        assert len(described.pictures) == 8

    def test_the_random_check_draws_only_frames_the_model_passed(self, drawn):
        answers = [{"person": False, "who": "unsure"}, {"person": True, "who": "unsure"}]
        found = self.run(drawn, Model(people=1, persons=answers), random_check=5)
        flagged = [entry["file"] for entry in found["frames"] if entry["problems"]]
        drawn_for_check = [entry["file"] for entry in found["frames"] if entry["random_check"]]
        assert len(flagged) == 2 and len(drawn_for_check) == 2 and not set(flagged) & set(drawn_for_check)
        assert len([entry for entry in self.run(drawn, Model(people=1), random_check=1)["frames"] if entry["random_check"]]) == 1


class TestChecks:
    def test_verdicts_count_toward_the_summary(self, drawn):
        found = TestLookAtRecording().run(drawn, Model(people=[2, 2, 2, 1]), random_check=1)
        flagged, passed = [entry["file"] for entry in found["frames"] if entry["problems"]], [entry["file"] for entry in found["frames"] if entry["random_check"]]
        assert len(flagged) == 3 and len(passed) == 1
        look.set_check(drawn.layout, RID, flagged[0], RIGHT, "anna")
        look.set_check(drawn.layout, RID, flagged[1], WRONG, "anna")
        look.set_check(drawn.layout, RID, passed[0], WRONG, "anna")
        counts = look.summary(drawn.layout, RID)
        assert (counts["flagged"], counts["flagged_checked"], counts["flagged_right"]) == (3, 2, 1)
        assert (counts["drawn"], counts["drawn_checked"], counts["drawn_wrong"]) == (1, 1, 1)
        assert look.describe(counts) == (
            "3 of 4 review frame(s) flagged by vision@abc, 1 of 2 checked confirmed by a person; "
            "a person found 1 of 1 frame(s) it passed, drawn at random, wrong"
        )

    def test_only_a_frame_waiting_for_a_person_takes_a_verdict(self, drawn):
        found = TestLookAtRecording().run(drawn, Model(people=1), random_check=1)
        untouched = next(entry["file"] for entry in found["frames"] if not entry["random_check"])
        with pytest.raises(MacawError, match="not among the frames"):
            look.set_check(drawn.layout, RID, untouched, RIGHT, "anna")
        chosen = next(entry["file"] for entry in found["frames"] if entry["random_check"])
        with pytest.raises(MacawError, match="right or wrong"):
            look.set_check(drawn.layout, RID, chosen, "maybe", "anna")

    def test_a_frame_asked_again_waits_for_a_person_again(self, drawn):
        found = TestLookAtRecording().run(drawn, Model(people=2))
        file = found["frames"][0]["file"]
        look.set_check(drawn.layout, RID, file, RIGHT, "anna")
        assert look.summary(drawn.layout, RID)["flagged_checked"] == 1
        TestLookAtRecording().run(drawn, Model(people=2), people={"parent": "an adult"})
        assert look.summary(drawn.layout, RID)["flagged_checked"] == 0
        look.set_check(drawn.layout, RID, file, None, "anna")
        assert look.read_checks(drawn.layout, RID) == {}


class TestSettings:
    def test_nothing_is_asked_without_a_model(self):
        with pytest.raises(MacawError, match="qc.look.model is not set"):
            look.backend_for(LookSection())

    def test_frames_sent_away_are_announced(self):
        assert look.notice(LookSection(provider="ollama")) is None
        assert "pictures of the people recorded" in look.notice(LookSection(provider="openai"))

    def test_the_server_settings_are_coding_s(self):
        settings = LookSection(provider="vllm", model="m", base_url="http://x/v1").llm()
        assert (settings.provider, settings.model, settings.base_url, settings.temperature) == ("vllm", "m", "http://x/v1", 0.0)


@pytest.fixture
def project(tmp_path):
    from macaw.config.loader import scaffold_project_config

    (tmp_path / "raw" / RID).mkdir(parents=True)
    (tmp_path / "raw" / RID / "video.mp4").write_bytes(b"")
    path = tmp_path / "project.yaml"
    scaffold_project_config(path, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    return path


def asked(project, frames):
    from macaw.config.loader import resolve_config
    from macaw.project.layout import layout_for

    layout = layout_for(resolve_config(project))
    entries = [{"file": f"{index:03d}_baseline.jpg", "t": float(index), "video": "video.mp4", "frame": index, "reason": BASELINE,
                "key": f"k{index}", "people_seen": 2, "boxes": [], "problems": found, "unreadable": 0, "random_check": drawn}
               for index, (found, drawn) in enumerate(frames)]
    layout.look_path(RID).parent.mkdir(parents=True, exist_ok=True)
    layout.look_path(RID).write_text(json.dumps({"model": "vision@abc", "provider": "ollama", "frames": entries}), encoding="utf-8")
    return layout


ONE = [{"kind": ON_NO_ONE, "box": "child", "says": "the box named child is not on a person"}]


class TestCommand:
    def test_a_verdict_is_kept_with_who_gave_it(self, project):
        from macaw.cli.main import app

        layout = asked(project, [(ONE, False), ([], True)])
        result = CliRunner().invoke(app, ["check-look", "-c", str(project), "-r", RID, "--frame", "000_baseline.jpg", "--verdict", "right", "--by", "anna"])
        assert result.exit_code == 0, result.output
        assert "1 of 1 checked confirmed" in result.output
        assert look.read_checks(layout, RID)["000_baseline.jpg"]["by"] == "anna"
        assert layout.look_checks_path(RID) in layout.human_made(RID)
        result = CliRunner().invoke(app, ["check-look", "-c", str(project), "-r", RID, "--frame", "000_baseline.jpg"])
        assert result.exit_code == 1 and "--verdict right or wrong, or --clear" in str(result.exception)

    def test_the_report_shows_what_the_model_said(self, project):
        from macaw.qc.report import _looked_at

        layout = asked(project, [(ONE, False), ([], True)])
        look.set_check(layout, RID, "001_baseline.jpg", WRONG, "anna")
        groups = _looked_at(layout, RID, layout.output_dir / "reports" / "page.html")
        assert [group["reason"] for group in groups] == ["flagged by the model", "passed by the model, drawn for a person to check"]
        assert groups[0]["shots"][0]["note"] == "the box named child is not on a person; not checked yet"
        assert groups[1]["shots"][0]["note"] == "nothing wrong, says the model; a person disagreed"

    def test_methods_says_what_the_model_was_asked_and_what_people_found(self, project):
        from macaw.config.loader import resolve_config
        from macaw.qc.boilerplate import _looked

        layout = asked(project, [(ONE, False), ([], True)])
        look.set_check(layout, RID, "000_baseline.jpg", RIGHT, "anna")
        look.set_check(layout, RID, "001_baseline.jpg", RIGHT, "anna")
        text = _looked(look.project_summary(resolve_config(project)))
        assert "the vision language model vision@abc (ollama)" in text
        assert "It flagged 1 of 2 frames, and a person confirmed 1 of the 1 checked." in text
        assert "who found 0 of them wrong" in text


class TestWorkbench:
    def test_the_review_page_shows_the_frames_to_check_with_buttons(self, project):
        from macaw.config.loader import resolve_config
        from macaw.interface.pages import review

        layout = asked(project, [(ONE, False), ([], True), ([], False)])
        card = review._model_look(resolve_config(project), layout, RID)
        text = json.dumps(card.to_plotly_json(), default=lambda item: item.to_plotly_json())
        assert "the box named child is not on a person" in text and "drawn at random to check" in text
        assert text.count('"look-right"') == 2

    def test_its_buttons_run_the_commands(self, project):
        from macaw.interface.pages import review

        assert review.action_command(project, RID, "look", "", "", {}, {})[-1] == "--look"
        argv = review.action_command(project, RID, "look-wrong", "001_baseline.jpg", "anna", {}, {})
        assert argv[:2] == ["macaw", "check-look"] and argv[-6:] == ["--frame", "001_baseline.jpg", "--verdict", "wrong", "--by", "anna"]

    def test_no_card_before_a_model_is_named(self, project):
        from macaw.config.loader import resolve_config
        from macaw.interface.pages import review
        from macaw.project.layout import layout_for

        config = resolve_config(project)
        assert review._model_look(config, layout_for(config), RID) is None

    def test_the_settings_form_takes_the_people_as_project_yaml_writes_them(self, project):
        from macaw.config.loader import resolve_config
        from macaw.interface import project_file

        typed = "{parent: an adult, child: 'a young child, in a high chair'}"
        assert project_file.control("qc.look.people") == {"kind": "mapping"}
        assert project_file.from_form("qc.look.people", typed) == {"parent": "an adult", "child": "a young child, in a high chair"}
        assert project_file.to_form("qc.look.people", project_file.from_form("qc.look.people", typed)) == typed
        assert project_file.from_form("qc.look.people", "") == {}
        text = project_file.with_values(project.read_text(encoding="utf-8"), {"qc.look.people": project_file.from_form("qc.look.people", typed)})
        assert "people: {parent: an adult, child: 'a young child, in a high chair'}" in text
        project.write_text(text, encoding="utf-8")
        assert resolve_config(project).qc.look.people["child"] == "a young child, in a high chair"


def test_methods_names_the_people_the_model_was_told_about(project):
    from macaw.config.loader import resolve_config
    from macaw.qc.boilerplate import _looked

    layout = asked(project, [(ONE, False)])
    look_file = layout.look_path(RID)
    look_file.write_text(json.dumps({**json.loads(look_file.read_text(encoding="utf-8")), "described": LOOKS}), encoding="utf-8")
    text = _looked(look.project_summary(resolve_config(project)))
    assert "whether it was a real person and, given a description of the child and parent, which of them it was;" in text

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.schema import LookSection, PoseSection, QCSection
from macaw.extractors.prep import _window
from macaw.extractors.video.pose import keypoints as pose_module
from macaw.extractors.video.pose.backends import BACKEND_KEYPOINT_SCHEMA, PoseFrames
from macaw.extractors.video.pose.keypoints import PoseDetectExtractor, PoseExtractor, inside_frame
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.project.recording import MediaFile, Recording, Subject
from macaw.store.schema import (
    KEYPOINT_SCHEMA,
    LONG_TABLE_SCHEMA,
    coerce_frame,
    empty_frame,
    validate_frame,
)

FPS = 10.0
N_FRAMES = 20
SWAP_AT = 10
# the picture the fake videos have; clicks are shares of it
PICTURE = (640, 400)


def backend_frame(positions, *, n_frames=N_FRAMES, swap_at=None, confidence=0.9):
    rows = []
    for frame in range(n_frames):
        order = list(positions)
        if swap_at is not None and frame >= swap_at:
            order = order[::-1]
        for track, x in enumerate(order):
            for joint in STANDARD_JOINTS:
                rows.append(
                    {
                        "frame": frame,
                        "joint": joint,
                        "x": float(x),
                        "y": 200.0,
                        "z": float("nan"),
                        "confidence": confidence,
                        "track_id": track,
                        "units": "pixel",
                        "metric_scale_valid": None,
                    }
                )
    if not rows:
        return empty_frame(BACKEND_KEYPOINT_SCHEMA)
    return coerce_frame(pd.DataFrame(rows), BACKEND_KEYPOINT_SCHEMA, "backend keypoints")


class FakeBackend:
    name = "fake"
    version = "9.9"
    joint_set = "coco17"

    def __init__(self, frames=None):
        self.frames = frames if frames is not None else backend_frame([100.0, 500.0], swap_at=SWAP_AT)
        self.calls = []

    def check_available(self):
        return None

    def infer(self, video_path, *, device, min_confidence, model=None, max_people=1, frames=None, input_size=None):
        self.calls.append({"path": video_path, "max_people": max_people, "frames": frames})
        return PoseFrames(keypoints=self.frames)


def media(name="room.mp4", *, subject_id=None, people=None, fps=FPS, offset_s=0.0, kind="video"):
    """A camera the setup reads for these people (the subject alone, else parent and child), or a microphone."""
    people = tuple(people) if people is not None else ((subject_id,) if subject_id else ("parent", "child"))
    uses = {use: people for use in ("body", "face", "gaze")} if kind == "video" else {}
    return MediaFile(
        path=Path(name),
        kind=kind,
        subject_id=people[0] if len(people) == 1 else None,
        device=name,
        uses=uses,
        fps=fps,
        offset_s=offset_s,
        duration_s=N_FRAMES / FPS,
    )


@pytest.fixture
def recording():
    def build(files, *, t_task_start=0.0, t_task_end=None, subject_ids=("parent", "child")):
        return Recording(
            recording_id="sub-003_ses-12mo",
            root=Path("."),
            group_id="003",
            session_id="12mo",
            subjects=[Subject(subject_id=name, role=name) for name in subject_ids],
            media=list(files),
            t_task_start=t_task_start,
            t_task_end=t_task_end,
        )

    return build


def fake_prep_timing(config, recording):
    """The fake media already carry fps and offset; prep only has to supply the window."""
    start, end = _window(recording)
    return SimpleNamespace(t_origin_s=start, duration_s=end - start)


class AnyVideo(dict):
    """prep.video's record of every fake video: frames from its first picture, at PICTURE size."""

    def get(self, key, default=None):
        return {"first_picture_s": 0.0, "frames_start_s": 0.0, "prepared_size": list(PICTURE), "size": list(PICTURE)}


@pytest.fixture
def offsets(tmp_path, monkeypatch):
    path = tmp_path / "offsets.json"
    monkeypatch.setattr(
        pose_module,
        "layout_for",
        lambda config, create=False: SimpleNamespace(
            offset_table_path=lambda rid: path,
            names_path=lambda rid: tmp_path / "names.json",
            artifacts_dir=lambda rid, name: tmp_path / "artifacts" / name,
        ),
    )
    monkeypatch.setattr(pose_module, "load_video_prep", lambda config, rid: AnyVideo())
    return path


def clicks(tmp_path, camera, at, **people):
    """A person's clicks at one moment of a camera, given in pixels of PICTURE."""
    path = tmp_path / "names.json"
    names = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    names.setdefault(camera, []).extend(
        {"person": person, "at": at, "x": x / PICTURE[0], "y": y / PICTURE[1], "by": "ann 2026-11-03T10:00:00"}
        for person, (x, y) in people.items()
    )
    path.write_text(json.dumps(names), encoding="utf-8")


class Chain:
    """pose.detect then pose.keypoints, the way the runner chains them."""

    def __init__(self, backend, look=None, **pose):
        config = SimpleNamespace(pose=PoseSection(**pose), qc=QCSection(look=look or LookSection()))
        self.fake = backend
        self.detect = PoseDetectExtractor(config)
        self.identify = PoseExtractor(config)

    def run(self, recording, workspace):
        found = self.detect.run(recording, workspace / "detect")
        artifacts = workspace / "artifacts" / PoseDetectExtractor.name
        artifacts.mkdir(parents=True, exist_ok=True)
        for name, source in found.artifacts.items():
            (artifacts / name).write_bytes(source.read_bytes())
        result = self.identify.run(recording, workspace)
        result.warnings[:0] = found.warnings
        result.detected = found
        return result


@pytest.fixture
def extractor(monkeypatch, offsets):
    def build(backend=None, look=None, **pose):
        backend = backend or FakeBackend()
        monkeypatch.setattr(pose_module, "resolve_backend", lambda name: backend)
        monkeypatch.setattr(pose_module, "apply_prep_timing", fake_prep_timing)
        monkeypatch.setattr(
            pose_module, "load_prep", lambda config, rid: SimpleNamespace(tracks={}, t_origin_s=0.0, duration_s=N_FRAMES / FPS)
        )
        return Chain(backend, look, **pose)

    return build


@pytest.fixture
def shared(extractor, recording, tmp_path):
    instance = extractor()
    clicks(tmp_path, "room.mp4", 0.5, parent=(100.0, 200.0), child=(500.0, 200.0))
    return instance, instance.run(recording([media()]), tmp_path)


class TestSharedCamera:
    def test_the_keypoints_satisfy_the_stored_schema(self, shared):
        validate_frame(shared[1].keypoints, KEYPOINT_SCHEMA, "keypoints")

    def test_the_rows_satisfy_the_long_table_schema(self, shared):
        validate_frame(shared[1].rows, LONG_TABLE_SCHEMA, "long table")

    def test_manifest_and_rows_agree(self, shared):
        """Every feature emitted is documented, and nothing documented is never emitted."""
        extractor, result = shared
        assert {spec.name for spec in result.specs} == set(result.rows.feature)

    def test_the_backend_is_asked_for_as_many_people_as_there_are_subjects(self, shared):
        assert shared[0].fake.calls[0]["max_people"] == 2

    def test_both_people_are_named(self, shared):
        assert set(shared[1].keypoints.subject_id.dropna()) == {"parent", "child"}

    def test_the_crossover_is_repaired(self, shared):
        """Each person stayed at one x, so one x per subject means the swap was undone."""
        assert shared[1].keypoints.groupby("subject_id")["x"].nunique().to_dict() == {
            "parent": 1,
            "child": 1,
        }

    def test_the_raw_track_id_survives_the_repair(self, shared):
        """The repair shows up in subject_id; track_id stays what the model actually said, so
        a disputed assignment can still be traced back."""
        keypoints = shared[1].keypoints
        assert sorted(keypoints.track_id.dropna().unique()) == [0, 1]
        assert keypoints[keypoints.frame == 0].track_id.nunique() == 2

    def test_time_comes_from_the_frame_rate(self, shared):
        keypoints = shared[1].keypoints
        assert keypoints.t.min() == pytest.approx(0.0)
        assert keypoints.t.max() == pytest.approx((N_FRAMES - 1) / FPS)

    def test_the_swap_count_belongs_to_nobody(self, shared):
        rows = shared[1].rows
        swaps = rows[rows.feature == "n_track_swaps"]
        assert len(swaps) == 1
        assert swaps.subject_id.isna().all()
        assert float(swaps.value.iloc[0]) == 1.0

    def test_the_crossovers_are_written_out_as_evidence(self, shared):
        """The identity review step needs to show why a track was relabelled."""
        assert shared[1].artifacts["swaps.json"].is_file()

    def test_a_crossover_is_reported(self, shared):
        assert any(warning.severity == "info" for warning in shared[1].warnings)

    def test_the_people_are_named_by_the_clicks(self, shared):
        owners = json.loads(shared[1].artifacts["owners"].read_text(encoding="utf-8"))
        assert owners["room.mp4"]["people"] == {"parent": [0], "child": [1]}
        assert owners["room.mp4"]["doubt"] is None and "clicked at 0.5 s" in owners["room.mp4"]["chosen_by"]["child"]

    def test_clicks_outside_the_analysed_part_are_left_out_with_a_note_not_a_doubt(self, extractor, recording, tmp_path):
        clicks(tmp_path, "room.mp4", 0.2, parent=(500.0, 200.0), child=(100.0, 200.0))
        clicks(tmp_path, "room.mp4", 0.7, parent=(100.0, 200.0), child=(500.0, 200.0))
        result = extractor().run(recording([media()], t_task_start=0.5), tmp_path)
        owners = json.loads(result.artifacts["owners"].read_text(encoding="utf-8"))
        assert owners["room.mp4"]["people"] == {"parent": [0], "child": [1]} and owners["room.mp4"]["doubt"] is None
        assert "clicked at 0.7 s" in owners["room.mp4"]["chosen_by"]["parent"]
        said = [w for w in result.warnings if "left out" in w.message]
        assert [w.severity for w in said] == ["info"] and "clicks at 0.2 s fall outside the analysed part (0.5 to 2.0 s" in said[0].message

    def test_with_every_click_outside_the_analysed_part_a_person_is_asked_to_click_inside_it(self, extractor, recording, tmp_path):
        clicks(tmp_path, "room.mp4", 0.2, parent=(100.0, 200.0), child=(500.0, 200.0))
        clicks(tmp_path, "room.mp4", 0.7, parent=(100.0, 200.0))
        result = extractor().run(recording([media()], t_task_start=0.5), tmp_path)
        doubt = json.loads(result.artifacts["owners"].read_text(encoding="utf-8"))["room.mp4"]["doubt"]
        assert "child was not clicked on room.mp4 within its analysed part, 0.5 to 2.0 s of its file" in doubt
        result = extractor().run(recording([media()], t_task_start=1.0), tmp_path)
        doubt = json.loads(result.artifacts["owners"].read_text(encoding="utf-8"))["room.mp4"]["doubt"]
        assert "every click on room.mp4 falls outside its analysed part" in doubt

    def test_without_clicks_nobody_is_named_and_a_person_is_asked(self, extractor, recording, tmp_path):
        result = extractor().run(recording([media()]), tmp_path)
        assert result.keypoints.subject_id.isna().all()
        owners = json.loads(result.artifacts["owners"].read_text(encoding="utf-8"))
        assert owners["room.mp4"]["people"] == {} and "nobody has said who is who on room.mp4" in owners["room.mp4"]["doubt"]


class TestPerSubjectCameras:
    @pytest.fixture
    def result(self, extractor, recording, tmp_path):
        instance = extractor(FakeBackend(backend_frame([100.0])))
        files = [media("a.mp4", subject_id="parent"), media("b.mp4", subject_id="child")]
        return instance, instance.run(recording(files), tmp_path)

    def test_one_person_is_asked_for_per_file(self, result):
        assert [call["max_people"] for call in result[0].fake.calls] == [1, 1]

    def test_the_subject_comes_from_the_setup_not_from_a_guess(self, result):
        assert set(result[1].keypoints.subject_id.dropna()) == {"parent", "child"}

    def test_no_repair_is_attempted(self, result):
        """A camera pointed at one person has already answered the identity question."""
        rows = result[1].rows
        assert float(rows[rows.feature == "n_track_swaps"].value.iloc[0]) == 0.0
        assert "swaps.json" not in result[1].artifacts



class TestWhichCamerasAreRead:
    def test_every_camera_the_setup_reads_for_someone_is_read_in_its_order(self, extractor, recording, tmp_path):
        instance = extractor()
        result = instance.run(recording([media("room_b.mp4"), media("room_a.mp4")]), tmp_path)
        assert [call["path"].name for call in instance.fake.calls] == ["room_b.mp4", "room_a.mp4"]
        assert set(result.keypoints.video) == {"room_a.mp4", "room_b.mp4"}

    def test_a_camera_whose_sound_alone_is_used_is_not_read_for_pictures(self, extractor, recording):
        files = [media("a.mp4", subject_id="parent"), media("hall.mp4", people=()), media("mic.wav", kind="audio")]
        assert extractor().detect.inputs_for(recording(files)) == [Path("a.mp4")]

    def test_the_offset_table_is_part_of_the_cache_key(self, extractor, recording, offsets):
        """Changing an offset moves every frame on the recording clock."""
        offsets.write_text("{}")
        assert extractor().detect.inputs_for(recording([media("room.mp4")])) == [Path("room.mp4"), offsets]


class TestWindowAndClock:
    def test_frames_outside_the_task_window_are_dropped(self, extractor, recording, tmp_path):
        instance = extractor()
        result = instance.run(recording([media()], t_task_start=0.5, t_task_end=1.0), tmp_path)
        assert result.keypoints.t.min() >= 0.5
        assert result.keypoints.t.max() <= 1.0

    def test_only_the_window_is_asked_for(self, extractor, recording, tmp_path):
        """In the file's own frames: the window 100 to 101 s at offset 99.5 is 0.5 to 1.5 s in."""
        instance = extractor()
        instance.run(recording([media(offset_s=99.5)], t_task_start=100.0, t_task_end=101.0), tmp_path)
        assert instance.fake.calls[0]["frames"] == (5, 16)

    def test_a_file_entirely_outside_the_window_is_not_read(self, extractor, recording, tmp_path):
        instance = extractor()
        result = instance.run(
            recording([media(offset_s=50.0)], t_task_start=0.0, t_task_end=10.0), tmp_path
        )
        assert instance.fake.calls == []
        assert any(warning.severity == "fail" for warning in result.warnings)

    def test_the_offset_is_applied_once_here(self, extractor, recording, tmp_path):
        """prep applies it once for audio; doing it again downstream would misalign the two
        modalities against each other rather than against the clock."""
        instance = extractor()
        result = instance.run(
            recording([media(offset_s=100.0)], t_task_start=100.0, t_task_end=102.0), tmp_path
        )
        assert result.keypoints.t.min() == pytest.approx(100.0)


class TestOutsideThePicture:
    def test_a_point_beyond_any_edge_becomes_missing(self):
        points = pd.DataFrame({"x": [-1.0, 10.0, 321.0, 10.0, 320.0], "y": [5.0, -0.5, 5.0, 241.0, 240.0], "z": 0.0})
        kept = inside_frame(points, (320, 240))
        assert kept["x"].notna().tolist() == [False, False, False, False, True]
        assert kept["z"].isna().tolist() == [True, True, True, True, False]

    def test_without_a_frame_size_nothing_is_dropped(self):
        points = pd.DataFrame({"x": [-50.0], "y": [5.0], "z": float("nan")})
        assert inside_frame(points, None)["x"].tolist() == [-50.0]

    def test_the_extractor_applies_it_to_what_the_backend_returns(self, extractor, recording, tmp_path):
        class SizedBackend(FakeBackend):
            def infer(self, *args, **kwargs):
                return PoseFrames(keypoints=self.frames, frame_size=(300, 400))

        instance = extractor(SizedBackend(backend_frame([100.0, 500.0])))
        keypoints = instance.run(recording([media()]), tmp_path).keypoints
        assert keypoints.loc[keypoints.track_id == 1, "x"].isna().all()
        assert keypoints.loc[keypoints.track_id == 0, "x"].notna().all()


class TestNothingDetected:
    @pytest.fixture
    def result(self, extractor, recording, tmp_path):
        instance = extractor(FakeBackend(backend_frame([100.0], n_frames=0)))
        return instance.run(recording([media()]), tmp_path)

    def test_an_empty_keypoint_table_is_still_valid(self, result):
        validate_frame(result.keypoints, KEYPOINT_SCHEMA, "keypoints")
        assert result.keypoints.empty

    def test_it_fails_loudly_rather_than_returning_nothing(self, result):
        assert any(warning.severity == "fail" for warning in result.warnings)

    def test_no_video_at_all_is_an_error(self, extractor, recording, tmp_path):
        from macaw.exceptions import UnreadableMediaError

        instance = extractor()
        with pytest.raises(UnreadableMediaError):
            instance.run(recording([media(kind="audio")]), tmp_path)


class TestQualityWarnings:
    def test_a_track_that_is_mostly_missing_is_reported(self, extractor, recording, tmp_path):
        frames = backend_frame([100.0, 500.0])
        frames.loc[frames.frame > 3, "x"] = None
        instance = extractor(FakeBackend(frames))
        clicks(tmp_path, "room.mp4", 0.2, parent=(100.0, 200.0), child=(500.0, 200.0))
        result = instance.run(recording([media()]), tmp_path)
        assert any("detected in only" in warning.message for warning in result.warnings)

    def test_an_unmatched_track_is_reported(self, extractor, recording, tmp_path):
        instance = extractor(FakeBackend(backend_frame([100.0, 300.0, 500.0])))
        clicks(tmp_path, "room.mp4", 0.2, parent=(100.0, 200.0), child=(500.0, 200.0))
        result = instance.run(recording([media()]), tmp_path)
        assert result.keypoints.subject_id.isna().any()
        assert any("no subject_id" in warning.message for warning in result.warnings)


def spread_frame(*people):
    """(track, frames, cx, size) per person, joints spread so each body has a size."""
    rows = []
    for track, frames, cx, size in people:
        for frame in frames:
            for index, joint in enumerate(STANDARD_JOINTS):
                rows.append(
                    {
                        "frame": frame,
                        "joint": joint,
                        "x": cx + size * ((index % 5) - 2) / 5,
                        "y": 200.0 + size * ((index // 5) - 1.5) / 5,
                        "z": float("nan"),
                        "confidence": 0.9,
                        "track_id": track,
                        "units": "pixel",
                        "metric_scale_valid": None,
                    }
                )
    return coerce_frame(pd.DataFrame(rows), BACKEND_KEYPOINT_SCHEMA, "backend keypoints")


class TestSeveralPeopleOnAPersonalCamera:
    """A backend that returns everyone in view, on a camera pointed at one person."""

    FRAMES = spread_frame((1, range(N_FRAMES), 200.0, 100.0), (2, range(5, 9), 60.0, 160.0))

    def run(self, extractor, recording, tmp_path):
        instance = extractor(FakeBackend(self.FRAMES))
        return instance.run(recording([media("a.mp4", subject_id="parent")]), tmp_path)

    def test_without_a_click_nobody_is_named_and_a_person_is_asked(self, extractor, recording, tmp_path):
        result = self.run(extractor, recording, tmp_path)
        assert result.keypoints.subject_id.isna().all()
        owners = json.loads(result.artifacts["owners"].read_text(encoding="utf-8"))
        assert "which the setup reads for parent alone" in owners["a.mp4"]["doubt"]

    def test_the_person_clicked_carries_the_subject_and_the_other_does_not(self, extractor, recording, tmp_path):
        clicks(tmp_path, "a.mp4", 0.6, parent=(200.0, 200.0))
        keypoints = self.run(extractor, recording, tmp_path).keypoints
        assert not keypoints.dropna(subset=["subject_id"]).duplicated(["subject_id", "frame", "joint"]).any()
        assert set(keypoints.loc[keypoints.subject_id == "parent", "track_id"]) == {1}
        assert keypoints.loc[keypoints.track_id == 2, "subject_id"].isna().all()

    def test_a_click_on_the_other_one_names_them_instead(self, extractor, recording, tmp_path):
        clicks(tmp_path, "a.mp4", 0.6, parent=(60.0, 200.0))
        keypoints = self.run(extractor, recording, tmp_path).keypoints
        assert set(keypoints.loc[keypoints.subject_id == "parent", "track_id"]) == {2}

    def test_with_suggestions_the_one_most_in_view_and_largest_is_taken(self, extractor, recording, tmp_path):
        instance = extractor(FakeBackend(self.FRAMES), suggest_names=True)
        result = instance.run(recording([media("a.mp4", subject_id="parent")]), tmp_path)
        assert set(result.keypoints.loc[result.keypoints.subject_id == "parent", "track_id"]) == {1}
        owners = json.loads(result.artifacts["owners"].read_text(encoding="utf-8"))
        assert owners["a.mp4"]["doubt"] is None and owners["a.mp4"]["chosen_by"] == {"parent": "suggested for being most in view"}

    def test_with_suggestions_someone_else_in_view_as_long_waits_for_a_person(self, extractor, recording, tmp_path):
        frames = spread_frame((1, range(N_FRAMES), 200.0, 100.0), (2, range(5, 15), 500.0, 100.0))
        result = extractor(FakeBackend(frames), suggest_names=True).run(recording([media("a.mp4", subject_id="parent")]), tmp_path)
        owners = json.loads(result.artifacts["owners"].read_text(encoding="utf-8"))
        assert "tracks [2] are in view nearly as long" in owners["a.mp4"]["doubt"] and owners["a.mp4"]["spans"] == [[0.5, 1.4]]

    def test_the_clicks_are_part_of_the_cache_key(self, extractor, recording, tmp_path):
        clicks(tmp_path, "a.mp4", 0.6, parent=(200.0, 200.0))
        assert tmp_path / "names.json" in extractor().identify.inputs_for(recording([media("a.mp4", subject_id="parent")]))


class Looker:
    """A model that reads pictures: whoever is cut out of the red left half is the parent, of the blue right half the child."""

    def __init__(self, unsure=()):
        self.asked, self.unsure = 0, set(unsure)

    def model_version(self):
        return "vision@1"

    def complete(self, system, user, *, seed=None, images=()):
        import cv2

        self.asked += 1
        picture = cv2.imdecode(np.frombuffer(images[0], np.uint8), cv2.IMREAD_COLOR)
        blue, _, red = picture.reshape(-1, 3).mean(axis=0)
        who = "parent" if red > blue else "child"
        return SimpleNamespace(text=json.dumps({"person": True, "who": "unsure" if who in self.unsure else who}))


class TestNamingByLook:
    """A shared camera with no click, no other camera and no microphone of anyone's own."""

    FRAMES = spread_frame((1, range(N_FRAMES), 150.0, 100.0), (2, range(N_FRAMES), 450.0, 100.0))
    LOOKS = LookSection(model="vision", people={"parent": "an adult", "child": "a young child"})

    @pytest.fixture
    def run(self, extractor, recording, tmp_path, monkeypatch):
        import macaw.qc.look as look_module
        import macaw.qc.overlay as overlay_module

        picture = np.zeros((PICTURE[1], PICTURE[0], 3), np.uint8)
        picture[:, : PICTURE[0] // 2] = (40, 40, 220)
        picture[:, PICTURE[0] // 2 :] = (220, 40, 40)
        # each frame a little different, as frames of a video are
        monkeypatch.setattr(overlay_module, "read_frames", lambda path, frames: ((frame, picture + frame) for frame in sorted(frames)))

        def go(model, look=self.LOOKS):
            def reach(cfg):
                if isinstance(model, Exception):
                    raise model
                return model

            monkeypatch.setattr(look_module, "backend_for", reach)
            result = extractor(FakeBackend(self.FRAMES), look=look, suggest_names=True).run(recording([media()]), tmp_path)
            kept = tmp_path / "artifacts" / PoseExtractor.name
            kept.mkdir(parents=True, exist_ok=True)
            for name, path in result.artifacts.items():
                (kept / name).write_bytes(path.read_bytes())
            return result, json.loads(result.artifacts["owners"].read_text(encoding="utf-8"))["room.mp4"]

        return go

    def test_clear_answers_name_the_people(self, run):
        model = Looker()
        result, owner = run(model)
        assert result.keypoints.groupby("track_id")["subject_id"].first().to_dict() == {1: "parent", 2: "child"}
        assert owner["doubt"] is None and set(owner["chosen_by"].values()) == {"suggested by the picture-reading model"}
        assert model.asked == 10 and "look_names.json" in result.artifacts

    def test_answers_are_kept_and_not_asked_again(self, run):
        run(Looker())
        again = Looker()
        _, owner = run(again)
        assert again.asked == 0 and owner["doubt"] is None

    def test_a_person_the_model_cannot_tell_waits_for_a_click(self, run):
        _, owner = run(Looker(unsure={"child"}))
        assert owner["people"] == {"parent": [1]} and "did not clearly tell child" in owner["doubt"]

    def test_a_model_that_cannot_be_reached_leaves_the_camera_waiting(self, run):
        from macaw.exceptions import MacawError

        _, owner = run(MacawError("no server at 127.0.0.1:11434"))
        assert owner["people"] == {} and "could not be asked (no server" in owner["doubt"]

    def test_people_nobody_described_are_not_asked_about(self, run):
        model = Looker()
        _, owner = run(model, LookSection(model="vision", people={"parent": "an adult"}))
        assert model.asked == 0 and owner["people"] == {} and "nothing to suggest" in owner["doubt"]


class TestJoinedFragments:
    def test_continuity_is_the_persons_not_the_shortest_fragments(self, extractor, recording, tmp_path):
        frames = spread_frame(
            (1, range(0, 10), 200.0, 100.0),
            (2, range(10, N_FRAMES), 205.0, 100.0),
            (3, range(0, N_FRAMES), 600.0, 40.0),
        )
        clicks(tmp_path, "a.mp4", 0.2, parent=(200.0, 200.0))
        result = extractor(FakeBackend(frames)).run(recording([media("a.mp4", subject_id="parent")]), tmp_path)
        rows = result.rows
        continuity = rows[(rows.feature == "track_continuity") & (rows.subject_id == "parent")]
        assert float(continuity.value.iloc[0]) == 1.0
        assert set(result.keypoints.loc[result.keypoints.subject_id == "parent", "track_id"]) == {1, 2}

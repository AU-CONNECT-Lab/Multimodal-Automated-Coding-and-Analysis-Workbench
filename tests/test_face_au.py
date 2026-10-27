from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.schema import FaceSection
from macaw.extractors.video.face import au as au_module
from macaw.extractors.video.face.au import FaceExtractor, name_faces
from macaw.store.keypoints import write_keypoints
from macaw.store.schema import KEYPOINT_SCHEMA, LONG_TABLE_SCHEMA, coerce_frame, validate_frame

FPS = 10.0


def heads(people, n_frames=10, video="room.mp4"):
    """people: subject -> (nose x, nose y); a full keypoint row set is not needed to name a face."""
    rows = [
        {"recording_id": "r1", "subject_id": subject, "source": "pose.keypoints", "video": video, "frame": frame,
         "t": frame / FPS, "joint": joint, "x": x, "y": y, "z": float("nan"), "confidence": 0.9, "track_id": 0,
         "units": "pixel", "metric_scale_valid": None}
        for frame in range(n_frames) for subject, (x, y) in people.items() for joint in ("nose", "left_eye", "right_eye")
    ]
    return coerce_frame(pd.DataFrame(rows), KEYPOINT_SCHEMA, "keypoints")


def faces(boxes, n_frames=10, au12=0.5):
    """boxes: list of (x, y, width, height, score); the same faces in every frame."""
    return pd.DataFrame(
        [{"frame": frame, "face_x": x, "face_y": y, "face_width": w, "face_height": h, "face_score": score,
          "pitch": 1.0, "roll": 2.0, "yaw": 3.0, "gaze_pitch": -10.0, "gaze_yaw": 20.0, "AU01": 0.1, "AU12": au12,
          "x_0": x + 1.0, "y_0": y + 1.0}
         for frame in range(n_frames) for x, y, w, h, score in boxes]
    )


class TestNameFaces:
    def test_each_face_goes_to_the_person_whose_nose_is_in_it(self):
        named = name_faces(faces([(80, 80, 40, 40, 0.9), (380, 80, 40, 40, 0.8)]),
                           heads({"parent": (400, 100), "child": (100, 100)}), ["parent", "child"])
        assert named.groupby("subject_id")["face_x"].first().to_dict() == {"child": 80, "parent": 380}

    def test_a_face_no_one_is_in_stays_out(self):
        named = name_faces(faces([(700, 80, 40, 40, 0.9)]), heads({"parent": (100, 100)}), ["parent"])
        assert named.empty

    def test_two_faces_on_one_person_keep_the_most_confident(self):
        named = name_faces(faces([(80, 80, 40, 40, 0.6), (85, 85, 30, 30, 0.95)]), heads({"parent": (100, 100)}), ["parent"])
        assert named["face_score"].unique().tolist() == [0.95] and len(named) == 10

    def test_someone_not_measured_on_this_camera_is_not_named_here(self):
        named = name_faces(faces([(80, 80, 40, 40, 0.9)]), heads({"parent": (100, 100)}), ["child"])
        assert named.empty


class FakeFaceBackend:
    name, version = "fake", "1.0"

    def __init__(self, table):
        self.table = table
        self.calls = []

    def infer(self, video_path, **kwargs):
        self.calls.append((Path(video_path).name, kwargs["frames"]))
        return self.table


@pytest.fixture
def run(tmp_path, monkeypatch):
    def build(keypoints, table, files, landmarks=True, gaze=False):
        stored = tmp_path / "pose.keypoints.parquet"
        write_keypoints(keypoints, stored)
        backend = FakeFaceBackend(table)
        monkeypatch.setattr(au_module, "_keypoints_file", lambda config, rid: stored)
        monkeypatch.setattr(au_module, "resolve_backend", lambda name: backend)
        monkeypatch.setattr(au_module, "apply_prep_timing", lambda config, rec: SimpleNamespace(t_origin_s=0.0, duration_s=0.9))
        recording = SimpleNamespace(
            recording_id="r1", subjects=[SimpleNamespace(subject_id="parent"), SimpleNamespace(subject_id="child")],
            media=list(files), entities=lambda: {"group_id": "1", "session_id": None, "task": None, "run": None},
        )
        config = SimpleNamespace(face=FaceSection(landmarks=landmarks, gaze=gaze))
        return backend, FaceExtractor(config).run(recording, tmp_path)

    return build


def video(name, subject_id=None):
    return SimpleNamespace(path=Path(name), kind="video", subject_id=subject_id, fps=FPS, offset_s=0.0)


class TestExtractor:
    def test_rows_manifest_and_names(self, run):
        _, result = run(heads({"parent": (400, 100), "child": (100, 100)}),
                        faces([(80, 80, 40, 40, 0.9), (380, 80, 40, 40, 0.8)]), [video("room.mp4")])
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        assert {spec.name for spec in result.specs} == set(result.rows.feature)
        assert {"au01", "au12", "face_yaw_deg"} <= set(result.rows.feature)
        assert set(result.rows.subject_id) == {"parent", "child"}
        assert result.artifacts["landmarks.parquet"].is_file()
        assert any("adult" in warning.message for warning in result.warnings)

    def test_a_person_with_their_own_camera_is_read_there_only(self, run):
        keypoints = pd.concat([heads({"parent": (100, 100)}, video="parent.mp4"),
                               heads({"parent": (400, 100), "child": (100, 100)}, video="room.mp4")])
        files = [video("parent.mp4", "parent"), video("room.mp4")]
        backend, _ = run(keypoints, faces([(80, 80, 40, 40, 0.9)]), files)
        assert sorted(name for name, _ in backend.calls) == ["parent.mp4", "room.mp4"]

    def test_gaze_rows_only_when_asked_for(self, run):
        people, boxes = heads({"parent": (100, 100)}), faces([(80, 80, 40, 40, 0.9)])
        _, without = run(people, boxes, [video("room.mp4")])
        backend, with_gaze = run(people, boxes, [video("room.mp4")], gaze=True)
        assert not {"gaze_pitch_deg", "gaze_yaw_deg"} & set(without.rows.feature)
        assert with_gaze.rows.loc[with_gaze.rows.feature == "gaze_yaw_deg", "value"].unique().tolist() == [20.0]
        assert {"gaze_pitch_deg", "gaze_yaw_deg"} <= {spec.name for spec in with_gaze.specs}

    def test_a_person_whose_face_is_rarely_found_is_reported(self, run):
        _, result = run(heads({"parent": (100, 100), "child": (400, 100)}), faces([(80, 80, 40, 40, 0.9)]), [video("room.mp4")])
        assert any(w.subject_id == "child" and "face was found in 0 of the 10" in w.message for w in result.warnings)


def test_pyfeat_columns_become_the_backend_contract():
    from macaw.extractors.video.face.backends import FACE_COLUMNS, GAZE_COLUMNS, _rows

    found = pd.DataFrame({"frame": [0, 1], "FaceRectX": [1.0, 2.0], "FaceRectY": [3.0, 4.0], "FaceRectWidth": [5.0, 5.0],
                          "FaceRectHeight": [6.0, 6.0], "FaceScore": [0.9, None], "Pitch": [1.0, 1.0], "Roll": [0.0, 0.0],
                          "Yaw": [2.0, 2.0], "AU01": [0.2, 0.3], "x_0": [1.5, 2.5], "y_0": [3.5, 4.5], "Identity_1": [0.0, 0.0]})
    out = _rows(found, offset=100, landmarks=True)
    assert list(out.columns) == [*FACE_COLUMNS, "AU01", "x_0", "y_0"]
    assert out["frame"].tolist() == [100]
    assert out["yaw"].iloc[0] == pytest.approx(2.0 * 180.0 / 3.141592653589793)
    assert list(_rows(found, offset=0, landmarks=False).columns) == [*FACE_COLUMNS, "AU01"]
    found["gaze_pitch"], found["gaze_yaw"], found["gaze_angle"] = 0.5, -0.25, 0.6
    with_gaze = _rows(found, offset=0, landmarks=False, gaze=True)
    assert list(with_gaze.columns) == [*FACE_COLUMNS, *GAZE_COLUMNS, "AU01"]
    assert with_gaze["gaze_yaw"].iloc[0] == pytest.approx(-0.25 * 180.0 / 3.141592653589793)


def test_gaze_needs_a_backend_that_gives_it(monkeypatch):
    from macaw.exceptions import BackendNotAvailableError

    monkeypatch.setattr(au_module, "resolve_backend", lambda name: SimpleNamespace(name="plain", provides_gaze=False))
    with pytest.raises(BackendNotAvailableError, match="no gaze direction"):
        FaceExtractor(SimpleNamespace(face=FaceSection(gaze=True))).check_available()


class TestAgreement:
    def frames(self, values_a, values_b, fps=10.0):
        t = pd.Series(range(len(values_a))) / fps
        return pd.DataFrame({"t": t, "au12": values_a}), pd.DataFrame({"t": t, "au12": values_b})

    def test_two_calibrations_of_the_same_movement_agree(self):
        import numpy as np

        rise = np.repeat(np.arange(30) % 7, 10).astype(float)
        a, b = self.frames(rise, 0.1 + rise ** 2 / 50)
        assert au_module.au_agreement(a, b)["au12"] == pytest.approx(1.0)

    def test_unrelated_series_do_not(self):
        import numpy as np

        rng = np.random.default_rng(0)
        a, b = self.frames(rng.random(400), rng.random(400))
        assert abs(au_module.au_agreement(a, b)["au12"]) < 0.4

    def test_too_little_overlap_is_missing(self):
        a, b = self.frames([0.1, 0.2] * 20, [0.3, 0.1] * 20)
        assert pd.isna(au_module.au_agreement(a, b)["au12"])


@pytest.mark.parametrize(
    "face, message",
    [
        ({}, "set face.check_model"),
        ({"check_model": "v1"}, "already uses"),
        ({"backend": "openface", "check_backend": "openface"}, "already uses"),
        ({"backend": "mediapipe", "check_model": "v2"}, "mediapipe gives none"),
        ({"check_backend": "mediapipe"}, "mediapipe gives none"),
    ],
)
def test_the_check_must_be_another_model_or_backend_with_action_units(face, message):
    from macaw.exceptions import ConfigError

    extractor = au_module.FaceAUCheckExtractor(SimpleNamespace(face=FaceSection(**face)))
    with pytest.raises(ConfigError, match=message):
        extractor.check_available()


@pytest.mark.parametrize("face", [{"check_model": "v2"}, {"check_backend": "openface"}, {"backend": "openface", "check_backend": "pyfeat"}])
def test_another_model_or_another_backend_is_a_check(face):
    au_module.FaceAUCheckExtractor(SimpleNamespace(face=FaceSection(**face))).check_available()


class TestMediaPipeFace:
    def test_a_head_turned_to_its_own_right_has_a_positive_yaw_and_a_raised_one_a_positive_pitch(self):
        import numpy as np

        from macaw.extractors.video.face.backends import head_angles

        assert head_angles(np.eye(4)) == pytest.approx((0.0, 0.0, 0.0))
        turn = np.radians(30.0)
        about_y = np.array([[np.cos(turn), 0, np.sin(turn)], [0, 1, 0], [-np.sin(turn), 0, np.cos(turn)]])
        pitch, roll, yaw = head_angles(about_y.T)
        assert yaw == pytest.approx(30.0) and pitch == pytest.approx(0.0, abs=1e-9)
        about_x = np.array([[1, 0, 0], [0, np.cos(turn), -np.sin(turn)], [0, np.sin(turn), np.cos(turn)]])
        assert head_angles(about_x)[0] == pytest.approx(-30.0)

    def test_the_mesh_becomes_68_points_in_pixels_and_a_box(self):
        from macaw.extractors.video.face.backends import FACE_COLUMNS, MEDIAPIPE_TO_68, mediapipe_face_rows

        points = [SimpleNamespace(x=index / 478.0, y=0.5) for index in range(478)]
        rows = mediapipe_face_rows([points], [np.eye(4)], frame_index=7, width=478, height=100, keep_landmarks=True)
        row = rows[0]
        assert set(FACE_COLUMNS) <= set(row) and row["frame"] == 7
        assert row["x_33"] == pytest.approx(MEDIAPIPE_TO_68[33]) and row["y_33"] == pytest.approx(50.0)
        assert row["face_width"] == pytest.approx(477.0) and "x_68" not in row
        assert "x_0" not in mediapipe_face_rows([points], [np.eye(4)], frame_index=7, width=478, height=100, keep_landmarks=False)[0]


def test_openface_csv_becomes_the_backend_contract():
    from macaw.extractors.video.face.backends import FACE_COLUMNS, GAZE_COLUMNS, openface_rows

    row = {"frame": 1, "face_id": 0, "confidence": 0.9, "success": 1, "pose_Rx": 0.1, "pose_Ry": 0.2, "pose_Rz": 0.3,
           "gaze_angle_x": 0.05, "gaze_angle_y": -0.1, "AU12_r": 2.5, "AU45_r": 1.0}
    row.update({f"x_{i}": 10.0 + i for i in range(68)})
    row.update({f"y_{i}": 20.0 + i for i in range(68)})
    found = pd.DataFrame([row, {**row, "frame": 2, "success": 0}])
    out = openface_rows(found, offset=100, landmarks=True, gaze=True, action_units={"AU12", "AU43"})
    assert out["frame"].tolist() == [100] and set(FACE_COLUMNS) | set(GAZE_COLUMNS) <= set(out.columns)
    assert out.at[0, "AU12"] == pytest.approx(0.5) and "AU45" not in out and "AU43" not in out
    assert out.at[0, "pitch"] == pytest.approx(-np.degrees(0.1)) and out.at[0, "yaw"] == pytest.approx(np.degrees(0.2))
    assert out.at[0, "gaze_pitch"] == pytest.approx(np.degrees(0.1)) and out.at[0, "face_width"] == pytest.approx(67.0)
    assert out.at[0, "x_67"] == pytest.approx(77.0)
    assert "x_0" not in openface_rows(found, offset=0, landmarks=False, gaze=False, action_units=set())


def test_openface_is_found_through_its_folder(tmp_path, monkeypatch):
    from macaw.exceptions import BackendNotAvailableError
    from macaw.extractors.video.face.backends import OPENFACE_ENV_VAR, OpenFaceBackend

    monkeypatch.setenv(OPENFACE_ENV_VAR, str(tmp_path))
    monkeypatch.setenv("PATH", "")
    with pytest.raises(BackendNotAvailableError, match="academic"):
        OpenFaceBackend().check_available()
    (tmp_path / "FaceLandmarkVidMulti.exe").write_bytes(b"")
    OpenFaceBackend().check_available()

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.qc import overlay as overlay_module
from macaw.qc.checks import CANDIDATE_SCHEMA
from macaw.qc.overlay import BASELINE, REVIEW_SCHEMA, draw_overlay, render_review, review_sample
from macaw.store.schema import KEYPOINT_SCHEMA, coerce_frame, empty_frame, validate_frame

RID = "sub-003_ses-12mo"


def candidates(rows):
    frame = pd.DataFrame(
        [
            {"recording_id": RID, "subject_id": "parent", "check": "track_quality", "reason": reason,
             "t_start": t, "t_end": t, "t": t, "score": score}
            for reason, t, score in rows
        ]
    )
    return coerce_frame(frame, CANDIDATE_SCHEMA, "candidates") if rows else empty_frame(CANDIDATE_SCHEMA)


def sample(found=(), *, max_candidates=40, n_baseline=25, seed=0, t_range=(0.0, 100.0)):
    return review_sample(
        candidates(list(found)), t_range, recording_id=RID,
        max_candidates=max_candidates, n_baseline=n_baseline, seed=seed,
    )


class TestReviewSample:
    def test_satisfies_its_schema(self):
        validate_frame(sample([("low_confidence", 3.0, 1.0)]), REVIEW_SCHEMA, "review sample")

    def test_the_baseline_has_one_frame_per_stratum(self):
        baseline = sample(n_baseline=10).t.to_numpy()
        assert len(baseline) == 10
        assert np.array_equal(np.floor(baseline / 10.0), np.arange(10))

    def test_the_baseline_ignores_what_the_checks_found(self):
        """Filtering the baseline by a check would hide exactly the confident failures."""
        alone = sample()
        alongside = sample([("low_confidence", 3.0, 9.0), ("bone_stretch", 50.0, 2.0)])
        assert np.allclose(alone.t, alongside[alongside.reason == BASELINE].t)

    def test_a_seed_fixes_the_baseline(self):
        assert np.allclose(sample(seed=7).t, sample(seed=7).t)
        assert not np.allclose(sample(seed=7).t, sample(seed=8).t)

    def test_reasons_take_turns_under_the_cap(self):
        """A noisy reason with many high scores cannot crowd out a rarer one."""
        found = [("low_confidence", float(t), 10.0 + t) for t in range(10)]
        found += [("bone_stretch", 50.0, 1.0), ("bone_stretch", 60.0, 0.5)]
        picked = sample(found, max_candidates=4, n_baseline=0)
        assert picked.reason.value_counts().to_dict() == {"low_confidence": 2, "bone_stretch": 2}
        assert sorted(picked[picked.reason == "low_confidence"].score) == [18.0, 19.0]

    def test_the_sample_is_in_time_order(self):
        picked = sample([("low_confidence", 70.0, 1.0), ("bone_stretch", 5.0, 1.0)])
        assert picked.t.is_monotonic_increasing

    def test_nothing_to_draw_is_an_empty_table(self):
        validate_frame(sample(n_baseline=0), REVIEW_SCHEMA, "review sample")
        assert sample(n_baseline=0).empty


cv2 = pytest.importorskip("cv2")

PARENT = (60, 180, 255)
UNASSIGNED = (170, 170, 170)


def body(subject_id, *, frame=5, centre=(100, 120), track_id=0, confidence=0.9, fps=10.0, video="room.avi"):
    offsets = {joint: (index % 3 * 10 - 10, index * 5 - 40) for index, joint in enumerate(STANDARD_JOINTS)}
    rows = [
        {
            "recording_id": RID, "subject_id": subject_id, "source": "pose.keypoints", "video": video, "frame": frame,
            "t": frame / fps, "joint": joint, "x": float(centre[0] + dx), "y": float(centre[1] + dy),
            "z": float("nan"), "confidence": confidence, "track_id": track_id, "units": "pixel",
            "metric_scale_valid": None,
        }
        for joint, (dx, dy) in offsets.items()
    ]
    return coerce_frame(pd.DataFrame(rows), KEYPOINT_SCHEMA, "keypoints")


def nose_pixel(image, people):
    nose = people[people.joint == "nose"].iloc[0]
    return tuple(int(value) for value in image[int(nose.y), int(nose.x)])


class TestDrawOverlay:
    def test_a_person_is_drawn_in_their_colour(self):
        people = body("parent")
        image = draw_overlay(np.zeros((240, 320, 3), np.uint8), people, ["parent", "child"], [])
        assert nose_pixel(image, people) == PARENT

    def test_an_unassigned_detection_is_grey(self):
        people = body(None, track_id=3)
        image = draw_overlay(np.zeros((240, 320, 3), np.uint8), people, ["parent", "child"], [])
        assert nose_pixel(image, people) == UNASSIGNED

    def test_the_input_frame_is_left_alone(self):
        blank = np.zeros((240, 320, 3), np.uint8)
        draw_overlay(blank, body("parent"), ["parent"], ["header"])
        assert not blank.any()

    def test_unusable_joints_are_not_drawn(self):
        people = body("parent")
        people.loc[people.joint == "nose", ["x", "y"]] = np.nan
        image = draw_overlay(np.zeros((240, 320, 3), np.uint8), people, ["parent"], [])
        assert image.any()

    def test_a_person_with_no_usable_joint_draws_nothing(self):
        people = body("parent").assign(x=np.nan, y=np.nan)
        assert not draw_overlay(np.zeros((240, 320, 3), np.uint8), people, ["parent"], []).any()

    def test_the_header_is_light_on_dark_even_on_a_white_frame(self):
        header = ["closeup  t=3.06s  frame 76  closeup.avi"]
        image = draw_overlay(np.full((288, 352, 3), 255, np.uint8), body("parent").iloc[:0], [], header)
        band = image[:30].min(axis=2)
        rows, cols = np.where(band < 128)
        backing = band[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        assert (backing > 200).mean() < 0.3

    def test_a_label_at_the_left_edge_stays_whole(self):
        def label_ink(centre_x):
            people = body("parent", centre=(centre_x, 120))
            image = draw_overlay(np.zeros((240, 320, 3), np.uint8), people, ["parent"], [])
            return int((image[:70] > 0).any(axis=2).sum())

        assert label_ink(-20) == pytest.approx(label_ink(150), rel=0.05)

    def test_the_label_confidence_ignores_joints_never_seen(self):
        """Masking joints inside the box changes the skeleton but must not change the label."""
        seen = body("parent")
        masked = seen.copy()
        hidden = masked.joint.isin(STANDARD_JOINTS[1::3])
        masked.loc[hidden, ["x", "y"]] = np.nan
        masked.loc[hidden, "confidence"] = 0.0
        top = int(seen.y.min()) - 4
        a = draw_overlay(np.zeros((240, 320, 3), np.uint8), seen, ["parent"], [])
        b = draw_overlay(np.zeros((240, 320, 3), np.uint8), masked, ["parent"], [])
        assert np.array_equal(a[:top], b[:top])


def write_video(path: Path, n_frames=10, fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    for _ in range(n_frames):
        writer.write(np.zeros((240, 320, 3), np.uint8))
    writer.release()
    return path


def media(path, subject_id=None, offset_s=0.0):
    return SimpleNamespace(path=path, kind="video", subject_id=subject_id, fps=10.0, offset_s=offset_s)


def recording(files):
    return SimpleNamespace(
        recording_id=RID,
        subjects=[SimpleNamespace(subject_id="parent"), SimpleNamespace(subject_id="child")],
        media=list(files),
    )


def moments(rows):
    return coerce_frame(
        pd.DataFrame(
            [{"recording_id": RID, "subject_id": s, "check": c, "reason": r, "t": t, "score": 1.0} for s, c, r, t in rows]
        ),
        REVIEW_SCHEMA,
        "review sample",
    )


class TestRenderReview:
    def test_each_moment_becomes_an_image_with_its_reason(self, tmp_path):
        video = write_video(tmp_path / "room.avi")
        keypoints = pd.concat([body("parent"), body("child", centre=(220, 120))])
        index = render_review(
            recording([media(video)]),
            keypoints,
            moments([(None, None, BASELINE, 0.2), ("parent", "track_quality", "low_confidence", 0.5)]),
            tmp_path / "renders",
        )
        assert list(index.reason) == [BASELINE, "low_confidence"]
        assert list(index.frame) == [2, 5]
        assert all((tmp_path / "renders" / name).is_file() for name in index.file)
        assert (tmp_path / "renders" / "index.csv").is_file()

    def test_the_skeleton_lands_on_the_right_frame(self, tmp_path):
        video = write_video(tmp_path / "room.avi")
        index = render_review(
            recording([media(video)]), body("parent", frame=5),
            moments([(None, None, BASELINE, 0.2), (None, None, BASELINE, 0.5)]), tmp_path / "renders",
        )
        drawn = [cv2.imread(str(tmp_path / "renders" / name))[60:, :] for name in index.file]
        # the header text is above row 60; below it only a skeleton can light a pixel
        assert not drawn[0].any()
        assert drawn[1].any()

    def test_the_file_offset_is_undone(self, tmp_path):
        """t is on the recording clock; the frame number is on the file's own."""
        video = write_video(tmp_path / "room.avi")
        index = render_review(
            recording([media(video, offset_s=0.3)]), body("parent"),
            moments([(None, None, BASELINE, 0.8)]), tmp_path / "renders",
        )
        assert list(index.frame) == [5]

    def test_a_moment_outside_the_file_is_skipped(self, tmp_path):
        video = write_video(tmp_path / "room.avi")
        index = render_review(
            recording([media(video)]), body("parent"),
            moments([(None, None, BASELINE, 5.0)]), tmp_path / "renders",
        )
        assert index.empty

    def test_a_run_that_found_nobody_still_renders_its_frames(self, tmp_path):
        video = write_video(tmp_path / "room.avi")
        index = render_review(
            recording([media(video)]), empty_frame(KEYPOINT_SCHEMA),
            moments([(None, None, BASELINE, 0.2)]), tmp_path / "renders",
        )
        assert list(index.frame) == [2]
        assert (tmp_path / "renders" / index.file.iloc[0]).is_file()

    def test_a_subject_is_seen_from_their_own_camera(self, tmp_path):
        a = write_video(tmp_path / "parent.avi")
        b = write_video(tmp_path / "child.avi")
        files = [media(a, "parent"), media(b, "child")]
        keypoints = pd.concat([body("parent", video="parent.avi"), body("child", video="child.avi")])
        index = render_review(
            recording(files), keypoints,
            moments([("child", "track_quality", "low_confidence", 0.5), (None, None, BASELINE, 0.5)]),
            tmp_path / "renders",
        )
        flagged = index[index.reason == "low_confidence"]
        assert list(flagged.video) == ["child.avi"]
        assert sorted(index[index.reason == BASELINE].video) == ["child.avi", "parent.avi"]

    def test_a_personal_camera_shows_everyone_it_saw_and_nobody_it_did_not(self, tmp_path, monkeypatch):
        """Others in view carry no subject; their track ids are what a person names in a review."""
        a = write_video(tmp_path / "parent.avi")
        b = write_video(tmp_path / "child.avi")
        drawn = {}

        def capture(image, people, subjects, header):
            drawn[header[0].split()[-1]] = people
            return image

        monkeypatch.setattr(overlay_module, "draw_overlay", capture)
        keypoints = pd.concat([
            body("parent", video="parent.avi"),
            body(None, track_id=7, video="parent.avi"),
            body("child", video="child.avi"),
        ])
        render_review(
            recording([media(a, "parent"), media(b, "child")]), keypoints,
            moments([(None, None, BASELINE, 0.5)]), tmp_path / "renders",
        )
        assert set(drawn["parent.avi"].track_id) == {0, 7}
        assert set(drawn["child.avi"].subject_id) == {"child"}

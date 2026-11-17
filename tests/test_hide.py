from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macaw.media.hide import blur_heads, head_circles, held_circles, not_hidden, video_shape, write_hidden
from macaw.media.ffmpeg import encode_frames
from macaw.media.probe import require_ffmpeg

from test_kinematics import person

cv2 = pytest.importorskip("cv2")
HEAD = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]


def has_ffmpeg() -> bool:
    try:
        require_ffmpeg()
    except Exception:
        return False
    return True


def noise(n_frames, width=160, height=120):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(n_frames)]


class TestHeadCircles:
    def test_a_circle_takes_in_every_head_point(self):
        people = person("parent", centre=(300.0, 200.0), n_frames=3)
        circles = head_circles(people)
        assert circles["who"].tolist() == ["parent"] * 3 and circles["frame"].tolist() == [0, 1, 2]
        first, points = circles.iloc[0], people[(people.frame == 0) & people.joint.isin(HEAD)]
        assert (np.hypot(points.x - first.x, points.y - first.y) < first.r).all()

    def test_someone_nobody_named_is_hidden_as_their_track(self):
        stray = person("parent", n_frames=2).assign(subject_id=pd.NA, track_id=3)
        assert set(head_circles(stray)["who"]) == {"track 3"}

    def test_a_head_without_points_sits_above_the_shoulders(self):
        people = person("parent", centre=(300.0, 200.0), n_frames=1)
        people.loc[people.joint.isin(HEAD), ["x", "y"]] = np.nan
        shoulders = people[people.joint.isin(["left_shoulder", "right_shoulder"])]
        circle = head_circles(people).iloc[0]
        assert circle.x == pytest.approx(shoulders.x.mean()) and circle.y < shoulders.y.min() and circle.r > 0

    def test_nobody_found_gives_no_circle(self):
        people = person("parent", n_frames=1)
        people[["x", "y"]] = np.nan
        assert head_circles(people).empty


class TestHeld:
    def circles(self):
        return pd.DataFrame({"who": "parent", "frame": [10, 11, 50], "x": [100.0, 110.0, 500.0], "y": 50.0, "r": 20.0})

    def test_a_head_stays_hidden_a_second_around_each_frame_it_was_found_in(self):
        held = held_circles(self.circles(), n_frames=55, fps=10.0)
        assert held["frame"].tolist() == [*range(0, 22), *range(40, 55)]
        assert held.loc[held.frame == 5, "x"].item() == 100.0
        assert held.loc[held.frame == 21, "x"].item() == pytest.approx(110.0 + 10 * 390.0 / 39)

    def test_the_stretches_without_a_circle_are_listed_for_each_named_person(self):
        held = held_circles(self.circles(), n_frames=55, fps=10.0)
        gaps = not_hidden(held, ["parent", "child"], n_frames=55, fps=10.0)
        assert gaps.values.tolist() == [["parent", 2.2, 4.0], ["child", 0.0, 5.5]]


def test_a_blurred_head_is_flat_and_the_rest_is_untouched():
    image = noise(1)[0]
    before = image.copy()
    blur_heads(image, [(80.0, 60.0, 20.0)])
    from_centre = np.hypot(*np.meshgrid(np.arange(160) - 80, np.arange(120) - 60))
    assert image[from_centre < 18].std() < 0.3 * before[from_centre < 18].std()
    assert (image[from_centre > 22] == before[from_centre > 22]).all()


@pytest.mark.skipif(not has_ffmpeg(), reason="needs ffmpeg")
class TestWritten:
    @pytest.fixture
    def source(self, tmp_path):
        return encode_frames(iter(noise(30)), tmp_path / "front.mp4", size=(160, 120), fps=30.0)

    def people(self):
        return person("parent", centre=(80.0, 80.0), scale=0.25, n_frames=20)

    def frame(self, path, index):
        capture = cv2.VideoCapture(str(path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        capture.release()
        assert ok
        return image

    def test_the_head_is_blurred_through_the_copy_and_the_room_stays(self, source, tmp_path):
        gaps = write_hidden(source, tmp_path / "out.mp4", self.people(), ["parent"], hide="faces", sound=False)
        assert video_shape(tmp_path / "out.mp4")[:3] == (160, 120, 30)
        circle = head_circles(self.people()).iloc[0]
        grid = np.hypot(*np.meshgrid(np.arange(160) - circle.x, np.arange(120) - circle.y))
        # the last frame is past the frames the person was found in, and within the second a head stays hidden
        for index in (5, 29):
            original, hidden = self.frame(source, index), self.frame(tmp_path / "out.mp4", index)
            assert hidden[grid < 0.7 * circle.r].std() < 0.4 * original[grid < 0.7 * circle.r].std()
            assert hidden[grid > 1.5 * circle.r].std() > 0.7 * original[grid > 1.5 * circle.r].std()
        assert gaps.empty

    def test_only_the_skeletons_are_left_on_a_plain_ground(self, source, tmp_path):
        gaps = write_hidden(source, tmp_path / "out.mp4", self.people(), ["parent"], hide="everything", sound=False)
        drawn, empty = self.frame(tmp_path / "out.mp4", 5), self.frame(tmp_path / "out.mp4", 25)
        assert abs(float(np.median(drawn)) - 128) < 8 and drawn.std() > 3 * empty.std()
        assert abs(float(empty.mean()) - 128) < 8 and empty.std() < 3
        assert gaps.empty

    def test_asking_for_the_sound_of_a_silent_video_still_writes_it(self, source, tmp_path):
        write_hidden(source, tmp_path / "out.mp4", self.people(), ["parent"], hide="faces", sound=True)
        assert (tmp_path / "out.mp4").is_file()

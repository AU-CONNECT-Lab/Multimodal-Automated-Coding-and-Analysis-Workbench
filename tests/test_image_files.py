from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

WIDE = "项目_ü"


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / WIDE
    path.mkdir()
    return path


def test_review_frames_are_written_into_a_folder_named_with_letters_beyond_ascii(folder):
    from macaw.media.images import write_image

    write_image(folder / "a.jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert (folder / "a.jpg").read_bytes()[:2] == b"\xff\xd8"


def test_a_depth_image_reads_back_from_a_folder_named_with_letters_beyond_ascii(folder):
    from macaw.extractors.video.pose.depth import read_png
    from macaw.media.images import write_image

    depth = np.arange(12, dtype=np.uint16).reshape(3, 4) * 1000
    write_image(folder / "000000.png", depth)
    assert np.array_equal(read_png(folder / "000000.png"), depth)


def test_an_imported_depth_recording_lands_in_a_folder_named_with_letters_beyond_ascii(folder):
    from macaw.extractors.video.pose.cameras import Camera
    from macaw.extractors.video.pose.depth import read_png
    from macaw.media.depth_import import Shot, Source, write_folder

    colour = Camera("colour", (64, 48), np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]]), np.zeros(5), np.zeros(3), np.zeros(3))
    picture = np.zeros((48, 64, 3), np.uint8)
    shots = (Shot(t, picture, np.full((48, 64), 1000 + i, np.uint16)) for i, t in enumerate([0.0, 0.1, 0.2]))
    write_folder(Source(colour, 0.001, 10.0, shots, []), folder / "cam.mp4", folder)
    assert [int(read_png(folder / f"{index:06d}.png").max()) for index in range(3)] == [1000, 1001, 1002]

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

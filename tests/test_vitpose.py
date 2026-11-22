from __future__ import annotations

import numpy as np

from macaw.extractors.video.pose.backends import COCO_KEYPOINT_ORDER, ViTPoseBackend, vitpose_rows


def person(x0: float) -> dict:
    return {"keypoints": np.array([[x0 + i, 10.0 + i] for i in range(17)]), "scores": np.linspace(0.1, 0.9, 17)}


def test_each_person_gives_the_17_coco_points_in_pixels():
    rows = vitpose_rows([person(100.0), person(500.0)], [0.9, 0.6], frame_index=42)
    assert len(rows) == 34 and {row["frame"] for row in rows} == {42}
    first = [row for row in rows if row["track_id"] == 0]
    assert [row["joint"] for row in first] == list(COCO_KEYPOINT_ORDER)
    assert (first[0]["x"], first[0]["y"], first[0]["units"]) == (100.0, 10.0, "pixel")
    assert all(np.isnan(row["z"]) for row in rows)
    assert {row["box_score"] for row in rows if row["track_id"] == 1} == {0.6}


def test_the_slot_is_no_track():
    assert ViTPoseBackend.tracks is False and ViTPoseBackend.default_model == "usyd-community/vitpose-plus-base"

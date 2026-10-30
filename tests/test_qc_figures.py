from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macaw.qc.checks import CANDIDATE_SCHEMA
from macaw.qc.figures import motion_figure, overview_figure, pitch_figure, present_spans, timeline_figure
from macaw.store.schema import coerce_frame

pytest.importorskip("matplotlib")

# the column types candidates come back from disk with, where a span of nobody's is NA
CANDIDATES = coerce_frame(
    pd.DataFrame(
        {"recording_id": "r", "subject_id": ["parent", None], "check": "c", "reason": ["octave_error", "track_swap"],
         "t_start": [2.0, 5.0], "t_end": [3.0, 5.0], "t": [2.5, 5.0], "score": 1.0}
    ),
    CANDIDATE_SCHEMA,
    "candidates",
)


class TestPresentSpans:
    def test_nearby_times_make_one_run(self):
        assert present_spans(np.array([0.1, 0.3, 0.5]), step=0.5) == [(0.0, 1.0)]

    def test_a_gap_of_a_bin_splits_the_run(self):
        assert present_spans(np.array([0.1, 1.2]), step=0.5) == [(0.0, 0.5), (1.0, 0.5)]

    def test_nothing_seen_draws_nothing(self):
        assert present_spans(np.empty(0)) == []


class TestFigures:
    def test_the_timeline_is_written_even_for_a_person_never_seen(self, tmp_path):
        seen = {"parent": present_spans(np.arange(0, 10, 0.04))}
        heard = {"parent": [(1.0, 3.0)], "child": [(5.0, 1.0)]}
        path = timeline_figure(["parent", "child"], seen, heard, "speaking", CANDIDATES, tmp_path / "t.png")
        assert path.stat().st_size > 0

    def test_a_voice_that_is_no_ones_gets_a_heard_lane_only(self, tmp_path, monkeypatch):
        from macaw.qc import figures

        drawn = []
        monkeypatch.setattr(figures, "_save", lambda figure, path: drawn.append(figure) or path)
        seen = {"parent": present_spans(np.arange(0, 10, 0.04))}
        heard = {"parent": [(1.0, 3.0)], "other": [(4.0, 2.0)], "none": []}
        timeline_figure(["parent", "child"], seen, heard, "speaking", CANDIDATES, tmp_path / "t.png")
        lanes = [label.get_text() for label in drawn[0].axes[0].get_yticklabels()]
        assert lanes == ["parent  seen", "parent  heard", "child  seen", "child  heard", "other  heard"]

    def test_the_pitch_panels_cover_a_person_without_pitch(self, tmp_path):
        pitch = {"parent": (np.arange(0, 10, 0.01), np.full(1000, 220.0))}
        path = pitch_figure(["parent", "child"], pitch, CANDIDATES, tmp_path / "p.png")
        assert path.stat().st_size > 0

    def test_the_motion_panels_shade_spans_of_nobody_too(self, tmp_path):
        energy = {"parent": (np.arange(0, 10, 0.04), np.full(250, 0.02))}
        path = motion_figure(["parent", "child"], energy, CANDIDATES, tmp_path / "m.png")
        assert path.stat().st_size > 0


class TestOverview:
    def draw(self, tmp_path, span, name="o.png"):
        t = np.arange(0.0, span[1], 0.5)
        lanes = [
            ("interpersonal_distance", [("parent to child", "parent", t, 100 + np.sin(t))], False),
            ("left_wrist_speed", [("parent", "parent", t, np.cos(t)), ("child", "child", t, np.sin(t))], False),
            ("not_made", [], False),
        ]
        codes = pd.DataFrame(
            {"tier": ["backchannel", "speech act"], "subject_id": ["child", None], "t_start": [10.0, 150.0], "t_end": [10.5, 152.0]}
        )
        talk = {"parent": [(0.0, 5.0)], "child": [(120.0, 4.0)], "other": [(60.0, 2.0)]}
        return overview_figure(["parent", "child"], talk, lanes, codes, CANDIDATES, span, tmp_path / name)

    def test_it_is_drawn_at_report_resolution(self, tmp_path):
        from matplotlib.image import imread

        assert imread(self.draw(tmp_path, (0.0, 200.0))).shape[1] == 12 * 300

    def test_a_recording_longer_than_a_row_wraps_onto_more_rows(self, tmp_path):
        from matplotlib.image import imread

        one = imread(self.draw(tmp_path, (0.0, 200.0), "one.png")).shape[0]
        two = imread(self.draw(tmp_path, (0.0, 400.0), "two.png")).shape[0]
        assert two > 1.8 * one

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macaw.qc.figures import pitch_figure, present_spans, timeline_figure

pytest.importorskip("matplotlib")

CANDIDATES = pd.DataFrame(
    {"subject_id": ["parent", None], "t_start": [2.0, 5.0], "t_end": [3.0, 5.0], "reason": ["octave_error", "track_swap"]}
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

    def test_the_pitch_panels_cover_a_person_without_pitch(self, tmp_path):
        pitch = {"parent": (np.arange(0, 10, 0.01), np.full(1000, 220.0))}
        path = pitch_figure(["parent", "child"], pitch, CANDIDATES[:1], tmp_path / "p.png")
        assert path.stat().st_size > 0

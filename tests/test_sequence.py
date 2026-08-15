from __future__ import annotations

import pandas as pd
import pytest

from macaw.store.sequence import (
    EMPTY_STATE,
    SEQUENCE_COLUMNS,
    export_state_sequence,
    to_joint_sequence,
    to_state_sequence,
)


@pytest.fixture
def make_intervals():
    def build(rows, recording_id="rec1"):
        return pd.DataFrame(
            [
                {
                    "recording_id": recording_id,
                    "subject_id": subject,
                    "tier": tier,
                    "t_start": float(start),
                    "t_end": float(end),
                    "label": label,
                }
                for subject, tier, start, end, label in rows
            ]
        )

    return build


class TestSampling:
    def test_columns_and_step(self, make_intervals):
        frame = make_intervals([("parent", "act", 0.0, 2.0, "praise")])
        sequence = to_state_sequence(frame, step_s=1.0)
        assert list(sequence.columns) == list(SEQUENCE_COLUMNS)
        assert list(sequence["t"]) == [0.0, 1.0]
        assert set(sequence["state"]) == {"praise"}

    def test_a_gap_becomes_the_empty_state_not_a_missing_row(self, make_intervals):
        """A sequence with holes is not a sequence; recurrence analysis needs every step."""
        frame = make_intervals(
            [("parent", "act", 0.0, 1.0, "praise"), ("parent", "act", 3.0, 4.0, "directive")]
        )
        sequence = to_state_sequence(frame, step_s=1.0)
        assert list(sequence["state"]) == ["praise", EMPTY_STATE, EMPTY_STATE, "directive"]

    def test_the_cell_goes_to_whoever_covers_most_of_it(self, make_intervals):
        frame = make_intervals(
            [("parent", "act", 0.0, 0.3, "praise"), ("parent", "act", 0.3, 1.0, "directive")]
        )
        assert list(to_state_sequence(frame, step_s=1.0)["state"]) == ["directive"]

    def test_an_even_split_goes_alphabetically(self, make_intervals):
        frame = make_intervals(
            [("parent", "act", 0.0, 0.5, "zeta"), ("parent", "act", 0.5, 1.0, "alpha")]
        )
        assert list(to_state_sequence(frame, step_s=1.0)["state"]) == ["alpha"]

    def test_one_series_per_subject_and_tier(self, make_intervals):
        frame = make_intervals(
            [("parent", "act", 0.0, 1.0, "praise"), ("child", "act", 0.0, 1.0, "reply")]
        )
        sequence = to_state_sequence(frame, step_s=1.0)
        assert sorted(sequence["series"].unique()) == ["child|act", "parent|act"]

    def test_every_series_covers_the_same_grid(self, make_intervals):
        """Two sequences that do not line up cannot be compared to each other."""
        frame = make_intervals(
            [("parent", "act", 0.0, 1.0, "praise"), ("child", "act", 2.0, 3.0, "reply")]
        )
        sequence = to_state_sequence(frame, step_s=1.0)
        counts = sequence.groupby("series").size()
        assert set(counts) == {3}

    def test_an_explicit_window_wins_over_the_data(self, make_intervals):
        frame = make_intervals([("parent", "act", 1.0, 2.0, "praise")])
        sequence = to_state_sequence(frame, step_s=1.0, t_start=0.0, t_end=4.0)
        assert list(sequence["t"]) == [0.0, 1.0, 2.0, 3.0]
        assert sequence["state"].tolist() == [EMPTY_STATE, "praise", EMPTY_STATE, EMPTY_STATE]

    def test_the_speaker_can_be_the_state(self, make_intervals):
        """Turns resampled this way give the floor sequence, which is what the source is."""
        frame = make_intervals(
            [("parent", "turn", 0.0, 1.0, "x"), ("child", "turn", 1.0, 2.0, "x")]
        )
        sequence = to_state_sequence(
            frame, step_s=1.0, state_column="subject_id", series_columns=()
        )
        assert list(sequence["series"]) == ["all", "all"]
        assert list(sequence["state"]) == ["parent", "child"]

    def test_recordings_stay_apart(self, make_intervals):
        frame = pd.concat(
            [
                make_intervals([("parent", "act", 0.0, 1.0, "a")], recording_id="rec1"),
                make_intervals([("parent", "act", 0.0, 1.0, "b")], recording_id="rec2"),
            ],
            ignore_index=True,
        )
        sequence = to_state_sequence(frame, step_s=1.0)
        assert sorted(sequence["recording_id"].unique()) == ["rec1", "rec2"]

    def test_an_unlabelled_interval_is_dropped(self, make_intervals):
        frame = make_intervals([("parent", "act", 0.0, 1.0, None)])
        assert to_state_sequence(frame, step_s=1.0).empty

    def test_a_zero_width_interval_is_dropped(self, make_intervals):
        frame = make_intervals([("parent", "act", 1.0, 1.0, "praise")])
        assert to_state_sequence(frame, step_s=1.0).empty

    def test_an_empty_input_returns_an_empty_frame(self):
        sequence = to_state_sequence(pd.DataFrame(), step_s=1.0)
        assert sequence.empty
        assert list(sequence.columns) == list(SEQUENCE_COLUMNS)

    def test_a_non_positive_step_is_a_programming_error(self, make_intervals):
        with pytest.raises(ValueError):
            to_state_sequence(make_intervals([("parent", "act", 0.0, 1.0, "a")]), step_s=0.0)


class TestJointSequence:
    def test_two_series_concatenate_into_one_state(self, make_intervals):
        frame = make_intervals(
            [
                ("A", "floor", 0.0, 1.0, "A"),
                ("B", "floor", 0.0, 2.0, "B"),
                ("A", "floor", 2.0, 3.0, "A"),
            ]
        )
        joint = to_joint_sequence(to_state_sequence(frame, step_s=1.0))
        assert list(joint["state"]) == ["AB", "nonB", "Anone"]

    def test_the_series_order_is_the_same_at_every_step(self, make_intervals):
        frame = make_intervals(
            [("zeta", "f", 0.0, 2.0, "z"), ("alpha", "f", 1.0, 2.0, "a")]
        )
        joint = to_joint_sequence(to_state_sequence(frame, step_s=1.0))
        assert list(joint["state"]) == [f"{EMPTY_STATE}z", "az"]

    def test_a_separator_can_be_asked_for(self, make_intervals):
        frame = make_intervals([("A", "f", 0.0, 1.0, "A"), ("B", "f", 0.0, 1.0, "B")])
        joint = to_joint_sequence(to_state_sequence(frame, step_s=1.0), separator="-")
        assert list(joint["state"]) == ["A-B"]

    def test_an_empty_sequence_stays_empty(self):
        assert to_joint_sequence(pd.DataFrame()).empty


class TestExport:
    def test_csv_is_written_and_readable(self, make_intervals, tmp_path):
        sequence = to_state_sequence(make_intervals([("parent", "act", 0.0, 2.0, "a")]), step_s=1.0)
        path = export_state_sequence(sequence, tmp_path / "out" / "seq.csv")
        assert path.is_file()
        assert len(pd.read_csv(path)) == 2

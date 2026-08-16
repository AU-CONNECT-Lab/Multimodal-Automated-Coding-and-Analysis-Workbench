from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.extractors.video.pose.track_repair import (
    SwapEvent,
    assign_subjects,
    detect_swaps,
    repair_tracks,
    track_quality,
)

FPS = 10.0


def scene(positions, *, n_frames=10, swap_at=None, confidence=0.9):
    """positions is one x per track; swap_at is the frame where they trade places, which is
    what a track swap looks like from the outside."""
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
                        "t": frame / FPS,
                        "track_id": track,
                        "joint": joint,
                        "x": float(x),
                        "y": 200.0,
                        "confidence": confidence,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def crossing():
    return scene([100.0, 500.0], swap_at=5)


class TestDetectSwaps:
    def test_two_people_holding_still_never_swap(self):
        assert detect_swaps(scene([100.0, 500.0]), max_gap_s=1.0) == []

    def test_a_crossover_is_found_at_the_frame_it_happens(self, crossing):
        events = detect_swaps(crossing, max_gap_s=1.0)
        assert len(events) == 1
        assert events[0].frame == 5
        assert {events[0].track_a, events[0].track_b} == {0, 1}

    def test_the_score_says_how_much_better_the_swapped_pairing_was(self, crossing):
        """A clean exchange leaves the straight pairing with all of the movement."""
        assert detect_swaps(crossing, max_gap_s=1.0)[0].score == pytest.approx(1.0)

    def test_a_track_that_was_gone_too_long_is_a_new_track_not_a_swap(self):
        gapped = scene([100.0, 500.0], swap_at=5)
        gapped = gapped[(gapped.frame < 3) | (gapped.frame > 6)]
        assert len(detect_swaps(gapped, max_gap_s=1.0)) == 1
        assert detect_swaps(gapped, max_gap_s=0.05) == []

    def test_two_tracks_in_the_same_place_cannot_be_told_apart(self):
        """Calling that a swap would be relabelling noise."""
        assert detect_swaps(scene([300.0, 300.0], swap_at=5), max_gap_s=1.0) == []

    def test_one_track_alone_has_nothing_to_swap_with(self):
        assert detect_swaps(scene([100.0]), max_gap_s=1.0) == []

    def test_a_track_swaps_with_at_most_one_partner_per_frame(self):
        three = scene([100.0, 300.0, 500.0], swap_at=5)
        events = [event for event in detect_swaps(three, max_gap_s=1.0) if event.frame == 5]
        claimed = [track for event in events for track in (event.track_a, event.track_b)]
        assert len(claimed) == len(set(claimed))

    def test_empty_input(self):
        assert detect_swaps(scene([100.0]).iloc[0:0], max_gap_s=1.0) == []


class TestRepairTracks:
    def test_each_person_ends_up_on_one_id(self, crossing):
        repaired = repair_tracks(crossing, detect_swaps(crossing, max_gap_s=1.0))
        assert repaired.groupby("track_id")["x"].nunique().to_dict() == {0: 1, 1: 1}

    def test_nothing_moves_when_there_was_no_swap(self):
        clean = scene([100.0, 500.0])
        assert repair_tracks(clean, []).equals(clean)

    def test_a_second_crossing_puts_the_labels_back(self):
        """Swaps compose, so the mapping is rebuilt from the raw ids each time rather than
        applied on top of itself."""
        clean = scene([100.0, 500.0])
        repaired = repair_tracks(
            clean,
            [
                SwapEvent(t=0.3, frame=3, track_a=0, track_b=1, evidence="test", score=1.0),
                SwapEvent(t=0.7, frame=7, track_a=0, track_b=1, evidence="test", score=1.0),
            ],
        )
        before = repaired.frame < 3
        during = (repaired.frame >= 3) & (repaired.frame < 7)
        after = repaired.frame >= 7
        assert list(repaired[before].track_id) == list(clean[before.to_numpy()].track_id)
        assert list(repaired[during].track_id) == [1 - track for track in clean[during.to_numpy()].track_id]
        assert list(repaired[after].track_id) == list(clean[after.to_numpy()].track_id)

    def test_the_other_columns_are_untouched(self, crossing):
        repaired = repair_tracks(crossing, detect_swaps(crossing, max_gap_s=1.0))
        assert len(repaired) == len(crossing)
        assert list(repaired.frame) == list(crossing.frame)
        assert list(repaired.x) == list(crossing.x)


class TestAssignSubjects:
    def test_anchors_are_read_as_an_ordering(self):
        """They are configured as a fraction of frame width while x is in pixels, and this
        function is never told the width, so only the ordering survives."""
        clean = scene([100.0, 500.0])
        left_is_child = assign_subjects(clean, ["parent", "child"], identity_anchors={"child": 0.1, "parent": 0.9})
        assert left_is_child.groupby("subject_id")["x"].first().to_dict() == {"child": 100.0, "parent": 500.0}

        left_is_parent = assign_subjects(clean, ["parent", "child"], identity_anchors={"child": 0.9, "parent": 0.1})
        assert left_is_parent.groupby("subject_id")["x"].first().to_dict() == {"child": 500.0, "parent": 100.0}

    def test_without_anchors_the_configured_subject_order_is_used_left_to_right(self):
        """Inventing an alphabetical order would silently disagree with what was configured."""
        clean = scene([100.0, 500.0])
        assert assign_subjects(clean, ["parent", "child"]).groupby("subject_id")["x"].first().to_dict() == {
            "parent": 100.0,
            "child": 500.0,
        }
        assert assign_subjects(clean, ["child", "parent"]).groupby("subject_id")["x"].first().to_dict() == {
            "child": 100.0,
            "parent": 500.0,
        }

    def test_a_spurious_extra_track_is_left_unassigned(self):
        """Forcing every track onto a person is how a reflection becomes a third participant."""
        blip = scene([900.0], n_frames=2)
        blip["track_id"] = 7
        noisy = pd.concat([scene([100.0, 500.0]), blip], ignore_index=True)
        assigned = assign_subjects(noisy, ["parent", "child"])
        assert assigned[assigned.track_id == 7].subject_id.isna().all()
        assert set(assigned.subject_id.dropna()) == {"parent", "child"}

    def test_fewer_tracks_than_people_leaves_the_missing_one_out(self):
        assigned = assign_subjects(scene([100.0]), ["parent", "child"])
        assert set(assigned.subject_id.dropna()) == {"parent"}

    def test_no_subjects_and_no_tracks(self):
        assert assign_subjects(scene([100.0]), []).subject_id.isna().all()
        assert assign_subjects(scene([100.0]).iloc[0:0], ["parent"]).empty


class TestTrackQuality:
    @pytest.fixture
    def holey(self):
        frame = scene([100.0, 500.0])
        frame = frame[frame.frame != 4].copy()
        frame.loc[frame.joint == "nose", "x"] = np.nan
        return track_quality(frame)

    def test_one_row_per_track(self, holey):
        assert list(holey.track_id) == [0, 1]

    def test_continuity_counts_frames_seen_over_frames_spanned(self, holey):
        """A track that vanishes for half its life scores 0.5 even if every frame it has is
        perfect, which is the thing a mean confidence would hide."""
        assert holey.continuity.tolist() == [pytest.approx(0.9)] * 2

    def test_a_joint_below_threshold_counts_as_missing(self, holey):
        assert holey.missing_fraction.tolist() == [pytest.approx(1 / 17)] * 2

    def test_the_time_span_is_reported(self, holey):
        assert holey.t_start.tolist() == [0.0, 0.0]
        assert holey.t_end.tolist() == [pytest.approx(0.9)] * 2

    def test_frames_are_counted_not_rows(self, holey):
        assert holey.n_frames.tolist() == [9, 9]

    def test_empty_input_keeps_the_columns(self):
        empty = track_quality(scene([100.0]).iloc[0:0])
        assert empty.empty
        assert "continuity" in empty.columns

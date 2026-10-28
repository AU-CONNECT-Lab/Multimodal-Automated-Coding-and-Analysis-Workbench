from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from macaw.config.schema import TextSection, TurnsSection
from macaw.store.schema import LONG_TABLE_SCHEMA, validate_frame
from macaw.turns.transitions import TransitionExtractor, classify_overlaps, detect_transitions

# the smooth transfer sits inside smooth_transfer_gap_s rather than on it, because
# subtracting these times lands an ulp either side of the threshold
TURNS = [
    ("parent", 10.0, 13.0),
    ("child", 12.9, 13.0),    # short overlap, parent carries on -> backchannel
    ("parent", 13.1, 15.0),   # +0.1 gap, smooth
    ("child", 14.2, 16.0),    # -0.8 overlap -> interruption
    ("parent", 17.0, 18.0),   # +1.0 gap
    ("parent", 20.5, 21.0),   # +2.5 hold
]


@pytest.fixture
def cfg():
    return TurnsSection()


@pytest.fixture
def extractor(cfg):
    return TransitionExtractor(SimpleNamespace(turns=cfg, text=TextSection()))


def build_scene(make_turns, make_recording, make_prep, cfg, rows):
    turns = make_turns(rows)
    transitions = classify_overlaps(
        detect_transitions(turns, cfg), interruption_s=cfg.interruption_overlap_s
    )
    transitions["recording_id"] = "sub-003_ses-12mo"
    return SimpleNamespace(
        turns=turns,
        transitions=transitions,
        recording=make_recording(["parent", "child"]),
        prep=make_prep(t_origin_s=10.0, duration_s=60.0),
    )


@pytest.fixture
def scene(make_turns, make_recording, make_prep, cfg):
    return build_scene(make_turns, make_recording, make_prep, cfg, TURNS)


@pytest.fixture
def rows(extractor, scene):
    return extractor._rows(scene.recording, scene.turns, scene.transitions, scene.prep)


def select(rows, feature, subject=None, partner=None):
    subset = rows[rows.feature == feature]
    subset = subset[subset.subject_id.isna() if subject is None else subset.subject_id == subject]
    return subset[
        subset.subject_id_b.isna() if partner is None else subset.subject_id_b == partner
    ]


def value(rows, feature, subject=None, partner=None):
    return float(select(rows, feature, subject, partner).value.iloc[0])


def test_scene_is_what_we_think_it_is(scene):
    assert list(scene.transitions.transition_type) == [
        "backchannel",
        "gap",
        "interruption",
        "gap",
        "hold",
    ]


def test_rows_satisfy_the_schema(rows):
    validate_frame(rows, LONG_TABLE_SCHEMA, "long table")


def test_manifest_and_rows_agree(extractor, rows):
    """Every feature emitted is documented, and nothing is documented that is never emitted."""
    assert {spec.name for spec in extractor._specs()} == set(rows.feature)


class TestEventRows:
    def test_the_responder_owns_the_row(self, rows):
        taken = select(rows, "fto_s", "child", "parent")
        assert sorted(taken.value.round(2)) == [-0.8, -0.1]

    def test_an_overlap_spans_the_crossing_forwards(self, rows):
        """t_end below t_start would slip past every time filter; the sign lives in the value."""
        events = select(rows, "fto_s", "child", "parent")
        assert (events.t_end >= events.t_start).all()
        assert sorted((events.t_end - events.t_start).round(2)) == [0.1, 0.8]

    def test_latency_is_absent_on_overlaps(self, rows):
        assert select(rows, "response_latency_s", "child", "parent").value.isna().all()

    def test_latency_is_present_on_gaps(self, rows):
        waits = select(rows, "response_latency_s", "parent", "child")
        assert sorted(waits.value.round(2)) == [0.1, 1.0]


class TestSubjectRows:
    def test_counts_are_per_responder(self, rows):
        assert value(rows, "n_turns_taken_from_other", "parent") == 2
        assert value(rows, "n_interruptions_made", "child") == 1
        assert value(rows, "n_backchannels_made", "child") == 1
        assert value(rows, "n_interruptions_made", "parent") == 0

    def test_latency_summary_skips_overlaps(self, rows):
        assert value(rows, "median_response_latency_s", "parent") == pytest.approx(0.55)
        assert pd.isna(value(rows, "median_response_latency_s", "child"))
        assert value(rows, "mean_fto_s", "child") == pytest.approx(-0.45)

    def test_rate_uses_the_task_window(self, rows):
        assert value(rows, "interruptions_per_minute", "child") == pytest.approx(1.0)


class TestPairRows:
    def test_every_ordered_pair_gets_a_cell(self, rows):
        assert len(select(rows, "transition_count", "parent", "child")) == 1
        assert value(rows, "transition_count", "parent", "child") == 2
        assert value(rows, "transition_count", "child", "parent") == 2

    def test_the_diagonal_holds_the_restarts(self, rows):
        assert value(rows, "transition_count", "parent", "parent") == 1
        assert value(rows, "transition_count", "child", "child") == 0

    def test_pair_median_follows_the_same_direction(self, rows):
        assert value(rows, "pair_median_fto_s", "parent", "child") == pytest.approx(-0.45)
        assert value(rows, "pair_median_fto_s", "child", "parent") == pytest.approx(0.55)

    def test_counts_sum_to_the_number_of_transitions(self, rows):
        cells = rows[(rows.feature == "transition_count") & rows.subject_id_b.notna()]
        assert cells.value.sum() == value(rows, "n_transitions")


class TestRecordingRows:
    def test_summary_belongs_to_nobody(self, rows):
        for feature in ("n_switches", "gap_ratio", "median_hold_s"):
            subset = rows[rows.feature == feature]
            assert len(subset) == 1
            assert subset.subject_id.isna().all()
            assert subset.subject_id_b.isna().all()

    def test_values(self, rows):
        assert value(rows, "n_transitions") == 5
        assert value(rows, "n_switches") == 4
        assert value(rows, "n_holds") == 1
        assert value(rows, "gap_ratio") == pytest.approx(0.5)
        assert value(rows, "smooth_transfer_ratio") == pytest.approx(0.5)
        assert value(rows, "median_hold_s") == pytest.approx(2.5)


def test_entities_ride_along(rows):
    assert (rows.group_id == "003").all()
    assert (rows.session_id == "12mo").all()
    assert rows.task.isna().all()


def test_provenance_is_left_to_the_runner(rows):
    assert rows.execution_id.isna().all()
    assert rows.config_hash.isna().all()


class TestWarnings:
    def test_short_recording(self, extractor, scene):
        checks = {w.check for w in extractor._warnings(scene.recording, scene.transitions, scene.prep)}
        assert "transitions_too_few" in checks
        assert "transitions_never_responds" not in checks
        assert "transitions_overlap_heavy" not in checks

    def test_a_speaker_who_never_picks_up(
        self, extractor, make_turns, make_recording, make_prep, cfg
    ):
        alone = build_scene(
            make_turns, make_recording, make_prep, cfg, [row for row in TURNS if row[0] == "parent"]
        )
        checks = {w.check for w in extractor._warnings(alone.recording, alone.transitions, alone.prep)}
        assert "transitions_never_responds" in checks

    def test_overlap_heavy_points_at_crosstalk(
        self, extractor, make_turns, make_recording, make_prep, cfg
    ):
        leaky = build_scene(
            make_turns,
            make_recording,
            make_prep,
            cfg,
            [
                ("parent", 10.0, 13.0),
                ("child", 12.5, 14.0),
                ("parent", 13.5, 16.0),
                ("child", 15.5, 17.0),
                ("parent", 18.0, 19.0),
            ],
        )
        checks = {w.check for w in extractor._warnings(leaky.recording, leaky.transitions, leaky.prep)}
        assert "transitions_overlap_heavy" in checks

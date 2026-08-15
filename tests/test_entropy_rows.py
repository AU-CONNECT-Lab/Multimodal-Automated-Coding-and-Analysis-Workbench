from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from macaw.config.schema import TurnsSection
from macaw.store.schema import LONG_TABLE_SCHEMA, validate_frame
from macaw.turns.entropy import EntropyExtractor
from macaw.turns.transitions import classify_overlaps, detect_transitions

# parent and child alternating, with two parent restarts so the matrix has a diagonal
TURNS = [
    ("parent", 10.0, 13.0),
    ("child", 13.5, 15.0),
    ("parent", 16.0, 18.0),
    ("child", 18.4, 19.0),
    ("parent", 20.0, 23.0),
    ("parent", 26.0, 28.0),
    ("child", 29.0, 31.0),
    ("parent", 32.0, 35.0),
    ("child", 36.0, 37.0),
    ("parent", 38.0, 41.0),
    ("child", 44.0, 46.0),
    ("parent", 47.0, 50.0),
]


@pytest.fixture
def cfg():
    return TurnsSection()


@pytest.fixture
def windowed_cfg():
    return TurnsSection(window_s=20.0, window_hop_s=10.0)


def build_scene(make_turns, make_recording, make_prep, cfg, rows=TURNS):
    turns = make_turns(rows)
    transitions = classify_overlaps(
        detect_transitions(turns, cfg), interruption_s=cfg.interruption_overlap_s
    )
    return SimpleNamespace(
        turns=turns,
        transitions=transitions,
        recording=make_recording(["parent", "child"]),
        prep=make_prep(t_origin_s=10.0, duration_s=60.0),
    )


@pytest.fixture
def scene(make_turns, make_recording, make_prep, cfg):
    return build_scene(make_turns, make_recording, make_prep, cfg)


@pytest.fixture
def extractor(cfg):
    return EntropyExtractor(SimpleNamespace(turns=cfg))


@pytest.fixture
def rows(extractor, scene):
    return extractor._rows(scene.recording, scene.turns, scene.transitions, scene.prep)


def select(rows, feature, subject=None, scale="recording"):
    subset = rows[(rows.feature == feature) & (rows.scale == scale)]
    return subset[subset.subject_id.isna() if subject is None else subset.subject_id == subject]


def value(rows, feature, subject=None, scale="recording"):
    return float(select(rows, feature, subject, scale).value.iloc[0])


def test_rows_satisfy_the_schema(rows):
    validate_frame(rows, LONG_TABLE_SCHEMA, "long table")


def test_manifest_and_rows_agree(extractor, rows):
    """Every feature emitted is documented, and nothing is documented that is never emitted."""
    assert {spec.name for spec in extractor._specs()} == set(rows.feature)


def test_entities_ride_along(rows):
    assert (rows.group_id == "003").all()
    assert rows.task.isna().all()


def test_provenance_is_left_to_the_runner(rows):
    assert rows.execution_id.isna().all()
    assert rows.config_hash.isna().all()


class TestRecordingRows:
    def test_bits_and_normalised_come_in_pairs(self, rows):
        for stem in ("transition_entropy", "conditional_entropy", "speaker_entropy"):
            assert len(select(rows, f"{stem}_bits")) == 1
            assert 0.0 <= value(rows, f"{stem}_norm") <= 1.0

    def test_normalisation_uses_the_speaker_count(self, rows):
        """Two speakers, so the joint ceiling is 2 bits and the marginal one is 1."""
        assert value(rows, "transition_entropy_norm") == pytest.approx(
            value(rows, "transition_entropy_bits") / 2.0
        )
        assert value(rows, "speaker_entropy_norm") == pytest.approx(
            value(rows, "speaker_entropy_bits")
        )

    def test_pattern_count_is_reported(self, rows, scene):
        switches = int(scene.transitions.is_switch.astype(bool).sum())
        assert value(rows, "permutation_entropy_n") == switches - 3 + 1

    def test_summary_rows_belong_to_nobody(self, rows):
        for feature in ("transition_entropy_bits", "speaker_entropy_bits"):
            subset = rows[rows.feature == feature]
            assert subset.subject_id.isna().all()


class TestPerSpeakerConditional:
    def test_one_row_per_speaker(self, rows):
        assert len(select(rows, "conditional_entropy_bits", "parent")) == 1
        assert len(select(rows, "conditional_entropy_bits", "child")) == 1

    def test_the_two_directions_are_separate_numbers(self, rows):
        """Asymmetry between them is the direction-of-influence reading."""
        assert value(rows, "conditional_entropy_bits", "parent") != value(
            rows, "conditional_entropy_bits", "child"
        )

    def test_child_always_hands_back_so_its_row_is_certain(self, rows):
        """Every child turn is followed by a parent turn; nothing to be uncertain about."""
        assert value(rows, "conditional_entropy_bits", "child") == pytest.approx(0.0)


class TestWindowRows:
    def test_no_windows_without_a_configured_length(self, rows):
        assert (rows.scale != "window").all()

    def test_windows_appear_once_configured(
        self, make_turns, make_recording, make_prep, windowed_cfg
    ):
        scene = build_scene(make_turns, make_recording, make_prep, windowed_cfg)
        extractor = EntropyExtractor(SimpleNamespace(turns=windowed_cfg))
        rows = extractor._rows(scene.recording, scene.turns, scene.transitions, scene.prep)

        windowed = rows[rows.scale == "window"]
        spans = windowed[["t_start", "t_end"]].drop_duplicates()
        # a 20 s window hopping 10 s over 60 s of task
        assert len(spans) == 5
        assert ((windowed.t_end - windowed.t_start) - 20.0).abs().max() < 1e-9
        assert spans.t_start.min() == pytest.approx(10.0)

    def test_windows_carry_only_the_recomputable_features(
        self, make_turns, make_recording, make_prep, windowed_cfg
    ):
        """Conditional entropy needs more pairs than a window holds, so it stays out."""
        scene = build_scene(make_turns, make_recording, make_prep, windowed_cfg)
        extractor = EntropyExtractor(SimpleNamespace(turns=windowed_cfg))
        rows = extractor._rows(scene.recording, scene.turns, scene.transitions, scene.prep)

        emitted = set(rows[rows.scale == "window"].feature)
        assert emitted == {
            "transition_entropy_bits",
            "transition_entropy_norm",
            "speaker_entropy_bits",
            "speaker_entropy_norm",
        }

    def test_an_empty_window_yields_missing_not_zero(
        self, make_turns, make_recording, make_prep, windowed_cfg
    ):
        """No turns inside means no estimate; a zero would read as perfect predictability."""
        quiet = [row for row in TURNS if row[1] < 25.0]
        scene = build_scene(make_turns, make_recording, make_prep, windowed_cfg, quiet)
        extractor = EntropyExtractor(SimpleNamespace(turns=windowed_cfg))
        rows = extractor._rows(scene.recording, scene.turns, scene.transitions, scene.prep)

        late = rows[(rows.scale == "window") & (rows.t_start >= 40.0)]
        assert not late.empty
        assert late.value.isna().all()


class TestWarnings:
    def test_short_recording_flags_both_estimators(self, extractor, scene):
        checks = {
            w.check
            for w in extractor._warnings(
                scene.recording, scene.turns, scene.transitions, scene.prep
            )
        }
        assert "entropy_few_transitions" in checks
        assert "entropy_short_fto_series" in checks

    def test_window_count_is_flagged_only_when_windows_are_configured(
        self, extractor, scene, make_turns, make_recording, make_prep, windowed_cfg
    ):
        without = {
            w.check
            for w in extractor._warnings(
                scene.recording, scene.turns, scene.transitions, scene.prep
            )
        }
        assert "entropy_too_few_windows" not in without

        windowed = build_scene(make_turns, make_recording, make_prep, windowed_cfg)
        other = EntropyExtractor(SimpleNamespace(turns=windowed_cfg))
        checks = {
            w.check
            for w in other._warnings(
                windowed.recording, windowed.turns, windowed.transitions, windowed.prep
            )
        }
        assert "entropy_too_few_windows" in checks

from __future__ import annotations

from types import SimpleNamespace

import pytest

from macaw.config.schema import TurnsSection
from macaw.store.schema import LONG_TABLE_SCHEMA, validate_frame
from macaw.turns.segmentation import TurnStructureExtractor, silences, utterances_to_turns

UTTERANCES = [
    ("parent", 10.0, 13.0, "a b c"),
    ("parent", 13.5, 15.0, "d e"),
    ("child", 17.0, 18.0, "f"),
    ("parent", 19.0, 21.0, "g h"),
]


@pytest.fixture
def cfg():
    return TurnsSection()


@pytest.fixture
def extractor(cfg):
    return TurnStructureExtractor(SimpleNamespace(turns=cfg))


@pytest.fixture
def scene(make_transcript, make_recording, make_prep, cfg):
    transcript = make_transcript(UTTERANCES)
    turns = utterances_to_turns(transcript, cfg)
    turns["recording_id"] = "sub-003_ses-12mo"
    return SimpleNamespace(
        transcript=transcript,
        turns=turns,
        quiet=silences(transcript, min_s=cfg.silence_min_s),
        recording=make_recording(["parent", "child"]),
        prep=make_prep(t_origin_s=10.0, duration_s=60.0),
    )


@pytest.fixture
def rows(extractor, scene):
    return extractor._rows(scene.recording, scene.turns, scene.quiet, scene.prep)


def value(rows, feature, subject=None):
    subset = rows[rows.feature == feature]
    subset = subset[subset.subject_id.isna() if subject is None else subset.subject_id == subject]
    return float(subset.value.iloc[0])


def test_scene_is_what_we_think_it_is(scene):
    assert list(scene.turns.subject_id) == ["parent", "child", "parent"]
    assert list(scene.quiet.t_start) == [15.0, 18.0]


def test_rows_satisfy_the_schema(rows):
    validate_frame(rows, LONG_TABLE_SCHEMA, "long table")


def test_manifest_and_rows_agree(extractor, rows):
    """Every feature emitted is documented, and nothing is documented that is never emitted."""
    assert {spec.name for spec in extractor._specs()} == set(rows.feature)


def test_floor_time(rows):
    assert value(rows, "talk_time_s", "parent") == pytest.approx(7.0)
    assert value(rows, "talk_time_s", "child") == pytest.approx(1.0)
    assert value(rows, "talk_ratio", "parent") + value(rows, "talk_ratio", "child") == pytest.approx(1.0)


def test_rates_use_the_task_window(rows):
    assert value(rows, "turns_per_minute", "parent") == pytest.approx(2.0)
    assert value(rows, "joint_silence_total_s") == pytest.approx(3.0)
    assert value(rows, "joint_silence_fraction") == pytest.approx(3.0 / 60.0)


def test_silence_events_belong_to_nobody(rows):
    assert rows[rows.feature == "joint_silence_s"].subject_id.isna().all()


def test_entities_ride_along(rows):
    assert (rows.group_id == "003").all()
    assert (rows.session_id == "12mo").all()
    assert rows.task.isna().all()


def test_provenance_is_left_to_the_runner(rows):
    assert rows.execution_id.isna().all()
    assert rows.config_hash.isna().all()


def test_event_rows_span_their_turn(rows):
    events = rows[rows.feature == "turn_duration_s"]
    assert (events.t_end > events.t_start).all()


class TestWarnings:
    def test_short_recording(self, extractor, scene):
        checks = {w.check for w in extractor._warnings(scene.recording, scene.turns, scene.prep)}
        assert "turns_too_few" in checks
        assert "turns_silent_speaker" not in checks

    def test_one_speaker_only(self, extractor, scene, make_transcript, cfg):
        alone = utterances_to_turns(
            make_transcript([row for row in UTTERANCES if row[0] == "parent"]), cfg
        )
        checks = {w.check for w in extractor._warnings(scene.recording, alone, scene.prep)}
        assert "turns_silent_speaker" in checks
        assert "turns_dominance" in checks

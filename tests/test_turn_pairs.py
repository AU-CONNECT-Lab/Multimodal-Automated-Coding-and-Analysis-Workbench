from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.schema import TurnsSection
from macaw.extractors.audio.prosody import turn_pairs as module
from macaw.extractors.audio.prosody.turn_pairs import TurnPairsExtractor
from macaw.store.longtable import build_long_rows, concat_long_tables, write_long_table
from macaw.store.schema import LONG_TABLE_SCHEMA, validate_frame
from macaw.turns import transitions as transitions_module
from macaw.turns.transitions import detect_transitions

TURNS = pd.DataFrame(
    {
        "recording_id": "r1",
        "subject_id": ["parent", "child", "parent", "parent", "child", "other"],
        "turn_index": [0, 1, 2, 3, 4, 5],
        "t_start": [0.0, 2.5, 4.0, 8.0, 9.5, 11.0],
        "t_end": [2.0, 3.5, 6.0, 9.0, 10.5, 12.0],
        "n_utterances": 1,
        "n_words": [4, 2, 6, 3, 1, 2],
        "text": "",
    }
)


def voice_rows(turns):
    measured = turns[turns.subject_id != "other"]
    return concat_long_tables([
        build_long_rows(
            values=pd.DataFrame({"t_start": group.t_start, "t_end": group.t_end, "turn_f0_mean_hz": 100.0 + 10 * group.turn_index}),
            recording_id="r1", source="prosody.voice_report", scale="event", subject_id=subject,
        )
        for subject, group in measured.groupby("subject_id")
    ])


@pytest.fixture
def result(tmp_path, monkeypatch):
    path = tmp_path / "long"
    write_long_table(voice_rows(TURNS), path)
    monkeypatch.setattr(module, "_voice_rows_file", lambda config, rid: path)
    monkeypatch.setattr(module, "load_turns", lambda config, rid: TURNS[TURNS.subject_id != "other"].reset_index(drop=True))
    monkeypatch.setattr(transitions_module, "load_transitions", lambda config, rid: detect_transitions(TURNS, TurnsSection()))
    recording = SimpleNamespace(
        recording_id="r1",
        subjects=[SimpleNamespace(subject_id="parent"), SimpleNamespace(subject_id="child")],
        entities=lambda: {"group_id": "1", "session_id": None, "task": None, "run": None},
    )
    return TurnPairsExtractor(SimpleNamespace(turns=TurnsSection())).run(recording, tmp_path)


def test_each_switch_between_participants_sets_the_turn_handed_over_beside_the_one_that_took_the_floor(result):
    validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
    assert {spec.name for spec in result.specs} >= set(result.rows.feature)
    f0 = result.rows[result.rows.feature.str.endswith("turn_f0_mean_hz")]
    wide = f0.pivot_table(index=["subject_id", "subject_id_b", "t_start"], columns="feature", values="value").reset_index()
    # the hold from turn 2 to 3 is no switch, and the switch to the other voice is left out
    assert wide[["subject_id", "subject_id_b", "t_start"]].values.tolist() == [
        ["child", "parent", 2.5], ["child", "parent", 9.5], ["parent", "child", 4.0],
    ]
    assert wide["prior_turn_f0_mean_hz"].tolist() == [100.0, 130.0, 110.0]
    assert wide["next_turn_f0_mean_hz"].tolist() == [110.0, 140.0, 120.0]


def test_speech_rate_comes_from_the_turns_and_a_measure_not_made_stays_missing(result):
    rate = result.rows[(result.rows.feature == "next_turn_speech_rate_wps") & (result.rows.subject_id == "parent")]
    assert rate.value.tolist() == [pytest.approx(3.0)]
    jitter = result.rows[result.rows.feature == "prior_jitter_local"]
    assert len(jitter) == 3 and jitter.value.isna().all()


def test_every_turn_measure_is_paired_both_ways():
    names = {spec.name for spec in TurnPairsExtractor(SimpleNamespace(turns=TurnsSection()))._specs()}
    assert {"prior_turn_intensity_db", "next_turn_intensity_db", "prior_cpps_db", "next_turn_speech_rate_wps"} <= names
    assert len(names) == 2 * len(module.PAIRED)
    assert np.isnan(module.per_turn(TURNS.head(1).assign(t_end=0.0), pd.DataFrame())["turn_speech_rate_wps"].iloc[0])

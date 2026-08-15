from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from macaw.coding.layer_a import (
    BACKCHANNEL_TIER,
    INTERRUPTION_TIER,
    PAUSE_TIER,
    LayerACodingExtractor,
    code_backchannels,
    code_interruptions,
    code_pause_types,
)
from macaw.config.schema import CodingSection, TurnsSection
from macaw.store.schema import (
    ANNOTATION_SCHEMA,
    LONG_TABLE_SCHEMA,
    validate_frame,
)
from macaw.turns.transitions import classify_overlaps, detect_transitions

UTTERANCES = [
    ("parent", 10.0, 12.0, "so what happened next"),
    ("parent", 13.5, 15.0, "tell me about it"),   # 1.5 s hole inside the parent turn
    ("child", 15.4, 15.7, "mm"),                  # backchannel
    ("parent", 16.0, 20.0, "and then the dog ran"),
    ("child", 19.0, 22.0, "no it was the cat"),   # 1 s overlap, takes the floor
    ("parent", 23.0, 25.0, "you are right"),
]


@pytest.fixture
def turns_cfg():
    return TurnsSection()


@pytest.fixture
def cfg():
    return CodingSection(backchannel_lexicon=["mm", "uh-huh", "yeah"])


@pytest.fixture
def scene(make_transcript, make_turns, make_recording, turns_cfg):
    from macaw.turns.segmentation import utterances_to_turns

    transcript = make_transcript(UTTERANCES)
    turns = utterances_to_turns(transcript, turns_cfg)
    return SimpleNamespace(
        transcript=transcript,
        turns=turns,
        transitions=classify_overlaps(
            detect_transitions(turns, turns_cfg),
            interruption_s=turns_cfg.interruption_overlap_s,
        ),
        recording=make_recording(["parent", "child"]),
    )


@pytest.fixture
def extractor(cfg, turns_cfg):
    return LayerACodingExtractor(SimpleNamespace(coding=cfg, turns=turns_cfg))


@pytest.fixture
def coded(scene, cfg, turns_cfg):
    return {
        BACKCHANNEL_TIER: code_backchannels(scene.turns, cfg),
        PAUSE_TIER: code_pause_types(scene.turns, turns_cfg, scene.transcript),
        INTERRUPTION_TIER: code_interruptions(scene.transitions, cfg),
    }


def test_scene_is_what_we_think_it_is(scene, coded):
    assert list(scene.turns.subject_id) == ["parent", "child", "parent", "child", "parent"]
    assert len(coded[BACKCHANNEL_TIER]) == 1
    assert len(coded[INTERRUPTION_TIER]) == 1
    assert "turn_internal" in set(coded[PAUSE_TIER].label)


class TestAnnotations:
    @pytest.fixture
    def annotations(self, extractor, scene, coded):
        return extractor._annotations(scene.recording, coded)

    def test_schema(self, annotations):
        validate_frame(annotations, ANNOTATION_SCHEMA, "annotations")

    def test_every_row_is_layer_a_and_machine_made(self, annotations):
        assert set(annotations.layer) == {"A"}
        assert set(annotations.annotator) == {"coding.layer_a"}

    def test_tiers_are_separate(self, annotations):
        assert set(annotations.tier) == {BACKCHANNEL_TIER, PAUSE_TIER, INTERRUPTION_TIER}

    def test_rule_based_codes_carry_no_confidence_and_need_no_review(self, annotations):
        assert annotations.confidence.isna().all()
        assert (~annotations.needs_review).all()
        assert (~annotations.abstained).all()

    def test_every_interval_runs_forwards(self, annotations):
        assert (annotations.t_end > annotations.t_start).all()

    def test_entities_ride_along(self, annotations):
        assert (annotations.group_id == "003").all()
        assert annotations.task.isna().all()

    def test_provenance_is_left_to_the_runner(self, annotations):
        assert annotations.execution_id.isna().all()
        assert annotations.config_hash.isna().all()

    def test_each_interval_is_attributed_to_a_person(self, annotations):
        assert annotations.subject_id.notna().all()

    def test_nothing_coded_still_gives_a_valid_table(self, extractor, scene, coded):
        empty = {tier: intervals.iloc[0:0] for tier, intervals in coded.items()}
        found = extractor._annotations(scene.recording, empty)
        assert found.empty
        validate_frame(found, ANNOTATION_SCHEMA, "annotations")


class TestRows:
    @pytest.fixture
    def rows(self, extractor, scene, coded):
        return extractor._rows(scene.recording, scene.turns, coded)

    def test_schema(self, rows):
        validate_frame(rows, LONG_TABLE_SCHEMA, "long table")

    def test_manifest_and_rows_agree(self, extractor, rows):
        assert {spec.name for spec in extractor._specs()} == set(rows.feature)

    def test_counts_land_on_the_right_person(self, rows):
        def value(feature, subject):
            subset = rows[(rows.feature == feature) & (rows.subject_id == subject)]
            return float(subset.value.iloc[0])

        assert value("n_backchannels_coded", "child") == 1
        assert value("n_backchannels_coded", "parent") == 0
        assert value("n_turn_internal_pauses", "parent") == 1
        assert value("n_turn_internal_pauses", "child") == 0

    def test_a_speaker_without_pauses_reports_missing_not_zero(self, rows):
        subset = rows[
            (rows.feature == "median_turn_internal_pause_s") & (rows.subject_id == "child")
        ]
        assert pd.isna(subset.value.iloc[0])

    def test_every_subject_gets_a_row(self, rows):
        assert set(rows.subject_id) == {"parent", "child"}


class TestWarnings:
    def test_an_empty_lexicon_says_so(self, scene, coded, turns_cfg):
        extractor = LayerACodingExtractor(
            SimpleNamespace(coding=CodingSection(), turns=turns_cfg)
        )
        found = extractor._warnings(scene.recording, scene.turns, coded)
        assert [w.check for w in found] == ["coding_no_backchannel_lexicon"]
        assert found[0].severity == "info"

    def test_a_lexicon_that_never_matches_is_flagged(self, scene, turns_cfg):
        """Standing in for the usual cause: the list is in the wrong language."""
        wrong = CodingSection(backchannel_lexicon=["嗯", "对"])
        extractor = LayerACodingExtractor(SimpleNamespace(coding=wrong, turns=turns_cfg))
        coded = {
            BACKCHANNEL_TIER: code_backchannels(scene.turns, wrong),
            PAUSE_TIER: code_pause_types(scene.turns, turns_cfg, scene.transcript),
            INTERRUPTION_TIER: code_interruptions(scene.transitions, wrong),
        }
        checks = {w.check for w in extractor._warnings(scene.recording, scene.turns, coded)}
        assert "coding_backchannel_lexicon_unmatched" in checks

    def test_a_working_lexicon_is_quiet(self, extractor, scene, coded):
        assert extractor._warnings(scene.recording, scene.turns, coded) == []

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("pympi", reason=".eaf interchange needs pympi-ling")

from macaw.exceptions import InputNotFoundError                       # noqa: E402
from macaw.interop.eaf import (                                       # noqa: E402
    DEFAULT_ANNOTATOR,
    DEFAULT_LAYER,
    export_eaf,
    import_eaf,
    roundtrip_check,
    tier_for_subject,
)
from macaw.store.schema import ANNOTATION_SCHEMA, coerce_frame, validate_frame  # noqa: E402


@pytest.fixture
def recording():
    return SimpleNamespace(recording_id="sub-003_ses-12mo", media=[])


@pytest.fixture
def make_annotations():
    def build(rows):
        frame = pd.DataFrame(
            [
                {
                    "recording_id": "sub-003_ses-12mo",
                    "group_id": "003",
                    "session_id": "12mo",
                    "task": None,
                    "run": None,
                    "subject_id": row.get("subject_id", "parent"),
                    "tier": row.get("tier", "speech_act"),
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "label": row.get("label", "praise"),
                    "annotator": row.get("annotator", "coder_a"),
                    "layer": row.get("layer", "B"),
                    "confidence": row.get("confidence", float("nan")),
                    "abstained": row.get("abstained", False),
                    "needs_review": row.get("needs_review", False),
                    "execution_id": "exec-1",
                    "config_hash": "abc123456789",
                }
                for row in rows
            ]
        )
        return coerce_frame(frame, ANNOTATION_SCHEMA, "annotations")

    return build


class TestTierNames:
    def test_the_subject_goes_into_the_name(self):
        assert tier_for_subject("parent", "speech_act") == "parent_speech_act"

    def test_no_subject_leaves_the_tier_alone(self):
        assert tier_for_subject("", "joint_silence") == "joint_silence"

    def test_a_missing_subject_leaves_the_tier_alone(self):
        assert tier_for_subject(None, "joint_silence") == "joint_silence"


class TestRoundTrip:
    def test_a_plain_frame_survives(self, make_annotations, recording, tmp_path):
        frame = make_annotations(
            [
                {"t_start": 0.0, "t_end": 1.0, "label": "praise"},
                {"t_start": 2.0, "t_end": 3.5, "label": "directive"},
            ]
        )
        assert roundtrip_check(frame, recording, tmp_path) == []

    def test_the_subject_comes_back_from_the_tier_attribute(self, make_annotations, recording, tmp_path):
        """Splitting the tier name would break the moment a subject id has an underscore in it."""
        frame = make_annotations(
            [{"subject_id": "parent_1", "tier": "speech_act", "t_start": 0.0, "t_end": 1.0}]
        )
        export_eaf(frame, recording, tmp_path / "a.eaf")
        back = import_eaf(tmp_path / "a.eaf", recording_id=recording.recording_id)
        assert list(back["subject_id"]) == ["parent_1"]
        assert list(back["tier"]) == ["speech_act"]

    def test_the_annotator_comes_back(self, make_annotations, recording, tmp_path):
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0, "annotator": "qy"}])
        export_eaf(frame, recording, tmp_path / "a.eaf")
        assert list(import_eaf(tmp_path / "a.eaf", recording_id="r")["annotator"]) == ["qy"]

    def test_the_layer_comes_back(self, make_annotations, recording, tmp_path):
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0, "layer": "A"}])
        export_eaf(frame, recording, tmp_path / "a.eaf")
        assert list(import_eaf(tmp_path / "a.eaf", recording_id="r")["layer"]) == ["A"]

    def test_two_coders_on_one_tier_stay_apart(self, make_annotations, recording, tmp_path):
        """One ELAN tier id cannot hold two coders, and agreement needs both sides back."""
        frame = make_annotations(
            [
                {"t_start": 0.0, "t_end": 1.0, "label": "praise", "annotator": "coder_a"},
                {"t_start": 0.0, "t_end": 1.0, "label": "directive", "annotator": "coder_b"},
            ]
        )
        assert roundtrip_check(frame, recording, tmp_path) == []

    def test_two_layers_on_one_tier_stay_apart(self, make_annotations, recording, tmp_path):
        frame = make_annotations(
            [
                {"t_start": 0.0, "t_end": 1.0, "label": "praise", "layer": "A"},
                {"t_start": 0.0, "t_end": 1.0, "label": "directive", "layer": "B"},
            ]
        )
        assert roundtrip_check(frame, recording, tmp_path) == []

    def test_several_subjects_survive(self, make_annotations, recording, tmp_path):
        frame = make_annotations(
            [
                {"subject_id": "parent", "t_start": 0.0, "t_end": 1.0},
                {"subject_id": "child", "t_start": 1.0, "t_end": 2.0},
            ]
        )
        assert roundtrip_check(frame, recording, tmp_path) == []

    def test_an_annotation_without_a_subject_survives(self, make_annotations, recording, tmp_path):
        frame = make_annotations(
            [{"subject_id": None, "tier": "joint_silence", "t_start": 0.0, "t_end": 1.0}]
        )
        assert roundtrip_check(frame, recording, tmp_path) == []

    def test_times_are_exact_to_the_millisecond(self, make_annotations, recording, tmp_path):
        frame = make_annotations([{"t_start": 1.234, "t_end": 5.678}])
        export_eaf(frame, recording, tmp_path / "a.eaf")
        back = import_eaf(tmp_path / "a.eaf", recording_id="r")
        assert list(back["t_start"]) == [1.234]
        assert list(back["t_end"]) == [5.678]

    def test_finer_than_a_millisecond_is_rounded(self, make_annotations, recording, tmp_path):
        """ELAN stores whole milliseconds, so this is documented loss rather than a bug."""
        frame = make_annotations([{"t_start": 1.23449, "t_end": 5.0}])
        export_eaf(frame, recording, tmp_path / "a.eaf")
        assert list(import_eaf(tmp_path / "a.eaf", recording_id="r")["t_start"]) == [1.234]

    def test_an_empty_frame_produces_a_readable_file(self, make_annotations, recording, tmp_path):
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0}]).iloc[0:0]
        path = export_eaf(frame, recording, tmp_path / "a.eaf")
        assert path.is_file()
        assert import_eaf(path, recording_id="r").empty


class TestWhatTheFormatCannotCarry:
    def test_confidence_and_review_flags_do_not_survive(self, make_annotations, recording, tmp_path):
        """The narrowed contract: an ELAN tier has nowhere to put these, so they are dropped."""
        frame = make_annotations(
            [{"t_start": 0.0, "t_end": 1.0, "confidence": 0.42, "needs_review": True}]
        )
        export_eaf(frame, recording, tmp_path / "a.eaf")
        back = import_eaf(tmp_path / "a.eaf", recording_id="r")
        assert back["confidence"].isna().all()
        assert not back["needs_review"].any()

    def test_the_execution_stamps_do_not_survive(self, make_annotations, recording, tmp_path):
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0}])
        export_eaf(frame, recording, tmp_path / "a.eaf")
        back = import_eaf(tmp_path / "a.eaf", recording_id="r")
        assert back["execution_id"].isna().all()
        assert back["config_hash"].isna().all()

    def test_the_entities_do_not_survive(self, make_annotations, recording, tmp_path):
        """An .eaf knows nothing about the wave or the task it came from."""
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0}])
        export_eaf(frame, recording, tmp_path / "a.eaf")
        back = import_eaf(tmp_path / "a.eaf", recording_id="r")
        assert back["group_id"].isna().all()
        assert back["session_id"].isna().all()

    def test_a_zero_length_annotation_is_dropped(self, make_annotations, recording, tmp_path):
        frame = make_annotations([{"t_start": 1.0, "t_end": 1.0}, {"t_start": 2.0, "t_end": 3.0}])
        export_eaf(frame, recording, tmp_path / "a.eaf")
        assert len(import_eaf(tmp_path / "a.eaf", recording_id="r")) == 1

    def test_a_dropped_zero_length_annotation_is_not_called_a_round_trip_failure(
        self, make_annotations, recording, tmp_path
    ):
        frame = make_annotations([{"t_start": 1.0, "t_end": 1.0}])
        assert roundtrip_check(frame, recording, tmp_path) == []


class TestExportOptions:
    def test_the_tier_filter_selects(self, make_annotations, recording, tmp_path):
        frame = make_annotations(
            [
                {"tier": "speech_act", "t_start": 0.0, "t_end": 1.0},
                {"tier": "pause", "t_start": 1.0, "t_end": 2.0},
            ]
        )
        export_eaf(frame, recording, tmp_path / "a.eaf", tiers=["pause"])
        assert list(import_eaf(tmp_path / "a.eaf", recording_id="r")["tier"]) == ["pause"]

    def test_a_controlled_vocabulary_is_accepted(self, make_annotations, recording, tmp_path):
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0}])
        path = export_eaf(
            frame,
            recording,
            tmp_path / "a.eaf",
            controlled_vocabulary={"speech_act": ["praise", "directive"]},
        )
        assert "praise" in path.read_text(encoding="utf-8")


class TestImport:
    def test_the_recording_id_comes_from_the_argument(self, make_annotations, recording, tmp_path):
        export_eaf(make_annotations([{"t_start": 0.0, "t_end": 1.0}]), recording, tmp_path / "a.eaf")
        back = import_eaf(tmp_path / "a.eaf", recording_id="other")
        assert list(back["recording_id"]) == ["other"]

    def test_the_result_matches_the_annotation_schema(self, make_annotations, recording, tmp_path):
        export_eaf(make_annotations([{"t_start": 0.0, "t_end": 1.0}]), recording, tmp_path / "a.eaf")
        validate_frame(import_eaf(tmp_path / "a.eaf", recording_id="r"), ANNOTATION_SCHEMA, "annotations")

    def test_a_foreign_file_gets_the_defaults(self, tmp_path):
        """A file ELAN made by hand declares no layer and no annotator; it still has to import."""
        from pympi.Elan import Eaf

        eaf = Eaf()
        eaf.add_tier("gaze")
        eaf.add_annotation("gaze", 0, 1000, "mutual")
        eaf.to_file(str(tmp_path / "foreign.eaf"))

        back = import_eaf(tmp_path / "foreign.eaf", recording_id="r")
        assert list(back["layer"]) == [DEFAULT_LAYER]
        assert list(back["annotator"]) == [DEFAULT_ANNOTATOR]
        assert list(back["tier"]) == ["gaze"]
        assert back["subject_id"].isna().all()

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(InputNotFoundError):
            import_eaf(tmp_path / "nope.eaf", recording_id="r")

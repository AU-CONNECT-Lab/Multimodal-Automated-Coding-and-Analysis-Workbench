from __future__ import annotations

import pandas as pd
import pytest

from macaw.coding.review_queue import (
    ABSTAINED,
    FLAGGED,
    LOW_CONFIDENCE,
    REVIEW_QUEUE_SCHEMA,
    apply_decisions,
    build_queue,
    queue_summary,
    read_queue,
    write_queue,
)
from macaw.exceptions import InputNotFoundError
from macaw.store.schema import ANNOTATION_SCHEMA, coerce_frame, validate_frame


def _has_pyarrow() -> bool:
    try:
        import pyarrow.parquet as parquet
    except (ImportError, AttributeError):
        return False
    return hasattr(parquet, "write_table")


needs_parquet = pytest.mark.skipif(not _has_pyarrow(), reason="needs a real pyarrow")


@pytest.fixture
def make_annotations():
    def build(rows):
        frame = pd.DataFrame(
            [
                {
                    "recording_id": row.get("recording_id", "sub-003_ses-12mo"),
                    "group_id": "003",
                    "session_id": "12mo",
                    "task": None,
                    "run": None,
                    "subject_id": row.get("subject_id", "parent"),
                    "tier": row.get("tier", "speech_act"),
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "label": row.get("label"),
                    "annotator": row.get("annotator", "llama3.1:8b"),
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


class TestBuildQueue:
    def test_low_confidence_queues(self, make_annotations):
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0, "label": "praise", "confidence": 0.4}])
        queue = build_queue(frame, threshold=0.7)
        assert list(queue["reason"]) == [LOW_CONFIDENCE]

    def test_confident_enough_does_not_queue(self, make_annotations):
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0, "label": "praise", "confidence": 0.9}])
        assert build_queue(frame, threshold=0.7).empty

    def test_the_threshold_is_strict(self, make_annotations):
        """Exactly at the threshold is confident enough; only below it queues."""
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0, "label": "a", "confidence": 0.7}])
        assert build_queue(frame, threshold=0.7).empty

    def test_an_abstention_queues_whatever_the_confidence(self, make_annotations):
        frame = make_annotations(
            [{"t_start": 0.0, "t_end": 1.0, "label": None, "confidence": 0.99, "abstained": True}]
        )
        assert list(build_queue(frame, threshold=0.7)["reason"]) == [ABSTAINED]

    def test_a_flagged_row_queues(self, make_annotations):
        frame = make_annotations(
            [{"t_start": 0.0, "t_end": 1.0, "label": "a", "confidence": 0.95, "needs_review": True}]
        )
        assert list(build_queue(frame, threshold=0.7)["reason"]) == [FLAGGED]

    def test_a_rule_code_without_confidence_never_queues(self, make_annotations):
        """Layer A carries no confidence, and a missing number is not a low one."""
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0, "label": "backchannel", "layer": "A"}])
        assert build_queue(frame, threshold=0.7).empty

    def test_abstention_outranks_the_other_reasons(self, make_annotations):
        frame = make_annotations(
            [
                {
                    "t_start": 0.0,
                    "t_end": 1.0,
                    "label": None,
                    "confidence": 0.1,
                    "abstained": True,
                    "needs_review": True,
                }
            ]
        )
        assert list(build_queue(frame, threshold=0.7)["reason"]) == [ABSTAINED]

    def test_rows_come_back_in_time_order(self, make_annotations):
        frame = make_annotations(
            [
                {"t_start": 5.0, "t_end": 6.0, "label": "b", "confidence": 0.1},
                {"t_start": 1.0, "t_end": 2.0, "label": "a", "confidence": 0.1},
            ]
        )
        assert list(build_queue(frame, threshold=0.7)["t_start"]) == [1.0, 5.0]

    def test_the_queue_matches_its_schema(self, make_annotations):
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0, "label": "a", "confidence": 0.1}])
        validate_frame(build_queue(frame, threshold=0.7), REVIEW_QUEUE_SCHEMA, "review queue")

    def test_an_empty_input_still_returns_a_valid_queue(self):
        queue = build_queue(pd.DataFrame(), threshold=0.7)
        assert queue.empty
        validate_frame(queue, REVIEW_QUEUE_SCHEMA, "review queue")

    def test_nothing_to_review_still_returns_a_valid_queue(self, make_annotations):
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0, "label": "a", "confidence": 1.0}])
        validate_frame(build_queue(frame, threshold=0.7), REVIEW_QUEUE_SCHEMA, "review queue")

    def test_decision_starts_empty(self, make_annotations):
        frame = make_annotations([{"t_start": 0.0, "t_end": 1.0, "label": "a", "confidence": 0.1}])
        assert build_queue(frame, threshold=0.7)["decision"].isna().all()


class TestFileRoundTrip:
    @needs_parquet
    def test_parquet_round_trip(self, make_annotations, tmp_path):
        queue = build_queue(
            make_annotations([{"t_start": 0.0, "t_end": 1.0, "label": "a", "confidence": 0.1}]),
            threshold=0.7,
        )
        back = read_queue(write_queue(queue, tmp_path / "queue.parquet"))
        pd.testing.assert_frame_equal(back, queue)

    def test_csv_round_trip(self, make_annotations, tmp_path):
        """csv is the path a reviewer without the GUI actually edits, so it has to survive."""
        queue = build_queue(
            make_annotations([{"t_start": 0.5, "t_end": 1.25, "label": "a", "confidence": 0.1}]),
            threshold=0.7,
        )
        back = read_queue(write_queue(queue, tmp_path / "queue.csv"))
        assert list(back["t_start"]) == [0.5]
        assert list(back["label"]) == ["a"]

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(InputNotFoundError):
            read_queue(tmp_path / "nope.parquet")


class TestApplyDecisions:
    @pytest.fixture
    def scene(self, make_annotations):
        frame = make_annotations(
            [
                {"t_start": 0.0, "t_end": 1.0, "label": "praise", "confidence": 0.4},
                {"t_start": 2.0, "t_end": 3.0, "label": "directive", "confidence": 0.95},
            ]
        )
        queue = build_queue(frame, threshold=0.7)
        return frame, queue

    def test_a_decision_adds_a_row_and_keeps_the_model_row(self, scene):
        """Overwriting would destroy the only evidence that the automatic pass agreed."""
        frame, queue = scene
        queue.loc[0, ["decision", "reviewer"]] = ["directive", "qy"]
        out = apply_decisions(frame, queue)
        assert len(out) == 3
        first = out[(out["t_start"] == 0.0)]
        assert set(first["annotator"]) == {"llama3.1:8b", "qy"}
        assert set(first["label"]) == {"praise", "directive"}

    def test_the_human_row_carries_the_entities(self, scene):
        frame, queue = scene
        queue.loc[0, ["decision", "reviewer"]] = ["directive", "qy"]
        added = apply_decisions(frame, queue).query("annotator == 'qy'")
        assert list(added["group_id"]) == ["003"]
        assert list(added["session_id"]) == ["12mo"]
        assert list(added["subject_id"]) == ["parent"]

    def test_the_human_row_belongs_to_no_execution(self, scene):
        frame, queue = scene
        queue.loc[0, ["decision", "reviewer"]] = ["directive", "qy"]
        added = apply_decisions(frame, queue).query("annotator == 'qy'")
        assert added["execution_id"].isna().all()
        assert added["config_hash"].isna().all()
        assert added["confidence"].isna().all()

    def test_the_reviewed_model_row_is_no_longer_waiting(self, make_annotations):
        frame = make_annotations(
            [{"t_start": 0.0, "t_end": 1.0, "label": "praise", "confidence": 0.9, "needs_review": True}]
        )
        queue = build_queue(frame, threshold=0.7)
        queue.loc[0, ["decision", "reviewer"]] = ["directive", "qy"]
        out = apply_decisions(frame, queue)
        assert not out.query("annotator == 'llama3.1:8b'")["needs_review"].any()

    def test_agreeing_with_the_model_still_adds_a_row(self, scene):
        """Agreement needs both sides recorded, including where they agree."""
        frame, queue = scene
        queue.loc[0, ["decision", "reviewer"]] = ["praise", "qy"]
        out = apply_decisions(frame, queue)
        assert len(out) == 3
        assert sorted(out.query("t_start == 0.0")["annotator"]) == ["llama3.1:8b", "qy"]

    def test_an_undecided_queue_changes_nothing(self, scene):
        frame, queue = scene
        pd.testing.assert_frame_equal(apply_decisions(frame, queue), frame)

    def test_a_blank_decision_is_not_a_decision(self, scene):
        frame, queue = scene
        queue.loc[0, ["decision", "reviewer"]] = ["   ", "qy"]
        pd.testing.assert_frame_equal(apply_decisions(frame, queue), frame)

    def test_an_empty_decision_frame_changes_nothing(self, scene):
        frame, _ = scene
        pd.testing.assert_frame_equal(apply_decisions(frame, pd.DataFrame()), frame)

    def test_applying_twice_does_not_double_the_human_row(self, scene):
        frame, queue = scene
        queue.loc[0, ["decision", "reviewer"]] = ["directive", "qy"]
        once = apply_decisions(frame, queue)
        twice = apply_decisions(once, queue)
        assert len(twice) == len(once)

    def test_a_decision_survives_the_csv_round_trip(self, scene, tmp_path):
        frame, queue = scene
        queue.loc[0, ["decision", "reviewer"]] = ["directive", "qy"]
        back = read_queue(write_queue(queue, tmp_path / "queue.csv"))
        assert len(apply_decisions(frame, back)) == 3

    def test_a_decision_pointing_at_nothing_is_counted(self, scene):
        frame, queue = scene
        queue.loc[0, ["t_start", "decision", "reviewer"]] = [99.0, "directive", "qy"]
        out = apply_decisions(frame, queue)
        assert len(out) == 2
        assert out.attrs["unmatched_decisions"] == 1

    def test_an_unnamed_reviewer_still_gets_an_annotator(self, scene):
        """Agreement groups on annotator, so a blank name would collapse into the wrong column."""
        frame, queue = scene
        queue.loc[0, "decision"] = "directive"
        out = apply_decisions(frame, queue)
        assert "human" in set(out["annotator"])

    def test_the_result_matches_the_annotation_schema(self, scene):
        frame, queue = scene
        queue.loc[0, ["decision", "reviewer"]] = ["directive", "qy"]
        validate_frame(apply_decisions(frame, queue), ANNOTATION_SCHEMA, "annotations")

    def test_deciding_on_an_abstention_produces_a_label(self, make_annotations):
        frame = make_annotations(
            [{"t_start": 0.0, "t_end": 1.0, "label": None, "abstained": True, "confidence": 0.2}]
        )
        queue = build_queue(frame, threshold=0.7)
        queue.loc[0, ["decision", "reviewer"]] = ["praise", "qy"]
        added = apply_decisions(frame, queue).query("annotator == 'qy'")
        assert list(added["label"]) == ["praise"]
        assert not added["abstained"].any()


class TestSummary:
    def test_counts_and_reasons(self, make_annotations):
        frame = make_annotations(
            [
                {"t_start": 0.0, "t_end": 1.0, "label": "a", "confidence": 0.1},
                {"t_start": 2.0, "t_end": 3.0, "label": None, "abstained": True},
                {"t_start": 4.0, "t_end": 5.0, "label": "b", "confidence": 0.2},
            ]
        )
        summary = queue_summary(build_queue(frame, threshold=0.7))
        assert summary["n_items"] == 3
        assert summary["by_reason"] == {LOW_CONFIDENCE: 2, ABSTAINED: 1}

    def test_decided_and_pending_split(self, make_annotations):
        frame = make_annotations(
            [
                {"t_start": 0.0, "t_end": 1.0, "label": "a", "confidence": 0.1},
                {"t_start": 2.0, "t_end": 3.0, "label": "b", "confidence": 0.2},
            ]
        )
        queue = build_queue(frame, threshold=0.7)
        queue.loc[0, "decision"] = "a"
        summary = queue_summary(queue)
        assert (summary["n_decided"], summary["n_pending"]) == (1, 1)

    def test_an_empty_queue_summarises_to_zero(self):
        summary = queue_summary(build_queue(pd.DataFrame(), threshold=0.7))
        assert summary["n_items"] == 0
        assert summary["by_reason"] == {}

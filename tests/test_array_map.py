from __future__ import annotations

import pytest

from macaw.exceptions import MacawError
from macaw.execution.array_map import (
    array_spec,
    build_map,
    read_map,
    recording_for_index,
    rerun_spec,
    write_map,
)


class TestBuilding:
    def test_indices_start_at_zero_and_follow_sorted_ids(self):
        assert build_map(["rec3", "rec1", "rec2"]) == {"rec1": 0, "rec2": 1, "rec3": 2}

    def test_input_order_does_not_change_the_mapping(self):
        """Element N of one array must pair with element N of the next, so the mapping cannot
        depend on the order the ids happened to arrive in."""
        assert build_map(["b", "a", "c"]) == build_map(["c", "b", "a"])

    def test_duplicates_collapse(self):
        assert build_map(["rec1", "rec1", "rec2"]) == {"rec1": 0, "rec2": 1}

    def test_no_recordings_gives_no_mapping(self):
        assert build_map([]) == {}


class TestFile:
    def test_round_trip(self, tmp_path):
        mapping = build_map(["rec1", "rec2", "rec3"])
        assert read_map(write_map(mapping, tmp_path / "array_map.tsv")) == mapping

    def test_the_file_is_written_in_index_order(self, tmp_path):
        path = write_map({"b": 1, "a": 0}, tmp_path / "m.tsv")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[1:] == ["0\ta", "1\tb"]

    def test_gaps_survive_a_round_trip(self, tmp_path):
        """Indices are never compacted, so a map with a hole in it has to read back as is."""
        mapping = {"rec1": 0, "rec3": 2}
        assert read_map(write_map(mapping, tmp_path / "m.tsv")) == mapping

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "m.tsv"
        path.write_text("index\trecording_id\n0\trec1\n\n1\trec2\n", encoding="utf-8")
        assert read_map(path) == {"rec1": 0, "rec2": 1}

    def test_a_malformed_line_names_the_line_number(self, tmp_path):
        path = tmp_path / "m.tsv"
        path.write_text("index\trecording_id\n0\trec1\nnonsense\n", encoding="utf-8")
        with pytest.raises(MacawError, match=":3:"):
            read_map(path)

    def test_the_parent_directory_is_created(self, tmp_path):
        assert write_map({"rec1": 0}, tmp_path / "deep" / "m.tsv").is_file()


class TestLookup:
    def test_an_index_resolves_to_its_recording(self):
        assert recording_for_index({"rec1": 0, "rec2": 1}, 1) == "rec2"

    def test_an_index_from_another_run_is_an_error(self):
        with pytest.raises(MacawError, match="different run"):
            recording_for_index({"rec1": 0}, 7)


class TestSpecs:
    def test_the_array_spec_spans_zero_to_the_last_index(self):
        assert array_spec(build_map(["a", "b", "c"])) == "0-2"

    def test_a_concurrency_limit_is_appended(self):
        assert array_spec(build_map(["a", "b", "c"]), max_concurrent=4) == "0-2%4"

    def test_the_spec_follows_the_highest_index_not_the_count(self):
        """A cached recording still holds its index, so the range can be wider than the map."""
        assert array_spec({"rec1": 0, "rec9": 9}) == "0-9"

    def test_an_empty_map_is_an_error(self):
        with pytest.raises(MacawError, match="zero recordings"):
            array_spec({})

    def test_a_rerun_lists_the_failed_indices(self):
        assert rerun_spec([7, 3, 12]) == "3,7,12"

    def test_a_rerun_deduplicates(self):
        assert rerun_spec([3, 3, 7]) == "3,7"

    def test_a_rerun_takes_a_concurrency_limit(self):
        assert rerun_spec([3, 7], max_concurrent=2) == "3,7%2"

    def test_a_rerun_with_nothing_to_rerun_is_an_error(self):
        with pytest.raises(MacawError, match="zero failed"):
            rerun_spec([])

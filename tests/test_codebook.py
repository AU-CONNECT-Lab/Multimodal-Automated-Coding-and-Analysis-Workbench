from __future__ import annotations

import pytest

from macaw.coding.codebook import (
    CodeDefinition,
    Codebook,
    RESERVED_LABELS,
    read_codebook,
    validate_codebook,
    write_codebook,
)
from macaw.exceptions import CodebookError, InputNotFoundError

GOOD = """
name: parenting
unit: turn
mutually_exclusive: true
codes:
  - code: praise
    description: The parent approves of what the child did.
    criteria: ["names the behaviour", "positive evaluation"]
    positive_examples: ["good job putting that away"]
    negative_examples: ["okay"]
    boundary_cases: ["thanks, said flatly"]
    layer: B
  - code: directive
    description: The parent tells the child to do something.
    criteria: ["imperative"]
"""


@pytest.fixture
def codebook_file(tmp_path):
    def build(text, name="codebook.yaml"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return build


class TestReading:
    def test_every_field_survives(self, codebook_file):
        book = read_codebook(codebook_file(GOOD))
        assert book.name == "parenting"
        assert book.unit == "turn"
        assert book.mutually_exclusive is True
        praise = book.get("praise")
        assert praise.criteria == ["names the behaviour", "positive evaluation"]
        assert praise.boundary_cases == ["thanks, said flatly"]
        assert praise.layer == "B"

    def test_labels_keep_file_order(self, codebook_file):
        assert read_codebook(codebook_file(GOOD)).labels() == ["praise", "directive"]

    def test_missing_lists_become_empty_not_none(self, codebook_file):
        directive = read_codebook(codebook_file(GOOD)).get("directive")
        assert directive.positive_examples == []
        assert directive.negative_examples == []

    def test_layer_defaults_to_b(self, codebook_file):
        assert read_codebook(codebook_file(GOOD)).get("directive").layer == "B"

    def test_a_single_string_is_read_as_a_one_item_list(self, codebook_file):
        text = "name: x\nunit: turn\ncodes:\n  - code: a\n    criteria: only one\n"
        assert read_codebook(codebook_file(text)).get("a").criteria == ["only one"]

    def test_name_falls_back_to_the_file_stem(self, codebook_file):
        text = "unit: turn\ncodes:\n  - code: a\n"
        assert read_codebook(codebook_file(text, "my_codes.yaml")).name == "my_codes"

    def test_unknown_key_is_an_error_not_a_silent_drop(self, codebook_file):
        text = "name: x\nunit: turn\ncodes:\n  - code: a\n    criteria_typo: []\n"
        with pytest.raises(CodebookError, match="criteria_typo"):
            read_codebook(codebook_file(text))

    def test_a_code_entry_without_a_code_key_is_an_error(self, codebook_file):
        with pytest.raises(CodebookError, match="'code' key"):
            read_codebook(codebook_file("name: x\nunit: turn\ncodes:\n  - description: hi\n"))

    def test_codes_must_be_a_list(self, codebook_file):
        with pytest.raises(CodebookError, match="must be a list"):
            read_codebook(codebook_file("name: x\ncodes: {a: b}\n"))

    def test_a_scalar_file_is_an_error(self, codebook_file):
        with pytest.raises(CodebookError, match="mapping at the top level"):
            read_codebook(codebook_file("just a string\n"))

    def test_broken_yaml_is_an_error(self, codebook_file):
        with pytest.raises(CodebookError, match="YAML"):
            read_codebook(codebook_file("name: [unclosed\n"))

    def test_a_missing_file_is_not_a_codebook_error(self, tmp_path):
        with pytest.raises(InputNotFoundError):
            read_codebook(tmp_path / "nope.yaml")


class TestRoundTrip:
    def test_write_then_read_reproduces_the_codebook(self, codebook_file, tmp_path):
        before = read_codebook(codebook_file(GOOD))
        after = read_codebook(write_codebook(before, tmp_path / "out.yaml"))
        assert after == before

    def test_a_second_round_trip_changes_nothing(self, codebook_file, tmp_path):
        once = write_codebook(read_codebook(codebook_file(GOOD)), tmp_path / "a.yaml")
        twice = write_codebook(read_codebook(once), tmp_path / "b.yaml")
        assert twice.read_text(encoding="utf-8") == once.read_text(encoding="utf-8")

    def test_unicode_stays_readable(self, tmp_path):
        book = Codebook(
            name="中文",
            unit="turn",
            codes=[CodeDefinition(code="回应", description="孩子作出回应", criteria=["有语音"])],
        )
        path = write_codebook(book, tmp_path / "cn.yaml")
        assert "中文" in path.read_text(encoding="utf-8")
        assert read_codebook(path) == book


class TestValidation:
    def test_a_complete_codebook_has_no_problems(self, codebook_file):
        text = """
name: x
unit: turn
codes:
  - code: a
    description: first
    criteria: ["one"]
"""
        assert validate_codebook(read_codebook(codebook_file(text))) == []

    def test_an_unknown_unit_is_reported(self):
        book = Codebook(name="x", unit="frame", codes=[CodeDefinition("a", "d", criteria=["c"])])
        assert any("frame" in problem for problem in validate_codebook(book))

    def test_no_codes_is_reported(self):
        assert any("no codes" in problem for problem in validate_codebook(Codebook("x", "turn")))

    def test_a_duplicate_code_is_reported(self):
        book = Codebook(
            name="x",
            unit="turn",
            codes=[CodeDefinition("a", "d", criteria=["c"]), CodeDefinition("a", "e", criteria=["c"])],
        )
        assert any("more than once" in problem for problem in validate_codebook(book))

    @pytest.mark.parametrize("label", RESERVED_LABELS)
    def test_a_reserved_label_is_reported(self, label):
        """The agreement report reads these two as blank cells, so a real code cannot use them."""
        book = Codebook(name="x", unit="turn", codes=[CodeDefinition(label, "d", criteria=["c"])])
        assert any("reserved" in problem for problem in validate_codebook(book))

    def test_a_missing_description_is_reported(self):
        book = Codebook(name="x", unit="turn", codes=[CodeDefinition("a", "", criteria=["c"])])
        assert any("no description" in problem for problem in validate_codebook(book))

    def test_no_criteria_and_no_example_is_reported(self):
        book = Codebook(name="x", unit="turn", codes=[CodeDefinition("a", "d")])
        assert any("neither criteria" in problem for problem in validate_codebook(book))

    def test_a_positive_example_is_enough_on_its_own(self):
        book = Codebook(
            name="x", unit="turn", codes=[CodeDefinition("a", "d", positive_examples=["hi"])]
        )
        assert validate_codebook(book) == []

    def test_an_unknown_layer_is_reported(self):
        book = Codebook(
            name="x", unit="turn", codes=[CodeDefinition("a", "d", criteria=["c"], layer="D")]
        )
        assert any("layer 'D'" in problem for problem in validate_codebook(book))

    def test_validation_never_raises(self):
        assert validate_codebook(Codebook(name="", unit="", codes=[])) != []


class TestLookup:
    def test_get_finds_the_definition(self, codebook_file):
        assert read_codebook(codebook_file(GOOD)).get("directive").description.startswith("The parent")

    def test_an_unknown_code_names_the_ones_that_exist(self, codebook_file):
        with pytest.raises(CodebookError, match="praise, directive"):
            read_codebook(codebook_file(GOOD)).get("nonsense")

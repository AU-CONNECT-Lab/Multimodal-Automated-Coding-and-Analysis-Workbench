from __future__ import annotations

import threading

import pytest

from macaw.store import _io
from macaw.store.longtable import read_long_table, write_no_rows


def test_a_write_that_was_killed_does_not_stop_the_table_being_read(tmp_path, monkeypatch):
    features = tmp_path / "features"
    write_no_rows(features, "one.step")

    def killed(staging, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(_io, "swap_in", killed)
    with pytest.raises(KeyboardInterrupt):
        write_no_rows(features, "other.step")
    (leftover,) = [path for path in features.iterdir() if path.suffix != ".parquet"]
    leftover.write_bytes(leftover.read_bytes()[:10])
    assert read_long_table(features).empty


def test_a_write_leaves_only_its_own_file(tmp_path):
    features = tmp_path / "features"
    write_no_rows(features, "one.step")
    assert [path.name for path in features.iterdir()] == ["one.step.parquet"]


def test_a_file_swapped_in_waits_for_whoever_is_reading_the_one_it_replaces(tmp_path):
    target, staging = tmp_path / "record.json", tmp_path / "record.json.tmp"
    target.write_text("old", encoding="utf-8")
    staging.write_text("new", encoding="utf-8")
    opened, release = threading.Event(), threading.Event()

    def read():
        with open(target, "r", encoding="utf-8"):
            opened.set()
            release.wait(5)

    reader = threading.Thread(target=read)
    reader.start()
    opened.wait(5)
    threading.Timer(0.2, release.set).start()
    _io.swap_in(staging, target)
    reader.join()
    assert target.read_text(encoding="utf-8") == "new"


def test_a_frame_number_with_a_fraction_is_refused_and_a_whole_number_id_read_as_a_number_keeps_its_text():
    import pandas as pd
    import pytest

    from macaw.exceptions import SchemaViolationError
    from macaw.store.schema import coerce_frame

    schema = {"frame": "int64", "subject_id": "string"}
    out = coerce_frame(pd.DataFrame({"frame": [2.0, 3.0], "subject_id": [1.0, None]}), schema, "rows")
    assert list(out["frame"]) == [2, 3] and out["subject_id"].iloc[0] == "1" and out["subject_id"].isna().iloc[1]
    with pytest.raises(SchemaViolationError, match="2.9999"):
        coerce_frame(pd.DataFrame({"frame": [2.9999, 3.0], "subject_id": ["a", "b"]}), schema, "rows")


def test_a_killed_write_of_what_people_marked_leaves_the_earlier_file_whole(tmp_path, monkeypatch):
    import json

    from macaw.project.gaze_labels import GazeLabel, read_labels, write_labels
    from macaw.project.names import read_names, write_names
    from macaw.project.review_notes import add_note, read_notes

    names, labels, notes = tmp_path / "names.json", tmp_path / "gaze_labels.json", tmp_path / "notes.json"
    write_names(names, {"cam": [{"person": "p1", "at": 1.0, "x": 0.5, "y": 0.5}]})
    write_labels([GazeLabel(t=1.0, subject="p1", target="p2")], labels)
    add_note(notes, [], "first", "ann")

    def killed(staging, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(_io, "swap_in", killed)
    for write in (
        lambda: write_names(names, {"cam": [{"person": "p2", "at": 2.0, "x": 0.1, "y": 0.1}]}),
        lambda: write_labels([GazeLabel(t=2.0, subject="p2", target="p1")], labels),
        lambda: add_note(notes, [], "second", "bo"),
    ):
        with pytest.raises(KeyboardInterrupt):
            write()
    assert read_names(names)["cam"][0]["person"] == "p1" and read_labels(labels)[0].subject == "p1"
    assert [note["text"] for note in read_notes(notes)] == ["first"] and json.loads(names.read_text(encoding="utf-8"))

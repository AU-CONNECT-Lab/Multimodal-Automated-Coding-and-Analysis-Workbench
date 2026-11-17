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

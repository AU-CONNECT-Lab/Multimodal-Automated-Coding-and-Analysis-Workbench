"""A cache hit has to look exactly like a real run to everything downstream. That invariant is
what these cover; it is also what the annotations bug in _stage_payload broke."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pandas as pd
import pytest

from macaw.exceptions import CacheCorruptError, InputNotFoundError
from macaw.execution.runner import _stage_payload
from macaw.project.layout import make_layout
from macaw.store import cache as cache_store

EXTRACTOR = "coding.layer_a"
RECORDING = "sub-003_ses-12mo"


def _write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def produced(tmp_path):
    """Everything one extractor leaves behind after a real run."""
    layout = make_layout(tmp_path / "derivatives")
    _write(layout.long_table_path(RECORDING) / f"{EXTRACTOR}.parquet", "rows")
    _write(layout.annotations_path(RECORDING) / "A.parquet", "annotations")
    _write(layout.manifest_fragment_path(RECORDING, EXTRACTOR), "{}")
    _write(layout.warnings_path(RECORDING, EXTRACTOR), "[]")
    _write(layout.artifacts_dir(RECORDING, EXTRACTOR) / "parent.wav", "audio")
    result = SimpleNamespace(annotations=pd.DataFrame({"layer": ["A", "A"]}))
    return layout, result


@pytest.fixture
def payload(produced, tmp_path):
    layout, result = produced
    target = tmp_path / "workspace" / "_payload"
    _stage_payload(layout, RECORDING, EXTRACTOR, result, target)
    return target


def _tree(root):
    return sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file())


class TestStaging:
    def test_the_long_table_is_staged(self, payload):
        assert f"features/{EXTRACTOR}.parquet" in _tree(payload)

    def test_annotations_are_staged(self, payload):
        """They are partitioned by layer, so the file name cannot be guessed from the extractor
        name the way the others can. Leaving them out made a second run lose the table."""
        assert "annotations/A.parquet" in _tree(payload)

    def test_several_layers_are_all_staged(self, produced, tmp_path):
        layout, _ = produced
        _write(layout.annotations_path(RECORDING) / "B.parquet", "more")
        result = SimpleNamespace(annotations=pd.DataFrame({"layer": ["A", "B"]}))
        target = tmp_path / "p2"
        _stage_payload(layout, RECORDING, EXTRACTOR, result, target)
        assert {"annotations/A.parquet", "annotations/B.parquet"} <= set(_tree(target))

    def test_artifacts_keep_their_extractor_directory(self, payload):
        """Without this level a restored hit lands one directory too high and the next
        extractor cannot find the tracks."""
        assert f"artifacts/{EXTRACTOR}/parent.wav" in _tree(payload)

    def test_the_manifest_fragment_and_warnings_are_staged(self, payload):
        tree = _tree(payload)
        assert f"manifest/{EXTRACTOR}.json" in tree
        assert f"warnings/{EXTRACTOR}.json" in tree

    def test_another_extractors_output_is_left_alone(self, produced, tmp_path):
        """One writer per file: a rerun of one extractor must not carry off its neighbour's."""
        layout, result = produced
        _write(layout.long_table_path(RECORDING) / "prosody.frame.parquet", "not mine")
        target = tmp_path / "p3"
        _stage_payload(layout, RECORDING, EXTRACTOR, result, target)
        assert "features/prosody.frame.parquet" not in _tree(target)

    def test_an_extractor_with_no_annotations_stages_the_rest(self, produced, tmp_path):
        layout, _ = produced
        target = tmp_path / "p4"
        _stage_payload(layout, RECORDING, EXTRACTOR, SimpleNamespace(annotations=None), target)
        tree = _tree(target)
        assert f"features/{EXTRACTOR}.parquet" in tree
        assert not any(item.startswith("annotations/") for item in tree)


class TestReplay:
    @pytest.fixture
    def key(self):
        return cache_store.CacheKey(
            extractor=EXTRACTOR, extractor_version="0.1.0", input_hash="aaa", config_hash="bbb"
        )

    def test_a_hit_reproduces_the_whole_tree(self, payload, key, tmp_path):
        entry = cache_store.store(tmp_path / "cache", key, payload)
        restored = tmp_path / "replayed"
        cache_store.restore(entry, restored)
        assert _tree(restored) == _tree(payload)

    def test_the_replayed_artifacts_sit_where_the_next_extractor_looks(self, payload, key, tmp_path):
        entry = cache_store.store(tmp_path / "cache", key, payload)
        restored = tmp_path / "replayed"
        cache_store.restore(entry, restored)
        assert (restored / "artifacts" / EXTRACTOR / "parent.wav").is_file()
        assert not (restored / "artifacts" / "parent.wav").exists()

    def test_the_complete_marker_is_not_replayed(self, payload, key, tmp_path):
        entry = cache_store.store(tmp_path / "cache", key, payload)
        restored = tmp_path / "replayed"
        cache_store.restore(entry, restored)
        assert not (restored / cache_store.COMPLETE_MARKER).exists()

    def test_replaying_over_existing_output_does_not_fail(self, payload, key, tmp_path):
        entry = cache_store.store(tmp_path / "cache", key, payload)
        restored = tmp_path / "replayed"
        cache_store.restore(entry, restored)
        cache_store.restore(entry, restored)
        assert _tree(restored) == _tree(payload)

    def test_a_stored_entry_is_a_hit(self, payload, key, tmp_path):
        cache_store.store(tmp_path / "cache", key, payload)
        assert cache_store.lookup(tmp_path / "cache", key) is not None

    def test_a_different_config_hash_is_a_miss(self, payload, key, tmp_path):
        cache_store.store(tmp_path / "cache", key, payload)
        other = cache_store.CacheKey(EXTRACTOR, "0.1.0", "aaa", "changed")
        assert cache_store.lookup(tmp_path / "cache", other) is None

    def test_a_new_extractor_version_is_a_miss(self, payload, key, tmp_path):
        cache_store.store(tmp_path / "cache", key, payload)
        other = cache_store.CacheKey(EXTRACTOR, "0.2.0", "aaa", "bbb")
        assert cache_store.lookup(tmp_path / "cache", other) is None

    def test_a_directory_without_the_marker_is_a_miss(self, key, tmp_path):
        """That is the wreckage of a job killed mid-write, not a usable result."""
        entry = cache_store.entry_path(tmp_path / "cache", key)
        _write(entry / "features" / "x.parquet")
        assert cache_store.lookup(tmp_path / "cache", key) is None

    def test_storing_leaves_no_partial_directory(self, payload, key, tmp_path):
        cache_store.store(tmp_path / "cache", key, payload)
        assert list((tmp_path / "cache" / EXTRACTOR).glob("*.partial*")) == []

    def test_storing_twice_replaces_rather_than_nests(self, payload, key, tmp_path):
        cache_store.store(tmp_path / "cache", key, payload)
        entry = cache_store.store(tmp_path / "cache", key, payload)
        assert _tree(entry) == sorted(_tree(payload) + [cache_store.COMPLETE_MARKER])

    def test_caching_nothing_is_an_error(self, key, tmp_path):
        with pytest.raises(CacheCorruptError):
            cache_store.store(tmp_path / "cache", key, tmp_path / "not-there")

    def test_invalidate_clears_one_extractor(self, payload, key, tmp_path):
        cache_store.store(tmp_path / "cache", key, payload)
        assert cache_store.invalidate(tmp_path / "cache", extractor=EXTRACTOR) == 1
        assert cache_store.lookup(tmp_path / "cache", key) is None


class TestInputHashing:
    def test_touching_a_file_does_not_change_its_hash(self, tmp_path):
        """Copying a dataset to the cluster changes every mtime, and a cache that noticed
        would be thrown away on arrival."""
        path = _write(tmp_path / "a.wav", "content")
        before = cache_store.hash_file(path)
        os.utime(path, (0, 0))
        assert cache_store.hash_file(path) == before

    def test_changing_the_content_changes_the_hash(self, tmp_path):
        path = _write(tmp_path / "a.wav", "content")
        before = cache_store.hash_file(path)
        path.write_text("different", encoding="utf-8")
        assert cache_store.hash_file(path) != before

    def test_the_same_file_in_a_different_directory_hashes_the_same(self, tmp_path):
        """Absolute paths stay out for the same reason they stay out of the config hash."""
        left = _write(tmp_path / "laptop" / "a.wav", "content")
        right = _write(tmp_path / "cluster" / "a.wav", "content")
        assert cache_store.hash_inputs([left]) == cache_store.hash_inputs([right])

    def test_a_different_file_name_changes_the_hash(self, tmp_path):
        left = _write(tmp_path / "parent.wav", "content")
        right = _write(tmp_path / "child.wav", "content")
        assert cache_store.hash_inputs([left]) != cache_store.hash_inputs([right])

    def test_input_order_does_not_change_the_hash(self, tmp_path):
        a = _write(tmp_path / "a.wav", "one")
        b = _write(tmp_path / "b.wav", "two")
        assert cache_store.hash_inputs([a, b]) == cache_store.hash_inputs([b, a])

    def test_hashing_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(InputNotFoundError):
            cache_store.hash_file(tmp_path / "nope.wav")

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from macaw.exceptions import ConfigError, InputNotFoundError, UnreadableSignalError

pytest.importorskip("h5py")


def write_snirf(path, *, unit="s", start=0.0, rate=10.0, n=50, spacing=False, haemo=False, columnar=False):
    """A small SNIRF file: two channels, a two-row and a one-row marker, and a gyroscope axis at twice the rate."""
    import h5py

    per_s = {"s": 1.0, "ms": 1000.0}[unit]
    times = start + np.arange(n) / rate
    labels = [(1, 1, 0, 99999, "HbO"), (1, 1, 0, 99999, "HbR")] if haemo else [(1, 1, 1, 1, ""), (1, 1, 2, 1, "")]
    with h5py.File(path, "w") as handle:
        handle["formatVersion"] = "1.1"
        nirs = handle.create_group("nirs")
        tags = nirs.create_group("metaDataTags")
        tags["TimeUnit"] = unit
        tags["MeasurementDate"] = "2020-01-01"
        tags["MeasurementTime"] = "10:00:00"
        data = nirs.create_group("data1")
        data["dataTimeSeries"] = np.column_stack([np.sin(times), np.cos(times)])
        data["time"] = np.array([start * per_s, per_s / rate]) if spacing else times * per_s
        if columnar:
            lists = data.create_group("measurementLists")
            for index, key in enumerate(("sourceIndex", "detectorIndex", "wavelengthIndex", "dataType")):
                lists[key] = np.array([label[index] for label in labels])
            lists["dataTypeLabel"] = np.array([label[4].encode() for label in labels])
        else:
            for number, (source, detector, wavelength, kind, label) in enumerate(labels, start=1):
                item = data.create_group(f"measurementList{number}")
                item["sourceIndex"], item["detectorIndex"], item["wavelengthIndex"], item["dataType"] = source, detector, wavelength, kind
                if label:
                    item["dataTypeLabel"] = label
        nirs.create_group("probe")["wavelengths"] = np.array([760.0, 850.0])
        first = nirs.create_group("stim1")
        first["name"], first["data"] = "start", np.array([[2.0 * per_s, 1.0 * per_s, 1.0], [3.0 * per_s, 0.5 * per_s, 1.0]])
        second = nirs.create_group("stim2")
        second["name"], second["data"] = "end", np.array([4.0 * per_s, 0.0, 1.0])
        aux = nirs.create_group("aux1")
        aux["name"], aux["dataUnit"] = "GYRO_X_1", "deg/s"
        aux["dataTimeSeries"] = np.arange(2 * n, dtype=float)
        aux["time"] = (start + np.arange(2 * n) / (2 * rate)) * per_s
    return path


@pytest.fixture
def project(tmp_path):
    from macaw.config.loader import scaffold_project_config

    (tmp_path / "raw").mkdir()
    path = tmp_path / "project.yaml"
    scaffold_project_config(path, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    return path


def test_a_snirf_file_gives_its_channels_markers_and_aux_on_its_own_clock(tmp_path):
    from macaw.signals.snirf import read_snirf

    found = read_snirf(write_snirf(tmp_path / "a.snirf", start=5.0))
    assert found.channels == ["S1_D1 760", "S1_D1 850"] and found.kinds == ["raw", "raw"]
    assert found.times[0] == pytest.approx(5.0) and found.rate_hz == pytest.approx(10.0)
    assert found.data.shape == (50, 2)
    assert found.markers.to_dict("list") == {"name": ["start", "start", "end"], "onset": [2.0, 3.0, 4.0], "duration": [1.0, 0.5, 0.0]}
    gyro = found.aux["GYRO_X_1"]
    assert gyro.unit == "deg/s" and gyro.times[1] - gyro.times[0] == pytest.approx(0.05)
    assert found.started == "2020-01-01T10:00:00"


def test_milliseconds_and_the_start_and_spacing_form_come_out_in_seconds(tmp_path):
    from macaw.signals.snirf import read_snirf

    seconds = read_snirf(write_snirf(tmp_path / "s.snirf", start=1.0))
    for name, kwargs in (("ms.snirf", {"unit": "ms"}), ("spacing.snirf", {"spacing": True}), ("both.snirf", {"unit": "ms", "spacing": True})):
        found = read_snirf(write_snirf(tmp_path / name, start=1.0, **kwargs))
        assert found.times == pytest.approx(seconds.times), name
        assert found.markers["onset"].tolist() == pytest.approx([2.0, 3.0, 4.0]), name
        assert found.markers["duration"].tolist() == pytest.approx([1.0, 0.5, 0.0]), name
        assert found.aux["GYRO_X_1"].times == pytest.approx(seconds.aux["GYRO_X_1"].times), name


def test_haemoglobin_is_named_by_its_label_in_either_form_of_the_channel_list(tmp_path):
    from macaw.signals.snirf import read_snirf

    for columnar in (False, True):
        found = read_snirf(write_snirf(tmp_path / f"h{columnar}.snirf", haemo=True, columnar=columnar))
        assert found.channels == ["S1_D1 hbo", "S1_D1 hbr"] and found.kinds == ["hbo", "hbr"]


def test_a_file_that_is_not_snirf_says_so(tmp_path):
    from macaw.signals.snirf import read_snirf

    (tmp_path / "x.snirf").write_text("not hdf5", encoding="utf-8")
    with pytest.raises(UnreadableSignalError, match="not a SNIRF file"):
        read_snirf(tmp_path / "x.snirf")


def test_the_signals_table_is_read_and_checked(project, tmp_path):
    from macaw.config.loader import resolve_config
    from macaw.signals.table import read_signals

    config = resolve_config(project)
    assert read_signals(config) == {}
    elsewhere = (tmp_path / "elsewhere" / "parent.snirf").resolve()
    (tmp_path / "signals.csv").write_text(
        f"recording,person,stage,file\n003,child,raw,nirs/child.snirf\n003,parent,,{elsewhere}\n003,,wtc,dyad.snirf\n", encoding="utf-8"
    )
    found = read_signals(config)["003"]
    assert [(item.person, item.stage, item.key) for item in found] == [
        ("child", "raw", "child_raw"), ("parent", "raw", "parent_raw"), (None, "wtc", "all_wtc"),
    ]
    assert found[0].path == (tmp_path / "nirs" / "child.snirf").resolve() and found[1].path == elsewhere
    for text, reason in (
        ("003,uncle,raw,a.snirf", "not one of project.roles"),
        ("003,child,pre proc,a.snirf", "should be letters"),
        ("003,child,raw,", "no file"),
        ("003,child,raw,a.snirf\n003,child,,b.snirf", "given twice"),
    ):
        (tmp_path / "signals.csv").write_text("recording,person,stage,file\n" + text + "\n", encoding="utf-8")
        with pytest.raises(ConfigError, match=reason):
            read_signals(config)
    (tmp_path / "signals.csv").write_text("recording,person,file\n003,child,a.snirf\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no stage column"):
        read_signals(config)


def test_the_read_step_keeps_each_file_on_its_own_clock(project, tmp_path):
    from macaw.config.loader import resolve_config
    from macaw.extractors import get_extractor
    from macaw.project.recording import Recording
    from macaw.signals.read import SignalReadExtractor, aux_name, info_name, samples_name
    from macaw.signals.table import signal_files

    config = resolve_config(project)
    assert get_extractor("signals.read") is SignalReadExtractor
    assert not SignalReadExtractor.wanted_by_default(config)
    (tmp_path / "nirs").mkdir()
    write_snirf(tmp_path / "nirs" / "child.snirf", start=5.0)
    (tmp_path / "signals.csv").write_text("recording,person,stage,file\n003,child,raw,nirs/child.snirf\n", encoding="utf-8")
    assert SignalReadExtractor.wanted_by_default(config)
    step, recording = SignalReadExtractor(config), Recording(recording_id="003", root=tmp_path)
    assert step.inputs_for(recording) == [(tmp_path / "nirs" / "child.snirf").resolve()]
    work = tmp_path / "work"
    work.mkdir()
    result = step.run(recording, work)
    (item,) = signal_files(config, "003")
    samples = pd.read_parquet(result.artifacts[samples_name(item)])
    assert list(samples.columns) == ["time", "S1_D1 760", "S1_D1 850"] and samples["time"].iloc[0] == pytest.approx(5.0)
    assert set(pd.read_parquet(result.artifacts[aux_name(item)])["aux"]) == {"GYRO_X_1"}
    info = json.loads(result.artifacts[info_name(item)].read_text(encoding="utf-8"))
    assert info["person"] == "child" and info["started"] == "2020-01-01T10:00:00" and info["rate_hz"] == pytest.approx(10.0)
    assert [marker["name"] for marker in info["markers"]] == ["start", "start", "end"]
    assert result.warnings == []
    (tmp_path / "signals.csv").write_text("recording,person,stage,file\n003,child,raw,nirs/gone.snirf\n", encoding="utf-8")
    with pytest.raises(InputNotFoundError, match="signals.csv"):
        step.run(recording, work)

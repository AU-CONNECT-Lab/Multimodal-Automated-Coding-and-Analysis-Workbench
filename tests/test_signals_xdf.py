from __future__ import annotations

import struct

import numpy as np
import pytest

pytest.importorskip("pyxdf")
pytest.importorskip("h5py")

from test_signals_align import TRUE_OFFSET, aligned_project, run_steps  # noqa: E402,F401  (the fixture is used by name)

FORMATS = {"double64": "<d", "float32": "<f", "int32": "<i"}


def _chunk(tag: int, content: bytes) -> bytes:
    body = struct.pack("<H", tag) + content
    return struct.pack("<BQ", 8, len(body)) + body


def _count(n: int) -> bytes:
    return struct.pack("<BQ", 8, n)


def write_xdf(path, streams, written="2020-01-01T10:00:00+0000"):
    """A small XDF file, chunk by chunk as the format lays one out: (name, type, format, rate, labels, times, values, source_id) per stream."""
    out = [b"XDF:", _chunk(1, f'<?xml version="1.0"?><info><version>1.0</version><datetime>{written}</datetime></info>'.encode())]
    for number, (name, kind, form, rate, labels, times, values, source) in enumerate(streams, start=1):
        channels = "".join(f"<channel><label>{label}</label></channel>" for label in labels)
        header = (f'<?xml version="1.0"?><info><name>{name}</name><type>{kind}</type><channel_count>{len(labels)}</channel_count>'
                  f"<nominal_srate>{rate}</nominal_srate><channel_format>{form}</channel_format><source_id>{source}</source_id>"
                  f"<desc><channels>{channels}</channels></desc></info>")
        out.append(_chunk(2, struct.pack("<I", number) + header.encode()))
        samples = bytearray(struct.pack("<I", number) + _count(len(times)))
        for at, value in zip(times, values):
            samples += struct.pack("<Bd", 8, at)
            for item in np.atleast_1d(value):
                if form == "string":
                    text = str(item).encode()
                    samples += _count(len(text)) + text
                else:
                    samples += struct.pack(FORMATS[form], item)
        out.append(_chunk(3, bytes(samples)))
        out.append(_chunk(6, struct.pack("<I", number) + b'<?xml version="1.0"?><info></info>'))
    path.write_bytes(b"".join(out))
    return path


def dyad_xdf(path):
    """Two caps streamed to one recorder from 1000 s on its clock, and the experiment program's triggers."""
    own = 1000.0 + np.arange(0.0, 400.0, 0.1)
    waves = np.column_stack([np.sin(own), np.cos(own)])
    return write_xdf(path, [
        ("Cap", "NIRS", "double64", 10, ["S1_D1 760", "S1_D1 850"], own, waves, "cap-parent"),
        ("Cap", "NIRS", "double64", 10, ["S1_D1 760", "S1_D1 850"], own, waves * 2, "cap-child"),
        ("Triggers", "Markers", "string", 0, ["code"], [1170.3, 1270.3, 1370.3], ["start", "12", "end"], "psychopy"),
        ("Codes", "Markers", "int32", 0, ["code"], [1200.0], [7], "port"),
    ])


def test_a_stream_of_numbers_is_read_with_its_channels_and_the_file_s_events(tmp_path):
    from macaw.exceptions import UnreadableSignalError
    from macaw.signals.xdf import read_xdf

    path = dyad_xdf(tmp_path / "dyad.xdf")
    found = read_xdf(path, "cap-child")
    assert found.channels == ["S1_D1 760", "S1_D1 850"] and found.kinds == ["nirs", "nirs"]
    assert found.times[0] == pytest.approx(1000.0) and found.rate_hz == pytest.approx(10.0)
    assert found.data[0, 0] == pytest.approx(2 * np.sin(1000.0))
    # every event stream of the file is the stream's, numbers named as written
    assert found.markers[["name", "onset"]].values.tolist() == [["start", 1170.3], ["7", 1200.0], ["12", 1270.3], ["end", 1370.3]]
    assert found.started == "2020-01-01T10:00:00"
    # two streams of one name are told apart by source_id; with none named the reader says what to write
    with pytest.raises(UnreadableSignalError, match="2 streams are called 'Cap'.*cap-parent"):
        read_xdf(path, "Cap")
    with pytest.raises(UnreadableSignalError, match="2 streams of numbers.*stream column"):
        read_xdf(path, None)
    with pytest.raises(UnreadableSignalError, match="no stream of numbers called 'EEG'"):
        read_xdf(path, "EEG")


def test_the_signals_table_names_a_stream_of_an_xdf_file(tmp_path):
    from macaw.config.loader import resolve_config, scaffold_project_config
    from macaw.exceptions import ConfigError
    from macaw.signals.table import read_signals, stream_mates

    project = tmp_path / "project.yaml"
    scaffold_project_config(project, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out", roles=("parent", "child"))
    config = resolve_config(project)
    table = tmp_path / "signals.csv"
    table.write_text("recording,person,stage,file,stream\n003,parent,raw,dyad.xdf,cap-parent\n003,child,raw,dyad.xdf,cap-child\n"
                     "003,child,raw,c.snirf,\n003,parent,ecg,dyad.xdf,ECG 1\n", encoding="utf-8")
    files = read_signals(config)["003"]
    assert [item.key for item in files] == ["parent_raw_cap-parent", "child_raw_cap-child", "child_raw", "parent_ecg_ECG-1"]
    # setting one stream's offset sets the file's other streams, which share its clock, and no other file
    assert [item.key for item in stream_mates(files, [files[1]])] == ["parent_raw_cap-parent", "child_raw_cap-child", "parent_ecg_ECG-1"]
    assert [item.key for item in stream_mates(files, [files[2]])] == ["child_raw"]
    table.write_text("recording,person,stage,file,stream\n003,child,raw,c.snirf,Cap\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not an XDF file"):
        read_signals(config)
    table.write_text("recording,person,stage,file,stream\n003,child,raw,a.xdf,Cap\n003,child,raw,a.xdf,Cap\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="the stream Cap at raw of child in 003 is given twice"):
        read_signals(config)


def test_an_xdf_file_s_streams_are_placed_together_by_its_events_and_set_together_by_hand(aligned_project, tmp_path):
    from typer.testing import CliRunner

    from macaw.cli.main import app
    from macaw.interface.pages import signals
    from macaw.media.align import MANUAL, UNALIGNED
    from macaw.qc.boilerplate import _signals, signals_placed
    from macaw.signals.align import MARKERS, computed_offsets, signal_offsets
    from macaw.signals.read import load_info, load_samples
    from macaw.signals.table import signal_files

    project, config, layout = aligned_project
    dyad_xdf(tmp_path / "nirs" / "dyad.xdf")
    (tmp_path / "signals.csv").write_text(
        "recording,person,stage,file,stream\n003,parent,raw,nirs/dyad.xdf,cap-parent\n003,child,lsl,nirs/dyad.xdf,cap-child\n",
        encoding="utf-8")
    run_steps(config, layout)
    items = signal_files(config, "003")
    assert load_info(config, "003", items[0])["format"] == "xdf"
    assert list(load_samples(config, "003", items[1]).columns) == ["time", "S1_D1 760", "S1_D1 850"]
    found = {entry.target: entry for entry in computed_offsets(config, "003")}
    assert found["parent_raw_cap-parent"].method == UNALIGNED and "XDF file's streams share" in found["parent_raw_cap-parent"].note

    # the trigger 12 at 1270.3 s on the file's clock is seen 120 s into the video
    (tmp_path / "marks.csv").write_text("recording,mark,start_s,end_s\n003,12,120.0,121\n", encoding="utf-8")
    run_steps(config, layout)
    found = {entry.target: entry for entry in computed_offsets(config, "003")}
    assert {key: (entry.method, round(entry.offset_s, 2)) for key, entry in found.items()} == {
        "parent_raw_cap-parent": (MARKERS, -1150.3), "child_lsl_cap-child": (MARKERS, -1150.3)}
    placed = signals_placed(config)
    assert (placed["files"], placed["xdf"], placed["markers"]) == (2, 2, 2)
    assert "Two of the signal files were streams of Lab Streaming Layer recordings (XDF)" in _signals(placed)

    # one row on the page for the file's clock, and a typed offset sets both streams
    card = signals.offsets_card(config, "003", items)
    assert "parent, child" in str(card) and "raw (cap-parent), lsl (cap-child)" in str(card)
    argv = signals.action_command(project, "003", "set", "parent_raw_cap-parent", "ann", -1150.0)
    assert argv[argv.index("--file") + 1] == "parent_raw_cap-parent" and argv[-4:] == ["--offset", "-1150", "--by", "ann"]
    said = CliRunner().invoke(app, argv[1:])
    assert said.exit_code == 0, said.output
    current = signal_offsets(config, "003")
    assert {key: (entry.method, entry.offset_s) for key, entry in current.items()} == {
        "parent_raw_cap-parent": (MANUAL, -1150.0), "child_lsl_cap-child": (MANUAL, -1150.0)}

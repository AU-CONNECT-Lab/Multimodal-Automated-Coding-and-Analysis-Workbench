from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pympi")

# a tier whose middle time slot has no time, as ELAN writes a Time Subdivision; a and c are timed, b is not
UNALIGNED = """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="" DATE="2026-01-01T00:00:00+00:00" FORMAT="3.0" VERSION="3.0"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
<HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"/>
<TIME_ORDER><TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="1000"/><TIME_SLOT TIME_SLOT_ID="ts2"/><TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="3000"/>
<TIME_SLOT TIME_SLOT_ID="ts4" TIME_VALUE="4000"/></TIME_ORDER>
<TIER LINGUISTIC_TYPE_REF="default" TIER_ID="gaze">
<ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2"><ANNOTATION_VALUE>x</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
<ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a2" TIME_SLOT_REF1="ts3" TIME_SLOT_REF2="ts4"><ANNOTATION_VALUE>y</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
</TIER>
<LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="default" TIME_ALIGNABLE="true"/>
</ANNOTATION_DOCUMENT>"""


def test_an_annotation_without_a_time_of_its_own_is_left_out_and_counted(tmp_path):
    from macaw.interop.coding_files import read_coding_file

    path = tmp_path / "coded.eaf"
    path.write_text(UNALIGNED, encoding="utf-8")
    table = read_coding_file(path)
    assert table["label"].tolist() == ["y"] and table.attrs["untimed"] == {"gaze": 1}


def test_frames_are_written_into_a_folder_named_with_letters_beyond_ascii(tmp_path):
    from macaw.qc.overlay import _write_jpeg

    folder = tmp_path / "项目_renders"
    folder.mkdir()
    _write_jpeg(folder / "a.jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert (folder / "a.jpg").read_bytes()[:2] == b"\xff\xd8"

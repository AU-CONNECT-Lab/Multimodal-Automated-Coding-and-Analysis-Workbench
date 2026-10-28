from __future__ import annotations

from fractions import Fraction

import pytest

from macaw.media.probe import StreamInfo, frame_rate, is_variable_rate


def stream(declared, average):
    return StreamInfo(
        index=0, codec_type="video", codec_name="h264",
        declared_fps=Fraction(declared) if declared else None, average_fps=Fraction(average) if average else None,
    )


@pytest.mark.parametrize(
    ("declared", "average", "rate", "variable"),
    [
        ("30000/1001", "30000/1001", Fraction(30000, 1001), False),
        # interlaced files state the field rate
        ("50", "25", Fraction(25), False),
        # some containers state their time base
        ("1000", "25", Fraction(25), False),
        # a phone that slowed down in the dark: what it states is what it meant
        ("30", "22.3", Fraction(30), True),
        ("30", "29.6", Fraction(30), True),
        # a time base stated, the average snapped to the rate it is near
        ("1000", "29.62", Fraction(30000, 1001), True),
        ("1000", "17.3", Fraction(173, 10), True),
        ("25", None, Fraction(25), False),
    ],
)
def test_the_rate_a_camera_meant(declared, average, rate, variable):
    assert frame_rate(stream(declared, average)) == rate
    assert is_variable_rate(stream(declared, average)) is variable

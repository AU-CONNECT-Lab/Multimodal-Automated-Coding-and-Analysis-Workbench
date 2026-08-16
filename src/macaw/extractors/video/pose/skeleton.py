"""The joint vocabulary every pose backend maps onto.

COCO-17 is the intersection of the backends worth supporting, so all of them can produce it,
and feature code can name a joint without asking which backend ran. Points a backend has
beyond these keep their native name behind a backend prefix: nothing is thrown away, but
nothing outside this tuple is ever read by a feature.

The same name is not the same anatomical point across backends (hips especially), so the
manifest records JOINT_SET and the backend next to every feature, exactly as it already does
for formants and HNR.
"""

from __future__ import annotations

JOINT_SET = "coco17"

STANDARD_JOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

_STANDARD = frozenset(STANDARD_JOINTS)

EXTRA_JOINT_SEPARATOR = ":"


def extra_joint(backend: str, native_name: str) -> str:
    """A backend's own point stays available without being mistakable for a standard one."""
    return f"{backend}{EXTRA_JOINT_SEPARATOR}{native_name}"


def is_standard(joint: str) -> bool:
    return joint in _STANDARD

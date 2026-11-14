from __future__ import annotations

from pydantic import BaseModel, Field

from macaw.config.fields import added
from macaw.config.hashing import hash_full, hash_section, hash_sections


class Camera(BaseModel):
    films: list[str] = Field(default_factory=list)


class CameraLater(Camera):
    ignore: list[list[float]] = added(default_factory=list)


class Window(BaseModel):
    start_s: float | None = None


class WindowLater(Window):
    keep_last_s: float | None = added(default=None, gt=0)


class Before(BaseModel):
    window: Window = Field(default_factory=Window)
    cameras: dict[str, Camera] = Field(default_factory=dict)
    several: list[Window] = Field(default_factory=list)


class After(BaseModel):
    window: WindowLater = Field(default_factory=WindowLater)
    cameras: dict[str, CameraLater] = Field(default_factory=dict)
    several: list[WindowLater] = Field(default_factory=list)
    gpu_workers: int = added(default=1)


def before():
    return Before(cameras={"front": Camera(films=["parent"])}, several=[Window(start_s=2.0)])


def after(**changes):
    values = {"cameras": {"front": CameraLater(films=["parent"])}, "several": [WindowLater(start_s=2.0)], **changes}
    return After(**values)


def test_a_key_added_later_changes_no_key_while_it_holds_its_default():
    old, new = before(), after()
    assert hash_full(old) == hash_full(new)
    for section in ("window", "cameras", "cameras.front", "several"):
        assert hash_section(old, section) == hash_section(new, section)
    assert hash_sections(old, ["window.start_s", "cameras"]) == hash_sections(new, ["window.start_s", "window.keep_last_s", "cameras", "gpu_workers"])


def test_once_set_it_changes_the_keys_that_read_it():
    plain = after()
    assert hash_section(plain, "window") != hash_section(after(window=WindowLater(keep_last_s=300)), "window")
    marked = after(cameras={"front": CameraLater(films=["parent"], ignore=[[0, 0, 10, 10]])})
    assert hash_section(plain, "cameras.front") != hash_section(marked, "cameras.front")
    assert hash_section(plain, "window") == hash_section(marked, "window")
    listed = after(several=[WindowLater(start_s=2.0, keep_last_s=60)])
    assert hash_section(plain, "several") != hash_section(listed, "several")
    names = ["window.start_s", "gpu_workers"]
    assert hash_sections(plain, names) != hash_sections(after(gpu_workers=2), names)

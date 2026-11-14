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


def settings_keys() -> dict[str, bool]:
    """Section.key of every settings key -> whether it is declared with added()."""
    import typing

    from macaw.config.fields import HASHED_WHEN_SET
    from macaw.config.schema import MacawConfig

    found: dict[str, bool] = {}
    seen: set[type] = set()

    def walk(model: type[BaseModel]) -> None:
        if model in seen:
            return
        seen.add(model)
        for name, field in model.model_fields.items():
            found[f"{model.__name__}.{name}"] = isinstance(field.json_schema_extra, dict) and bool(field.json_schema_extra.get(HASHED_WHEN_SET))
            kinds = [field.annotation]
            while kinds:
                kind = kinds.pop()
                if isinstance(kind, type) and issubclass(kind, BaseModel):
                    walk(kind)
                kinds.extend(typing.get_args(kind))

    walk(MacawConfig)
    return found


def test_a_new_settings_key_is_declared_with_added():
    from pathlib import Path

    listed = set(Path(__file__).with_name("hashed_keys.txt").read_text(encoding="utf-8").split())
    keys = settings_keys()
    new = sorted(key for key, later in keys.items() if not later and key not in listed)
    assert not new, f"declare {', '.join(new)} with added(), so that the results people already have stay as they are"
    assert not sorted(listed - set(keys)), "hashed_keys.txt lists keys the settings no longer have"
    assert not sorted(key for key in listed if keys[key]), "a key in hashed_keys.txt has always been in the cache keys; added() would change them"

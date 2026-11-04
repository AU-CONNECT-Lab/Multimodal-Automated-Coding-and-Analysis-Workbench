"""A model that reads pictures looks at the review frames and says where the people drawn disagree with what it sees;
a person then checks the model on the frames it flagged and on a few it passed."""

from __future__ import annotations

import hashlib
import json
import zlib
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.coding.llm.backends import LLMBackend, resolve_llm_backend
from macaw.coding.llm.parsing import repair_json
from macaw.config.schema import LookSection
from macaw.exceptions import MacawError
from macaw.extractors import get_backend
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.qc.overlay import cv2_module, draw_overlay

MISSED, TWICE, ON_NO_ONE, WRONG_NAME = "missed", "twice", "on_no_one", "wrong_name"
RIGHT, WRONG = "right", "wrong"
UNSURE = "unsure"
PLAIN = "plain"           # the review frames undrawn, beside the drawn ones
LONG_SIDE = 1280          # px of the whole frame as sent
CROP_SIDE = 768           # px of one person cut out
CROP_MARGIN = 0.25        # of the person's size, around the cut
# changing a question asks every frame again
QUESTIONS = 3

COUNT = (
    "How many people can you see in this video frame? Count everyone whose head or upper body is visible, even partly or "
    "at an edge of the picture. Do not count people in pictures, on posters or on screens. "
    'Answer in JSON: {"people": <number>}.'
)

INTRO = (
    "You check the work of a pose tracker. The picture is cut out of a video frame around one thing the tracker took for "
    "a person; it drew a skeleton on it, lines between body joints, inside a thin box."
)
IS_PERSON = "- person: is it a real person, rather than furniture, a wall, a poster, a screen, a toy or a reflection?"
PERSON = f"""{INTRO}

Answer about what the skeleton is drawn on:
{IS_PERSON}

Answer in JSON: {{"person": true or false}}."""
PERSON_AND_WHO = f"""{INTRO}
The people in this study are: {{people}}.

Answer two questions about what the skeleton is drawn on:
{IS_PERSON}
- who: which of these people is it? Give the name only when the picture clearly shows it; otherwise answer "unsure".

Answer in JSON: {{{{"person": true or false, "who": "<a name>" or "unsure"}}}}."""


def backend_for(cfg: LookSection) -> LLMBackend:
    if not cfg.model:
        raise MacawError("qc.look.model is not set: name a model that reads pictures, for example qwen3-vl:8b-instruct on ollama")
    backend = resolve_llm_backend(cfg.llm(), settings="qc.look")
    backend.check_available()
    return backend


def notice(cfg: LookSection, sent: str = "the review frames, pictures of the people recorded,") -> str | None:
    """What to say before frames leave this machine; None for a server on it."""
    from macaw.coding.llm.backends import BACKEND_GROUP

    if get_backend(BACKEND_GROUP, cfg.provider).is_local:
        return None
    return (
        f"qc.look.provider {cfg.provider}: {sent} are sent to its servers. "
        "Whether that is allowed is for your ethics approval to say."
    )


def drawn_people(people: pd.DataFrame) -> list[dict]:
    """Each skeleton drawn on a frame, as draw_overlay draws it: who it was given to, its track, its bounds and its rows."""
    standard = people[people["joint"].isin(STANDARD_JOINTS) & np.isfinite(people["x"]) & np.isfinite(people["y"])]
    found = []
    subject = standard["subject_id"].astype(object)
    for (_, track), rows in standard.groupby([subject.fillna(""), standard["track_id"]], sort=False, dropna=False):
        name = rows["subject_id"].iloc[0]
        found.append({
            "name": str(name) if pd.notna(name) else None,
            "track": int(track) if pd.notna(track) else None,
            "bounds": (float(rows["x"].min()), float(rows["y"].min()), float(rows["x"].max()), float(rows["y"].max())),
            "rows": rows,
        })
    return found


def _jpeg(image: np.ndarray, side: int) -> bytes:
    cv2 = cv2_module()
    scale = side / max(image.shape[:2])
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def person_picture(image: np.ndarray, person: dict, subjects: list[str]) -> bytes:
    """One person cut out with only their own skeleton and no name on it, so the model says what it sees, not what it reads."""
    drawn = draw_overlay(image, person["rows"], subjects, [], labels=False)
    height, width = drawn.shape[:2]
    left, top, right, bottom = person["bounds"]
    pad_x, pad_y = CROP_MARGIN * (right - left) + 20, CROP_MARGIN * (bottom - top) + 20
    x0, y0 = int(max(0, left - pad_x)), int(max(0, top - pad_y))
    x1, y1 = int(min(width, right + pad_x)), int(min(height, bottom + pad_y))
    return _jpeg(drawn[y0:y1, x0:x1], CROP_SIDE)


def parsed_answer(text: str) -> dict | None:
    try:
        parsed = json.loads(repair_json(text))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def question(described: dict[str, str]) -> str:
    """Who each person is, asked only for the people described: a name such as p1 tells the model nothing to go on."""
    if not described:
        return PERSON
    return PERSON_AND_WHO.format(people=", ".join(f'"{name}" ({looks})' for name, looks in described.items()))


def problems(people_seen: int | None, boxes: list[dict], names: list[str]) -> list[dict]:
    """Where the answers disagree with what was drawn: someone in view without a skeleton, someone drawn twice, a box on no one, a name on the wrong person."""
    found = []
    on_people = sum(box["person"] is not False for box in boxes)
    if people_seen is not None and people_seen > on_people:
        found.append({"kind": MISSED, "box": None, "says": f"{people_seen} people in view, {on_people} with a skeleton on them"})
    elif people_seen is not None and people_seen < on_people:
        found.append({"kind": TWICE, "box": None, "says": f"{on_people} skeletons on {people_seen} people in view: someone is drawn twice"})
    # a name belongs to one person: given to two boxes, it says the model cannot tell them apart
    given = Counter(box["who"] for box in boxes if box["person"] is not False)
    for box in boxes:
        shown = box["name"] or "unassigned"
        if box["person"] is False:
            found.append({"kind": ON_NO_ONE, "box": box["name"], "says": f"the box named {shown} is not on a person"})
        elif box["name"] and box["who"] in names and box["who"] != box["name"] and given[box["who"]] == 1:
            found.append({"kind": WRONG_NAME, "box": box["name"], "says": f"the box named {box['name']} looks like {box['who']}"})
    return found


def look_at_frame(backend: LLMBackend, image: np.ndarray, people: pd.DataFrame, subjects: list[str], described: dict[str, str]) -> dict:
    """described is qc.look.people for this recording's people; only their names are checked."""
    counted = parsed_answer(backend.complete(COUNT, "Count the people.", seed=0, images=[_jpeg(image, LONG_SIDE)]).text)
    seen = counted.get("people") if counted else None
    unreadable = int(not isinstance(seen, (int, float)) or isinstance(seen, bool))
    people_seen = None if unreadable else int(seen)
    system = question(described)
    names = list(described)
    boxes = []
    for person in drawn_people(people):
        said = parsed_answer(backend.complete(system, "Check this person.", seed=0, images=[person_picture(image, person, subjects)]).text)
        is_person = said.get("person") if said else None
        unreadable += int(not isinstance(is_person, bool))
        who = str(said.get("who") or UNSURE).strip() if said else UNSURE
        boxes.append({
            "name": person["name"], "track": person["track"], "person": is_person if isinstance(is_person, bool) else None,
            "who": who if who in names else UNSURE,
        })
    return {"people_seen": people_seen, "boxes": boxes, "problems": problems(people_seen, boxes, names), "unreadable": unreadable}


def _key(picture: bytes, people: list[dict], model: str, described: dict[str, str]) -> str:
    drawn = [(person["name"], person["track"], [round(value, 1) for value in person["bounds"]]) for person in people]
    text = json.dumps([QUESTIONS, model, described, drawn])
    return hashlib.sha1(picture + text.encode("utf-8")).hexdigest()[:16]


def look_at_recording(
    backend: LLMBackend, cfg: LookSection, layout, recording, keypoints: pd.DataFrame, index: pd.DataFrame, *, seed: int,
) -> dict:
    """Asks about each review frame not already answered for the same picture, people, model and descriptions, then draws the random check."""
    rid = recording.recording_id
    cv2 = cv2_module()
    subjects = [subject.subject_id for subject in recording.subjects]
    described = {name: looks for name, looks in cfg.people.items() if name in subjects}
    model = backend.model_version()
    earlier = {entry["file"]: entry for entry in read_look(layout, rid).get("frames", [])}
    plain = layout.renders_dir(rid) / PLAIN
    entries = []
    for row in index.itertuples():
        path = plain / row.file
        if not path.is_file():
            continue
        image = cv2.imread(str(path))
        people = keypoints[(keypoints["frame"] == row.frame) & (keypoints["video"] == row.video)]
        key = _key(path.read_bytes(), drawn_people(people), model, described)
        entry = earlier.get(row.file)
        if entry is None or entry.get("key") != key:
            entry = {"file": row.file, "t": float(row.t), "video": row.video, "frame": int(row.frame), "reason": row.reason, "key": key,
                     **look_at_frame(backend, image, people, subjects, described)}
        # read again from the answers, so a changed reading needs no new question
        entries.append({**entry, "problems": problems(entry["people_seen"], entry["boxes"], list(described)), "random_check": False})
    passed = [position for position, entry in enumerate(entries) if not entry["problems"] and not entry["unreadable"]]
    rng = np.random.default_rng([seed, zlib.crc32(rid.encode("utf-8"))])
    for position in rng.choice(passed, size=min(cfg.random_check, len(passed)), replace=False) if passed else []:
        entries[int(position)]["random_check"] = True
    look = {"model": model, "provider": cfg.provider, "described": described, "asked_at": datetime.now().isoformat(timespec="seconds"), "frames": entries}
    path = layout.look_path(rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(look, indent=2, ensure_ascii=False), encoding="utf-8")
    return look


def read_look(layout, recording_id: str) -> dict:
    path = layout.look_path(recording_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_checks(layout, recording_id: str) -> dict[str, dict]:
    path = layout.look_checks_path(recording_id)
    return json.loads(path.read_text(encoding="utf-8")).get("frames", {}) if path.is_file() else {}


def to_check(look: dict) -> list[dict]:
    """The frames a person is asked about: every one the model flagged, and the ones it passed that were drawn at random."""
    return [entry for entry in look.get("frames", []) if entry["problems"] or entry.get("random_check")]


def verdict(entry: dict, checks: dict[str, dict]) -> str | None:
    """right or wrong about the model, only for the answer the person saw: a frame asked again is checked again."""
    check = checks.get(entry["file"])
    return check["verdict"] if check and check.get("key") == entry["key"] else None


def set_check(layout, recording_id: str, file: str, decided: str | None, by: str) -> Path:
    """decided is right when the model was right about the frame, wrong when not, None to take a verdict back."""
    entry = next((item for item in to_check(read_look(layout, recording_id)) if item["file"] == file), None)
    if entry is None:
        raise MacawError(f"{file} is not among the frames of {recording_id} that wait for a person; see macaw qc --look")
    checks = read_checks(layout, recording_id)
    if decided is None:
        checks.pop(file, None)
    elif decided not in (RIGHT, WRONG):
        raise MacawError(f"a verdict is {RIGHT} or {WRONG}, not {decided}")
    else:
        checks[file] = {"verdict": decided, "key": entry["key"], "by": by, "at": datetime.now().isoformat(timespec="seconds")}
    path = layout.look_checks_path(recording_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"frames": checks}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def summary(layout, recording_id: str) -> dict | None:
    look = read_look(layout, recording_id)
    if not look:
        return None
    checks = read_checks(layout, recording_id)
    frames = look.get("frames", [])
    flagged = [entry for entry in frames if entry["problems"]]
    drawn = [entry for entry in frames if entry.get("random_check")]
    said = {entry["file"]: verdict(entry, checks) for entry in flagged + drawn}
    return {
        "model": look.get("model"), "provider": look.get("provider"), "described": sorted(look.get("described") or {}), "frames": len(frames), "flagged": len(flagged),
        "flagged_checked": sum(said[entry["file"]] is not None for entry in flagged),
        "flagged_right": sum(said[entry["file"]] == RIGHT for entry in flagged),
        "drawn": len(drawn), "drawn_checked": sum(said[entry["file"]] is not None for entry in drawn),
        "drawn_wrong": sum(said[entry["file"]] == WRONG for entry in drawn),
        "unreadable": sum(entry.get("unreadable", 0) for entry in frames),
        "kinds": dict(Counter(item["kind"] for entry in flagged for item in entry["problems"])),
    }


def describe(found: dict) -> str:
    """One line on what the model flagged and what people made of it."""
    text = f"{found['flagged']} of {found['frames']} review frame(s) flagged by {found['model']}"
    if found["flagged"]:
        text += f", {found['flagged_right']} of {found['flagged_checked']} checked confirmed by a person" if found["flagged_checked"] else ", none checked yet"
    if found["drawn"]:
        text += (
            f"; a person found {found['drawn_wrong']} of {found['drawn_checked']} frame(s) it passed, drawn at random, wrong"
            if found["drawn_checked"] else f"; {found['drawn']} frame(s) it passed wait for a person"
        )
    return text


def project_summary(config) -> dict | None:
    """The recordings' summaries added up, for the Methods text."""
    from macaw.execution.runner import resolve_recordings
    from macaw.project.layout import layout_for

    layout = layout_for(config, create=False)
    found = [item for item in (summary(layout, rid) for rid in resolve_recordings(config, None)) if item]
    if not found:
        return None
    counts = ("frames", "flagged", "flagged_checked", "flagged_right", "drawn", "drawn_checked", "drawn_wrong")
    return {
        **{key: sum(item[key] for item in found) for key in counts},
        "recordings": len(found), "models": sorted({str(item["model"]) for item in found}),
        "providers": sorted({str(item["provider"]) for item in found}),
        "described": sorted({name for item in found for name in item["described"]}),
    }

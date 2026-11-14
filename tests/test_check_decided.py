import json
from types import SimpleNamespace

from macaw.config.schema import MacawConfig
from macaw.extractors import prep
from macaw.extractors.audio.content import diarization
from macaw.extractors.audio.content.diarization import AUTOMATIC, SpeakerReviewGate
from macaw.extractors.base import PERSON, SOFTWARE
from macaw.extractors.prep import AlignReviewGate
from macaw.extractors.video.pose.keypoints import PoseExtractor, PoseReviewGate
from macaw.project.layout import layout_for
from macaw.project.names import add_clicks


def _config(tmp_path, **sections):
    return MacawConfig.model_validate({
        "project": {"name": "t", "raw_dir": str(tmp_path / "raw"), "output_dir": str(tmp_path / "out"), "n_subjects": 2, "roles": ["parent", "child"]},
        **sections,
    })


RECORDING = SimpleNamespace(recording_id="r1", media=[object(), object()])


def test_who_is_who_was_decided_by_a_click_or_left_to_the_software(tmp_path):
    config = _config(tmp_path)
    layout = layout_for(config)
    gate = PoseReviewGate(config)
    assert gate.decided_by(RECORDING) is None
    owners = layout.artifacts_dir("r1", PoseExtractor.name) / "owners"
    owners.parent.mkdir(parents=True, exist_ok=True)
    owners.write_text(json.dumps({"room": {"people": {"parent": [1]}, "chosen_by": {"parent": "suggested for being most in view"}, "doubt": None}}))
    assert gate.decided_by(RECORDING) == SOFTWARE
    add_clicks(layout.names_path("r1"), "room", 1.0, {"parent": (0.3, 0.5)}, "ann")
    assert gate.decided_by(RECORDING) == PERSON


def test_voices_named_by_the_suggestion_or_by_a_person(tmp_path, monkeypatch):
    config = _config(tmp_path, text={"diarization": "pyannote"})
    monkeypatch.setattr(diarization, "read_voices", lambda config, rid: {"S0": {}})
    gate, path = SpeakerReviewGate(config), layout_for(config).voice_owners_path("r1")
    assert gate.decided_by(RECORDING) is None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"people": {"S0": "child"}, "by": AUTOMATIC}))
    assert gate.decided_by(RECORDING) == SOFTWARE
    path.write_text(json.dumps({"people": {"S0": "child"}, "by": "ann 2026-11-13T10:00:00"}))
    assert gate.decided_by(RECORDING) == PERSON
    assert SpeakerReviewGate(_config(tmp_path)).decided_by(RECORDING) is None


def test_offsets_kept_or_set_by_a_person_or_left_as_found(tmp_path, monkeypatch):
    config = _config(tmp_path)
    gate = AlignReviewGate(config)
    monkeypatch.setattr(AlignReviewGate, "inputs_for", lambda self, recording: [tmp_path / "offsets.json"])
    found = [SimpleNamespace(target="a", reference="a", method="gcc_phat", accepted=None),
             SimpleNamespace(target="b", reference="a", method="gcc_phat", accepted=None)]
    monkeypatch.setattr(prep, "read_offset_table", lambda path: found)
    assert gate.decided_by(RECORDING) == SOFTWARE
    found[1].accepted = "ann 2026-11-13"
    assert gate.decided_by(RECORDING) == PERSON
    assert AlignReviewGate(_config(tmp_path, align={"method": "none"})).decided_by(RECORDING) is None

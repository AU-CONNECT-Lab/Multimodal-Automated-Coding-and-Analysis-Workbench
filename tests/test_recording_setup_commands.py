from __future__ import annotations

import wave

from typer.testing import CliRunner

from macaw.cli.main import app


def two_setups(base):
    """A project whose setup is sa, with a recording r1 that chose setup sb and its one microphone."""
    (base / "project.yaml").write_text(
        "project:\n  name: p\n  raw_dir: raw\n  output_dir: out\n  roles: [a, b]\nsetup:\n  name: sa\n", encoding="utf-8",
    )
    (base / "setups").mkdir()
    (base / "setups" / "sa.yaml").write_text("microphones:\n  mic_a:\n    hears: [a, b]\n", encoding="utf-8")
    (base / "setups" / "sb.yaml").write_text("microphones:\n  mic_b:\n    hears: [a, b]\n", encoding="utf-8")
    (base / "recordings").mkdir()
    (base / "recordings" / "r1.yaml").write_text("setup:\n  name: sb\n", encoding="utf-8")
    folder = base / "raw" / "r1" / "mic_b"
    folder.mkdir(parents=True)
    with wave.open(str(folder / "x.wav"), "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(16000)
        sound.writeframes(b"\0\0" * 1600)
    return base / "project.yaml"


def test_a_command_on_one_recording_finds_the_devices_of_the_setup_it_chose(tmp_path):
    config = two_setups(tmp_path)
    outcome = CliRunner().invoke(app, ["set-offset", "-c", str(config), "-r", "r1", "--device", "mic_b", "--offset", "0.5", "--by", "ann"])
    # found among r1's own devices, where the project's setup has none: its one file is the reference
    assert "mic_b is the reference" in str(outcome.exception) and "not a device" not in str(outcome.exception)


def test_a_recording_without_sound_to_align_by_leaves_the_others_aligned(tmp_path, monkeypatch):
    from macaw.cli import align as align_module
    from macaw.exceptions import AlignmentFailedError

    config = two_setups(tmp_path)
    (tmp_path / "recordings" / "r2.yaml").write_text("setup:\n  name: sb\n", encoding="utf-8")
    (tmp_path / "raw" / "r2" / "mic_b").mkdir(parents=True)
    (tmp_path / "raw" / "r2" / "mic_b" / "x.wav").write_bytes((tmp_path / "raw" / "r1" / "mic_b" / "x.wav").read_bytes())
    aligned = []

    def fake(recording, cfg, infos, on_pair=None):
        if recording.recording_id == "r1":
            raise AlignmentFailedError("r1: no file has an audio stream to align by")
        aligned.append(recording.recording_id)
        return []

    monkeypatch.setattr(align_module, "align_recording", fake)
    monkeypatch.setattr(align_module, "_clip", lambda *args, **kwargs: None)
    outcome = CliRunner().invoke(app, ["align", "-c", str(config)])
    assert aligned == ["r2"] and "r1: not aligned" in outcome.output and "not aligned: r1" in str(outcome.exception)

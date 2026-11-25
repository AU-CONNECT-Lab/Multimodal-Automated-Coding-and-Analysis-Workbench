from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import archive_resolved, resolve_config, scaffold_project_config
from macaw.exceptions import ConfigError, MacawError, RecordingNotFoundError
from macaw.project.recording import build_recording, camera_clips
from macaw.project.setup import require_setup, setup_path, write_setup
from setup_helpers import EVERYONE, camera, mics, put, write_setup as write_devices


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=raw, output_dir=tmp_path / "out", roles=("parent", "child"))
    raw.mkdir()
    return config


def built(project, recording="003"):
    return build_recording(recording, resolve_config(project))


class TestDeviceFolders:
    def test_each_file_is_its_folders_device_and_the_setup_says_whom_it_records(self, project):
        raw = project.parent / "raw"
        put(raw, "003", "front", "00001.MTS")
        put(raw, "003", "side", "00001.MTS")
        put(raw, "003", "parent_mic", "take 1.wav")
        write_devices(project, cameras={"front": camera(["child"]), "side": camera(uses=("body",), hears=EVERYONE)}, microphones=mics(["parent"]))
        media = {item.name: item for item in built(project).media}
        assert set(media) == {"front", "side", "parent_mic"}
        assert (media["front"].subject_id, media["side"].subject_id, media["parent_mic"].subject_id) == ("child", None, "parent")
        assert media["front"].path != media["side"].path
        assert media["side"].hears == ("child", "parent") and media["parent_mic"].hears == ("parent",)

    def test_a_microphone_is_read_for_its_sound_whatever_file_holds_it(self, project):
        put(project.parent / "raw", "003", "recorder", "a.mp4")
        write_devices(project, microphones={"recorder": {"hears": list(EVERYONE)}})
        (item,) = built(project).media
        assert item.kind == "audio"

    def test_two_files_in_one_device_folder_stop_the_recording(self, project):
        raw = project.parent / "raw"
        put(raw, "003", "front", "00001.MTS")
        put(raw, "003", "front", "00002.MTS")
        write_devices(project, cameras={"front": camera()})
        with pytest.raises(RecordingNotFoundError, match="front folder holds 2 files.*00001.MTS, 00002.MTS"):
            built(project)

    def test_a_camera_folder_holding_no_video_is_refused(self, project):
        put(project.parent / "raw", "003", "front", "a.wav")
        write_devices(project, cameras={"front": camera()})
        with pytest.raises(RecordingNotFoundError, match="front is a camera"):
            built(project)

    def test_what_is_missing_or_left_unread_is_noted_not_guessed(self, project):
        raw = project.parent / "raw"
        put(raw, "003", "front", "a.mp4")
        put(raw, "003", "Front2", "b.mp4")
        (raw / "003" / "loose.wav").write_bytes(b"")
        write_devices(project, cameras={"front": camera()}, microphones=mics())
        recording = built(project)
        assert [item.name for item in recording.media] == ["front"]
        notes = recording.metadata["setup_notes"]
        assert "no parent_mic folder, so this recording has no parent_mic" in notes
        assert "the Front2 folder is not a device of setup main, so it was not read" in notes
        assert "loose.wav lies outside the device folders, so it was not read" in notes

    @pytest.mark.parametrize("name", ["00001.MPG", "clip.mpeg", "00001.M2TS", "clip.ts", "clip.wmv"])
    def test_program_and_transport_streams_are_videos(self, project, name):
        put(project.parent / "raw", "003", "front", name)
        write_devices(project, cameras={"front": camera()})
        (item,) = built(project).media
        assert item.kind == "video" and item.path.name == name

    def test_a_file_of_a_kind_not_read_is_named_not_skipped_in_silence(self, project):
        raw = project.parent / "raw"
        put(raw, "003", "front", "clip.vob")
        put(raw, "003", "side", "a.mp4")
        put(raw, "003", "side", "a.mxf")
        put(raw, "003", "side", ".hidden.dat")
        write_devices(project, cameras={"front": camera(), "side": camera(uses=("body",))})
        recording = built(project)
        assert [item.name for item in recording.media] == ["side"]
        notes = recording.metadata["setup_notes"]
        held = next(note for note in notes if note.startswith("the front folder"))
        assert "holds only clip.vob, which the package does not read as audio or video, so this recording has no front" in held
        assert ".mpg" in held
        assert "the side folder also holds a.mxf, which the package does not read as audio or video, so it was left out" in notes
        assert not any(".hidden" in note for note in notes)

    def test_the_run_plan_says_what_was_left_unread(self, project):
        put(project.parent / "raw", "003", "front", "clip.vob")
        put(project.parent / "raw", "003", "side", "a.mp4")
        write_devices(project, cameras={"front": camera(), "side": camera(uses=("body",))})
        result = CliRunner().invoke(app, ["run", "-c", str(project), "--dry-run"])
        assert "note: in 003, the front folder holds only clip.vob" in result.output

    def test_a_board_clip_belongs_to_the_camera_whose_folder_holds_it(self, tmp_path):
        put(tmp_path, "lab", "left", "MVI_0001.MP4")
        put(tmp_path, "lab", "right", "MVI_0001.MP4")
        assert {name: path.parent.name for name, path in camera_clips(tmp_path / "lab").items()} == {"left": "left", "right": "right"}
        (tmp_path / "lab" / "stray.mp4").write_bytes(b"")
        with pytest.raises(RecordingNotFoundError, match="named after its camera"):
            camera_clips(tmp_path / "lab")


class TestSetupFile:
    @pytest.mark.parametrize(
        "cameras, microphones, message",
        [
            ({"front": camera(["teacher"])}, {}, "names teacher, which project.roles does not have"),
            ({"front": {"body": [], "hears": []}}, {}, "read for nobody"),
            ({"front": camera()}, {"front": {"hears": ["parent"]}}, "both a camera and a microphone"),
            ({"front cam": camera()}, {}, "folder's name"),
            ({}, {"a": {"hears": ["parent"]}, "b": {"hears": ["parent"]}}, "both carry parent's voice alone"),
            ({"room": camera(hears=EVERYONE)}, {"table": {"hears": list(EVERYONE)}}, "voices are separated on one"),
        ],
    )
    def test_a_setup_that_cannot_be_read_one_way_is_refused(self, project, cameras, microphones, message):
        write_devices(project, cameras=cameras, microphones=microphones)
        with pytest.raises(ConfigError, match=message):
            resolve_config(project)

    def test_a_recording_can_take_another_setup(self, project):
        raw = project.parent / "raw"
        put(raw, "003", "front", "a.mp4")
        put(raw, "004", "camcorder", "a.MTS")
        write_devices(project, cameras={"front": camera()})
        write_devices(project, cameras={"camcorder": camera()}, name="home")
        (project.parent / "recordings").mkdir()
        (project.parent / "recordings" / "004.yaml").write_text("setup:\n  name: home\n", encoding="utf-8")
        assert [item.name for item in built(project, "003").media] == ["front"]
        assert [item.name for item in build_recording("004", resolve_config(project, recording_id="004")).media] == ["camcorder"]

    def test_a_task_reads_the_other_setup_alone_from_the_archived_settings(self, project):
        write_devices(project, cameras={"front": camera()})
        write_devices(project, cameras={"camcorder": camera()}, name="home")
        (project.parent / "recordings").mkdir()
        (project.parent / "recordings" / "004.yaml").write_text("setup:\n  name: home\n", encoding="utf-8")
        archived = archive_resolved(resolve_config(project), project.parent / "out")
        assert list(resolve_config(archived, recording_id="004").setup.cameras) == ["camcorder"]
        assert list(resolve_config(archived, recording_id="003").setup.cameras) == ["front"]

    def test_a_path_in_a_recordings_file_follows_the_project_however_the_settings_are_read(self, project):
        from macaw.execution.runner import recording_configs

        (project.parent / "recordings").mkdir()
        (project.parent / "recordings" / "004.yaml").write_text("coding:\n  codebook: books/own.yaml\n", encoding="utf-8")
        mine = (project.parent / "books" / "own.yaml").resolve()
        archived = archive_resolved(resolve_config(project), project.parent / "out")
        assert resolve_config(project, recording_id="004").coding.codebook == mine
        assert resolve_config(archived, recording_id="004").coding.codebook == mine
        assert recording_configs(resolve_config(project), ["004"])["004"].coding.codebook == mine

    def test_a_recording_naming_a_setup_without_a_file_is_refused(self, project):
        write_devices(project, cameras={"front": camera()})
        (project.parent / "recordings").mkdir()
        (project.parent / "recordings" / "004.yaml").write_text("setup:\n  name: hom\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="setup hom has no file"):
            resolve_config(project, recording_id="004")

    def test_the_file_is_named_by_its_name_and_written_without_it(self, project):
        write_devices(project, cameras={"front": camera()})
        path = setup_path(resolve_config(project))
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        path.write_text(yaml.safe_dump({**data, "name": "other"}), encoding="utf-8")
        with pytest.raises(ConfigError, match="named by its file"):
            resolve_config(project)
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        config = resolve_config(project)
        again = write_setup(path, config.setup)
        assert "name" not in yaml.safe_load(again.read_text(encoding="utf-8"))
        assert resolve_config(project).setup == config.setup

    def test_saving_the_setup_again_keeps_what_a_person_wrote_beside_it(self, project):
        write_devices(project, cameras={"front": camera(), "side": camera(uses=("body",))})
        path = setup_path(resolve_config(project))
        path.write_text("# the lab's playroom\n" + path.read_text(encoding="utf-8").replace("side:", "side:  # by the door"), encoding="utf-8")
        config = resolve_config(project)
        changed = config.setup.model_copy(update={"ignore": {"front": [(0.0, 0.0, 0.5, 0.5)]}})
        text = write_setup(path, changed).read_text(encoding="utf-8")
        assert "# the lab's playroom" in text and "# by the door" in text
        assert resolve_config(project).setup == changed
        dropped = changed.model_copy(update={"cameras": {"front": changed.cameras["front"]}, "ignore": {}})
        text = write_setup(path, dropped).read_text(encoding="utf-8")
        assert "# the lab's playroom" in text and "side" not in text and resolve_config(project).setup == dropped


class TestRequired:
    def test_nothing_runs_before_the_setup_is_saved(self, project):
        put(project.parent / "raw", "003", "front", "a.mp4")
        outcome = CliRunner().invoke(app, ["run", "-c", str(project), "--dry-run"])
        assert outcome.exit_code != 0 and "not saved yet" in str(outcome.exception)
        write_devices(project)
        with pytest.raises(MacawError, match="names no camera or microphone"):
            require_setup(resolve_config(project))
        write_devices(project, cameras={"front": camera()})
        outcome = CliRunner().invoke(app, ["run", "-c", str(project), "--dry-run"])
        # past the setup; what stops it now is a step this machine cannot run
        assert outcome.exit_code == 0 or "nothing was started" in str(outcome.exception)
        assert "003:prep.media" in outcome.output

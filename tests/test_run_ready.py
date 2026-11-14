import pytest

from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.config.schema import MacawConfig
from macaw.exceptions import ConfigError
from macaw.execution import runner
from macaw.extractors.audio.content.diarization import unread_voices
from macaw.project.layout import layout_for


def _config(cameras=None, microphones=None, diarization="per_track", **sections):
    return MacawConfig.model_validate({
        "project": {"name": "t", "raw_dir": "raw", "output_dir": "out", "n_subjects": 2, "roles": ["parent", "child"]},
        "setup": {"name": "main", "cameras": cameras or {}, "microphones": microphones or {}},
        "text": {"diarization": diarization},
        **sections,
    })


ONE_CAMERA = {"camera": {"body": ["parent", "child"], "hears": ["parent", "child"]}}


class TestUnreadVoices:
    def test_a_camera_hearing_both_under_a_microphone_per_person_is_named_with_what_to_set(self):
        message = unread_voices(_config(ONE_CAMERA))
        assert message.startswith("camera carries the voices of child, parent, but no device carries the voice of child or parent alone")
        assert message.endswith("set text.diarization: pyannote to separate the voices on camera")

    def test_only_the_person_without_a_device_of_their_own_is_named(self):
        message = unread_voices(_config(ONE_CAMERA, {"mic": {"hears": ["parent"]}}))
        assert "no device carries the voice of child alone" in message

    def test_separated_voices_or_a_device_for_everyone_need_nothing(self):
        assert unread_voices(_config(ONE_CAMERA, diarization="pyannote")) is None
        assert unread_voices(_config(ONE_CAMERA, {"mic_p": {"hears": ["parent"]}, "mic_c": {"hears": ["child"]}})) is None


class TestCheckReady:
    def test_a_step_that_cannot_run_here_stops_everything_with_its_reason(self):
        with pytest.raises(ConfigError, match=r"nothing was started[\s\S]*pose\.detect: pose\.backend is not set"):
            runner.check_ready({"rec1": _config(ONE_CAMERA, diarization="pyannote")}, ["pose.detect"])

    def test_voices_left_unread_are_said_only_when_a_step_reads_them(self):
        config = _config(ONE_CAMERA)
        runner.check_ready({"rec1": config}, ["prep.media"])
        with pytest.raises(ConfigError, match=r"set text\.diarization: pyannote"):
            runner.check_ready({"rec1": config}, ["prep.media", "prosody.frame"])

    def test_steps_finished_for_every_recording_are_not_checked(self):
        from macaw.execution.dag import build_dag

        dag = build_dag(_config(ONE_CAMERA, diarization="pyannote"), ["pose.detect"])
        done = {f"{r}:{name}" for r in "ab" for name in dag.topological_order() if name != "pose.detect"}
        assert runner.pending_steps(dag, ["a", "b"], done | {"a:pose.detect"}) == ["pose.detect"]
        assert runner.pending_steps(dag, ["a"], done | {"a:pose.detect"}) == []


def test_a_run_whose_pose_backend_is_not_set_starts_no_task(tmp_path):
    project = tmp_path / "project.yaml"
    scaffold_project_config(project, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    config = resolve_config(project)
    with pytest.raises(ConfigError, match=r"pose\.detect: pose\.backend is not set"):
        runner.run(config, recordings=["rec1"], extractors=["pose.detect"])
    assert not layout_for(config).job_db_path.exists()

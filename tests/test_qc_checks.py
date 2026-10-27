from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.schema import QCSection
from macaw.exceptions import ConfigError
from macaw.extractors.base import QCWarning
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.project.layout import Layout
from macaw.qc.checks import (
    CANDIDATE_SCHEMA,
    RecordingResults,
    candidate_spans,
    check_motion_energy,
    check_mouth_match,
    check_octave_error,
    check_transcript_match,
    check_track_quality,
    octave_jumps,
    read_candidates,
    run_checks,
    warnings_to_frame,
    write_candidates,
)
from macaw.store.keypoints import write_keypoints
from macaw.store.longtable import build_long_rows, write_long_table
from macaw.store.schema import KEYPOINT_SCHEMA, LONG_TABLE_SCHEMA, Scale, coerce_frame, empty_frame, validate_frame

FPS = 30.0
RID = "sub-003_ses-12mo"
RECORDING = SimpleNamespace(recording_id=RID)

# x offsets from the person's centre; y offsets below the shoulders
LAYOUT = {
    "nose": (0, -60), "left_eye": (8, -68), "right_eye": (-8, -68), "left_ear": (16, -64),
    "right_ear": (-16, -64), "left_shoulder": (40, 0), "right_shoulder": (-40, 0),
    "left_elbow": (50, 60), "right_elbow": (-50, 60), "left_wrist": (55, 120),
    "right_wrist": (-55, 120), "left_hip": (25, 150), "right_hip": (-25, 150),
    "left_knee": (25, 230), "right_knee": (-25, 230), "left_ankle": (25, 310),
    "right_ankle": (-25, 310),
}


def person(subject_id, *, centre=300.0, frames=range(60), confidence=None, override=None):
    """confidence maps frame -> value; override maps (frame, joint) -> (x, y)."""
    rows = []
    for frame in frames:
        for joint in STANDARD_JOINTS:
            dx, dy = LAYOUT[joint]
            x, y = (override or {}).get((frame, joint), (centre + dx, 200.0 + dy))
            rows.append(
                {
                    "recording_id": RID, "subject_id": subject_id, "source": "pose.keypoints",
                    "video": "room.mp4", "frame": frame, "t": frame / FPS, "joint": joint, "x": x, "y": y,
                    "z": float("nan"),
                    # alternating, so the person's own spread is not zero
                    "confidence": (confidence or {}).get(frame, 0.88 if frame % 2 else 0.92),
                    "track_id": 0, "units": "pixel", "metric_scale_valid": None,
                }
            )
    return coerce_frame(pd.DataFrame(rows), KEYPOINT_SCHEMA, "keypoints")


def fake_results(keypoints=None, f0=None, swaps=None, extractor_warnings=()):
    return SimpleNamespace(
        keypoints=lambda: keypoints if keypoints is not None else empty_frame(KEYPOINT_SCHEMA),
        long_table=lambda source, features=None: f0 if f0 is not None else pd.DataFrame(),
        artifact=lambda extractor, name: swaps,
        extractor_warnings=lambda: list(extractor_warnings),
    )


def two_people(**overrides):
    return pd.concat([person("parent", centre=150.0, **overrides), person("child", centre=450.0)])


def reasons(result):
    return list(result.candidates["reason"])


class TestCandidateSpans:
    def test_a_run_of_flags_is_one_span_pointing_at_its_worst_sample(self):
        spans = candidate_spans(
            RID, "parent", "c", "r",
            t=[0.0, 0.1, 0.2, 0.3, 0.4], flagged=[False, True, True, True, False],
            score=[0, 1, 5, 2, 0],
        )
        assert len(spans) == 1
        assert (spans.t_start.iloc[0], spans.t_end.iloc[0], spans.t.iloc[0]) == (0.1, 0.3, 0.2)
        assert spans.score.iloc[0] == 5

    def test_separate_runs_stay_separate(self):
        spans = candidate_spans(RID, None, "c", "r", [0, 1, 2, 3], [True, False, True, True], [1, 1, 1, 1])
        assert list(spans.t_start) == [0.0, 2.0]

    def test_nothing_flagged_is_an_empty_table_with_the_columns(self):
        spans = candidate_spans(RID, None, "c", "r", [0, 1], [False, False], [0, 0])
        validate_frame(spans, CANDIDATE_SCHEMA, "candidates")
        assert spans.empty


class TestTrackQuality:
    def test_a_clean_scene_points_at_nothing(self):
        result = check_track_quality(RECORDING, None, fake_results(two_people()))
        assert result.candidates.empty
        assert result.warnings == []

    def test_candidates_satisfy_the_schema(self):
        stray = person("parent", frames=range(5, 8)).assign(subject_id=pd.NA)
        result = check_track_quality(RECORDING, None, fake_results(pd.concat([two_people(), stray])))
        validate_frame(result.candidates, CANDIDATE_SCHEMA, "candidates")

    def test_an_unmatched_detection_is_flagged_and_belongs_to_nobody(self):
        """Statistics cannot tell a poster from a person; someone has to look at the frame."""
        stray = person("parent", centre=700.0, frames=range(5, 8)).assign(subject_id=pd.NA)
        result = check_track_quality(RECORDING, None, fake_results(pd.concat([two_people(), stray])))
        flagged = result.candidates[result.candidates.reason == "unassigned_detection"]
        assert len(flagged) == 1
        assert flagged.subject_id.isna().all()
        assert flagged.t_start.iloc[0] == pytest.approx(5 / FPS)
        assert flagged.t_end.iloc[0] == pytest.approx(7 / FPS)

    def test_a_repaired_swap_is_a_place_to_look(self, tmp_path):
        swaps = tmp_path / "swaps.json"
        swaps.write_text(json.dumps({"room.mp4": [{"t": 1.0, "frame": 30, "track_a": 0, "track_b": 1, "evidence": "x", "score": 0.7}]}))
        result = check_track_quality(RECORDING, None, fake_results(two_people(), swaps=swaps))
        flagged = result.candidates[result.candidates.reason == "track_swap"]
        assert list(flagged.t) == [1.0]
        assert list(flagged.score) == [0.7]

    def test_a_long_absence_is_flagged(self):
        frames = [*range(0, 10), *range(40, 60)]       # gone for a second
        keypoints = pd.concat([person("parent", centre=150.0, frames=frames), person("child", centre=450.0)])
        flagged = check_track_quality(RECORDING, None, fake_results(keypoints)).candidates
        flagged = flagged[flagged.reason == "missing_subject"]
        assert list(flagged.subject_id) == ["parent"]
        assert flagged.t_start.iloc[0] == pytest.approx(9 / FPS)
        assert flagged.t_end.iloc[0] == pytest.approx(40 / FPS)

    def test_a_short_dropout_is_not(self):
        frames = [*range(0, 10), *range(15, 60)]
        keypoints = pd.concat([person("parent", centre=150.0, frames=frames), person("child", centre=450.0)])
        assert "missing_subject" not in reasons(check_track_quality(RECORDING, None, fake_results(keypoints)))

    def test_absence_at_the_start_counts_too(self):
        keypoints = pd.concat([person("parent", centre=150.0, frames=range(30, 60)), person("child", centre=450.0)])
        flagged = check_track_quality(RECORDING, None, fake_results(keypoints)).candidates
        assert flagged[flagged.reason == "missing_subject"].t_start.tolist() == [0.0]

    def test_a_confidence_collapse_is_flagged_against_the_persons_own_level(self):
        keypoints = two_people(confidence={20: 0.3, 21: 0.3})
        flagged = check_track_quality(RECORDING, None, fake_results(keypoints)).candidates
        flagged = flagged[flagged.reason == "low_confidence"]
        assert list(flagged.subject_id) == ["parent"]
        assert flagged.t_start.iloc[0] == pytest.approx(20 / FPS)

    def test_a_stretched_limb_is_flagged(self):
        """A forearm three times its usual length is a keypoint on the wrong thing."""
        keypoints = two_people(override={(15, "left_wrist"): (150.0 + 55.0, 200.0 + 360.0)})
        flagged = check_track_quality(RECORDING, None, fake_results(keypoints)).candidates
        flagged = flagged[flagged.reason == "bone_stretch"]
        assert list(flagged.t) == [pytest.approx(15 / FPS)]

    def test_a_shortened_limb_is_only_foreshortening(self):
        keypoints = two_people(override={(15, "left_wrist"): (150.0 + 50.0, 200.0 + 70.0)})
        assert "bone_stretch" not in reasons(check_track_quality(RECORDING, None, fake_results(keypoints)))

    def test_each_reason_gets_one_summary_line(self):
        keypoints = two_people(confidence={20: 0.3, 40: 0.3})
        result = check_track_quality(RECORDING, None, fake_results(keypoints))
        assert len(result.warnings) == 1
        assert result.warnings[0].details == {"reason": "low_confidence", "n_spans": 2}
        assert result.warnings[0].severity == "info"

    def test_without_keypoints_it_says_so(self):
        assert check_track_quality(RECORDING, None, fake_results()).skipped


def contour(hz, step=0.01):
    return np.arange(len(hz)) * step, np.asarray(hz, dtype=float)


class TestOctaveJumps:
    def test_a_doubled_stretch_between_jumps_is_caught(self):
        spans, n_error, n_voiced = octave_jumps(*contour([200.0] * 40 + [400.0] * 10 + [200.0] * 40))
        assert n_error == 10
        assert n_voiced == 90
        assert spans[0]["t_start"] == pytest.approx(0.40)
        assert spans[0]["t_end"] == pytest.approx(0.49)
        assert spans[0]["score"] == pytest.approx(1.0)

    def test_a_halved_stretch_is_caught(self):
        _, n_error, _ = octave_jumps(*contour([200.0] * 40 + [100.0] * 10 + [200.0] * 40))
        assert n_error == 10

    def test_one_jump_is_enough(self):
        """A stretch running into the end of voicing is still entered by a jump."""
        _, n_error, _ = octave_jumps(*contour([200.0] * 40 + [400.0] * 10))
        assert n_error == 10

    def test_a_high_voice_without_a_jump_is_a_voice(self):
        """An utterance pitched high from its first frame is not a tracker error."""
        hz = [200.0] * 40 + [np.nan] * 5 + [400.0] * 10 + [np.nan] * 5 + [200.0] * 40
        assert octave_jumps(*contour(hz))[1] == 0

    def test_a_large_but_possible_step_is_not_a_jump(self):
        assert octave_jumps(*contour([200.0] * 40 + [260.0] * 10 + [200.0] * 40))[1] == 0

    def test_too_little_voicing_is_nothing(self):
        assert octave_jumps(*contour([np.nan, 200.0, np.nan])) == ([], 0, 1)


def f0_rows(hz_by_subject):
    frames = []
    for subject_id, hz in hz_by_subject.items():
        t, values = contour(hz)
        frames.append(
            build_long_rows(
                recording_id=RID, source="prosody.frame", scale=Scale.FRAME, subject_id=subject_id,
                values=pd.DataFrame({"t_start": t, "f0_hz": values}),
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestOctaveCheck:
    def test_many_jumped_frames_warn(self):
        f0 = f0_rows({"parent": [200.0] * 40 + [400.0] * 10 + [200.0] * 40, "child": [300.0] * 90})
        result = check_octave_error(RECORDING, None, fake_results(f0=f0))
        assert [(w.subject_id, w.severity) for w in result.warnings] == [("parent", "warn")]
        assert list(result.candidates.subject_id) == ["parent"]
        validate_frame(result.candidates, CANDIDATE_SCHEMA, "candidates")

    def test_a_trace_of_them_is_only_noted(self):
        f0 = f0_rows({"parent": [200.0] * 500 + [400.0] * 3 + [200.0] * 500})
        assert [w.severity for w in check_octave_error(RECORDING, None, fake_results(f0=f0)).warnings] == ["info"]

    def test_without_prosody_it_says_so(self):
        assert check_octave_error(RECORDING, None, fake_results(f0=pd.DataFrame())).skipped


def config(checks):
    return SimpleNamespace(qc=QCSection(checks=checks))


class TestRunChecks:
    def test_an_unknown_check_is_a_config_error(self):
        with pytest.raises(ConfigError, match="trak_quality"):
            run_checks(RECORDING, config(["trak_quality"]), fake_results())

    def test_an_unimplemented_check_is_reported_not_silently_dropped(self, monkeypatch):
        from macaw.qc import checks

        def unfinished(recording, config, results):
            raise NotImplementedError

        monkeypatch.setitem(checks._CHECKS, "unfinished", unfinished)
        result = run_checks(RECORDING, config(["unfinished"]), fake_results())
        assert [(w.check, w.message) for w in result.warnings] == [("unfinished", "not run: not implemented yet")]

    def test_a_check_without_input_says_why(self):
        result = run_checks(RECORDING, config(["track_quality"]), fake_results())
        assert result.warnings[0].message == "not run: no pose.keypoints output"

    def test_the_extractors_own_warnings_come_along(self):
        earlier = QCWarning(check="prosody.frame", recording_id=RID, subject_id="child", severity="warn", message="m")
        result = run_checks(RECORDING, config([]), fake_results(extractor_warnings=[earlier]))
        assert result.warnings == [earlier]

    def test_candidates_from_every_check_are_collected(self):
        keypoints = two_people(confidence={20: 0.3})
        f0 = f0_rows({"parent": [200.0] * 40 + [400.0] * 10 + [200.0] * 40})
        result = run_checks(RECORDING, config(["track_quality", "octave_error"]), fake_results(keypoints, f0=f0))
        assert set(result.candidates.check) == {"track_quality", "octave_error"}

    def test_only_overrides_the_configured_list(self):
        result = run_checks(RECORDING, config(["track_quality"]), fake_results(), only=["octave_error"])
        assert [w.check for w in result.warnings] == ["octave_error"]


class TestStorage:
    def test_recording_results_reads_what_a_run_left(self, tmp_path):
        layout = Layout(output_dir=tmp_path, work_dir=tmp_path)
        write_keypoints(person("parent", frames=range(3)), layout.keypoints_path(RID))
        write_long_table(f0_rows({"parent": [200.0] * 5}), layout.long_table_path(RID))
        warnings_file = layout.warnings_path(RID, "prosody.frame")
        warnings_file.parent.mkdir(parents=True)
        warnings_file.write_text(json.dumps([{"check": "c", "recording_id": RID, "subject_id": None, "severity": "info", "message": "m", "t_start": None, "t_end": None, "details": {}}]))

        results = RecordingResults(layout, RID)
        assert len(results.keypoints()) == 3 * len(STANDARD_JOINTS)
        assert len(results.long_table("prosody.frame", features=["f0_hz"])) == 5
        assert results.long_table("turns.structure").empty
        assert [w.message for w in results.extractor_warnings()] == ["m"]
        assert results.artifact("pose.keypoints", "swaps.json") is None

    def test_candidates_survive_a_round_trip(self, tmp_path):
        spans = candidate_spans(RID, "parent", "c", "r", [0.0, 0.1], [True, True], [1.0, 2.0])
        path = write_candidates(spans, tmp_path / "candidates.parquet")
        pd.testing.assert_frame_equal(read_candidates(path), spans)

    def test_warnings_become_a_flat_table(self):
        frame = warnings_to_frame([QCWarning("c", RID, None, "warn", "m", details={"n": 1})])
        assert frame.details.tolist() == ['{"n": 1}']
        assert list(warnings_to_frame([]).columns) == list(frame.columns)


def motion_rows(by_subject, fps=10.0):
    return pd.concat(
        [
            build_long_rows(
                recording_id=RID, source="motion.mea", scale=Scale.FRAME, subject_id=subject_id,
                values=pd.DataFrame({"t_start": np.arange(len(values)) / fps, "motion_energy": values}),
            )
            for subject_id, values in by_subject.items()
        ],
        ignore_index=True,
    )


def picture_file(tmp_path, share, identical, fps=10.0):
    path = tmp_path / "picture.parquet"
    pd.DataFrame({"frame": np.arange(len(share)), "t": np.arange(len(share)) / fps, "whole_picture": share,
                  "identical": identical, "video": "room.avi"}).to_parquet(path)
    return path


class TestMotionCheck:
    def run(self, tmp_path, energy, share, stalled=()):
        identical = np.zeros(len(share), bool)
        identical[list(stalled)] = True
        results = SimpleNamespace(
            long_table=lambda source, features=None: motion_rows(energy),
            artifact=lambda extractor, name: picture_file(tmp_path, share, identical),
        )
        return check_motion_energy(RECORDING, None, results)

    def test_a_flash_and_a_stall_are_pointed_at_for_everyone(self, tmp_path):
        share = np.full(300, 0.02)
        share[50] = 0.9
        share[100:130] = 0.0
        result = self.run(tmp_path, {"parent": np.full(300, 0.01)}, share, stalled=range(100, 130))
        spans = result.candidates.set_index("reason")
        assert spans.loc["picture_change", "t"] == pytest.approx(5.0)
        assert spans.loc["frozen_video", "t_start"] == pytest.approx(10.0)
        assert result.candidates.subject_id.isna().all()

    def test_a_short_identical_stretch_is_not_a_stall(self, tmp_path):
        share = np.full(300, 0.02)
        share[100:105] = 0.0
        assert self.run(tmp_path, {"parent": np.full(300, 0.01)}, share, stalled=range(100, 105)).candidates.empty

    def test_a_person_still_for_a_long_time_is_pointed_at(self, tmp_path):
        parent = np.full(300, 0.01)
        parent[100:250] = 0.0
        child = np.full(300, 0.01)
        child[100:150] = 0.0
        result = self.run(tmp_path, {"parent": parent, "child": child}, np.full(300, 0.02))
        assert list(zip(result.candidates.subject_id, result.candidates.reason)) == [("parent", "no_motion")]
        assert result.candidates.score.iloc[0] == pytest.approx(14.9)

    def test_a_still_room_below_the_threshold_is_not_a_stall(self, tmp_path):
        assert self.run(tmp_path, {"parent": np.full(300, 0.01)}, np.zeros(300)).candidates.empty

    def test_without_motion_energy_it_is_skipped(self):
        empty = SimpleNamespace(long_table=lambda source, features=None: pd.DataFrame(), artifact=lambda e, n: None)
        assert check_motion_energy(RECORDING, None, empty).skipped


def speaking_track(path, spans, seconds=60.0, rate=8000):
    """A quiet track with a loud tone in each (start, end) span."""
    import wave

    rng = np.random.default_rng(1)
    t = np.arange(int(seconds * rate)) / rate
    audio = rng.normal(0, 0.003, t.size)
    for start, end in spans:
        here = (t >= start) & (t < end)
        audio[here] += 0.3 * np.sin(2 * np.pi * 200 * t[here])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def transcript_file(path, spans):
    from macaw.extractors.audio.content.asr import TRANSCRIPT_SCHEMA

    rows = [{"recording_id": RID, "subject_id": "parent", "utterance_index": i, "t_start": a, "t_end": b, "text": "hi",
             "n_words": 1, "confidence": 1.0, "avg_logprob": np.nan, "no_speech_prob": np.nan, "language": "en"}
            for i, (a, b) in enumerate(spans)]
    coerce_frame(pd.DataFrame(rows), TRANSCRIPT_SCHEMA, "transcript").to_parquet(path)
    return path


SPEECH = [(start, start + 1.5) for start in (2.0, 7.0, 9.5, 15.0, 21.0, 22.8, 30.0, 33.0, 41.0, 48.0, 52.5, 57.0)]


class TestTranscriptMatch:
    def run(self, tmp_path, monkeypatch, spoken, written):
        from macaw.extractors import prep as prep_module

        track = speaking_track(tmp_path / "parent.wav", spoken)
        monkeypatch.setattr(
            prep_module, "load_prep",
            lambda config, rid: SimpleNamespace(t_origin_s=0.0, tracks={"parent": "parent.wav"}, track_path=lambda s: track),
        )
        path = transcript_file(tmp_path / "transcript.parquet", written)
        return check_transcript_match(RECORDING, None, SimpleNamespace(artifact=lambda extractor, name: path))

    def test_a_transcript_of_this_mic_passes(self, tmp_path, monkeypatch):
        result = self.run(tmp_path, monkeypatch, SPEECH, SPEECH)
        assert [w.severity for w in result.warnings] == ["info"]
        assert result.candidates.empty

    def test_a_transcript_of_something_else_warns_and_points_at_quiet_utterances(self, tmp_path, monkeypatch):
        elsewhere = [(start + 3.0, end + 3.0) for start, end in SPEECH]
        result = self.run(tmp_path, monkeypatch, SPEECH, elsewhere)
        assert result.warnings[0].severity == "warn"
        assert set(result.candidates.reason) == {"quiet_utterance"}

    def test_one_made_up_utterance_is_pointed_at_without_failing_the_rest(self, tmp_path, monkeypatch):
        result = self.run(tmp_path, monkeypatch, SPEECH, SPEECH + [(44.0, 45.0)])
        assert result.warnings[0].severity == "info"
        assert result.candidates.t_start.tolist() == [44.0]

    def test_without_a_transcript_it_is_skipped(self):
        assert check_transcript_match(RECORDING, None, SimpleNamespace(artifact=lambda e, n: None)).skipped


def talk_turns(seconds=240.0, turn=6.0):
    """parent and child take turns of `turn` seconds with short pauses; returns subject -> spans."""
    spans, t, who = {"parent": [], "child": []}, 1.0, "parent"
    rng = np.random.default_rng(3)
    while t + turn < seconds:
        length = turn * rng.uniform(0.5, 1.5)
        spans[who].append((t, t + length))
        t += length + rng.uniform(0.5, 3.0)
        who = "child" if who == "parent" else "parent"
    return spans


def mouth_rows(moves_with, spans, seconds=240.0, fps=25.0):
    """Mouth movement per frame for each subject, high while the person in moves_with[subject] talks."""
    from macaw.qc.checks import _within

    rng = np.random.default_rng(4)
    t = np.arange(0, seconds, 1 / fps)
    parts = []
    for subject, source in moves_with.items():
        starts, ends = np.array(spans[source]).T
        value = rng.gamma(1.0, 0.2, t.size) + 0.6 * _within(t, starts, ends) * rng.random(t.size)
        parts.append(build_long_rows(values=pd.DataFrame({"t_start": t, "mouth_movement": value}), scale=Scale.FRAME,
                                     recording_id=RID, source="face.movement", entities={}, subject_id=subject))
    return pd.concat(parts, ignore_index=True)


def two_person_transcript(path, spans):
    from macaw.extractors.audio.content.asr import TRANSCRIPT_SCHEMA

    rows = [{"recording_id": RID, "subject_id": subject, "utterance_index": i, "t_start": a, "t_end": b, "text": "hi",
             "n_words": 1, "confidence": 1.0, "avg_logprob": np.nan, "no_speech_prob": np.nan, "language": "en"}
            for subject, items in spans.items() for i, (a, b) in enumerate(items)]
    coerce_frame(pd.DataFrame(rows), TRANSCRIPT_SCHEMA, "transcript").to_parquet(path)
    return path


class TestMouthMatch:
    def run(self, tmp_path, moves_with, spans=None):
        spans = spans or talk_turns()
        path = two_person_transcript(tmp_path / "transcript.parquet", spans)
        mouth = mouth_rows(moves_with, spans if all(spans.values()) else talk_turns())
        results = SimpleNamespace(artifact=lambda extractor, name: path, long_table=lambda source, features=None: mouth)
        return {w.subject_id: w for w in check_mouth_match(RECORDING, None, results).warnings}

    def test_mouths_that_move_with_their_own_talk_pass(self, tmp_path):
        found = self.run(tmp_path, {"parent": "parent", "child": "child"})
        assert {s: w.severity for s, w in found.items()} == {"parent": "info", "child": "info"}
        assert found["parent"].details["rho"] > found["parent"].details["rival_rho"]

    def test_swapped_cameras_warn_and_name_whose_talk_the_mouth_follows(self, tmp_path):
        found = self.run(tmp_path, {"parent": "child", "child": "parent"})
        assert {s: w.severity for s, w in found.items()} == {"parent": "warn", "child": "warn"}
        assert found["parent"].details["rival"] == "child"

    def test_someone_who_barely_talks_is_not_judged(self, tmp_path):
        spans = talk_turns()
        spans["child"] = spans["child"][:2]
        found = self.run(tmp_path, {"parent": "parent", "child": "parent"}, spans)
        assert found["child"].severity == "info" and "too little" in found["child"].message

    def test_without_a_transcript_or_mouth_rows_it_is_skipped(self, tmp_path):
        assert check_mouth_match(RECORDING, None, SimpleNamespace(artifact=lambda e, n: None)).skipped
        path = two_person_transcript(tmp_path / "t.parquet", talk_turns())
        results = SimpleNamespace(artifact=lambda e, n: path, long_table=lambda source, features=None: empty_frame(LONG_TABLE_SCHEMA))
        assert "face.movement" in check_mouth_match(RECORDING, None, results).skipped


class TestAUAgreement:
    def results(self, rho_by_subject):
        rows = pd.concat(
            [
                build_long_rows(
                    recording_id=RID, source="face.au_check", scale=Scale.RECORDING, subject_id=subject,
                    values=pd.DataFrame({"t_start": [0.0], **{f"{unit}_agreement": [value] for unit, value in rho.items()},
                                         "au_agreement_median": [0.0]}),
                )
                for subject, rho in rho_by_subject.items()
            ],
            ignore_index=True,
        )
        return SimpleNamespace(long_table=lambda source, features=None: rows)

    def test_each_person_hears_which_units_the_models_agree_on(self):
        from macaw.qc.checks import check_au_agreement

        result = check_au_agreement(RECORDING, None, self.results({"a": {"au43": 0.8, "au12": 0.5, "au09": -0.2}, "b": {"au43": 0.7, "au09": 0.1, "au11": 0.0}}))
        by = {w.subject_id: w for w in result.warnings}
        assert by["a"].severity == "info" and "2 of 3 action units (au12, au43); read au09 with care" in by["a"].message
        assert by["b"].severity == "warn"

    def test_without_the_second_model_it_says_how_to_get_one(self):
        from macaw.qc.checks import check_au_agreement

        empty = SimpleNamespace(long_table=lambda source, features=None: empty_frame(LONG_TABLE_SCHEMA))
        assert "face.check_model" in check_au_agreement(RECORDING, None, empty).skipped

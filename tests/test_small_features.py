from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from macaw.config.schema import FaceSection, KinematicsSection, TextSection
from macaw.exceptions import ConfigError
from macaw.extractors.audio.content import features as text_module
from macaw.extractors.audio.content.features import TextFeatureExtractor, any_phrase, is_question, read_phrases, sentences, tokenize
from macaw.extractors.video.face.shares import FaceShareExtractor, frames_between
from macaw.extractors.video.pose import listening as listening_module
from macaw.extractors.video.pose.kinematics import UPPER_BODY, upper_body_speed, velocity
from macaw.extractors.video.pose.listening import ListeningExtractor, alone_spans, within
from macaw.project.recording import MediaFile
from macaw.qc.boilerplate import _face_shares, _listening
from macaw.store.schema import LONG_TABLE_SCHEMA, validate_frame
from test_kinematics import FPS, person

ENTITIES = {"group_id": "1", "session_id": None, "task": None, "run": None}


class TestSentences:
    def test_a_line_is_split_after_each_end_of_sentence(self):
        assert sentences("How was it? Fine. Really?! ok") == ["How was it?", "Fine.", "Really?!", "ok"]
        assert sentences("It costs 3.5 dollars. He said 'go.' Then left") == ["It costs 3.5 dollars.", "He said 'go.'", "Then left"]

    def test_a_question_ends_in_a_question_mark(self):
        assert [is_question(s) for s in ("How was it?", "Really?!", "What?\"", "Fine.", "why not")] == [True, True, True, False, False]

    def test_a_list_holds_words_and_phrases_matched_as_whole_words(self, tmp_path):
        path = tmp_path / "negative.txt"
        path.write_text("# feelings\nAnnoyed\nso nervous\n\n", encoding="utf-8")
        phrases = read_phrases(path)
        assert phrases == [("annoyed",), ("so", "nervous")]
        assert any_phrase(tokenize("I'm so annoyed"), phrases) and any_phrase(tokenize("I am so nervous today"), phrases)
        assert not any_phrase(tokenize("so very nervous, annoyedly"), phrases)


def run_text(monkeypatch, tmp_path, rows, **text):
    from macaw.turns import segmentation

    frame = pd.DataFrame(rows, columns=["subject_id", "t_start", "text"])
    frame["t_end"] = frame["t_start"] + 1.0
    frame["utterance_index"] = frame.groupby("subject_id").cumcount()
    monkeypatch.setattr(text_module, "load_transcript", lambda config, rid, others=False: frame)
    monkeypatch.setattr(segmentation, "load_turns", lambda config, rid, others=False: frame)
    monkeypatch.setattr(text_module, "load_prep", lambda config, rid: SimpleNamespace(t_origin_s=0.0, duration_s=120.0))
    recording = SimpleNamespace(recording_id="r1", entities=lambda: ENTITIES)
    extractor = TextFeatureExtractor(SimpleNamespace(text=TextSection(**text)))
    return extractor, extractor.run(recording, tmp_path)


def test_questions_and_sentences_with_listed_words_per_minute(monkeypatch, tmp_path):
    words = tmp_path / "negative.txt"
    words.write_text("annoyed\nso nervous\n", encoding="utf-8")
    rows = [
        ("parent", 0.0, "What happened? Tell me. Why?"),
        ("child", 5.0, "I'm so nervous. I was annoyed, so annoyed."),
        ("child", 9.0, "Can you help?"),
    ]
    extractor, result = run_text(monkeypatch, tmp_path, rows, word_lists={"negative": words})
    validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
    values = result.rows.set_index(["subject_id", "feature"])["value"]
    # two minutes analysed
    assert values["parent", "questions_per_minute"] == 1.0 and values["child", "questions_per_minute"] == 0.5
    assert values["child", "negative_per_minute"] == 1.0 and values["parent", "negative_per_minute"] == 0.0
    assert words in extractor._word_lists()
    spec = next(spec for spec in result.specs if spec.name == "negative_per_minute")
    assert spec.unit == "1/min" and "negative.txt" in spec.description and spec.params["word_lists"] == {"negative": "negative.txt"}


def test_a_word_list_name_has_to_fit_in_a_feature_name():
    with pytest.raises(ValidationError, match="feature name"):
        TextSection(word_lists={"Negative words": "x.txt"})


class TestUpperBody:
    def test_the_mean_speed_of_the_upper_body_points_found(self):
        n = 30
        moving = {"left_wrist": np.column_stack([300 + 55 + 3.0 * np.arange(n), np.full(n, 320.0)])}
        kp = person("child", n_frames=n, moves=moving)
        found = upper_body_speed(kp)
        wrist = velocity(kp, "left_wrist")
        middle = found.index[10]
        assert found[middle] == pytest.approx(wrist[middle] / len(UPPER_BODY)) and wrist[middle] == pytest.approx(3.0 * FPS)

    def test_a_frame_where_no_upper_body_point_is_found_has_none(self):
        kp = person("child", n_frames=12)
        kp.loc[(kp["frame"] == 11) & kp["joint"].isin(UPPER_BODY), ["x", "y"]] = np.nan
        assert np.isnan(upper_body_speed(kp)[11])


class TestFaceShares:
    def test_frames_of_a_video_inside_a_span_end_excluded(self):
        video = MediaFile(path=None, kind="video", fps=10.0, offset_s=1.0, duration_s=5.0)
        assert frames_between(video, 0.0, 10.0) == 50
        assert frames_between(video, 2.0, 3.0) == 10
        assert frames_between(video, 7.0, 9.0) == 0

    def run(self, au_above):
        video = MediaFile(path=None, kind="video", fps=10.0, offset_s=0.0, duration_s=10.0)
        recording = SimpleNamespace(recording_id="r1", entities=lambda: ENTITIES, views=lambda kind: {"child": video, "parent": video})
        # a face in 40 of the child's 100 frames, au12 above 0.5 in 10 of them; the parent's face never found
        times = np.arange(40) / 10.0
        faces = pd.concat([
            pd.DataFrame({"subject_id": "child", "feature": "face_yaw_deg", "t_start": times, "value": 0.0}),
            pd.DataFrame({"subject_id": "child", "feature": "au12", "t_start": times, "value": np.where(np.arange(40) < 10, 0.8, 0.1)}),
        ], ignore_index=True)
        step = FaceShareExtractor(SimpleNamespace(face=FaceSection(au_above=au_above)))
        return step, step._rows(recording, faces, 0.0, 10.0).set_index(["subject_id", "feature"])["value"]

    def test_the_share_of_frames_with_a_face_and_of_those_above_the_threshold(self):
        step, values = self.run({"au12": 0.5})
        assert values["child", "face_found_share"] == pytest.approx(0.4) and values["child", "au12_above_share"] == pytest.approx(0.25)
        assert values["parent", "face_found_share"] == 0.0 and np.isnan(values["parent", "au12_above_share"])
        assert {spec.name for spec in step._specs()} == {"face_found_share", "au12_above_share"}

    def test_an_action_unit_the_face_model_does_not_have_is_refused(self):
        with pytest.raises(ConfigError, match="au99"):
            FaceShareExtractor(SimpleNamespace(face=FaceSection(au_above={"au99": 0.5}))).check_available()

    def test_the_methods_name_the_thresholds(self):
        spec = SimpleNamespace(params={"au_above": {"au12": 0.5, "au06": 0.4}})
        assert "action unit 6 above 0.4 and action unit 12 above 0.5" in _face_shares(None, spec, set())


class TestListening:
    TALK = pd.DataFrame(
        [("parent", 0.0, 4.0), ("child", 3.0, 6.0), ("other", 7.0, 8.0), ("parent", 9.0, 10.0)],
        columns=["subject_id", "t_start", "t_end"],
    )

    def test_stretches_where_one_voice_talks_alone(self):
        assert alone_spans(self.TALK, 0.0, 10.0) == [
            ("parent", (0.0, 3.0)), ("child", (4.0, 6.0)), ("other", (7.0, 8.0)), ("parent", (9.0, 10.0)),
        ]
        assert alone_spans(self.TALK, 2.0, 5.0) == [("parent", (2.0, 3.0)), ("child", (4.0, 5.0))]
        assert within(np.array([0.0, 2.9, 3.0, 9.5]), [(0.0, 3.0), (9.0, 10.0)]).tolist() == [True, True, False, True]

    def test_each_person_s_movement_while_another_person_talks_alone(self, monkeypatch):
        monkeypatch.setattr(listening_module, "load_transcript", lambda config, rid, others=False: self.TALK)
        times = np.arange(0.0, 10.0, 0.5)
        series = pd.concat([
            pd.DataFrame({"subject_id": "child", "feature": "upper_body_speed", "t_start": times, "value": np.where(times < 3.0, 2.0, 10.0)}),
            pd.DataFrame({"subject_id": "parent", "feature": "upper_body_speed", "t_start": times, "value": 4.0}),
        ], ignore_index=True)
        monkeypatch.setattr(ListeningExtractor, "_series", lambda self, recording: series)
        recording = SimpleNamespace(
            recording_id="r1", entities=lambda: ENTITIES,
            subjects=[SimpleNamespace(subject_id="parent"), SimpleNamespace(subject_id="child")],
        )
        step = ListeningExtractor(SimpleNamespace(kinematics=KinematicsSection(listening=["upper_body_speed"])))
        values = step._rows(recording, 0.0, 10.0).set_index(["subject_id", "feature"])["value"]
        # the child listens while the parent talks alone (0 to 3 and 9 to 10), not while the researcher does
        assert values["child", "listening_s"] == 4.0 and values["child", "upper_body_speed_listening"] == pytest.approx((6 * 2.0 + 2 * 10.0) / 8)
        assert values["parent", "listening_s"] == 2.0 and values["parent", "upper_body_speed_listening"] == 4.0
        assert values["child", "upper_body_speed_listening_frames"] == 8 and values["parent", "upper_body_speed_listening_frames"] == 4

    def test_the_steps_that_give_each_series_come_first_and_others_are_refused(self):
        config = SimpleNamespace(kinematics=KinematicsSection(listening=["upper_body_speed", "motion_energy"]))
        assert set(ListeningExtractor.dependencies(config)) >= {"pose.kinematics", "motion.mea", "text.asr"}
        assert ListeningExtractor.wanted_by_default(config)
        assert not ListeningExtractor.wanted_by_default(SimpleNamespace(kinematics=KinematicsSection()))
        with pytest.raises(ConfigError, match="it takes"):
            ListeningExtractor(SimpleNamespace(kinematics=KinematicsSection(listening=["smiles"]))).check_available()

    def test_the_methods_name_the_series(self):
        assert "upper body speed and motion energy was averaged" in _listening(None, SimpleNamespace(params={"listening": ["upper_body_speed", "motion_energy"]}), set())

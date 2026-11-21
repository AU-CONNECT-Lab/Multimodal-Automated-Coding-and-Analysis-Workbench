from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from macaw.extractors.video.face import light_asd, speaking

cv2 = pytest.importorskip("cv2")


def test_a_box_is_placed_from_the_face_points_as_a_detector_draws_it():
    xs = np.array([[100.0, 160.0, 130.0]])
    ys = np.array([[200.0, 200.0, 280.0]])
    x, y, s = light_asd.face_boxes(xs, ys)
    assert x[0] == 130.0 and s[0] == pytest.approx(light_asd.BOX_SIZE * 40) and y[0] == pytest.approx(240 - light_asd.BOX_RAISE * 40)


def test_a_face_seen_at_30_frames_a_second_is_read_at_25_with_short_gaps_bridged():
    t = np.r_[np.arange(0, 60) / 30, np.arange(66, 90) / 30, np.arange(120, 160) / 30] + 10.0
    ones = np.ones_like(t)
    tracks = light_asd.plan_tracks(t, ones * 50, ones * 60, t, fps=30.0, offset_s=10.0)
    # the 6-frame gap is under 10 frames at 25 a second and bridged; the 30-frame one splits the run
    assert len(tracks) == 2
    first = tracks[0]
    assert np.allclose(np.diff(first.times), 1 / 25) and first.times[0] == 10.0 and first.times[-1] <= 89 / 30 + 10.0
    assert np.array_equal(first.frames, np.round((first.times - 10.0) * 30).astype(int))
    assert np.allclose(first.s[6:-6], first.times[6:-6], atol=1 / 30)


def test_a_track_shorter_than_the_demo_scores_is_left_out():
    t = np.arange(0, 10) / 25
    assert light_asd.plan_tracks(t, t, t, t, fps=25.0, offset_s=0.0) == []


def test_the_crop_is_grey_where_the_face_runs_off_the_picture():
    image = np.full((100, 100, 3), 255, np.uint8)
    face = light_asd.crop(image, 5.0, 5.0, 40.0)
    assert face.shape == (112, 112) and face.dtype == np.uint8
    assert face.min() == 110 and face.max() == 255


def test_the_sound_under_a_run_starts_on_its_first_moment_and_reaches_one_frame_past_its_last():
    samples = np.arange(16000, dtype=np.int16)
    sound = light_asd.sound_for(samples, 0.5, 10)
    assert sound.size == 11 * 640 and sound[0] == 8000 and sound[-1] == 8000 + 11 * 640 - 1
    early = light_asd.sound_for(samples, -0.01, 2)
    assert (early[:160] == 0).all() and early[160] == 0 and early[161] == 1


def write_counting_video(path, n_frames, fps=30.0, size=(160, 120)):
    """Each frame is flat grey at its own frame number, so a crop says which frame it came from."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
    for frame in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), frame % 250, np.uint8))
    writer.release()
    return path


def test_each_moment_is_scored_from_its_own_frame_and_the_sound_under_it_in_pieces(tmp_path, monkeypatch):
    video = write_counting_video(tmp_path / "camera.avi", 200)
    sound = tmp_path / "shared.wav"
    import soundfile

    rate = light_asd.SAMPLE_RATE
    soundfile.write(sound, (np.arange(rate * 9) // 640 % 30000).astype(np.int16), rate, subtype="PCM_16")
    pieces = []

    def fake_score(net, sound, faces, device):
        pieces.append((len(faces), int(sound[0] // 1)))
        # the frame a face came from, as the score
        return faces[:, 56, 56].astype(float)

    monkeypatch.setattr(light_asd, "score", fake_score)
    monkeypatch.setattr(light_asd, "CHUNK_FRAMES", 50)
    t = 2.0 + np.arange(30, 190) / 30
    ones = np.ones_like(t)
    tracks = light_asd.plan_tracks(t, ones * 80, ones * 60, ones * 20, fps=30.0, offset_s=2.0)
    found = speaking.score_video(video, {"child": (tracks, sound)}, net=None, device="cpu", t_origin_s=1.0)
    times, values = found["child"]
    assert np.array_equal(times, tracks[0].times)
    # a grey frame decodes within a level or two of its number
    assert np.abs(values - tracks[0].frames).max() <= 2
    assert [n for n, _ in pieces] == [50, 50, tracks[0].times.size - 100]
    # the first piece's sound starts at its first moment on the track's clock: 1 s after the track's start, in 640-sample frames
    assert pieces[0][1] == int((times[0] - 1.0) * rate) // 640


def test_the_crop_is_held_at_the_ends_of_a_run_not_pulled_towards_the_corner():
    t = np.arange(0, 12) / 25
    falling = 120.0 - np.arange(12)
    track = light_asd.plan_tracks(t, falling, falling, falling, fps=25.0, offset_s=0.0)[0]
    assert track.x[0] == 120.0 and track.x[-1] == 109.0


def test_a_copy_of_the_weights_cut_short_is_fetched_again(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    cached = tmp_path / "checkpoints" / light_asd.WEIGHTS_FILE
    cached.parent.mkdir()
    cached.write_bytes(b"half")
    calls = []

    def load(url, **kwargs):
        calls.append(url)
        if cached.exists():
            raise RuntimeError("PytorchStreamReader failed reading zip archive")
        return {"weights": 1}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(hub=SimpleNamespace(load_state_dict_from_url=load, get_dir=lambda: str(tmp_path))))
    monkeypatch.setattr(light_asd, "network", lambda state, device: state)
    assert speaking.FaceSpeakingExtractor._network(SimpleNamespace(), "cpu") == {"weights": 1}
    assert len(calls) == 2 and not cached.exists()


def test_the_sounds_scored_against_are_the_ones_in_the_key():
    from types import SimpleNamespace

    from macaw.extractors.video.face import speaking as speaking_module
    from macaw.extractors.video.face.speaking import FaceSpeakingExtractor

    prep = SimpleNamespace(
        shared_tracks={"room": None, "camera": None}, tracks={"parent": None, "child": None},
        shared_track_path=lambda name: Path(f"{name}.wav"), track_path=lambda name: Path(f"{name}.wav"),
    )
    step = FaceSpeakingExtractor.__new__(FaceSpeakingExtractor)
    step.config = None
    step._files = lambda rid: (Path("landmarks"), Path("faces"))
    recording = SimpleNamespace(recording_id="r1", subjects=[SimpleNamespace(subject_id="parent"), SimpleNamespace(subject_id="child")])
    original = speaking_module.load_prep
    speaking_module.load_prep = lambda config, rid: prep
    try:
        inputs = step.inputs_for(recording)
    finally:
        speaking_module.load_prep = original
    # two shared microphones: each face is scored against its own track, so those are what the key holds
    assert set(inputs[2:]) == {Path("parent.wav"), Path("child.wav")}

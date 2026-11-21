from macaw.media.ffmpeg import make_preview, run_ffmpeg
from macaw.media.probe import probe
from macaw.project.review_notes import add_note, notes_table, remove_note


def test_a_silent_camera_is_played_with_another_files_sound_placed_by_its_offset(tmp_path):
    video, sound = tmp_path / "cam.mp4", tmp_path / "mic.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "color=gray:s=320x240:r=10:d=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)])
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=4", str(sound)])
    for shift in (1.5, -1.5):
        out = make_preview(video, tmp_path / f"copy{shift}.mp4", sound=sound, sound_at_s=shift)
        info = probe(out)
        assert info.video_streams() and info.audio_streams()
    own = make_preview(video, tmp_path / "own.mp4")
    assert probe(own).video_streams() and not probe(own).audio_streams()


def _beep_file(path, picture_shift: float, sound_shift: float):
    beep = "aevalsrc='if(between(t\\,1\\,1.2)\\,sin(2*PI*1000*t)\\,0)':s=16000:d=3"
    run_ffmpeg([
        "-itsoffset", str(picture_shift), "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=3",
        "-itsoffset", str(sound_shift), "-f", "lavfi", "-i", beep,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", str(path),
    ])
    return path


def test_a_copy_starts_at_the_first_picture_with_its_own_sound_moved_to_match(tmp_path):
    import numpy as np

    from macaw.media.ffmpeg import decode_audio
    from macaw.media.probe import picture_lead

    for name, pictures, sound, lead in (("late", 0.4, 0.0, 0.4), ("early", 0.0, 0.3, -0.3)):
        source = _beep_file(tmp_path / f"{name}.mov", pictures, sound)
        assert abs(picture_lead(probe(source)) - lead) < 0.03
        info = probe(make_preview(source, tmp_path / f"{name}_copy.mp4"))
        # a browser cannot then place the copy's 0 on the sound rather than the first picture
        assert abs(info.video_streams()[0].start_s) < 0.01 and abs(info.audio_streams()[0].start_s) < 0.03
        samples = decode_audio(tmp_path / f"{name}_copy.mp4", sample_rate=16000)
        onset = np.argmax(np.abs(samples) > 0.1) / 16000
        assert abs(onset - (1.0 - lead)) < 0.03, (name, onset)


def test_the_copys_time_0_is_the_first_pictures_moment_whether_or_not_the_video_steps_read_a_normalised_copy(monkeypatch, tmp_path):
    from macaw.extractors import prep
    from macaw.project import previews
    from macaw.project.recording import MediaFile

    # pictures 0.4 s after the sound; the sound's start is 10 s into the recording at clock scale 2
    for frames_start in (0.4, 0.0):
        item = MediaFile(path=tmp_path / "cam.mp4", kind="video", device="cam", offset_s=10.0 + frames_start * 2, clock_scale=2.0)
        video = {"first_picture_s": 0.4, "frames_start_s": frames_start}
        monkeypatch.setattr(prep, "load_video_prep", lambda config, rid: {"cam": video})
        assert previews.first_picture_at(None, "r", item, timed=True) == 10.8


def test_notes_are_listed_one_row_each_and_taken_back_by_id(tmp_path):
    class Layout:
        def review_notes_path(self, rid):
            return tmp_path / rid / "review_notes.json"

    first = add_note(Layout().review_notes_path("a"), ["rerun", "poor sound"], "  again  ", "ann")
    add_note(Layout().review_notes_path("b"), [], "fine", "bo")
    table = notes_table(Layout(), ["a", "b", "c"])
    assert list(table["recording_id"]) == ["a", "b"] and table.loc[0, "tags"] == "rerun; poor sound" and table.loc[0, "text"] == "again"
    assert remove_note(Layout().review_notes_path("a"), first["id"]) and not Layout().review_notes_path("a").exists()
    assert not remove_note(Layout().review_notes_path("a"), "gone")

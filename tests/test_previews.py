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

# Changelog

## [Unreleased]

### Added
- Turn taking: turn segmentation, floor transfers, and entropy measures.
- Rule-based coding of backchannels, pauses, and interruptions.
- Inter-coder agreement report with Krippendorff's alpha and Cohen's kappa.
- Codebooks defined in YAML, with validation.
- Human review queue for low-confidence and abstained labels.
- Two-way ELAN (.eaf) interchange for annotations.
- Export to CSV, Parquet, ELAN, and fixed-rate state sequences.
- Command line interface for setting up a project, running extractors, and inspecting config.
- Body movement measures from pose, quality checks that point at the frames worth reviewing (`macaw qc --render`), your own choice and grouping of what runs (`pipeline:`), and a template for adding your own extractors (`macaw new-extractor`).
- Lining up files from separate devices by their sound (`macaw align`), with a figure per file to check the result, and setting an offset by hand (`macaw set-offset`).
- Pose from YOLO as well as MediaPipe, chosen explicitly (`pose.backend` has no default), runs that stop for a person only where something is in doubt (`macaw accept-offset`, `macaw identify`, `macaw accept-owner`), and usage reporting by the tools pymacaw calls turned off by default.
- People on a shared camera kept together when the tracker loses and finds them, matched to who they are by seat or by a person (`macaw accept-owner --rename`), an HTML quality report (`macaw qc --report`), and your own backends (`macaw new-backend`).
- Correcting who is who no longer reruns the pose model, resuming notices changed settings or inputs, the quality report draws each person over time, mics hint at who is who on a shared camera, and speech and prosody backend templates (`macaw new-backend --kind`).
- Motion energy per person from a hand-drawn region (`motion.mea`) with its own quality check and report figure, a shared camera read next to each person's own camera (`pose.read_shared`), which names its people by their movement and gives the distances between them, a check that a transcript belongs to the audio it is filed under, mic bleed judged fairly between mics recorded at different gains, every room camera read and named, talk exported as per-person and floor sequences, openSMILE feature sets per turn (`prosody.standard_sets`) and facial action units through py-feat (`face.au`), both installed by the user.
- Mouth movement, eye opening and blinks per person from the face landmarks (`face.movement`), gaze direction per frame from the face model (`face.gaze`), and a check that each person's mouth moves when they are the one talking.
- A voice report per turn (`prosody.voice_report`: pitch statistics and slope, jitter, shimmer, HNR, CPPS), lexical measures per speaker with optional sentence embeddings (`text.features`), a Methods section with its BibTeX for what a project ran (`macaw methods`), and a second face model to check the action units against (`face.check_model`, `au_agreement`); transcripts that reach past the recording are cut to it.
- A shared microphone (`text.diarization: pyannote`), alone or beside some people's own microphones: the voices are found by diarization, named by a person with a suggestion from their own microphone or mouth movement (`macaw accept-voices`), and then transcribed and measured for pitch and voice quality on each person's own stretches; the words counted for who brings in shared vocabulary are your choice (`text.first_use_*`).
- Every step that decides by itself can be checked or handed to a person: gates take `always`, `flagged` or `skip` (including the new `text.voice_review`), the quality report lists each decision and who made it, speech left out of a transcript is kept with the reason, and a transcript can be corrected in ELAN (`macaw transcript`); a workbench to run, review and browse results (`macaw gui`); syntax per speaker (`text.syntax`); and face landmarks from MediaPipe or OpenFace (`face.backend`).
- Speech acts coded by a language model from a codebook (`coding.judge`, `macaw code`), on your own model server or a cloud one, each unit asked several times so the agreement of the answers is its confidence; doubtful units go to a person (`macaw review --export / --apply`, or the Review page of `macaw gui`). A second face backend can check the action units of the first (`face.check_backend`), model and transcript folders written relative to project.yaml are found from there, and a blink rate is left empty when the face backend's eyelids never close.
- A redesigned `macaw gui`, light or dark: the Run page shows which steps are up to date, waiting for a person or due to run again, and a Settings page edits project.yaml as a form or as text, checking the settings and naming the steps they put out of date before saving. A project's roles must now match its number of people, and a language model setting no longer reruns the rule-based coding.
- Videos are prepared once before any video step (`prep.video`): constant frame rate, upright, square pixels, deinterlaced, with frame 0 at the file's first sound, so every step reads the same frames on the same clock; `video.fps` brings all videos to one rate where it divides them evenly, the quality report lists differing frame rates, and audio tracks from files whose sound and pictures start apart now land on the aligned clock.
- A recording filmed by one camera with no separate microphone is analysed from the camera's own sound, and `text.extra_voices` lets a shared microphone hold voices besides the participants (a researcher in the room) that are named `none` and left out. `segment.researcher_exit_marker`, which had no effect, is gone.

## [0.1.0-alpha] – 2026-08-13

### Added
- Initial package scaffold and CLI command surface.

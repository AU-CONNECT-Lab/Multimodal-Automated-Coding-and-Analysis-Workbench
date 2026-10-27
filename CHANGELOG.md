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
- One shared microphone for everyone (`text.diarization: pyannote`): the voices are found by diarization, named by a person with a suggestion from mouth movement (`macaw accept-voices`), and then transcribed and measured for pitch and voice quality on each person's own stretches; the words counted for who brings in shared vocabulary are your choice (`text.first_use_*`).

## [0.1.0-alpha] – 2026-08-13

### Added
- Initial package scaffold and CLI command surface.

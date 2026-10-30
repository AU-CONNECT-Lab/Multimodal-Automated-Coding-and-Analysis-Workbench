# Roadmap

## Vision

Turn recordings of natural multi-person interaction into analysable multimodal feature time series, covering the part annotation tools do not: automatic alignment, feature extraction, quality control, and assisted coding.

## Scope

A preprocessing and feature-extraction tool. It produces derivatives, descriptors and a quality-control report; the analysis stays yours.

- Nothing is pooled across recordings. Longitudinal trends and group comparisons are left to you.
- Within a recording it emits descriptions, not inferences. Where a measure's key parameter is the research question, it exports the input series instead.

Turn counts, gap and overlap distributions and floor-transfer measures are in scope; recurrence quantification, Granger causality and burstiness are not. Outputs export as event tables, fixed-rate time series, and binary spike trains.

Steps that cannot be decided reliably (track repair, speaker identity, who is who on a camera, low-confidence coding) produce candidates for review rather than silent guesses.

Reviewing the output works the same way. The quality report points at the frames and segments worth looking at, and review stays sampled rather than exhaustive.

## Completed features

### Unreleased
- Configuration, recording discovery, a feature store, and local runs that resume where they stopped.
- Alignment of separate devices by their sound, videos prepared once to one clock and one format, and a camera's own sound when there is no microphone.
- Transcription from each person's microphone or a shared one, voices told apart and named, prosody and voice quality per turn, lexical and syntactic measures.
- Turn structure, floor transfer and turn-taking entropy.
- Pose from MediaPipe or YOLO with people kept apart across cameras, body movement, motion energy, facial action units, head angles, gaze direction, mouth movement and blinks.
- Rule-based coding, language-model coding from a codebook with human review, coder agreement, and ELAN interchange both ways.
- Quality checks and an HTML report listing every automatic decision, a Methods text with references, exports, a GUI, and Slurm job arrays for clusters.

### 0.1.0-alpha
- Framework scaffold.

## Milestones

- Scale calibration and body-type classification, so distances can be given in real units.
- A language-model observation pass before coding, transcript cleanup, and codebook refinement.
- Who looks at whom, from gaze direction, on video where faces are large enough to tell.
- Gesture and speech alignment coding.
- Several kept segments per recording, and picking them on the video in the GUI.
- Metric 3D output; the storage schema reserves room for it.

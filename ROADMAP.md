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
- A setup that says what each camera and microphone records and for whom, edited with the room in 3D and three flat views, with notes on what each camera sees, the room seen from a camera, and a drawing for a paper, and who is who on each camera said by a click, with optional suggestions from head motion, a microphone or a model that reads pictures.
- Alignment of separate devices by their sound, videos prepared once to one clock and one format, and a camera's own sound when there is no microphone.
- Transcription from each person's microphone or a shared one, voices told apart and named, prosody and voice quality per turn, lexical and syntactic measures.
- Turn structure, floor transfer and turn-taking entropy.
- Pose from MediaPipe or YOLO with people kept apart across cameras, body movement, motion energy, facial action units, head angles, gaze direction, mouth movement and blinks.
- Rule-based coding, language-model coding from a codebook with human review, several versions of it side by side, a choice of what the model is shown (timing, voice, video frames, a note, examples people coded) and whether it reasons first or gives reasons, its agreement with itself, coder agreement, and ELAN interchange both ways.
- People's own coding brought in from ELAN or a table and compared with other coders or the machine, on shared intervals, one coder's intervals or a time grid, over whole recordings or within a segment, from the command line or the workbench.
- Quality checks and an HTML report listing every automatic decision, a Methods text with references, exports, a GUI, and Slurm job arrays for clusters.
- A model that reads pictures checking the review frames, and a person checking the model.
- Your own body measures, and a project scripts folder for measure functions, steps and backends used without installing.
- Kinematics of each coded gesture, and a summary of how each code was used.
- Lengths in one unit for the project (centimetres, metres or inches) beside pixels for each person's place, from an object of known length measured in the recording or in a calibration clip, or a number set by hand.
- 3D from two or more cameras calibrated with a printed board or a calibration file: each person's points in metres, measures in 3D (your own and coded gestures included), a 3D view, and the cameras calibrated on the Setup page and placed in its room from the calibration.
- 3D from a depth camera: a RealSense, Azure Kinect or Orbbec Femto recording imported with the maker's own alignment, or depth images beside any video, each person's points placed on the depth in metres, with the same 3D measures.
- Body points exported per recording, in pixels and in 3D.
- Who looks at whom and at which named things, from the picture or from where the room places people and things, mutual gaze, joint attention with who led, and labelled frames to check it.
- Paired values for convergence between speakers (each turn's voice measures beside the previous speaker's, lexical overlap with the previous turn), time spent below a threshold the study sets, and body orientation toward the other person in 3D.
- Named segments of a recording, each with every recording-level measure and the turn checks worked out again as if the recording had been cut there, marked on the video in the workbench.
- A codebook made clearer by the language model from the units a person and the model coded differently, or drafted from the transcripts, and transcript lines the model marks as possibly misheard, each for a person to read and decide.

### 0.1.0-alpha
- Framework scaffold.

## Milestones

- A language-model observation pass before coding.
- 3D from a single ordinary camera, with a body model that covers every age; the storage schema reserves room for it.

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
- Motion energy per person from a hand-drawn region (`motion.mea`), and a shared camera read next to each person's own camera (`pose.read_shared`), which names its people by their movement and gives the distances between them.

## [0.1.0-alpha] – 2026-08-13

### Added
- Initial package scaffold and CLI command surface.

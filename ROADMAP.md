# Roadmap

## Vision

Turn recordings of natural multi-person interaction into analysable multimodal feature
time series, covering the part annotation tools do not: automatic alignment, feature
extraction, quality control, and assisted coding.

## Scope

A preprocessing and feature-extraction tool. It produces derivatives, descriptors and a
quality-control report; the analysis stays yours.

- Nothing is pooled across recordings. Longitudinal trends and group comparisons are
  left to you.
- Within a recording it emits descriptions, not inferences. Where a measure's key
  parameter is the research question, it exports the input series instead.

Turn counts, gap and overlap distributions and floor-transfer measures are in scope;
recurrence quantification, Granger causality and burstiness are not. Outputs export as
event tables, fixed-rate time series, and binary spike trains.

Steps that cannot be decided reliably (scale calibration, track repair, speaker identity,
low-confidence coding) produce candidates for review rather than silent guesses.

## Completed features

### 0.1.0-alpha
- Framework scaffold. No working implementation yet.

## Milestones

- **0.2** Configuration, recording discovery, feature store, local execution, prosody
- **0.3** Transcription, turn structure, floor transfer
- **0.4** Speaker identity, coder agreement, export formats, generated methods text
- **0.5** Ingest quality assessment, pose, body-type classification, motion energy,
  scale calibration
- **0.6** Face, quality control reporting, rule-based and LLM-assisted coding
- **0.7** ELAN interchange, GUI, cluster job scripts

Metric 3D output is planned after 0.7; the storage schema reserves room for it.

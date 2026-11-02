# Questions

(privacy)=
## Do the recordings leave the computer?

MACAW itself sends nothing anywhere. Two choices in the settings can:

- A cloud language model (OpenAI or Anthropic) receives the transcript lines it codes, and the video frames when these are selected ({ref}`coding`).
- The MediaPipe tools, one of the options for body pose and for the face, send usage statistics to Google while they run. They do not send the video, but this cannot be switched off.

When no data may leave the computer, select YOLO for body pose and a language model running on the lab's own computer or server. A run first downloads any model that is missing, before processing starts; models already on the computer are used as they are, and the processing steps themselves never go online.

## What happens when a setting is changed?

Only the results that depend on that setting are computed again. The settings page lists them before saving, so that nothing is recomputed unexpectedly.

## Can a run be interrupted and resumed later?

Yes. Close the window or press **Stop**. The next run continues where the previous one stopped.

## How long does processing take?

It depends mainly on the length of the videos and on whether the computer has a graphics card. Transcription and the face and body steps take the longest. Processing one recording first gives a reliable estimate for the rest.

## An experimenter is present in the room. Is their speech mixed into the results?

With a shared microphone or the camera's sound, set *Other voices on a shared microphone* to 1 in the settings, and assign that voice *none* when reviewing the voices. The experimenter's speech still counts as a turn in the turn-taking measures, but is excluded from the participants' voice and language measures.

## A session contains several tasks. Can they be analysed separately?

Yes. When the tasks were filmed in one file, name each task as a segment ({ref}`segments`); every measure that is one number per recording is then also given within each. When each task is a separate file, each file is a recording of its own and needs no segments.

## Children have higher voices. Is pitch measured correctly?

The pitch range can be set separately for each role, for example a higher range for the child. A second setting makes the quality-control report flag a pitch track that lies an octave too high throughout, by giving the highest plausible median for a role. Both are added in the *project.yaml* box of the settings page:

```yaml
prosody:
  f0_range_by_role: {child: [100, 600]}   # Hz, lowest and highest
qc:
  f0_median_max_by_role: {mother: 330}    # Hz
```

## Can errors in the transcript be corrected?

Yes, in ELAN; see {ref}`transcript`. Everything that uses the transcript is updated on the next run.

## The project has hundreds of recordings.

MACAW can run on a computing cluster that uses Slurm, with one job per recording. See [Optional features](extras.md), section *Large projects on a computing cluster*.

## What should be reported in the paper?

Run `macaw methods`. It writes a Methods section for the steps the project actually ran, with the references to cite.

## A step failed.

A failed step states the reason at the bottom of the Run page. Correct the cause and press **Run** again; completed work is kept. Unresolved problems can be reported on the [GitHub page](https://github.com/AU-CONNECT-Lab/Multimodal-Automated-Coding-and-Analysis-Workbench/issues).

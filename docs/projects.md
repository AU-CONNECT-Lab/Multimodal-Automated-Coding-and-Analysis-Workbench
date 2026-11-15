# Segments, exports and extensions

This page covers parts of a recording analysed on their own, the description written beside every export, processing on a computing cluster, and new processing steps written for a study. Settings given as a block can be pasted into the *project.yaml* box on the settings page and adapted to the study.

(segments)=
## Several parts of one recording

When a recording contains several phases, for example free play followed by clean-up, each phase can be named as a segment. Every measure that is a single number per recording, such as each participant's share of the talk, the number of turns or the median floor transfer offset, is then also computed within each segment. The recording is still processed as a whole: transcription, the separation of voices and the tracking of participants use the full recording, and decisions such as which voice belongs to whom are made once and hold in every segment.

### Marking segments

On the Segments page, select the recording and press **Make a copy to play here**. Under the player, drag along an empty stretch of the *Segments* lane to make a segment, drag its edges or its middle to change it, and click it to name it or to set its ends to the hundredth of a second; **Start here** and **End here** set an end to where the player stands. Each change is saved when the mouse is released. The loudness of the sound is drawn behind the lane, so pauses and changes of activity are easy to find. All segments of the recording are listed as a table below the strip. Segments can also be written directly into the file `segments.csv` next to `project.yaml`, one row per segment, which is convenient when the times come from session logs:

```text
recording,segment,start_s,end_s
dyad01,free_play,35,335
dyad01,clean_up,350,470
dyad02,free_play,28,330
```

Times are in seconds from the start of the recording's main video (with several devices, of the file the others are aligned to). Segments may overlap, and a recording without rows is summarised only as a whole. After saving, press **Run**: only the summaries within the segments are computed, so a segment can be moved without processing the recording again.

### How a segment is summarised

:::{important}
A segment is summarised as if the recording had been cut at its edges. An utterance, turn, code, look or blink that crosses an edge counts only with its part inside, a word belongs to the segment in which its middle falls, and turns, pauses and floor transfers are formed again from what remains, so a pause across an edge belongs to neither segment. Rates are per minute of the segment. Values scaled to a participant's level over the whole recording, such as relative pitch, keep that reference. A short segment rests on few turns and events, so measures from short segments call for caution.
:::

Several of the quality report's checks are made again within each segment, with the same thresholds as for the whole recording: on turns, too few turns, floor transfers, or turn pairs for the entropies, a participant who never takes a turn or never takes the floor, one participant holding most of it, and heavy overlap; on speech, a participant with nothing transcribed, low recogniser confidence, or too few voiced frames; on tracking, a participant detected in too few frames, lost and found again too often, or not detected at all. Their warnings are listed under the segment. The other checks concern the whole recording.

### Segments and the analysed part

The analysed part, set by *Task starts at* and *Task ends at* or by dragging the two handles of the *Analysed* lane on the Segments page, restricts what every step reads, and changing it processes the recording again from the start. Segments only add summaries within the analysed part. A segment that reaches outside it is left out, and the quality report says so.

### Results

`macaw export` adds a `segment` column, which is empty for values of the whole recording and holds the segment's name for values within a segment. With `macaw export --scale recording`, each participant has one row for the whole recording and one for each segment. The quality report lists the segments, the warnings within each and their measures in a section of its own, the Results page shows a measure's value within each segment beside the whole recording, `macaw methods` describes how the segments were summarised, and `macaw agreement --segment` compares coders within a segment.

(data-dictionary)=
## The data dictionary of an export

`macaw export` writes a data dictionary beside the table, for example `features_dictionary.csv` beside `features.csv`, and the same content as `features_dictionary.json`, a sidecar in the style of BIDS keyed by column name with the keys `Description` and `Units` and the fields below. It describes only the columns of that export, under the settings with which its recordings were processed, so that it can be archived and shared with the table. In the long layout, where the measures are values of the column `feature`, each measure has an entry of its own after the columns of the layout.

| Field | Content |
|---|---|
| `column` | The name of the column, or of the measure in the long layout. |
| `meaning` | What the column holds. |
| `unit` | Its unit, where it has one. |
| `who` | Whose rows hold the measure: `person` (one participant, in `subject_id`), `pair` (two participants, in `subject_id` and `subject_id_b`) or `recording` (the recording as a whole). |
| `rows` | Which rows hold it: `frame`, `event`, `window`, `recording` (the whole analysed part) or `segment` (a named segment, {ref}`segments`). In the wide table a measure has no value in rows of another kind; a measure given per frame, for example, is empty in the row of the whole recording. |
| `aggregation` | How the step proposes that values of the measure be combined over a longer span. |
| `missing` | What an empty cell means in a row of the measure's own kind, for example that no face of the participant was found in that frame, as opposed to an action unit that was not active; for counts and proportions, also what a value of 0 means. |
| `step`, `backend` | The step and the model or library that computed the measure, with their versions and the version of pymacaw that wrote the dictionary. |
| `settings` | The settings of the step that bear on the measure. Where recordings were processed under settings of their own (`recordings/<recording>.yaml`), each variant is listed with its recordings. A note to the language model given by task is listed once, with the note of each task. |
| `comparable_across_people` | Whether values may be compared between participants; loudness, for example, depends on each participant's distance to the microphone. |
| `comparable_across_backends` | Whether values from different backends may be pooled; action units of different face models, for example, may not. |
| `validation` | How the measure has been checked: `none`, `synthetic` (against data generated with known values), `against_reference` (against a reference tool or annotation) or `published` (a published validation of the method). |
| `converted_from` | For a measure given in the project's unit of length ({ref}`scale`), the measure in pixels or metres it was converted from. |

A measure computed by a step of the lab's own ({ref}`extending`) that does not state what its empty cells mean is marked accordingly.


(cluster)=
## Large projects on a computing cluster

MACAW can run on a computing cluster that uses Slurm, with one job per recording. Describe the cluster in a file `hpc.yaml` next to `project.yaml`, starting from the template included in the package (`macaw/config/hpc.yaml`): the partitions, the graphics card to request, the commands that load modules and activate the Python environment, and a shared folder for model weights. Then, on a login node:

```bash
macaw hpc prefetch-models -c project.yaml
macaw hpc generate -c project.yaml
```

The first command, run once, downloads the model weights into the shared folder, as the jobs themselves run offline. The second writes the job arrays and prints the command that submits them. Steps that need a graphics card and steps that do not run in separate job arrays, and each recording proceeds to its next array by itself. A recording that pauses for a review waits there; once the review is done, run `macaw hpc generate` again, which leaves out whatever has finished.

:::{note}
Cluster execution has not yet been tested on a production cluster. Please report any problems.
:::

(extending)=
## Extending MACAW

The `scripts` folder of a project can hold, besides measure functions, whole new processing steps and new backends for existing steps, used without installing anything. `macaw new-extractor` and `macaw new-backend` write a working example; copying its `extractor.py` or `backend.py` into the `scripts` folder is sufficient.

:::{warning}
Every file in the `scripts` folder is executed as Python whenever MACAW reads the project. Place only trusted files there.
:::

# Using MACAW

This page describes the standard procedure, from organising the recordings to exporting the data set. Optional analyses, such as language-model coding, imported human coding, gaze and three-dimensional pose, are described in [Optional features](extras.md).

## Before you start

### Requirements

- Audio or video recordings of the interaction, or both. A separate microphone per participant is preferable; a single camera with its built-in microphone is also supported.
- A computer with an NVIDIA graphics card for more than a handful of recordings. Without one, every step still runs, but considerably more slowly.
- Free disk space roughly equal to the size of the videos, since MACAW stores one normalised copy of each video.

### Scope

- The original recordings are never modified. All output is written to a separate project folder.
- Transcription supports many languages. Lexical and syntactic measures are currently available for English only.
- Dyadic recordings are the most thoroughly tested case. Groups of three or more participants are supported.
- MACAW describes each recording. It does not perform statistical inference or compare groups; the analysis remains with the researcher.
- Where a decision cannot be made reliably by software, MACAW pauses and asks the researcher rather than guessing.

### Installation

MACAW is installed once per computer, usually by whoever maintains it.

1. Install Python 3.10, 3.11 or 3.12, git and ffmpeg.
2. Install PyTorch for the computer's graphics card, following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/). Installing it first ensures that the graphics card is used.
3. Install MACAW with all its components:

   ```bash
   pip install "pymacaw[audio,asr,text,vision,gui,report] @ git+https://github.com/AU-CONNECT-Lab/Multimodal-Automated-Coding-and-Analysis-Workbench.git"
   ```

Three tools are not bundled because of their licences: YOLO (body pose), OpenFace (face) and openSMILE (standard acoustic feature sets). They need to be installed only if they are selected in the settings. Separating voices on a shared microphone uses a model hosted on Hugging Face: accept its terms once on the Hugging Face website and log in to Hugging Face on the computer.

Once MACAW is installed, the *This computer* panel on its settings page reports whether ffmpeg and the optional programs are found.

## 1. Organise the recordings

{bdg-warning}`You do this`

Create one folder per recording. Inside it, create one folder per recording device (each camera and each microphone or sound recorder) and place that device's file in it. The device folders carry the same names in every recording; the names are chosen freely and are described once in {ref}`step 3 <setup>`. The names of the files themselves play no part, so files as a camera writes them, such as `00001.MTS`, need not be renamed.

```text
recordings/
  dyad01/
    front/
      00001.MTS           camera facing the child
    side/
      00001.MTS           camera filming both participants
    parent_mic/
      take1.wav           microphone worn by the parent
  dyad02/
    ...
```

Each device folder holds exactly one file. A recording in which a device folder holds two files is not processed until one of them is removed; a recording that lacks a device, or holds a folder or file outside the device folders, is processed with a note in the quality-control report.

Single camera
: The sound is taken from the camera and the voices are separated automatically. In the setup, the camera is said to carry the voices of all participants, and *Microphones* is set to `pyannote` (one shared microphone) in the settings. Each voice is then assigned to a participant in step 6.

Several sessions per participant
: Use one folder per family or participant with one subfolder per session, for example `sub-01/ses-1`, and describe this layout once under *Recording folder pattern* in the settings, here `sub-{group_id}/ses-{session_id}`.

Excluding a recording
: A recording can be excluded without moving any files, by adding `excluded_recordings: [dyad07]` under `project:` in the *project.yaml* box of the settings page.

## 2. Create a project and open the workbench

{bdg-warning}`You do this`

This is the only step that requires the command line. In a terminal, run the following three commands, substituting the project name and the folder that holds the recordings:

```bash
macaw init parent-child-play --name "Parent-child play" --raw-dir "D:/recordings"
cd parent-child-play
macaw gui
```

The first command creates a project folder containing a settings file (`project.yaml`) and a results folder. The last command opens the workbench. In later sessions the last two commands suffice. `macaw gui --browser` opens the workbench in a browser tab instead of a separate window.

The workbench has seven pages, an eighth for projects with several cameras, and a settings page behind the gear icon. **Open folder** opens the project folder on the computer.

Overview
: The page the workbench opens on. It summarises the project (recordings, participants and minutes), lists what needs attention next (an unsaved setup, failed steps, pending reviews, steps not yet run), and shows the status of every step for every recording. Clicking a square opens the step's results when it is up to date, the recording's reviews when it waits for one, and otherwise the Run page with that recording and step selected.

Setup
: The cameras and microphones of the recordings: what each device folder holds, whom each camera is used for, and whose voices each device carries ({ref}`step 3 <setup>`).

Run
: Selecting recordings and steps, starting a run and following its progress.

Review
: Decisions that await the researcher. The number on the tab gives their count.

Agreement
: Agreement between coders, people and machine, on one tier across the recordings ({ref}`own-coding`).

Segments
: Naming parts of a recording, such as a free-play and a clean-up phase, within which every measure is computed again ({ref}`segments`), and setting the part of a recording that is analysed.

Results
: The quality-control report and every measure plotted over time.

Cameras
: Shown only when two or more cameras film the same participants. Used to calibrate them for three-dimensional measures ({ref}`three-d`).

Settings (gear icon)
: All settings of the project.

Processing steps appear under short names, for example `text.asr` for transcription and `pose.kinematics` for movement measures.

(setup)=
## 3. Describe the setup

{bdg-warning}`You do this`

Before anything can run, the recording devices are described on the Setup page. Nothing is inferred from the names of folders or files: the description entered here is the only record of which file belongs to whom.

People
: The roles of the participants, as set under *Project* in the settings.

Cameras
: One row per camera: the name of its folder, and for each participant whether the camera is used for their body (pose and movement), their face (facial action units, head orientation, blinks) and their gaze. A camera that films a single participant is that participant's own camera. *Voices* states whose voices the camera's own microphone carries; a camera whose sound is not to be used, for example because every participant wears a microphone, carries none.

Microphones
: One row per microphone or sound recorder: the name of its folder and whose voices it carries. A device that carries one participant's voice alone serves as that participant's own microphone; the voices on a device that carries several participants are separated automatically and assigned to participants by the researcher (step 6).

**Save the setup** checks the description and writes it to the file `setups/main.yaml` in the project folder. The table *What each recording holds* then lists, for every recording, the file found in each device folder, together with any device that is missing and any folder or file that was not read. This table should be checked before the first run.

When the devices differed between sessions, for example after a change of camera, a second setup is started under *New setup*, and each recording made with it names it in its own settings file in the `recordings` folder of the project (here for `dyad07`, in `recordings/dyad07.yaml`):

```yaml
setup:
  name: home_visits
```

## 4. Configure the project

{bdg-warning}`You do this`

The settings most studies adjust are arranged as a form in six groups. All other settings can be edited as text in the *project.yaml* box below the form. On saving, MACAW validates the settings and lists the finished results that the change would make out of date.

Project
: Location of the recordings, the number of participants per recording and their roles. *Task starts at* and *Task ends at*, in seconds, restrict the analysis to the task period, for example to exclude the experimenter's instructions.

Video
: Whether videos are converted to one format and frame rate before analysis, and at what quality.

Steps
: Which measures a run produces, and how many recordings are processed in parallel.

Tools
: The program used for each kind of measurement, the spoken language, and whether each participant has a microphone or the participants share one. **Body pose has no default**: select MediaPipe or YOLO explicitly (see {ref}`privacy`).

Checks by a person
: For each check described in {ref}`step 6 <review>`, whether MACAW always waits for the researcher, waits only when uncertain, or never waits.

Coding
: The codebook and the language model, needed only for {ref}`language-model coding <coding>`.

:::{tip}
When the task starts or ends at a different time in each recording, give each recording its own times in a file named after it, such as `dyad01.yaml`, in the `recordings` folder of the project. **Analyse only this part** on the Segments page writes this file from times marked while the recording plays. Such a file can override other settings for that recording as well.

```yaml
segment:
  task_start_s: 42
  task_end_s: 402
```
:::

:::{important}
Backchannels (short responses such as "mm-hm") are coded only when the words that count as backchannels are listed, because these depend on the language spoken. An utterance is coded as a backchannel when it consists only of listed words, is short, and does not take the floor. Add the list in the *project.yaml* box:

```yaml
coding:
  backchannel_lexicon: [mhm, mm, uh-huh, yeah, okay, right]
```
:::

## 5. Run the analysis

{bdg-primary}`MACAW does this`

On the Run page, select the recordings and the steps, then press **Run**. Steps that a selected step depends on are included automatically. The **Run** button at the end of a step's row runs that step alone, with its prerequisites, for the selected recordings. **Preview** lists what would run without running it, and **Stop** ends a run cleanly. Processing a single recording first is a quick way to confirm that the setup works.

The status of each step is shown as follows:

| Status | Meaning |
|---|---|
| {bdg-success}`up to date` | Completed with the current settings. It is not run again. |
| {bdg-info}`not run yet` | Runs the next time **Run** is pressed. |
| {bdg-warning}`to run again` | A setting or an input file it depends on has changed since it ran. |
| {bdg-warning}`waiting for a person` | Paused at a check. See {ref}`step 6 <review>`. |
| {bdg-danger}`failed last time` | The output at the bottom of the page gives the reason. |

The window may be closed at any time. The next run resumes where the previous one stopped and does not repeat completed work.

(review)=
## 6. Review automatic decisions

{bdg-warning}`You do this`

Some decisions cannot be made reliably by software. When MACAW reaches one, processing of that recording pauses at that point while the other recordings continue. The Review page lists what is waiting; select a recording to see its items.

### Offsets between files

{bdg-secondary-line}`only when uncertain`

Separate devices start recording at different moments, and MACAW aligns them by their sound. Inspect the figure for the file, then keep the estimated offset or enter one by hand.

### Who is on camera

{bdg-secondary-line}`only when uncertain`

MACAW determines which person in the picture is which participant. Press **Draw review frames**, check the names on the boxes, and keep them or swap two that are mixed up.

### Voices on the shared microphone

{bdg-secondary-line}`always`

MACAW separates the voices and proposes a participant for each. Listen to the clips and assign each voice to a participant. A voice that belongs to none of the participants, such as an experimenter's, is assigned *none*. The proposal can also draw on what each voice says and on how high it is ({ref}`voice-hints`).

(transcript)=
### Transcript

{bdg-secondary-line}`optional`

**Export to ELAN** writes the transcript as an ELAN file. Words, boundaries and speakers can be corrected in ELAN, and the corrected file is read back with **Import a corrected transcript (.eaf)**. Speech the recogniser omitted is placed on a tier of its own so that it can be restored. **Back to the recogniser's** returns to the automatic transcript at any time.

### Speech acts

{bdg-secondary-line}`with language-model coding only`

The units the language model was uncertain about, together with a random sample of units it was certain about. Select the correct code for each and press **Save decisions**. The researcher's decision is stored alongside the model's.

### After a review

Once an item is resolved, press **Run** again; the recording continues from where it paused. Every decision is stored with who made it and when, and is listed in the quality-control report.

How often each check asks is set under *Checks by a person*; the labels above are the defaults. The remaining cards on the Review page (*What a model saw on the review frames*, *Pixels to centimetres*, *Where people look*, *Your coding*) belong to optional features.

(results)=
## 7. Inspect the results

{bdg-primary}`MACAW does this` {bdg-warning}`You do this`

The Results page has two parts.

Quality review
: Press **Write the QC page**, then open it. The report lists what the automatic checks flagged (for example a transcript that does not seem to match its audio, a mouth that does not move while its owner is said to be speaking, participants the tracker repeatedly lost, or a pitch track that jumps by an octave), every decision made by the software or by a person, an overview of each participant's speech, movement, distance and codes on a common time axis, the meaning and unit of every measure, and a set of frames to inspect. These frames include the moments the checks pointed to and a random selection, so that ordinary moments are inspected as well. The review is thus based on a sample rather than on every frame. A model that reads pictures can check these frames first ({ref}`model-look`).

Measures
: Select a step and a measure to plot it for each participant across the recording. A measure that is one number per recording is listed per participant, beside its value within each segment ({ref}`segments`).

:::{note}
When a result looks wrong, adjust the relevant setting and run again. Only the steps that depend on that setting are recomputed.
:::

## 8. Export the data

{bdg-success}`What you get`

Run one of the following commands in the project folder. Each reports where it saved its files.

`macaw export`
: One table (CSV) containing every measure for every participant over time. `--rmd` adds an R Markdown file that loads it. For recordings with a pixel scale, measures in pixels are also given in centimetres ({ref}`scale`). Values within named segments carry the segment's name ({ref}`segments`).

`macaw export -f sequence`
: For each participant, whether they are speaking at each second, for sequence analysis.

`macaw export -f eaf`
: One ELAN file per recording containing the machine codes, as a starting point for manual coding, which can be imported again ({ref}`own-coding`).

`macaw export -f skeleton`
: Each participant's body keypoints frame by frame, one CSV per recording, in pixels with the detector's confidence for each point. With several calibrated cameras ({ref}`three-d`), a second file gives the points in metres.

`macaw methods`
: A Methods section describing the steps that were actually run, with references.

Every run records its settings and the software versions it used, so that the same data set can be reproduced later.

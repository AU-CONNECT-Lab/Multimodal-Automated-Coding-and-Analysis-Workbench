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
2. Install PyTorch for the computer's graphics card, following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/), in the release that the speech recognition component (whisperx) requires: PyTorch 2.8 for whisperx 3.8, found under the previous versions on that site. Installing it first ensures that the graphics card is used: step 3 replaces any other release with one from the general package index, which on Windows runs on the processor only.
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
: Use one folder per family or participant with one subfolder per session, for example `sub-01/ses-1`, and describe this layout once under *Recording folder pattern* in the settings, here `sub-{group_id}/ses-{session_id}`. Several tasks per session are described in the same way, for example `sub-{group_id}/ses-{session_id}/task-{task}_run-{run}`.

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

The workbench has seven pages and a settings page behind the gear icon. **Open folder** opens the project folder on the computer.

```{figure} screens/overview.png
:width: 100%
:alt: The Overview page: four tiles with the number of recordings, the participants, the task time and the share of work up to date, a list of what to do next, and a grid with a square for each recording and step.

The Overview page of a project with six recordings. One recording failed at the first step, one waits for the researcher to check an offset, and one has a changed input file.
```

Overview
: The page the workbench opens on. It summarises the project (recordings, participants and minutes), lists what needs attention next (an unsaved setup, failed steps, pending reviews by kind of check, steps not yet run), draws the workflow (each step in the order it runs, coloured by its most pressing state across the recordings; clicking a step opens it on the Run page, or a check on the Review page), and shows the status of every step for every recording. Clicking a square opens the step's results when it is up to date, the recording's reviews when it waits for one, and otherwise the Run page with that recording and step selected. Each check by a person ({ref}`step 6 <review>`) has a round square of its own beside the steps: filled when a person decided it, an empty ring when it was left to the software, orange while it waits, and dashed when the recording had nothing to check. Steps are named in plain words; pointing at a name shows the step's own name, which the commands use.

Setup
: The cameras and microphones of the recordings: what each device folder holds, whom each camera is used for, whose voices each device carries, where everyone and everything is in the room, and who is who on each camera ({ref}`step 3 <setup>`). Projects with several cameras filming the same participants also calibrate them here for three-dimensional measures ({ref}`three-d`).

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

Settings (gear icon)
: All settings of the project. Under *Only for you*, the workbench's style (Lab, Playroom or Blueprint) and light or dark mode are chosen for the current browser only.

Processing steps appear under short names, for example `text.asr` for transcription and `pose.kinematics` for movement measures.

(setup)=
## 3. Describe the setup

{bdg-warning}`You do this`

Before anything can run, the participants, cameras and microphones are described on the Setup page. Nothing is inferred from the names of folders or files: the description entered here is the only record of which file belongs to whom.

The card *The setup* lists the participants, cameras, microphones and things on the left, shows the room in three dimensions in the middle and the settings of the entry chosen on the right, with a top, a front and a side view below. Choose an entry in the list, or click it in any view, to see and change its settings.

```{figure} screens/setup.png
:width: 100%
:alt: The setup card: a list of the participants, cameras and things on the left, a room in three dimensions with a parent, a child, a table and two cameras whose fields of view are drawn, and a top, a front and a side view below.

The card *The setup*, with two participants at a table and two cameras, each with its field of view.
```

People
: The participants, by role. **Add** with *Person* adds a role to the project, so that every recording has that participant; a role is renamed or removed under *Project* in the settings.

Cameras
: Each camera carries the name of its folder in every recording. A grid states, for each participant, whether the camera is used for their body (pose and movement), their face (facial action units, head orientation, blinks) and their gaze, and whether the camera's own microphone carries their voice. A camera used for a single participant is that participant's own camera. A camera whose sound is not to be used, for example because every participant wears a microphone, carries no voice.

Microphones
: Each microphone or sound recorder carries the name of its folder and states whose voices it carries. A device that carries one participant's voice alone serves as that participant's own microphone; the voices on a device that carries several participants are separated automatically and assigned to participants by the researcher (step 6).

Things
: Objects of interest, such as a table or a toy, which serve as targets of gaze ({ref}`gaze`) and appear in the drawings.

**Add** adds a camera, microphone, thing or participant, and *Remove this camera* (or microphone, or thing) takes it out of the setup. **Save the setup** checks the whole description and writes it to the file `setups/main.yaml` in the project folder; until then the card shows *unsaved changes*. The table *What each recording holds* then lists, for every recording, the file found in each device folder, together with any device that is missing and any folder or file that was not read. This table should be checked before the first run.

When the devices differed between sessions, for example after a change of camera, a second setup is started under *New setup*, and each recording made with it names it in its own settings file in the `recordings` folder of the project (here for `dyad07`, in `recordings/dyad07.yaml`):

```yaml
setup:
  name: home_visits
```

(room)=
### The room

Where the participants, devices and things are is optional, and a project runs without it. It is used to infer whom a participant looks at from their head direction ({ref}`gaze`), it yields notes on what each camera sees, and it provides a drawing of the arrangement for a publication.

Positions are given seen from above, across and along the room from a corner chosen as zero, with heights from the floor, in the project's unit of length: centimetres by default, metres or inches when chosen as *Length unit* under *Project* in the settings. Changing the unit there converts the lengths already entered. Each participant has a position, an eye height and the direction they face; each camera a position, the height of its lens, the direction it faces, how far it is tilted down and the angle its picture spans across (often given in the camera's specifications, otherwise found as described below); each microphone a position and a height; each thing a position, a height and, for drawing, a width and a depth. Directions are given in degrees clockwise from straight along the room as seen from above. The room's width and length, entered first in the *Room* row at the top of the list, draw its walls. A value left blank is drawn faintly at a typical height and is not used by the analysis.

An entry not yet in the room, such as one just added, is placed by clicking where it stands in the plan, the three-dimensional view or the top view, or by dragging it from the list into one of these; *Put in the middle* places it in the middle of the room. *Plan*, above the large view, shows the room from above as in the drawing for a paper, and is shown first while nothing is placed.

Values can be typed in the settings of the chosen entry or set by dragging. In the three-dimensional view and the top view, an entry is dragged across the floor; in the front and side views, it is dragged across and up or down. When an entry is chosen, the ring around it turns it, the knob in front of a camera tilts it, and the arrow above it changes its height. The R key turns the chosen entry by 15°, and Delete takes it out of the room while keeping its other settings. Dragging the empty floor turns the three-dimensional view, the scroll wheel zooms any view, and **Fit** shows the whole room again. The flat views are drawn either in the style of the drawing for a paper (*Drawing*) or as views of the three-dimensional model (*Model*). A camera or microphone used for one participant alone is shown in that participant's colour. When the arrangement differed in some sessions, those recordings can change parts of it in their own settings files.

When a camera is chosen, *What it sees* lists each participant it is used for, with their distance from it, the angle at which it sees their face (a face seen from more than 90° is seen from behind) and whether they fall inside its picture. When it is used for the body of two participants or more, a line states how much more length a pixel represents at the farther one than at the nearer; a single pixel scale then misstates one of them ({ref}`scale`). A camera that leaves someone outside its picture or sees a face from behind is marked in the list.

*From a camera*, above the three-dimensional view, shows the room from where a chosen camera stands, with its direction, tilt and picture angle, half transparent over a frame of that camera's file from a chosen recording and moment. Drag in the view to turn and tilt the camera, and scroll to change its picture angle, until the drawing lies on the picture; the camera's settings change accordingly.

With several calibrated cameras, **From the calibration** fills in each camera's position, height, direction, tilt and picture angle from the calibration, tied to the room by the values already entered and by the board lying flat in the calibration clips ({ref}`three-d`).

**Draw for a paper**, or `macaw draw-room`, writes a top view and a side view of the saved arrangement at 300 dpi as `setups/main.png`, with the participants, the cameras and what each sees, the microphones, the things and a scale bar, and no further annotation.

## 4. Configure the project

{bdg-warning}`You do this`

The settings most studies adjust are arranged as a form in six groups. All other settings can be edited as text in the *project.yaml* box below the form; a setting the file leaves out takes its default, and `macaw config show` lists every setting with the value in effect. On saving, MACAW validates the settings and lists the finished results that the change would make out of date.

```{figure} screens/settings.png
:width: 100%
:alt: The settings page: links to its groups, the Project group as a form with a short note under each field, and on the right the choices kept in the browser and the programs found on the computer.

The settings page. The links at the top lead to each group of settings.
```

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

When every recording should be cut to the same length counted back from its end, set *Keep only the last (s)* instead of the start, `segment: {keep_last_s: 300}` in *project.yaml*. The part kept ends at *Task ends at* when that is set, otherwise where the recording ends. A recording shorter than that is analysed whole, with a warning in its quality report, and the Methods text says how many there were.
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

Before any step starts, a run checks that every step it has to run can run on this computer with the settings of each recording: that its program is installed, that a setting without a default (such as the pose backend) is filled in, and that every voice can be read, which a shared microphone or a camera's own sound allows only when the voices are separated (*Microphones* set to `pyannote`). If anything is missing, nothing is started, and the output lists each reason; **Preview** reports the same.

```{figure} screens/run.png
:width: 100%
:alt: The Run page: the recordings as ticked chips, and the steps in the order they run, each with the count of its tasks in every state and a Run button of its own.

The Run page. Each step shows how many of its tasks are in each state.
```

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

Some decisions cannot be made reliably by software. When MACAW reaches one, processing of that recording pauses at that point while the other recordings continue. The Review page lists what is waiting; select a recording to see its items. The list of recordings gives, after each name, how many items wait on it and the tags of its notes, and the arrows on either side of it move to the recording before or after. Enter a name under *Your name* first: each decision is stored with it, and the browser keeps it for later sessions.

The *Play* card at the top of the page plays one camera, with its own sound or, for a camera without sound, with the sound of the device that hears the most participants, placed by the offsets between the files. It is made into a small copy once, with **Make a copy to play here**, or for every recording at once in the background with **Make copies of every recording** (`macaw previews`). The bar below the player marks the analysed part of the recording. The card stays in view while the page scrolls, and every time shown on the page in the same colour as a link, such as a voice's clip, a doubtful transcript line or a speech act, plays the camera from that moment.

```{figure} screens/review.png
:width: 100%
:alt: The Review page for one recording: the check that waits, and the offset between its two cameras with the confidence, a Keep button and the figure to inspect.

The Review page of a recording whose two cameras could not be aligned with confidence.
```

### Offsets between files

{bdg-secondary-line}`only when uncertain`

Separate devices start recording at different moments, and MACAW aligns them by their sound. Inspect the figure for the file and play the clip, then keep the estimated offset or enter one by hand.

The clip shows ten seconds from the start, the middle and the end of the recording, with every camera side by side, a waveform for each sound and all the sound mixed together. A file that is out of step sounds doubled or echoed, and the peaks of its waveform fall beside those of the others. For a camera without sound the clip is the only check: compare a clap or the movement of the lips with what is heard. The clip is made again whenever an offset is set by hand.

A device whose clock runs at a slightly different rate also drifts away from the others over a long recording. When the offset of successive segments moves along a straight line by more than 20 ms across the recording, MACAW corrects the drift along that line, resampling the file's sound and rescaling its frame times, and holds the file here; the figure shows the segment offsets and the line. An offset entered by hand is applied as it is, without drift correction.

(naming)=
### Who is on camera

{bdg-secondary-line}`only when uncertain`

Which person in a picture is which participant is stated by the researcher, once per camera and recording, under *Name the people on the camera playing* on this card. Find a moment at which the participants the camera is used for are all in view and clearly apart: drag the timeline above the frame, type a moment of the camera's file in seconds, or press **The player's time**; the frame follows. Choose a participant, each shown in a colour of their own, and click on them; choose the next and click again. A click on a participant's face or chest is the safest. Once the people have been detected (`pose.detect`), they are drawn on the frame in grey, and a click colours the person it falls on, or says that it falls on no one or on two people at once; a person then appears in the participant's colour exactly as MACAW will read the click. Press **Save the names**. MACAW names whoever was detected under each click and follows them through the rest of the recording. The clicks saved for the camera are listed above the frame, and **Clear this camera's names** removes them.

A camera used for one participant alone needs no click as long as no one else is ever in view on it at the same time. Every other camera needs one, and its recording waits here until it has one. A camera also waits when a click fell on no one or on two overlapping people, when two names were clicked on one person, or when someone who was not named is in view nearly as long as the participants who were; the card states the reason and the times at which to look. When the tracker loses a participant for a while, for example after they leave the picture, a further click at a moment after their return names them again. Press **Draw review frames** to see each person's box with the name it carries, and run again after clicking. **Keep as it stands** accepts the naming as it is, doubts included; the acceptance is recorded with who made it. Suggestions in place of clicks are available as an option ({ref}`suggest-names`).

From the command line, the same is done with `macaw name-people -r dyad01 --camera front --at 12.5 --person parent=0.3,0.5 --person child=0.7,0.6`, where each position is given as a share of the picture's width and height, measured from its top left corner.

### Voices on the shared microphone

{bdg-secondary-line}`always`

MACAW separates the voices. Read what each voice says, listen to its clips, and assign it to a participant. A voice that belongs to none of the participants, such as an experimenter's, is assigned *none*. MACAW proposes a participant for each voice from the participants' own microphones, their mouth movement and, where set, what each voice says and how high it is ({ref}`voice-hints`), and fills in the proposal; a weak one is marked to be listened to. Mouth movement (`face.movement`) is read from the faces by itself whenever the run reads the faces (`face.au`). By default every shared microphone waits for the researcher to confirm or change the proposal and press **Save these names**; play a clip on the video to see whose mouth moves. With *Voices on a shared microphone* set to wait only when uncertain, MACAW keeps a proposal it is sure of by itself and waits only when it is uncertain; the Overview then shows that check as left to the software.

(transcript)=
### Transcript

{bdg-secondary-line}`optional`

**Export to ELAN** writes the transcript as an ELAN file. Words, boundaries and speakers can be corrected in ELAN, and the corrected file is read back with **Import a corrected transcript (.eaf)** on the same card. Speech the recogniser omitted is placed on a tier of its own so that it can be restored. **Back to the recogniser's** returns to the automatic transcript at any time.

With a language model set under *Coding* ({ref}`coding`), **Ask the model for doubtful lines**, or `macaw transcript check -r dyad01`, has it read the transcript and list the lines that look misheard or invented, such as a phrase repeated many times or many words in a fraction of a second, and apart from them the lines that look cut off, a word or two that only make sense with the line before or after, each with a short reason. A line marked as cut off may still be what was said, since people do speak in single words; it points to where the recogniser may have split an utterance. The transcript itself is not changed. The ELAN export places these lines on tiers of their own, so that they can be checked against the recording while correcting. The model reads the text only and does not hear the recording: it misses errors that read naturally and may mark lines that are correct, so the list directs attention but does not replace listening.

### Speech acts

{bdg-secondary-line}`with language-model coding only`

The units the language model was uncertain about, together with a random sample of units it was certain about. Select the correct code for each and press **Save decisions**. The researcher's decision is stored alongside the model's.

### Notes

Anything to remember about a recording, such as a start to cut or a step to run again, is written on the *Notes* card: tick its tags, write the note and press **Save the note**. A note changes nothing by itself. The tags offered are listed under `qc.note_tags` in *project.yaml* and can be changed there. Notes are listed in the quality-control report, and `macaw export` writes them beside the measures as `features_review_notes.csv`. From the command line: `macaw note -r dyad01 --tag rerun --text "pose again after naming"`.

### After a review

Once an item is resolved, press **Run** again; the recording continues from where it paused. Every decision is stored with who made it and when, and is listed in the quality-control report.

How often each check asks is set under *Checks by a person*; the labels above are the defaults. The remaining cards on the Review page (*What a model saw on the review frames*, *Pixels to centimetres* (or to the unit chosen), *Where people look*, *Your coding*) belong to optional features.

(results)=
## 7. Inspect the results

{bdg-primary}`MACAW does this` {bdg-warning}`You do this`

The Results page has two parts.

Quality review
: Press **Write the QC page**, then open it. The report lists what the automatic checks flagged (for example a transcript that does not seem to match its audio, a mouth that does not move while its owner is said to be speaking, participants the tracker repeatedly lost, or a pitch track that jumps by an octave), every decision made by the software or by a person, an overview of each participant's speech, movement, distance and codes on a common time axis, the meaning and unit of every measure, and a set of frames to inspect. These frames include the moments the checks pointed to and a random selection, so that ordinary moments are inspected as well. The review is thus based on a sample rather than on every frame. A model that reads pictures can check these frames first ({ref}`model-look`).

Measures
: Select a step and a measure to plot it for each participant across the recording. A measure that is one number per recording is listed per participant, beside its value within each segment ({ref}`segments`). Each measure is labelled with its unit. Measures in three dimensions are shown in the project's unit of length; measures in pixels remain in pixels, since converting them holds only under the condition described in {ref}`scale`.

:::{note}
When a result looks wrong, adjust the relevant setting and run again. Only the steps that depend on that setting are recomputed.
:::

## 8. Export the data

{bdg-success}`What you get`

Run one of the following commands in the project folder. Each reports where it saved its files.

`macaw export`
: One table (CSV) containing every measure for every participant over time. Measures in metres, and measures in pixels for recordings with a pixel scale, are also given in the project's unit of length ({ref}`scale`). Values within named segments carry the segment's name ({ref}`segments`). A data dictionary beside the table describes each of its columns, including what an empty cell means ({ref}`data-dictionary`).

`macaw export -f parquet`
: The same table, with the same data dictionary, as a Parquet file, which is smaller and faster to read than a CSV when measures are exported frame by frame; R reads it with the arrow package.

`macaw export -f sequence`
: For each participant, whether they are speaking at each second, for sequence analysis.

`macaw export -f eaf`
: One ELAN file per recording containing the machine codes, as a starting point for manual coding, which can be imported again ({ref}`own-coding`). A linked file that began before the reference device is placed by its offset; one that began after it is linked without an offset and plays early in ELAN, and a drift correction is not carried into ELAN. The export names any such file.

`macaw export -f skeleton`
: Each participant's body keypoints frame by frame, one CSV per recording, in pixels with the detector's confidence for each point. With several calibrated cameras ({ref}`three-d`), a second file gives the points in metres.

`macaw methods`
: A Methods section describing the steps that were actually run, with references.

Every run records its settings and the software versions it used, so that the same data set can be reproduced later.

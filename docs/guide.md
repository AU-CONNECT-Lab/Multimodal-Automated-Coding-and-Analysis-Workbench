# Using MACAW

## Before you start

### You need

- Recordings: video, audio, or both. A microphone per person helps, but a single camera with its own sound works too.
- A computer with an NVIDIA graphics card for anything beyond a few recordings. Without one everything still runs, only much slower.
- Free disk space about the size of your videos, because MACAW makes one tidy copy of each video.

### Good to know

- Your recording folder is never changed. Results go to a folder of their own.
- Transcription works in many languages. Word and sentence measures are English only for now.
- Two-person recordings are the most tested. Groups of three or more are supported.

### What MACAW does not do

- Run statistics or compare groups. Each recording is described on its own, and the analysis stays yours.
- Guess silently. When it cannot decide something reliably, it asks you.

### Installing

This part is for whoever sets up the computer. MACAW is installed once per computer.

1. Install Python 3.10, 3.11 or 3.12, git, and ffmpeg.
2. Install PyTorch for the computer's graphics card, following [pytorch.org](https://pytorch.org/get-started/locally/). Doing this first keeps the graphics card in use.
3. Install MACAW with all its parts:

   ```bash
   pip install "pymacaw[audio,asr,text,vision,gui,report] @ git+https://github.com/AU-CONNECT-Lab/Multimodal-Automated-Coding-and-Analysis-Workbench.git"
   ```

Some tools are not included because of their licences: YOLO (body pose), OpenFace (face) and openSMILE (voice feature sets). Install them only if you choose them in the settings. Telling voices apart on a shared microphone uses a model on Hugging Face: accept its terms there once and log in to Hugging Face on this computer.

Once MACAW runs, the *This computer* section of its settings page shows whether it finds ffmpeg and the other programs.

## 1. Organise your recordings

{bdg-warning}`You do this`

Make one folder per recording. End each file's name with the person it belongs to, using your study's role names (`parent` and `child` unless you change them). A file whose name ends in anything else, such as `room`, belongs to everyone in the recording.

```text
recordings/
  dyad01/
    dyad01_parent.wav     the parent's microphone
    dyad01_child.wav      the child's microphone
    dyad01_room.mp4       a camera both share
  dyad02/
    ...
```

One camera only
: MACAW takes the sound from the video and tells the voices apart. Set *Microphones* to pyannote, the shared-microphone option, in the settings. You then say which voice is whose.

Several visits per family
: Use a folder per family with a folder per visit inside, and describe that layout once in the settings under *Recording folder pattern*.

Leaving a recording out
: It can be excluded in the settings without moving any files.

## 2. Create a project and open the workbench

{bdg-warning}`You do this`

This is the only part that needs typing. Open a terminal and run these three lines, with your own project name and the folder that holds your recordings:

```bash
macaw init parent-child-play --name "Parent-child play" --raw-dir "D:/recordings"
cd parent-child-play
macaw gui
```

The first line creates a project folder with a settings file and a results folder. The last line opens the workbench. Next time, the last two lines are enough. If you prefer a browser tab to a window, use `macaw gui --browser`.

The workbench has four pages along the top, a fifth for projects with several cameras, and a gear for the settings. **Open folder** opens the project folder on your computer.

Overview
: The page the workbench opens on. How many recordings, people and minutes the project has, what to do next (what failed, what waits for you, what is left to run), and a square for each recording and step showing where it stands.

Run
: Choose recordings and steps, start a run and watch it.

Review
: The things waiting for you. The number on the tab says how many.

Results
: The quality page and every measure over time.

Cameras
: Only when two or more cameras film the same people: calibrating them for 3D ({ref}`three-d`).

Settings (the gear)
: Everything about the project.

Steps appear under short names, such as `text.asr` for transcription and `pose.kinematics` for body movement.

## 3. Fill in the settings

{bdg-warning}`You do this`

Click the gear. The choices most studies change are laid out as a form, in five groups. Everything else is below the form as text. When you save, MACAW checks the settings and tells you which finished results they would put out of date.

Project
: Where the recordings are, how many people each one has and their roles. *Task starts at* and *Task ends at*, in seconds, keep only the part you study, for example to skip the researcher's introduction.

Steps
: Which measures a run makes, and how many recordings run side by side.

Tools
: Which program does each kind of measurement, the language spoken, and whether each person has a microphone or they share one. **Body pose has no default**: choose MediaPipe or YOLO yourself (see {ref}`privacy`).

Checks by a person
: For each check in {ref}`step 5 <review>`: does MACAW always wait for you, wait only when it is unsure, or never wait.

Coding
: The codebook and the language model. Only needed for {ref}`coding with a language model <coding>`.

:::{tip}
Does the task start at a different time in each recording? Each recording can have its own start and end. Put them in a small file named after the recording, such as `dyad01.yaml`, in the `recordings` folder inside the project.
:::

## 4. Run

{bdg-primary}`MACAW does this`

On the Run page, tick the recordings and the steps you want, then press **Run**. Whatever a step needs runs with it. The **Run** at the end of a step's row runs only that step, and what it needs, for the ticked recordings. **Preview** lists what would run without running it, and **Stop** ends a run cleanly. Try one recording first to see that everything works.

Each step shows where it stands:

| Status | What it means |
|---|---|
| {bdg-success}`up to date` | Done with the current settings. It is not run again. |
| {bdg-info}`not run yet` | Runs the next time you press Run. |
| {bdg-warning}`to run again` | A setting or a file it depends on changed since it ran. |
| {bdg-warning}`waiting for a person` | Stopped at a check. See {ref}`step 5 <review>`. |
| {bdg-danger}`failed last time` | The output at the bottom of the page says why. |

You can close the window at any time. The next run picks up where the last one stopped and does not repeat finished work.

(review)=
## 5. Check what MACAW decided

{bdg-warning}`You do this`

Some things software cannot decide reliably. When MACAW meets one, that recording stops at that point and waits for you, while the rest of the project carries on. Open the Review page and choose a recording to see what is waiting.

### Which file is whose

{bdg-secondary-line}`once per project`

MACAW shows the files it assigned to each person, read from the file names. Check the list and press **Confirm for the whole project**.

### Offsets between files

{bdg-secondary-line}`only when unsure`

Separate devices start recording at different moments, and MACAW lines them up by their sound. Look at the file's figure, then keep the offset or set it by hand.

### Who is on camera

{bdg-secondary-line}`only when unsure`

MACAW decides which person in the picture is which. Press **Draw review frames**, look at the names on the boxes, and keep them or swap two that are mixed up.

### Voices on a shared microphone

{bdg-secondary-line}`always`

MACAW finds the separate voices and suggests whose each one is. Listen to the clips and pick the right person for each voice. A voice that belongs to nobody in the study, such as a researcher, is marked *none*.

:::{tip}
The suggestion can also use what each voice says and how high it is, if you tell it what to look for in `project.yaml`. Nothing is assumed about who your participants are: you write the rules.

```yaml
identity:
  voice_hints:
    - says: '^\s*(hey |look )?(mommy|mom|mama)[,.!?]'   # calling for mom: likely the child
      role: child
    - says: "\\b(mommy|mama)'s (gonna|going|not)\\b"  # speaking of herself: likely the mother
      role: mother
  pitch_order: [mother, child]   # lowest voice first; used when two voices differ by 40 Hz or more
```

`says` is a regular expression, matched against each line a voice said, ignoring case. A voice goes to a role when more of its lines match that role than any other. Where the words and the pitch disagree, or either disagrees with a microphone or a face, the suggestion says so, and with review set to flagged it waits for you.
:::

(transcript)=
### Transcript

{bdg-secondary-line}`whenever you want`

Export the transcript to ELAN, correct words, boundaries or speakers there, and import the corrected file. Speech the software left out sits on a tier of its own, so you can put it back. You can always go back to the automatic transcript.

### Speech acts

{bdg-secondary-line}`only with language-model coding`

The units the language model was unsure about. Pick the right code for each and save. Your choice is kept beside the model's.

### After a check

When a check is cleared, press Run again and that recording carries on from where it stopped. Every decision is kept with who made it and when, and is listed on the quality page.

How often each check asks is up to you: the labels above are the starting choices, and each check can be changed under *Checks by a person* in the settings.

## 6. Look at the results

{bdg-primary}`MACAW does this` {bdg-warning}`You do this`

The Results page has two parts.

Quality page
: Press **Write the QC page**, then open it. It lists what the checks flagged (a transcript that does not seem to fit its audio, a mouth that does not move while its owner is said to be talking, people the tracker kept losing, a pitch that jumps an octave, or that sits an octave high throughout when `qc.f0_median_max_by_role` gives the highest believable median for a role), every decision made by the software or by a person, an overview with each person's talk, movement, distance and codes on one clock (list other measures for it under `qc.overview_features` in `project.yaml`), and a set of frames to look at. Those frames are the moments the checks pointed at, plus frames picked at random so you also see ordinary moments. You look at a sample, not at every frame.

Measures over time
: Pick a step and a measure to see it for each person across the recording.

:::{note}
Something looks off? Change the setting and run again. Only the steps that setting affects are run again.
:::

## 7. Take your data with you

{bdg-success}`What you get`

Type one of these in the project folder. Each one prints where it saved its files.

`macaw export`
: One table (CSV) with every measure, for every person, over time. Add `--rmd` for an R Markdown file that loads it. Recordings with a scale also get their pixel measures in centimetres ({ref}`scale`).

`macaw export -f sequence`
: Who is talking at each second, per person, for sequence analysis.

`macaw export -f eaf`
: One ELAN file per recording with the machine codes, as a starting point for your own coding.

`macaw export -f skeleton`
: Every person's body points, one CSV per recording: a row per person, video and frame, with each point's position in pixels and how sure the detector was (`nose_x`, `nose_y`, `nose_conf`, and so on), the same columns a function in the `scripts` folder gets ({ref}`measures`). Nothing is smoothed, and a point the detector was less sure of than `pose.min_confidence` has no position. With 3D ({ref}`three-d`), `<recording>_3d.csv` gives the points in metres, with `_z` added, on the calibration's axes: after `macaw calibrate-cameras` these are the first camera's (x to its right, y down, z straight ahead).

`macaw methods`
: A Methods section describing exactly the steps you ran, with its references.

Every run keeps its settings and the software versions it used, so anyone can rebuild the same table later.

(coding)=
## Coding speech with a language model

This is optional. A language model can code each utterance or turn of the transcript with the codes in your codebook, such as question, instruction or praise. You write the codebook: each code with a description, the criteria, examples, counterexamples and borderline cases.

Where it runs
: On a model on your own computer or lab server, or on a cloud service (OpenAI, Anthropic). With a cloud service the transcript text is sent there, so check that your ethics approval allows it.

Whose talk
: Everyone's, unless you name roles under *People to code*, for example only the child. The others' talk, and a researcher's when other voices are on, is still shown to the model around each unit so it can tell an answer from a new topic, but it is not coded.

How sure it is
: Each unit is asked several times (five by default). The share of matching answers is its confidence. The model may also answer "cannot tell".

What comes to you
: Units below 70% agreement, and every "cannot tell", appear under *Speech acts* on the Review page.

Agreement
: MACAW reports how well your choices and the model's agree, as Krippendorff's alpha and Cohen's kappa.

How the codes were used
: `macaw describe-codes` lists, for each recording, coder and person, how many units and how much time each code took, in how many runs of the same code, and how many of those runs last a single unit. Many single-unit runs mean the coding flips from unit to unit, which is worth a look before you analyse it. Add `--out codes.csv` to keep the table.

To start, open *Coding* in the settings, add layer B, choose your codebook file and the model, then run as usual.

(measures)=
## Your own body measures

This is optional, and is written in `project.yaml` rather than in the workbench. Besides the body measures MACAW always makes, you can define your own: from a point on one person to a point on another person, or on the same person. Each one becomes a measure over time like any other, with its own name.

```yaml
kinematics:
  measures:
    - name: hand_to_child_head
      person: parent
      joints: [left_wrist, right_wrist]
      to_person: child
      to_joints: [nose]
      method: distance
      per_body_size: true
    - name: head_toward_parent
      person: child
      joints: [nose]
      to_person: parent
      to_joints: [nose]
      method: offset_toward
      per_body_size: true
```

person, to_person
: A role from your project's roles. `to_person: self` measures within one person, for example a hand to their own face.

joints, to_joints
: The skeleton's points (`nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`), plus `shoulder_center` and `hip_center`. With several listed, the nearest pair counts.

method
: `distance`; `approach_speed`, how fast that distance shrinks (positive while coming closer); `offset_toward`, how far one point sits from the person's own shoulder midpoint toward the other person, such as a head turned toward them; or a function of your own (below).

per_body_size
: Divides by the person's shoulder width, so a measure does not depend on how far from the camera they sit.

Two people are measured only on a camera that shows both of them.

### Gestures you coded

If you code gestures in ELAN on a tier of each person, MACAW can measure every coded gesture from the wrists: peak speed, submovements (the speed peaks it is made of), holds between the first and last submovement, how regular the submovements are, the area the hand covers, how high it reaches relative to the body, the gesture space it uses, and in how many directions around the body's centre it reaches out (above, below, to either side and the four corners). Each gesture gives one row per measure and hand.

```yaml
kinematics:
  gestures:
    tier: gesture
    still_speed: 15
    peak_prominence: 40
```

still_speed, peak_prominence
: In pixels per second, so they depend on your camera: a hand slower than `still_speed` is still, and a speed peak counts as a submovement when it rises at least `peak_prominence` above its surroundings. The wrist speeds in the results help to choose them. Set `peak_prominence` well above the jitter of the tracking, or every wobble counts as a submovement: before trusting the counts, check a few gestures against the video. Speeds, distances and areas stay in pixels; exports add centimetres when a recording has a scale ({ref}`scale`).

annotator
: When two people coded the tier, name whose gestures to measure.

holds
: `each_hand` (the default) counts the holds of each hand on its own. `both_hands` counts a hold only while both hands are still, once per gesture; a hand that is out of view throughout leaves it empty.

Height and gesture space need the hips in view. When a table hides them, those two stay empty.

:::{important}
Gesture space is drawn on the picture, so it follows the body only while the person faces the camera or has their back to it. Seen from the side, the body looks narrow and the hands seem to reach far out of it. Height is much less affected. With several cameras, the 3D gesture space below has no such limit.
:::

Tracking jitter adds a little to every speed and area, most to small, slow movements. Compare them between people and conditions measured the same way rather than reading them as true sizes.

With several calibrated cameras ({ref}`three-d`), the same measures are also taken in the room, named with `_3d` at the end, once you set two more thresholds in metres per second:

```yaml
kinematics:
  gestures:
    still_speed_3d: 0.05
    peak_prominence_3d: 0.1
```

These numbers are only an example: as in the picture, there is no default, and the 3D wrist speeds in the results help to choose them. In 3D the area, the height and the gesture space are taken in the person's own frontal plane, set by their hips and shoulders, so they hold whichever way the person faces a camera; the hips and shoulders must be placed in 3D.

### A method of your own

Put a Python file in the `scripts` folder next to `project.yaml`, and name its function as `method: file.function`. The function receives two tables, one per person, with a row per video frame, a column `t` for the time, and `<joint>_x` and `<joint>_y` for every point. It returns one number per row.

```python
# scripts/my_measures.py
def wrist_height_gap(person, other):
    return other["left_wrist_y"] - person["left_wrist_y"]
```

```yaml
    - name: wrist_height_gap
      person: parent
      to_person: child
      method: my_measures.wrist_height_gap
```

With several calibrated cameras ({ref}`three-d`), add `in_3d: true` to a measure whose function also makes sense on points in the room. It is then run a second time on tables that also have `<joint>_z`, with every position in metres, and gives `wrist_height_gap_3d`. Without it, a function written for the picture is not run in 3D.

When you change the file, the next run recomputes this measure. The `scripts` folder can also hold whole new steps or backends: `macaw new-extractor` and `macaw new-backend` write a working example, and copying its `extractor.py` or `backend.py` into `scripts` is enough, with nothing to install.

:::{warning}
Every file in `scripts` runs as Python whenever MACAW reads the project. Only put files there that you trust.
:::

(scale)=
## Centimetres instead of pixels

This is optional. Distances, speeds and areas from the video are in pixels. They can be given in centimetres too, with one condition.

:::{important}
A scale is right only at the distance from the camera where it was measured. Something twice as far away looks half as big, so a scale taken at the parent's seat makes a child sitting further back look smaller than they are. Measure an object where each person sits. When people sit at different distances, measure at each person's place, and say whose place it was: each person then gets their own scale. A measure between two people, such as the distance between them, gets centimetres only when both have the same scale.
:::

Film it before you record (calibration)
: Before the session, with the camera where it will stay, film an object of known length (a ruler or a sheet of paper) held at each person's place. Put the clip in a `calibration` folder next to `project.yaml`, and write `scale: {calibration: lab.mp4}` in `project.yaml` so every recording uses it. When the camera was moved for some sessions, film a new clip and name it in those recordings' own files, such as `recordings/003.yaml`. Moving or zooming the camera after the clip makes it wrong, and a clip carries over only to video of the same size.

Measure it in the recording
: When the object is in the recording itself, measure it there instead. A recording's own measurements come before a calibration clip.

Write the number
: Put `scale: {cm_per_px: 0.11}` in `project.yaml` for everyone, or `scale: {cm_per_px_by_role: {child: 0.12, parent: 0.09}}` for each person's place; in `recordings/003.yaml` for one recording. Numbers written this way come before any measurement.

To measure, open the Review page and go to *Pixels to centimetres*. Choose the recording's video or a calibration clip, show a frame, click the two ends of the object, type its length, say whose place it was, and save. Measure two or three times: the page lists each measurement, and their spread shows how precise the clicks are. In a terminal the same is `macaw set-scale --calibration lab.mp4 --points 412,630,1508,641 --length-cm 30 --person child`, or `-r 003` in place of `--calibration` for a recording's own video.

The results stay in pixels. `macaw export` adds every measure in pixels a second time in centimetres, named with `_cm` at the end, and writes `scale.csv` beside the table with each recording's scale for each person.

The quality page gives each person's nose-to-shoulders length in centimetres at their scale (a value far from a real one means they sit nearer or further than the object was), and flags a person who looks more than 10% bigger or smaller at some point in the recording (`scale.max_size_change`), since then they moved toward or away from the camera. It compares stretches of 30 s (`scale.size_window_s`), which suits people who stay seated: a few seconds bent over or looking down does not count. When people walk about, set it to about 5 s, or a walk to the far side of the room and back goes unseen; a person who bends over for a few seconds is then flagged too.

(three-d)=
## Several cameras: 3D

This is optional. When two or more cameras film the same people from different sides, MACAW can place each person's body points in the room, in metres. Distances then no longer depend on how far someone sits from a camera, and it becomes possible to say where a person's head points relative to the other person.

It needs four things:

- Every body point you care about is seen by at least two cameras, from clearly different directions.
- The cameras film at the same time. MACAW lines them up by their sound, as it does for any recording.
- The cameras are calibrated: MACAW knows each lens and where the cameras stand. This comes from a short clip of a printed board, filmed before the session.
- The cameras do not move or zoom after that clip. When they do, film the board again.

:::{note}
Placing people in 3D has been checked on simulated cameras and on a public recording of an adult and a toddler filmed by four calibrated cameras, against that recording's own 3D reconstruction. Finding the cameras from a board has been checked on simulated cameras only.
:::

### 1. Print the board

The workbench has a *Cameras* page for every step here; it appears once a recording has two or more videos, or a setup is in the `calibration` folder. Each button on it shows the command it runs.

*Draw the board* on that page, or `macaw calibration-board`, writes a ChArUco board (a chessboard with small codes in its white squares) as `calibration/board_A2.png`. Print it at 100%, glue it flat on something stiff such as foam board, and measure one square with a ruler. Type that length on the page and save it, or write it in `project.yaml`:

```yaml
pose3d:
  board:
    square_mm: 74.6
```

The default board fills A2. For A3, set `square_mm: 50` and `marker_mm: 37` before printing (`--paper A3`).

### 2. Film it before the session

With the cameras where they will stay, clap once in view of all of them, then move the board slowly through the space where people will sit, tilting it towards and away from the cameras, for about a minute. Keep it in view of two or more cameras at a time, and near enough that it fills a good part of the picture. Put one clip per camera in a folder such as `calibration/lab`, each named like that camera in the recordings: `room.mp4` for the camera whose recordings are named `003_room.mp4`.

### 3. Find the cameras

```bash
macaw calibrate-cameras lab
```

or choose the setup on the *Cameras* page and press *Find the cameras*. It takes a few minutes and reports, for each camera, how far the board's corners land from where the camera saw them (under 1 or 2 pixels is good), how often each pair of cameras saw the board together, and where the cameras stand, drawn in 3D on the page. The page also says when a clip or the board's size changed since, so the cameras need finding again. Then press *Use for this project*, or name the setup in `project.yaml`, or in a recording's own file when some sessions used another setup:

```yaml
pose3d:
  calibration: lab
```

A lab that has calibrated its cameras with another tool can put that calibration file in the `calibration` folder instead and name it, such as `calibration: lab.toml`. MACAW reads Anipose's `calibration.toml`; set `pose3d.units` to what its positions are in when the file does not say.

### 4. Run and look

Run as usual. Each person must carry the same name on every camera (see *Who is on camera* in step 5 above). The results gain measures ending in `_3d`, in metres, beside the measures from the picture:

`left_wrist_speed_3d`, `hand_distance_3d`, ...
: Each person's wrist speed, acceleration and jerk, and the distance between their wrists.

`interpersonal_distance_3d`, `head_distance_3d`
: Between two people's shoulder midpoints and between their heads.

`head_toward_deg_3d`
: The angle between where a person's head points and the direction to the other person's head; 0 means the head points straight at them. This is head direction, not gaze.

`gesture_left_peak_speed_3d`, ...
: The measures of each gesture you coded, in the room, when `still_speed_3d` and `peak_prominence_3d` are set (see *Gestures you coded*).

`<name>_3d`
: Your own measures ({ref}`measures`) in the room. `distance`, `approach_speed` and `offset_toward` always get one; a method of your own only with `in_3d: true`. In 3D, `offset_toward` is how far the point sits from the shoulder midpoint in the direction of the other point, depth included. Two people are measured whenever both are placed in 3D, not only when one camera shows them both.

On the Results page, *In 3D* shows everyone's skeleton and the cameras; drag to turn it and move the slider to go through the recording. The quality page says, for each person, how often they were placed in 3D, how well the cameras agree on them, and warns when a bone keeps changing length, which means some of its points were misplaced.

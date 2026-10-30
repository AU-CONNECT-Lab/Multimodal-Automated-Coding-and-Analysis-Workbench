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

The workbench has three pages along the top and a gear for the settings:

Run
: Choose recordings and steps, start a run and watch it.

Review
: The things waiting for you. The number on the tab says how many.

Results
: The quality page and every measure over time.

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

On the Run page, tick the recordings and the steps you want, then press **Run**. Whatever a step needs runs with it. **Preview** lists what would run without running it, and **Stop** ends a run cleanly. Try one recording first to see that everything works.

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
: One table (CSV) with every measure, for every person, over time. Add `--rmd` for an R Markdown file that loads it.

`macaw export -f sequence`
: Who is talking at each second, per person, for sequence analysis.

`macaw export -f eaf`
: One ELAN file per recording with the machine codes, as a starting point for your own coding.

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

When you change the file, the next run recomputes this measure. The `scripts` folder can also hold whole new steps or backends: `macaw new-extractor` and `macaw new-backend` write a working example, and copying its `extractor.py` or `backend.py` into `scripts` is enough, with nothing to install.

:::{warning}
Every file in `scripts` runs as Python whenever MACAW reads the project. Only put files there that you trust.
:::

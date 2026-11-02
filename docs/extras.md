# Optional features

The features on this page extend the standard procedure described in [Using MACAW](guide.md), and each can be used on its own. Where a feature is configured in the text part of the settings, a block is given that can be pasted into the *project.yaml* box on the settings page and adapted to the study.

(coding)=
## Coding speech acts with a language model

A language model can assign each utterance or turn of the transcript a code from a codebook written by the researcher, such as question, directive or praise.

:::{important}
With a cloud service (OpenAI or Anthropic), the transcript text, and video frames if selected, are sent to that service. Check that the ethics approval and the participants' consent cover this. A model running on the lab's own computer or server keeps all data local.
:::

### The codebook

The codebook is a YAML file in the project folder, for example `codebook.yaml`. Each code has a description and, ideally, criteria, examples, counterexamples and borderline cases.

```yaml
name: speech_acts          # the tier the codes are written to
unit: turn                 # utterance or turn
codes:
  - code: praise
    description: The parent approves of something the child did.
    criteria: [names the behaviour, positive evaluation]
    positive_examples: ["good job putting that away"]
    negative_examples: ["okay"]
    boundary_cases: ["thanks, said flatly"]
  - code: directive
    description: The parent tells the child to do something.
    criteria: [imperative or request for action]
```

A codebook that lists the codes by name only, without descriptions, is coded as it stands, which shows what the model makes of the names alone. A codebook in which only some codes are described is treated as an unfinished draft, and coding stops.

### Setting up

Open the *Coding* group in the settings, add layer B under *Coding layers*, select the codebook under *Codebook*, and choose the *Language model server* and the *Language model*. Then run as usual. For a cloud service, the key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) is set as an environment variable before the workbench is started; the *This computer* panel shows whether it is found.

### A model on the lab's own computer

Ollama is the simplest way to run a model locally. After installing it from [ollama.com](https://ollama.com), download a model once:

```bash
ollama pull qwen3:8b
```

For coding with video frames, use a model that reads images and answers directly, such as `qwen3-vl:8b-instruct`. Then set *Language model server* to `ollama` and *Language model* to the model's name, and leave *Server address* empty when Ollama runs on the same computer. A model of this size needs about 10 GB of graphics memory, somewhat more with frames; a model that does not fit in graphics memory runs many times more slowly. A model server on another machine (Ollama, vLLM or llama.cpp) is reached by entering its address under *Server address*.

:::{tip}
Ollama may reserve several gigabytes of working memory for a prompt cache, which brings no benefit here because every prompt differs. Starting the server with the environment variable `LLAMA_ARG_CACHE_RAM=0` disables the cache.
:::

### Whose speech is coded

By default, every participant's speech is coded. *People to code* restricts coding to the roles named, for example the child only. The other participants' speech is still shown to the model around each unit, so that a reply can be told from a new topic, but it is not coded.

### What the model is given

By default, the model receives each line's speaker, words and timing (onset, duration, and the pause or overlap before it). Several settings in the *Coding* group change this:

*What the model is shown besides the words*
: Removes the timing, or adds pitch, loudness and speech rate, each relative to that speaker's median in the recording, so that a child's higher voice is not read as excitement. It can also add three video frames per unit, taken near its start, middle and end from the camera that films the speaker. Frames require a model that reads images and make coding considerably slower. They carry no annotation, so the note to the model should state who is who and where they sit.

*Note to the model*
: Free text added to the prompt as written, for information the model cannot know, such as "A is the mother, seated on the left; B is the child."

*Examples people coded*
: A table of units coded by people, shown to the model as examples (see below).

*Model says why*
: The model names in a few words what each code rests on: the words, the timing, the voice or the picture. The reasons are shown on the Review page. Asking for reasons can itself change the answers, so a coding with reasons and one without should be compared as two versions rather than switched halfway.

*Model reasons before answering*
: Some models reason at length before answering, which makes each request several times slower. *No* asks for an immediate answer. This applies to models on the lab's own computer or server.

### Reliability and review

Each unit is coded several times (*Answers per unit*, five by default), and the share of identical answers is the unit's confidence. The model may also answer "cannot tell". Units below *Review below this agreement* (0.7 by default) and every "cannot tell" are placed on the Review page under *Speech acts*.

A model can give the same wrong code every time it is asked, so consistency across answers detects only some of its errors. A random sample of units the model was certain about (*Sure units checked at random*, five per recording) is therefore placed on the Review page as well, marked *random check*. Once these are decided, the quality-control report gives the proportion of them that the researcher coded differently, an estimate of how often the model's confident answers are wrong.

Agreement between the researcher's decisions, the model and imported human coding is computed as described in {ref}`own-coding`.

### Examples drawn from human coding

Once human coding has been imported ({ref}`own-coding`), a few units of each code can be drawn into a table:

```bash
macaw draw-examples examples.csv --coder anna --per-code 3
```

Each row gives the code, the speaker, the words, the preceding line and the source. Remove or edit rows as needed, and add a reason in the `why` column where the model should read one. Then select the file under *Examples people coded*. A recording is never shown its own examples. Units used to estimate agreement should not be the ones the examples were drawn from.

### Comparing codebooks, models or inputs

To test whether a fuller codebook, another model or different input changes the coding, the same units can be coded in several versions at once. Each version states only what it changes; everything else follows the *Coding* group.

```yaml
coding:
  judges:
    names_only: {codebook: codebook_names.yaml}
    bigger: {model: qwen3:14b}
    words_only: {show: []}
```

Each version runs as a step of its own, so changing one recodes only that one. When all codebooks carry the same `name`, all versions write to the same tier, and `macaw agreement --tier speech_acts` sets each version beside the agreement between human coders, code by code. Versions do not add units to the Review page. The Methods text describes each version.

A model asked twice does not always give the same code, so two versions can differ by chance alone. `macaw agreement --tier speech_acts --samples` treats each of the model's repeated answers as a separate coder; the resulting agreement shows how consistent the model is with itself. A difference between versions smaller than this is not interpretable. This requires more than one answer per unit.

(own-coding)=
## Human coding and inter-rater agreement

Coding done by people, in ELAN or another program, can be imported beside the machine codes, compared with other coders, with the language model or with the automatic gaze codes, and summarised.

### Importing

On the Review page, enter a name under *Your name*, select the recording and press **Import your coding (.eaf, .csv, .xlsx)**. Many files can be imported at once from the command line, each file name starting with its recording (for example `dyad01_anna.eaf`):

```bash
macaw import-codes coding/anna --coder anna
```

Importing again under the same name replaces that coder's codes for the same participant and tier; other coders' codes are kept.

Accepted files are ELAN files (`.eaf`) and tables (`.csv`, `.tsv` or `.xlsx`) with the columns `tier`, `start`, `end` and `code`, and optionally `person`, as most coding programs can export. Start and end are given in seconds or as a clock time such as `1:02.5`, counted from the start of the recording's video, as ELAN shows them when that file is coded.

Which participant each tier belongs to, and under which name it is stored, is stated once in the settings, using each tier's name as it appears in the files:

```yaml
coding:
  imports:
    Child Gaze:
      person: child
      tier: gaze_target
      labels: {Mom: parent, Away: elsewhere}
```

`person` is one of the study's roles; it can be omitted when the file itself states whose the tier is (the tier's *Participant* in ELAN, or the `person` column). `tier` is the name the codes receive in MACAW; to compare them with machine codes, use the machine's tier name, `gaze_target` for gaze or the codebook's `name` for language-model codes. `labels` renames codes to match; unlisted codes keep their names. Tiers that are neither listed nor assigned to a participant in the file are skipped, and the import reports them. Files exported by MACAW itself are imported without such a list.

### Agreement

Coders who coded the same intervals
: `macaw agreement --tier speech_acts`

Coders who each set their own boundaries
: `macaw agreement --tier speech_acts --units anna` takes each of anna's intervals as a unit and gives it the code of every other coder that covers most of it.

States that last, such as gaze
: `macaw agreement --tier gaze_target --grid-s 0.5` divides time into half-second bins and compares the bins.

The report gives Krippendorff's alpha across all coders, Cohen's kappa for each pair, and the agreement on each code for each pair, so that with two human coders and a language model, the model's agreement with each person stands beside the agreement between the people. `--out agreement` additionally writes a folder with a table of every unit, each coder's code and the words spoken, which shows which codes are confused with which and helps when revising a codebook.

:::{note}
A coder who coded only part of a recording leaves the rest uncoded, which counts as disagreement. Restrict the comparison to the part all coders coded.
:::

### Summary of code use

`macaw describe-codes` lists for each recording, coder and participant how many units and how much time each code took, in how many runs of the same code, and how many of those runs last a single unit. Many single-unit runs indicate that the coding alternates from unit to unit, which deserves a closer look before analysis. `--out codes.csv` saves the table.

(gaze)=
## Gaze: who looks at whom

For each participant in each frame, MACAW estimates the point in the picture they look at, and from it whom they look at, which named object, somewhere else in the picture, or something outside it. The estimate is made from the participant's head and the whole scene, so it also works when a face is partly turned away. The first run downloads a model of about 1.3 GB.

:::{important}
Whom a participant looks at can be determined only when the person looked at is visible in the same picture, so this requires a camera that shows everyone involved. A camera that films one participant alone yields where in the picture they look, not at whom; for that case see *Seat positions instead of the picture* below. The estimate reflects head and eye orientation as visible in the video and is not a substitute for an eye tracker: it distinguishes the other person's face from the table or the toys, but not one toy from the one beside it.
:::

Objects to be distinguished are named in plain English under *Things people look at* in the *Tools* group, for example `bottle, book, toy car`. The quality-control report warns when a name is rarely found; another word for the same object often works better. When objects are detected that are not there, raise the detection threshold (`gaze.object_score`, 0.35 by default); when objects that are there are missed, lower it.

For each pair of participants the results give, at every moment, whether one looks at the other and whether the two look at each other (mutual gaze), and for the whole recording the proportion of time spent looking at the other, the number of looks and the proportion of mutual gaze. A glance of a frame or two toward something else is merged into the preceding look, so that it does not count as a new look. Every look is also stored as a code, so that it can be exported to ELAN, compared with human gaze coding ({ref}`own-coding`) and summarised with `macaw describe-codes`.

### Checking the estimate on the study's own video

On the Review page, *Where people look* shows a frame with each participant's head marked. Indicate whom or what each participant looks at and press **Save what they look at**; **A frame at random** selects the next frame. Frames should be drawn at random rather than chosen for clarity, otherwise the check overstates accuracy. About 20 frames per participant give a useful estimate. The quality-control report then gives, for each participant, how often MACAW agreed with these labels and lists the frames where it did not.

### Seat positions instead of the picture

When the person looked at is often outside the picture, for example a parent seated beside the camera that films the child, whom a participant looks at can be inferred from the direction of their head (and, with eye direction enabled, their eyes), provided the seat positions are known. Give the positions of the participants and the camera as seen from above, and the height of their eyes, all in one unit (centimetres, for example), with the camera at the origin and the first axis pointing to the camera's right:

```yaml
gaze:
  backend: angles
  seats:
    camera: [0, 0, 110]
    parent: [-50, 120, 115]
    child: [30, 140, 85]
  tolerance_deg: 20
face:
  gaze: true                 # use eye direction as well as head direction
```

A look within `tolerance_deg` of the direction to a person counts as a look at that person. There is no default, as the appropriate value depends on how far apart the participants sit. The method assumes that participants remain in their seats and that the camera is level; a recording with a different arrangement gets its own seats in its recording file (for example `recordings/dyad03.yaml`). Head angles estimated from a single camera carry a bias of their own, so labelling frames as described above is recommended: MACAW then corrects each participant's offset from the labelled frames, and once a participant has three or more labelled frames looking at a target, it takes that target's direction from the frames rather than from the seats. Labelled frames alone, without seats, are also sufficient, and can name objects that stay in place, such as a toy on the table.

(voice-hints)=
## Assigning voices on a shared microphone

When participants share a microphone, the proposed assignment of voices to participants can draw on what each voice says and on how high it is. MACAW assumes nothing about the participants; the rules are written by the researcher:

```yaml
identity:
  voice_hints:
    - says: '^\s*(hey |look )?(mommy|mom|mama)[,.!?]'   # calling for the mother: likely the child
      role: child
    - says: "\\b(mommy|mama)'s (gonna|going|not)\\b"  # referring to herself: likely the mother
      role: mother
  pitch_order: [mother, child]   # from the lowest voice to the highest
```

`says` is a regular expression matched against each line a voice said, ignoring case. A voice is proposed for a role when more of its lines match that role than any other. `pitch_order` is used when two voices differ by 40 Hz or more. Where the words and the pitch disagree, or either disagrees with a microphone or a face, the proposal says so, and the voices wait for the researcher when *Voices on a shared microphone* is set to wait only when uncertain.

(motion)=
## Motion energy in a region of the picture

Motion energy measures the amount of movement within a rectangle of the picture, for each participant separately. Give each participant a region as `[x, y, width, height]` in pixels of their own camera's picture, or of the shared camera when they have none, as read off a still frame in an image viewer:

```yaml
motion:
  rois:
    parent: [80, 120, 640, 900]
    child: [1100, 200, 600, 800]
```

When the camera was placed differently in some sessions, those recordings get their own regions in their recording files.

(measures)=
## Custom body measures

Besides the body measures MACAW always computes, measures between two keypoints can be defined, from a point on one participant to a point on another participant or on the same participant. Each becomes a measure over time under its own name.

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

`person`, `to_person`
: Roles of the study. `to_person: self` measures within one participant, for example a hand to their own face.

`joints`, `to_joints`
: Keypoints of the skeleton: `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`, and the derived `shoulder_center` and `hip_center`. When several are listed, the nearest pair counts.

`method`
: `distance`; `approach_speed`, the rate at which that distance shrinks (positive while approaching); `offset_toward`, how far a point lies from the participant's own shoulder midpoint in the direction of the other participant, such as a head turned toward them; or a function of one's own (below).

`per_body_size`
: Divides by the participant's shoulder width, so that the measure does not depend on the distance from the camera.

Measures between two participants are computed only on a camera that shows both of them.

A measure of one's own is a Python function in a file in the `scripts` folder of the project, named as `method: file.function`. The function receives two tables, one per participant, with one row per video frame, a column `t` for time, and the columns `<joint>_x` and `<joint>_y` for every keypoint, and returns one number per row:

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

When the file changes, the next run recomputes the measure. See {ref}`extending` before placing files in the `scripts` folder.

(gestures)=
## Gesture kinematics

When gestures have been coded on a tier for each participant and imported ({ref}`own-coding`), MACAW measures each coded gesture from the wrists: peak speed, the number of submovements (the speed peaks it consists of), holds between the first and the last submovement, the regularity of the submovements, the area the hand covers, how high it reaches relative to the body, the gesture space it uses, and in how many directions around the body's centre it extends. Each gesture yields one value per measure and hand.

```yaml
kinematics:
  gestures:
    tier: gesture
    still_speed: 15
    peak_prominence: 40
```

`still_speed` and `peak_prominence` have no default, because they are in pixels per second and therefore depend on the camera. A hand slower than `still_speed` counts as still, and a speed peak counts as a submovement when it rises at least `peak_prominence` above its surroundings. The wrist speeds in the results help to choose them. `peak_prominence` should lie well above the jitter of the tracking, otherwise every wobble counts as a submovement; check a few gestures against the video before relying on the counts. When two people coded the tier, `annotator` names whose gestures are measured. `holds: both_hands` counts a hold only while both hands are still, instead of for each hand separately.

Height and gesture space require the hips to be visible; when a table hides them, these two measures remain empty.

:::{important}
Gesture space is measured in the picture, so it follows the body only while the participant faces the camera or has their back to it. Seen from the side, the body appears narrow and the hands seem to reach far beyond it. Height is much less affected. With several calibrated cameras, the three-dimensional gesture space described below is free of this limitation.
:::

Tracking jitter adds slightly to every speed and area, most to small, slow movements. These values are therefore best compared between participants and conditions measured in the same way rather than read as absolute sizes.

With several calibrated cameras ({ref}`three-d`), the same measures are also computed in three dimensions once two further thresholds are set, in metres per second. As in the picture, there is no default, and the three-dimensional wrist speeds in the results help to choose them:

```yaml
kinematics:
  gestures:
    still_speed_3d: 0.05
    peak_prominence_3d: 0.1
```

In three dimensions, area, height and gesture space are taken in the participant's own frontal plane, defined by their hips and shoulders, so they hold whichever way the participant faces.

(scale)=
## Centimetres instead of pixels

Distances, speeds and areas from the video are measured in pixels. They can additionally be given in centimetres, subject to one condition.

:::{important}
A pixel scale is valid only at the distance from the camera at which it was measured. An object twice as far away appears half as large, so a scale measured at the parent's seat makes a child seated further back appear smaller than they are. Measure an object at each participant's position and state whose position it was; each participant then receives their own scale. A measure between two participants, such as the distance between them, is given in centimetres only when both share the same scale.
:::

There are three ways to obtain a scale:

Calibration clip filmed before the session
: With the camera in its final position, film an object of known length (a ruler or a sheet of paper) held at each participant's position. Place the clip in a `calibration` folder in the project and name it in the settings as `scale: {calibration: lab.mp4}`, so that every recording uses it. When the camera was moved for some sessions, film a new clip and name it in those recordings' own files. Moving or zooming the camera after the clip invalidates it, and a clip applies only to videos of the same size.

Measurement in the recording itself
: When an object of known length is visible in the recording, measure it there. A recording's own measurement takes precedence over a calibration clip.

A known value
: `scale: {cm_per_px: 0.11}` for everyone, or `scale: {cm_per_px_by_role: {child: 0.12, parent: 0.09}}` for each participant's position. A value written this way takes precedence over any measurement.

To measure, open *Pixels to centimetres* on the Review page, choose the recording's video or a calibration clip, show a frame, click the two ends of the object, enter its length, state whose position it was, and press **Save this measurement**. Measuring two or three times is recommended; the spread of the measurements shows how precise the clicks are.

The results remain in pixels. `macaw export` adds every pixel measure a second time in centimetres, with `_cm` appended to its name, and writes a file beside the table with each recording's scale for each participant.

The quality-control report gives each participant's nose-to-shoulder length in centimetres at their scale (a value far from a plausible one indicates that the participant sat nearer or further than the object), and flags a participant whose apparent size changes by more than 10% during the recording, which indicates movement toward or away from the camera. The check compares stretches of 30 seconds, which suits seated participants. When participants walk about, shorten the stretches to about 5 seconds; brief bending over is then flagged as well:

```yaml
scale:
  size_window_s: 5
```

(three-d)=
## Three-dimensional pose from several cameras

When two or more cameras film the same participants from different sides, MACAW places each participant's keypoints in the room, in metres. Distances then no longer depend on how far a participant sits from a camera, and the direction of a participant's head relative to the other participant can be measured.

Four conditions must be met:

- Every keypoint of interest is seen by at least two cameras from clearly different directions.
- The cameras record simultaneously. MACAW aligns them by their sound, as for any recording.
- The cameras are calibrated, that is, the lens of each camera and the positions of the cameras are known. This is obtained from a short clip of a printed board filmed before the session.
- The cameras are not moved or zoomed after that clip. When they are, the board is filmed again.

:::{note}
Placing participants in three dimensions has been validated on simulated cameras and on a public four-camera recording of an adult and a toddler, against that recording's own reconstruction. Finding the cameras from a board has been validated on simulated cameras only.
:::

The *Cameras* page of the workbench covers every step below. It appears once a recording has two or more videos, or once a camera setup exists in the `calibration` folder.

### 1. Print the board

*Draw the board* on the Cameras page (or `macaw calibration-board`) writes a ChArUco board, a chessboard with small markers in its white squares, as an image in the `calibration` folder. Print it at 100%, mount it flat on a rigid backing such as foam board, and measure one square with a ruler. Enter that length on the page and press **Save the square**. The default board fills an A2 sheet. For A3, set smaller squares and markers before drawing the board, then draw it with `macaw calibration-board --paper A3`:

```yaml
pose3d:
  board:
    square_mm: 50
    marker_mm: 37
```

### 2. Film the board before the session

With the cameras in their final positions, clap once in view of all cameras, then move the board slowly through the space where the participants will sit, tilting it toward and away from the cameras, for about a minute. Keep it in view of at least two cameras at a time and close enough to fill a good part of the picture. Place one clip per camera in a folder such as `calibration/lab`, each named like that camera in the recordings: `room.mp4` for the camera whose recordings are named `dyad03_room.mp4`.

### 3. Find the cameras

Select the setup on the Cameras page and press **Find the cameras** (or run `macaw calibrate-cameras lab`). This takes a few minutes and reports, for each camera, how far the board's corners fall from where the camera saw them (below 1 to 2 pixels is good), how often each pair of cameras saw the board together, and where the cameras stand, drawn in three dimensions. The page also indicates when a clip or the board size has changed since, so that the cameras need to be found again. Then press **Use for this project**. When some sessions used another setup, name it in those recordings' own files:

```yaml
pose3d:
  calibration: lab
```

A lab that has calibrated its cameras with Anipose can place its `calibration.toml` in the `calibration` folder and name it instead, as `calibration: lab.toml`.

### 4. Run and inspect

Run as usual. Each participant must carry the same name on every camera (see *Who is on camera* in {ref}`step 5 <review>`). The results gain measures in metres, with `_3d` appended to their names: wrist speed, acceleration and jerk, the distance between the hands, the distances between participants' shoulder midpoints and between their heads, the angle between the direction of a participant's head and the direction to the other participant's head (0 when the head points straight at them; this is head direction, not gaze), gesture kinematics when the three-dimensional thresholds are set ({ref}`gestures`), and custom body measures ({ref}`measures`). A custom measure defined by a function of one's own is computed in three dimensions only when `in_3d: true` is added to it; the function then also receives the columns `<joint>_z`, with every position in metres.

On the Results page, *In 3D* shows every participant's skeleton and the cameras; drag to rotate the view and move the slider to step through the recording. The quality-control report states, for each participant, how often they were placed in three dimensions and how well the cameras agree on them, and warns when a bone keeps changing length, which indicates misplaced keypoints.

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

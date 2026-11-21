# Further measures

These measures extend the standard set listed on the first page. Each is switched on separately. Settings given as a block can be pasted into the *project.yaml* box on the settings page and adapted to the study.

(gaze)=
## Gaze: who looks at whom

For each participant in each frame, MACAW estimates the point in the picture they look at, and from it whom they look at, which named object, somewhere else in the picture, or something outside it. The estimate is made from the participant's head and the whole scene, so it also works when a face is partly turned away. The first run downloads a model of about 1.3 GB.

:::{important}
Whom a participant looks at can be determined only when the person looked at is visible in the same picture, so this requires a camera that shows everyone involved. A camera that films one participant alone yields where in the picture they look, not at whom; for that case see *Positions in the room instead of the picture* below. The estimate reflects head and eye orientation as visible in the video and is not a substitute for an eye tracker: it distinguishes the other person's face from the table or the toys, but not one toy from the one beside it.
:::

Objects to be distinguished are named under *Things to follow* ({ref}`objects`); MACAW follows each as an outline, and a look whose point falls inside an object's outline is a look at that object.

For each pair of participants the results give, at every moment, whether one looks at the other and whether the two look at each other (mutual gaze), and for the whole recording the proportion of time spent looking at the other, the number of looks and the proportion of mutual gaze. A moment at which a participant looks outside the picture while the other is not in it is left open, since the look may be directed at that person, and counts toward neither proportion. A glance of a frame or two toward something else is merged into the preceding look, so that it does not count as a new look. Every look is also stored as a code, so that it can be exported to ELAN, compared with human gaze coding ({ref}`own-coding`) and summarised with `macaw describe-codes`.

When objects are named, or placed in the room (see *Positions in the room instead of the picture* below), the results also give joint attention: the moments in which two participants look at the same named object. For each pair they give the proportion of time spent in joint attention, the number of stretches of it, and how many of these stretches each participant led, that is, looked at the object first. For each stretch, `joint_attention_lag_s` gives how long after the first participant the second one joined, so that a study can apply its own window for following another's attention. A look that began before the participant's gaze could first be read counts from that moment. Each stretch is stored as a code on a tier of its own for each pair, labelled with the object. Two participants looking at the same third person is not counted as joint attention; it can be read from whom each looks at. The limitation above applies: two objects close together in the picture are not told apart.

For each named object the results also give, for each participant, whether they look at it in each frame, the proportion of time spent looking at it and the number of looks: for a ball, `looks_at_ball`, `looks_at_ball_share` and `looks_at_ball_n`, with the words of a longer name joined by underscores as for objects ({ref}`objects`). The model's estimate is a map of where a participant may be looking, from which the single most likely point is taken. The results keep more of the map: how sharply it places that point (`gaze_peak`, on the model's own scale), and how much of it lies on each other participant's head (`gaze_share_on_head`, from 0 to 1) and inside each followed object's outline (`gaze_share_on_ball`). These graded values allow a stricter or a more lenient criterion than the single point, which matters most when two heads or objects are close together in the picture.

### Checking the estimate on the study's own video

On the Review page, *Where people look* shows a frame with each participant's head marked. Indicate whom or what each participant looks at and press **Save what they look at**; the dice button in the card's corner moves the player to a moment at random. Frames should be drawn at random rather than chosen for clarity, otherwise the check overstates accuracy. About 20 frames per participant give a useful estimate. The quality-control report then gives, for each participant, how often MACAW agreed with these labels and lists the frames where it did not. Saving again for the same frame replaces one's own label; a label another person saved for that frame is kept, and where two people labelled the same frames the report also gives how often they named the same target.

### Positions in the room instead of the picture

When the person looked at is often outside the picture, for example a parent seated beside the camera that films the child, whom a participant looks at can be inferred from the direction of their head (and, with eye direction enabled, their eyes), provided their positions are known. Enter the room on the Setup page ({ref}`room`): the position and eye height of each participant, and the position, height, direction and tilt of the camera used for each participant's gaze. Things placed in the room with a height, such as a toy on the table, become targets too. Then set:

```yaml
gaze:
  backend: angles
  tolerance_deg: 20
face:
  gaze: true                 # use eye direction as well as head direction
```

A look within `tolerance_deg` of the direction to a person or thing counts as a look at it. There is no default, as the appropriate value depends on how far apart the participants sit. Each participant's directions are taken as seen from the camera that reads their face, from which their head and eye angles come, so participants filmed by cameras of their own are read as well. The method assumes that participants remain at their positions; a recording with a different arrangement changes the room in its own settings file. Head angles estimated from a single camera carry a bias of their own, so labelling frames as described above is recommended: MACAW then corrects each participant's offset from the labelled frames, and once a participant has three or more labelled frames looking at a target, it takes that target's direction from the frames rather than from the room. Labelled frames alone, without the room, are also sufficient.

With this method, the angle between where a participant looks and where the room puts the other participant is given as `gaze_off_deg`, and for each object placed in the room or named in labelled frames as, for a toy, `gaze_off_deg_toy`, beside `looks_at_toy` and its proportion and count.

(objects)=
## Things in the picture

Objects such as toys can be followed through the video, so that looks and hands can be related to them. They are named in plain English under *Things to follow* in the *Tools* group, for example `ball, book, toy car`, or in *project.yaml*:

```yaml
objects:
  names: [ball, book, toy car]
```

Each object is first found by its name, in the first ten seconds of video in which it appears, and from there followed frame by frame as an outline, back to the beginning of the analysed part of the recording and on to its end, which keeps a toy apart from the one beside it and holds it while a hand covers part of it. Both are done with SAM 3, whose model of about 3.4 GB is downloaded on the first run. Following needs a graphics card and takes longer than the video itself: on a fast card, about three minutes per minute of video for one object, and about another minute for each further object. An object is followed only on the cameras from which someone's body or gaze is read.

:::{important}
Finding an object by its name can pick the wrong thing, for example the parent when the name is *child*, or one of two balls. The outlines are therefore not used until a person has looked at where each object starts on each camera. Two objects that look alike need names that tell them apart, such as `red ball, blue ball`. An object that leaves the picture and comes back elsewhere may be lost or picked up on something else.
:::

### Checking where each object starts

Until a person has looked, the recording waits under *Waiting for a person*, and the steps that use the outlines do not run. On the Review page, *Things followed* lists each object on each camera: where it starts, whether it was found by name or started by a person, how much of the video it was in view, its longest time out of view, and what waits. A start time moves the player there, and the frame beside the controls shows the outlines as followed at the player's moment, with the box the name found drawn dashed on the starting frame. **Keep the starts found by name** keeps them as they are. To start an object elsewhere, click on it in the frame, pick its name and press **Start it here**; the next run follows it from that point. A second start at a later time puts an object back where it is after it was lost or picked up on something else. **Not on this camera** says that an object is not there, and **Forget** takes back what was said.

The same is done in a terminal: `macaw objects -r REC` lists the starts, `--keep` keeps those found by name, `--start NAME --camera CAM --at T --point X,Y` gives a start by a point on the object at time *T* (in seconds), and `--not-here NAME --camera CAM` says that an object is not on that camera. With `objects.review: flagged` only the starts in doubt wait: more than one candidate in the starting frame, the object in view in fewer than half of the frames after its start, or not found at all. With `skip` nothing waits, and the quality-control report lists the starts in doubt that were used without a look.

After the start, the review frames of the quality-control report show the outlines too, so that an object picked up on something else can be seen there. With a model looking at the review frames ({ref}`model-look`), it is also asked of each outline whether it is on the object named.

When an object is not found by its name, another word for it often works; *finder_score* (0.35 by default) can also be lowered so that weaker matches count, at the cost of more wrong ones. SAM 3 is given on request on Hugging Face under its own licence: accept its terms on the page of `facebook/sam3` there and log in with `hf auth login` on the computer. Where SAM 3 cannot be had, because access was not granted, the installed *transformers* is too old for it, or the computer is offline and has not downloaded it, objects are found with Grounding DINO and followed with SAM 2.1 instead, two open models of about 1 GB together; the run says so before it starts and in the quality-control report, the Methods text names the models used, and the next run once SAM 3 can be had follows the objects again with it. To use the open models on purpose, set `objects.finder: grounding_dino` and `objects.tracker: sam2`: SAM 2.1 follows about half again as fast and needs less graphics memory, while SAM 3 tends to tell a named object from similar ones better and counts all the candidates in a frame.

### Where objects are

Once the starts are confirmed, the results give, for each object on each camera it was followed on, whether it is in view in each frame, where the middle of its outline lies in the picture (in pixels from the left and top edges) and how much of the picture the outline covers (in square pixels): for a ball followed on the camera `front`, `ball_front_in_view`, `ball_front_x`, `ball_front_y` and `ball_front_area`. They belong to the recording, not to a participant, and are empty in frames where the object is out of view. For the whole recording and each segment, `ball_front_in_view_share` gives the share of the frames in which the object was in view and `ball_front_longest_gap_s` its longest time out of view, which bound how far measures of the object, such as looks at it or hands on it, can be trusted. `ball_front_score` gives the tracker's own score in each frame, from 0 to 1; the object counts as in view from 0.5, and the outlines scored below that are kept in the results as well. The place is the middle of the outline, so it is right only as long as the outline stays on the object, which the review frames show.

### Hands on objects

With objects followed, MACAW can also give, for each participant, how far each hand is from each object, frame by frame, and when a hand is on one:

```yaml
objects:
  names: [ball, book]
  touch_px: 40
```

`touch_px` is the distance in pixels within which a wrist puts the hand on an object; it has no default, since it depends on the size of the people in the picture. The pose model marks the wrist, not the fingers, so it is best set to about the length of a hand in the picture, checked on a few frames. The results give `left_wrist_to_ball` and `right_wrist_to_ball` (pixels to the outline, 0 inside it; exports add the project's unit beside them when a scale is set) and `left_hand_on_ball` and `right_hand_on_ball` (1 or 0) per frame, and each participant's stretches with a hand on an object as codes on a tier per object (`on_ball`, labelled left, right or both), which can be exported to ELAN. Words in an object's name are joined by underscores in these names, so *toy car* gives `left_hand_on_toy_car`.

:::{important}
The distance is read from the picture, so a hand in front of or behind an object counts as on it: the measure marks a hand over the object as the camera sees it, not contact.
:::

(motion)=
## Motion energy in a region of the picture

Motion energy measures the amount of movement within a rectangle of the picture, for each participant separately. Give each participant a region as `[x, y, width, height]` in pixels of the picture of the camera the setup uses for their body (their own camera when one is used for them alone, otherwise the first camera listed for their body), as read off a still frame in an image viewer:

```yaml
motion:
  rois:
    parent: [80, 120, 640, 900]
    child: [1100, 200, 600, 800]
```

When the camera was placed differently in some sessions, those recordings get their own regions in their recording files.

Motion energy grows with the time between frames, so values are comparable only between videos at the same frame rate; `video.fps` brings every video to one. `motion_energy` is averaged over `motion.smoothing_s`; `motion_energy_raw` gives the same share frame by frame before averaging, for analyses that apply their own filter. A change of the whole picture, such as a light switched on or a camera knocked, also changes each region and is not movement. For each participant's camera, `picture_change` gives the share of the whole picture that changed in each frame (above 0.5 such an event rather than movement), and `picture_frozen` marks with 1 each frame identical to the one before, in which a stalled recording shows no movement whatever the participant did. Frames marked by either are best left out before motion energy is analysed.

(counts)=
## Counts and shares per person

A few simple summaries per person complement the measures every run gives. Each is computed for the whole analysed part and for every named segment ({ref}`segments`). They often serve as rule-based counterparts of a rating scale.

### Questions and listed words

Each person's sentences ending in a question mark are counted per minute (`questions_per_minute`). Sentences are cut after a full stop, question mark or exclamation mark followed by a space, so the count rests on the punctuation of the transcript: the recogniser's own, or that of the person who corrected it.

Lists of words or phrases in the project folder, one entry per line, give the number of sentences per minute that hold at least one entry:

```yaml
text:
  word_lists:
    negative: negative_words.txt
```

This gives `negative_per_minute` for each person. An entry matches whole words regardless of case, a phrase matches its words in a row, and `#` starts a comment. No list is supplied with MACAW.

### Faces and action units

For each person, `face_found_share` is the share of the frames, on the camera their face is read from, in which a face was found and assigned to them. In each frame with a face, `face_score` gives how sure the face detector was that it is a face, on the scale of the face backend (MediaPipe gives none), and `face_size_px` the height of the face in pixels; action units read from small or doubtful faces are less reliable, so these two serve to leave such frames out or weigh them down. A low share means that the face was turned away, hidden or out of view; that person's action units and face angles rest on the frames that remain. With thresholds set,

```yaml
face:
  au_above: {au12: 0.5}
```

`au12_above_share` is the share of the frames with that person's face in which the action unit exceeds the threshold. Thresholds have no default, because the scale of the values depends on the face backend and model, as the description of each action unit in the results states.

### Movement while listening

`upper_body_speed` is the mean speed of the head, shoulder, elbow and wrist points found in a frame, in pixels per second. With

```yaml
kinematics:
  listening: [upper_body_speed]
```

each person's upper-body speed is averaged over the frames in which another participant is talking and nobody else is, judged from the start and end of each utterance in the transcript, or on a shared microphone from the separated voices, so that the listener's own laughter or murmurs count as talking (`upper_body_speed_listening`). The results also give how long this was the case (`listening_s`) and how many frames the average rests on; an average over few frames is easily moved by a single jump of the tracker. Motion energy (`motion_energy`, see {ref}`motion`) can be listed as well. Speech in a voice other than the participants', such as a researcher's, does not count as another participant talking. Speeds in pixels depend on how far each person sits from the camera and on the camera's resolution, so comparisons between people or recordings call for lengths in a common unit ({ref}`scale`), or for speeds relative to body size:

```yaml
kinematics:
  listening: [upper_body_speed, motion_energy]
  listening_per_body_size: true
```

divides each listed length or speed by the person's shoulder width, the 95th percentile of the distance between their shoulders over the recording on the camera their body is measured from, as `per_body_size` does for the project's own measures ({ref}`measures`); `upper_body_speed_listening` is then in shoulder widths per second. This holds where each person stays at about the same distance from the camera and faces it for part of the recording. A person whose two shoulders are never found in the same frame has no value. Motion energy and the head angles are not lengths and stay as they are; the run plan says which series were divided.

(synchrony)=
## Synchrony, convergence and response rates

MACAW leaves indices of synchrony and convergence to dedicated tools, since each involves choices (window, lag, normalisation) that belong to the study. It provides the series and the paired values these tools take.

### Movement synchrony with rMEA or SUSY

Motion energy ({ref}`motion`) is measured in every video frame, and therefore at a constant rate. Export the frames with `macaw export --scale frame`, then give each recording's two series to the R package rMEA:

```r
library(dplyr)
library(tidyr)
library(rMEA)

frames <- readr::read_csv("exports/features.csv", show_col_types = FALSE)
dir.create("mea", showWarnings = FALSE)
frames |>
  filter(!is.na(motion_energy)) |>
  select(recording_id, subject_id, t_start, motion_energy) |>
  pivot_wider(names_from = subject_id, values_from = motion_energy) |>
  arrange(recording_id, t_start) |>
  group_by(recording_id) |>
  group_walk(~ write.table(select(.x, parent, child), file.path("mea", paste0(.y$recording_id, ".txt")),
                           row.names = FALSE, col.names = FALSE))

mea <- readMEA("mea", s1Col = 1, s2Col = 2, sampRate = 25, s1Name = "parent", s2Name = "child", idOrder = NA)
mea <- MEAccf(mea, lagSec = 5, winSec = 60, incSec = 30)
summary(mea)
```

`sampRate` is the frame rate of the video, which is the inverse of the step between consecutive values of `t_start`; `parent` and `child` are the participants' identifiers. The R package SUSY takes one recording's two columns directly, with the segment length and the largest lag in seconds:

```r
library(SUSY)
one <- frames |>
  filter(recording_id == "001", !is.na(motion_energy)) |>
  select(subject_id, t_start, motion_energy) |>
  pivot_wider(names_from = subject_id, values_from = motion_energy) |>
  arrange(t_start)
susy(as.data.frame(select(one, parent, child)), segment = 30, Hz = 25, maxlag = 5)
```

Other series at the frame rate, such as wrist speed or head angles, can be passed to these tools in the same way.

### Prosodic convergence

At each change of speaker, the results set the voice measures of the turn just handed over beside those of the turn that took the floor: `prior_<measure>` for the previous speaker's turn and `next_<measure>` for the new one, on one row per change, where `subject_id` is the new speaker and `subject_id_b` the previous one. The measures are those computed for every turn: pitch (mean, median, spread, range and slope), the voiced fraction, mean intensity, jitter, shimmer, harmonics-to-noise ratio, cepstral peak prominence, and speech rate in words per second. Indices of entrainment, such as the difference between adjacent turns or its change over the session, are computed from these pairs. Intensity and the amplitude measures depend on the microphone and its distance from the speaker, so they are usually standardised within each speaker before being compared.

### Lexical overlap

For each turn taken directly after another participant's, `lexical_overlap` gives the proportion of the distinct words in the turn that the previous speaker used in the turn just before it, and `mean_lexical_overlap` gives its mean for each pair of speakers, in each direction. Words are compared as written, after lower-casing, so that *play* and *played* count as different words. Function words, fillers and backchannels are left out by default, as they are when MACAW counts who introduces the words a pair come to share. `text.exclude_words` sets this list for both measures: `english_stopwords` (the default), `none`, or the path of a word list with one word per line.

### Responses within a time window

The proportion of one participant's turns that the other answers within a few seconds, as in the conversational turn counts of LENA, depends on the window, which varies between studies (2, 3 or 5 s). MACAW therefore gives, for every change of speaker, the floor transfer offset `fto_s` (the gap before the new turn, negative when it overlaps the previous one), and the rate is computed from it. For a 5 s window:

```r
features <- readr::read_csv("exports/features.csv", show_col_types = FALSE)
turns <- features |>
  filter(!is.na(turn_duration_s)) |>
  count(recording_id, subject_id, name = "turns")
answered <- features |>
  filter(!is.na(fto_s), fto_s <= 5) |>
  count(recording_id, subject_id = subject_id_b, name = "answered_within_5s")
turns |>
  left_join(answered, by = c("recording_id", "subject_id")) |>
  mutate(answered_within_5s = coalesce(answered_within_5s, 0L), response_rate = answered_within_5s / turns)
```

The counts differ from those of LENA itself, which segments the audio in its own way.

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
: `distance`; `approach_speed`, the rate at which that distance shrinks (positive while approaching); `speed_toward`, how fast the participant's own point moves along the line to the other point (positive while moving toward it), which tells who closes the distance: the `speed_toward` of each participant toward the other add up to about the `approach_speed` between the same points; `offset_toward`, how far a point lies from the participant's own shoulder midpoint in the direction of the other participant, such as a head turned toward them; or a function of one's own (below).

`per_body_size`
: Divides by the participant's shoulder width, so that the measure is in shoulder widths and does not depend on the distance from the camera. In the picture the width is the 95th percentile of the distance between the shoulders over the recording, which holds where the participant stays at about the same distance and faces the camera for part of the recording; in the room it is the median. The width itself is in the results as `body_size` (in pixels) and `body_size_3d`, one value per participant for the whole recording, so that other lengths and speeds can be divided by it as well. A measure toward another participant is divided by the first participant's shoulder width on the camera that films both, so that both lengths come from one picture; where each participant also has a camera of their own, this width differs from `body_size`.

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

### Time below a threshold

A measure can also report how much of the time it stays below a value, for example how long two participants are within reach of each other. Add `below` to it:

```yaml
    - name: shoulders_apart
      person: parent
      joints: [shoulder_center]
      to_person: child
      to_joints: [shoulder_center]
      below: 250
```

The results then give the proportion of the frames in which the measure was below the value, out of the frames in which it could be measured, and the number of unbroken stretches below it. Each stretch is also stored as a code, so that it can be inspected in ELAN or compared with human coding. `below` is in the unit of the measure: pixels for a distance in the picture, and shoulder widths with `per_body_size`. There is no default, because the appropriate value depends on the camera and on the study. A threshold given in centimetres in the literature, such as 76 cm (30 inches), can be converted to pixels with the scale of the recording ({ref}`scale`), which holds only while the participants stay at the distance from the camera at which the scale was measured. With several calibrated cameras ({ref}`three-d`), `below_3d` sets the threshold for the measure in three dimensions in the project's unit of length, which needs no such condition.

(moving)=
## Stretches of hand movement to code gestures from

Finding the gestures in a recording takes a coder longer than judging them. With a speed set, MACAW marks every stretch in which a wrist moves faster than that speed and writes the stretches to a tier named `moving`, one per participant, labelled `left`, `right` or `both` after the hand that moved. Exported to ELAN ({ref}`own-coding`), the tier lets a coder step from stretch to stretch, copy those that are gestures to the gesture tier and correct their boundaries, instead of watching the whole recording.

```yaml
kinematics:
  moving:
    speed: 15
    min_s: 0.15
    join_s: 0.15
```

`speed` has no default, because it is in pixels per second and therefore depends on the camera; the wrist speeds in the results help to choose it. Stretches less than `join_s` apart count as one, and stretches shorter than `min_s` are left out; those go to a tier of their own, `moving_short`, so that what the setting removed can be seen. The same decision is exported frame by frame as `left_hand_moving` and `right_hand_moving` (1 while the hand moves). With several calibrated cameras or another source of 3D points ({ref}`three-d`), `speed_3d` sets the same threshold for the wrists placed in 3D, in the project's unit of length per second; those stretches go to a tier of their own, `moving_3d`, and frame by frame to `left_hand_moving_3d` and `right_hand_moving_3d`. A speed in the room does not depend on the camera or on how far from it the participant sits, so one value serves every recording.

:::{important}
The tier says that a hand moved, not that it gestured: reaching, fidgeting and handling objects are marked as well, and which stretches are gestures remains the coder's decision. A coder who looks only at the marked stretches does not see a gesture slower than `speed`, so the speed is best set low, at the cost of more stretches to skip, and checked against a passage coded without the tier.
:::

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

The moments behind the counts are kept as codes: each submovement peak on the tier `gesture_submovement`, one frame long and labelled with the hand, and each hold on `gesture_hold`, labelled with the hand, or *both* with `holds: both_hands`. Exported to ELAN, they show where the thresholds placed peaks and holds in each gesture, and they can be related in time to speech.

Height and gesture space require the hips to be visible; when a table hides them, these two measures remain empty.

The height level, gesture space and direction of each wrist are also given in every frame for every participant, whether or not gestures were coded, coded as above: `left_wrist_height_level` (0 to 5), `left_wrist_gesture_space` (1 to 4) and `left_wrist_gesture_sector` (0 within the centre space, otherwise 1 to 8, from up and to the left in the picture, row by row, to down and to the right), and the same for the right wrist. With points in three dimensions they are given in the participant's own frontal plane as well, with `_3d` appended, seen from in front of the participant.

:::{important}
Gesture space is measured in the picture, so it follows the body only while the participant faces the camera or has their back to it. Seen from the side, the body appears narrow and the hands seem to reach far beyond it. Height is much less affected. With several calibrated cameras, the three-dimensional gesture space described below is free of this limitation.
:::

Tracking jitter adds slightly to every speed and area, most to small, slow movements. These values are therefore best compared between participants and conditions measured in the same way rather than read as absolute sizes.

With several calibrated cameras ({ref}`three-d`), the same measures are also computed in three dimensions once two further thresholds are set, in the project's unit of length per second. As in the picture, there is no default, and the three-dimensional wrist speeds in the exported table help to choose them (here in centimetres per second):

```yaml
kinematics:
  gestures:
    still_speed_3d: 5
    peak_prominence_3d: 10
```

In three dimensions, area, height and gesture space are taken in the participant's own frontal plane, defined by their hips and shoulders, so they hold whichever way the participant faces.

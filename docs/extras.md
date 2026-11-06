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

### Refining or drafting a codebook

The model set under *Coding* can also propose a codebook, which the researcher reads and edits before any coding uses it. Neither command below changes the codebook in use or any coding.

`macaw codebook refine codebook_refined.yaml`, or **Suggest a clearer codebook** on the Agreement page, takes the units in which a person's codes and the model's differ, together with some on which they agree, and asks the model to rewrite the description, criteria and borderline cases of each code concerned so that a coder reading only the codebook would code those units as the person did. The model also picks examples among these units, which are added in the participants' own words. When several people coded the tier, `--coder` names the person whose codes are followed. When nobody has coded it, the units on which the model's repeated answers were split are used instead. A code without such units is left unchanged. The proposed codebook keeps the name of the one in use and is written to a new file, beside `codebook_refined.changes.md`, which lists for each code what changed and the units that prompted it.

Whether the revision helps is tested by coding with it as a version, `coding.judges: {refined: {codebook: codebook_refined.yaml}}`, and comparing both versions with the person's coding. Agreement on the units the revision was made from overstates its benefit, so the comparison is made on other recordings: refine with `-r` on some recordings (on the Agreement page, those selected under *Recordings*) and compare on the rest.

`macaw codebook draft first.yaml --aim "how parents guide their child's play"` drafts a codebook when none exists. The model describes in a few words what each of 200 units drawn at random does, with the stated aim in mind (`--units` changes the number), and then groups these descriptions into at most twelve codes, each with a description, criteria, borderline cases and examples. A model that proposes more is asked to merge them; if it still gives more than twelve, the twelve that hold the most units are kept. `first.draft.md` lists the descriptions each code gathers and those that fit no code, and says when codes were merged or left out. The draft is a starting point: selecting, merging and defining the final codes remains the researcher's work.

:::{important}
Examples in a proposed or drafted codebook are participants' own words from the transcripts. Check them before the codebook is shared or published.
:::

(own-coding)=
## Human coding and inter-rater agreement

Coding done by people, in ELAN or another program, can be imported beside the machine codes, compared with other coders, with the language model or with the automatic gaze codes, and summarised.

### Importing

On the Review page, enter a name under *Your name*, select the recording and press **Import your coding (.eaf, .csv, .xlsx)** on the *Your coding* card. Many files can be imported at once from the command line, each file name starting with its recording (for example `dyad01_anna.eaf`):

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

Within one phase of the recordings
: `macaw agreement --tier speech_acts --segment free_play` compares only the coding inside each recording's segment of that name ({ref}`segments`), cut at the segment's edges as the segment summaries are. It combines with `--units` and `--grid-s`. Recordings without such a segment are left out and listed.

The report gives Krippendorff's alpha across all coders, Cohen's kappa for each pair, and the agreement on each code for each pair, so that with two human coders and a language model, the model's agreement with each person stands beside the agreement between the people. `--out agreement` additionally writes a folder with a table of every unit, each coder's code and the words spoken, which shows which codes are confused with which and helps when revising a codebook.

The Agreement page of the workbench runs the same comparison. Choose the tier, how units are formed (*One coder's units*, *A time grid* or *Shared intervals*) and, if needed, a segment, then press **Compare**. The page shows the report's tables and writes the folder named under *Save to folder*. The *Your coding* card on the Review page opens it with that tier and coder selected.

:::{note}
A coder who coded only part of a recording leaves the rest uncoded, which counts as disagreement. Mark the part all coders coded as a segment and compare within it with `--segment`.
:::

### Summary of code use

`macaw describe-codes` lists for each recording, coder and participant how many units and how much time each code took, in how many runs of the same code, and how many of those runs last a single unit. Many single-unit runs indicate that the coding alternates from unit to unit, which deserves a closer look before analysis. `--out codes.csv` saves the table.

(segments)=
## Several parts of one recording

When a recording contains several phases, for example free play followed by clean-up, each phase can be named as a segment. Every measure that is a single number per recording, such as each participant's share of the talk, the number of turns or the median floor transfer offset, is then also computed within each segment. The recording is still processed as a whole: transcription, the separation of voices and the tracking of participants use the full recording, and decisions such as which voice belongs to whom are made once and hold in every segment.

### Marking segments

On the Segments page, select the recording and press **Make a copy to play here**. While it plays, press **Start here** at the beginning of a phase and **End here** at its end, type a name and press **Save as a segment**. Segments can also be written directly into the file `segments.csv` next to `project.yaml`, one row per segment, which is convenient when the times come from session logs:

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

The analysed part, set by *Task starts at* and *Task ends at* or with **Analyse only this part** on the Segments page, restricts what every step reads, and changing it processes the recording again from the start. Segments only add summaries within the analysed part. A segment that reaches outside it is left out, and the quality report says so.

### Results

`macaw export` adds a `segment` column, which is empty for values of the whole recording and holds the segment's name for values within a segment. With `macaw export --scale recording`, each participant has one row for the whole recording and one for each segment. The quality report lists the segments, the warnings within each and their measures in a section of its own, the Results page shows a measure's value within each segment beside the whole recording, `macaw methods` describes how the segments were summarised, and `macaw agreement --segment` compares coders within a segment.

(gaze)=
## Gaze: who looks at whom

For each participant in each frame, MACAW estimates the point in the picture they look at, and from it whom they look at, which named object, somewhere else in the picture, or something outside it. The estimate is made from the participant's head and the whole scene, so it also works when a face is partly turned away. The first run downloads a model of about 1.3 GB.

:::{important}
Whom a participant looks at can be determined only when the person looked at is visible in the same picture, so this requires a camera that shows everyone involved. A camera that films one participant alone yields where in the picture they look, not at whom; for that case see *Positions in the room instead of the picture* below. The estimate reflects head and eye orientation as visible in the video and is not a substitute for an eye tracker: it distinguishes the other person's face from the table or the toys, but not one toy from the one beside it.
:::

Objects to be distinguished are named in plain English under *Things people look at* in the *Tools* group, for example `bottle, book, toy car`. The quality-control report warns when a name is rarely found; another word for the same object often works better. When objects are detected that are not there, raise the detection threshold (`gaze.object_score`, 0.35 by default); when objects that are there are missed, lower it.

For each pair of participants the results give, at every moment, whether one looks at the other and whether the two look at each other (mutual gaze), and for the whole recording the proportion of time spent looking at the other, the number of looks and the proportion of mutual gaze. A glance of a frame or two toward something else is merged into the preceding look, so that it does not count as a new look. Every look is also stored as a code, so that it can be exported to ELAN, compared with human gaze coding ({ref}`own-coding`) and summarised with `macaw describe-codes`.

When objects are named, or placed in the room (see *Positions in the room instead of the picture* below), the results also give joint attention: the moments in which two participants look at the same named object. For each pair they give the proportion of time spent in joint attention, the number of stretches of it, and how many of these stretches each participant led, that is, looked at the object first. For each stretch, `joint_attention_lag_s` gives how long after the first participant the second one joined, so that a study can apply its own window for following another's attention. A look that began before the participant's gaze could first be read counts from that moment. Each stretch is stored as a code on a tier of its own for each pair, labelled with the object. Two participants looking at the same third person is not counted as joint attention; it can be read from whom each looks at. The limitation above applies: two objects close together in the picture are not told apart.

### Checking the estimate on the study's own video

On the Review page, *Where people look* shows a frame with each participant's head marked. Indicate whom or what each participant looks at and press **Save what they look at**; **A frame at random** selects the next frame. Frames should be drawn at random rather than chosen for clarity, otherwise the check overstates accuracy. About 20 frames per participant give a useful estimate. The quality-control report then gives, for each participant, how often MACAW agreed with these labels and lists the frames where it did not.

### Positions in the room instead of the picture

When the person looked at is often outside the picture, for example a parent seated beside the camera that films the child, whom a participant looks at can be inferred from the direction of their head (and, with eye direction enabled, their eyes), provided their positions are known. Enter the room on the Setup page ({ref}`room`): the position and eye height of each participant, and the position, height, direction and tilt of the camera used for each participant's gaze. Things placed in the room with a height, such as a toy on the table, become targets too. Then set:

```yaml
gaze:
  backend: angles
  tolerance_deg: 20
face:
  gaze: true                 # use eye direction as well as head direction
```

A look within `tolerance_deg` of the direction to a person or thing counts as a look at it. There is no default, as the appropriate value depends on how far apart the participants sit. Each participant's directions are taken as seen from the camera used for their gaze, so participants filmed by cameras of their own are read as well. The method assumes that participants remain at their positions; a recording with a different arrangement changes the room in its own settings file. Head angles estimated from a single camera carry a bias of their own, so labelling frames as described above is recommended: MACAW then corrects each participant's offset from the labelled frames, and once a participant has three or more labelled frames looking at a target, it takes that target's direction from the frames rather than from the room. Labelled frames alone, without the room, are also sufficient.

(voice-hints)=
## Assigning voices on a shared microphone

When participants share a microphone and MACAW is set to propose whose voice is which (*Voices on a shared microphone* set to wait only when uncertain, see {ref}`step 6 <review>`), the proposal can draw on what each voice says and on how high it is. MACAW assumes nothing about the participants; the rules are written by the researcher:

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

Motion energy measures the amount of movement within a rectangle of the picture, for each participant separately. Give each participant a region as `[x, y, width, height]` in pixels of the picture of the camera the setup uses for their body (their own camera when one is used for them alone, otherwise the first camera listed for their body), as read off a still frame in an image viewer:

```yaml
motion:
  rois:
    parent: [80, 120, 640, 900]
    child: [1100, 200, 600, 800]
```

When the camera was placed differently in some sessions, those recordings get their own regions in their recording files.

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

(suggest-names)=
## Suggesting who is who on a camera

Instead of clicking every camera ({ref}`step 3 <naming>`), the researcher can let MACAW suggest who is who on the cameras nobody clicked, with *Suggest who is who where nobody clicked* under *Checks by a person* in the settings:

```yaml
pose:
  suggest_names: true
```

A suggestion is a decision made by software. A suggestion that is not clear waits for the researcher at *Who is on camera* ({ref}`step 6 <review>`), exactly as a camera nobody named, and the quality-control report and the Methods text state which participants were suggested and on what basis.

On a camera used for one participant alone
: When someone else comes into view, the person most in view and largest in the picture is taken as that participant. The camera waits when another person is in view at least half as long, or when the person chosen is in fewer than half of the frames in which anyone is in view.

On a camera used for several participants
: Each person's head motion is compared with each participant's head motion on another camera on which every participant was named; failing that, with the speech on each participant's own microphone. A match with another camera names the participant, and waits for the researcher when it is weak. A match with a microphone names the participant only when it is clear.

From a description of each participant
: Where neither another camera nor individual microphones are available, as with a single camera that films everyone, the suggestion is made by a model that reads pictures, the one set up for checking the review frames ({ref}`model-look`), from a written description of what each participant looks like:

```yaml
qc:
  look:
    model: qwen3-vl:8b-instruct
    people: {parent: an adult, child: a young child}
```

Each person in view is cut out of up to five frames, spread over the time they are in view, and the model is asked which of the described participants it is. A participant is named only when the model gives the same participant on more than half of those frames and gives no other; a participant given for two people is given to neither, and every participant not named clearly waits to be clicked. The description should state what tells the participants apart, such as age or clothing; participants of similar appearance cannot be told apart in this way. The model's answers are kept, so that running again asks only about pictures it has not seen. With a cloud provider, the pictures of the participants are sent to that provider, which is stated before the run starts.

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

With several calibrated cameras ({ref}`three-d`), the same measures are also computed in three dimensions once two further thresholds are set, in the project's unit of length per second. As in the picture, there is no default, and the three-dimensional wrist speeds in the exported table help to choose them (here in centimetres per second):

```yaml
kinematics:
  gestures:
    still_speed_3d: 5
    peak_prominence_3d: 10
```

In three dimensions, area, height and gesture space are taken in the participant's own frontal plane, defined by their hips and shoulders, so they hold whichever way the participant faces.

(scale)=
## Lengths instead of pixels

Distances, speeds and areas from the video are measured in pixels. They can additionally be given in the project's unit of length (centimetres, metres or inches, chosen as *Length unit* in the settings), subject to one condition.

:::{important}
A pixel scale is valid only at the distance from the camera at which it was measured. An object twice as far away appears half as large, so a scale measured at the parent's seat makes a child seated further back appear smaller than they are. Measure an object at each participant's position and state whose position it was; each participant then receives their own scale. A measure between two participants, such as the distance between them, is converted only when both share the same scale. When the room is laid out on the Setup page, its notes give each participant's distance from the camera ({ref}`room`).
:::

There are three ways to obtain a scale:

Calibration clip filmed before the session
: With the camera in its final position, film an object of known length (a ruler or a sheet of paper) held at each participant's position. Place the clip in a `calibration` folder in the project and name it in the settings as `scale: {calibration: lab.mp4}`, so that every recording uses it. When the camera was moved for some sessions, film a new clip and name it in those recordings' own files. Moving or zooming the camera after the clip invalidates it, and a clip applies only to videos of the same size.

Measurement in the recording itself
: When an object of known length is visible in the recording, measure it there. A recording's own measurement takes precedence over a calibration clip.

A known value
: `scale: {per_px: 0.11}` for everyone, or `scale: {per_px_by_role: {child: 0.12, parent: 0.09}}` for each participant's position, in the project's unit per pixel. A value written this way takes precedence over any measurement.

To measure, open *Pixels to centimetres* (or to the unit chosen) on the Review page, choose the recording's video or a calibration clip, show a frame, click the two ends of the object, enter its length, state whose position it was, and press **Save this measurement**. Measuring two or three times is recommended; the spread of the measurements shows how precise the clicks are.

The results remain in pixels. `macaw export` adds every pixel measure a second time in the project's unit, with `_cm`, `_m` or `_in` appended to its name, and writes a file beside the table with each recording's scale for each participant.

The quality-control report gives each participant's nose-to-shoulder length in the project's unit at their scale (a value far from a plausible one indicates that the participant sat nearer or further than the object), and flags a participant whose apparent size changes by more than 10% during the recording, which indicates movement toward or away from the camera. The check compares stretches of 30 seconds, which suits seated participants. When participants walk about, shorten the stretches to about 5 seconds; brief bending over is then flagged as well:

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

The section *Cameras for three dimensions* at the end of the Setup page covers every step below. It appears once the setup has two or more cameras, or once a calibration exists in the `calibration` folder.

### 1. Print the board

*Draw the board* in that section (or `macaw calibration-board`) writes a ChArUco board, a chessboard with small markers in its white squares, as an image in the `calibration` folder. The board sits in the lower left corner of the page, and an arrow in the margin starts at its marked corner, 15 mm from the page's left and bottom edges, and runs along its lower edge. Print it at 100%, mount it flat on a rigid backing such as foam board with the page's edges along the backing's, and measure one square with a ruler. Enter that length, in the project's unit, and press **Save the square**. The default board fills an A2 sheet. For A3, set smaller squares before drawing the board (here in centimetres; the markers keep their share of the square), then draw it with `macaw calibration-board --paper A3`:

```yaml
pose3d:
  board:
    square: 5
```

### 2. Film the board before the session

With the cameras in their final positions, clap once in view of all cameras, then move the board slowly through the space where the participants will sit, tilting it toward and away from the cameras, for about a minute. Keep it in view of at least two cameras at a time and close enough to fill a good part of the picture. Place each camera's clip in a folder named like that camera's device folder in the recordings, inside a folder for the calibration such as `calibration/lab`: the clip of the camera whose recordings sit in `front` goes in `calibration/lab/front`.

At the end of the clips, lay the board flat and still, face up, on the floor or on the table for about three seconds, in view of at least one camera. This shows which way is up and where the floor is, so that the cameras can later be placed in the room from the calibration (step 4). Laid at a known point, for example on the floor in the room's zero corner with the page's edges against the two walls and the arrow along the wall of the x axis, it also shows where the cameras stand and which way they face, and then no camera needs to be measured.

### 3. Find the cameras

Select the calibration in that section and press **Find the cameras** (or run `macaw calibrate-cameras lab`). This takes a few minutes and reports, for each camera, how far the board's corners fall from where the camera saw them (below 1 to 2 pixels is good), how often each pair of cameras saw the board together, and where the cameras stand, in the project's unit and drawn in three dimensions. The page also indicates when a clip or the board size has changed since, so that the cameras need to be found again. Then press **Use for this project**. When some sessions used another calibration, name it in those recordings' own files:

```yaml
pose3d:
  calibration: lab
```

A lab that has calibrated its cameras with Anipose can place its `calibration.toml` in the `calibration` folder and name it instead, as `calibration: lab.toml`.

### 4. Place the cameras in the room

This step is optional. The cameras' positions and directions in the room ({ref}`step 3 <setup>`) can be taken from the calibration instead of being measured. In *Board lying flat*, drag to a moment when the board lies flat, check that at least one camera finds it, choose what it lay on (the floor, or a thing in the room whose height is given) and press **Save the moment**, or run `macaw calibration-floor lab --at 63.5 --on table`. When it lay at a known point, also enter where its marked corner lay, across and along in the room, and where its arrow pointed, in degrees clockwise from the y axis as a camera's direction (90 is along the x axis). With the page in the zero corner as above, the corner lies 1.5 cm from each wall: `macaw calibration-floor lab --at 63.5 --corner 1.5 1.5 --toward 90`. Then press **From the calibration** above the setup's editor, or run `macaw place-cameras`.

The calibration gives where the cameras stand relative to each other and the angle each picture spans. What ties them to the room is taken from everything the room already gives: the board lying flat and where it lay, and any camera positions, heights, directions and tilts entered. With the board lying flat at a known point, nothing else is needed; lying flat elsewhere, the positions of two cameras suffice; without it, the positions and heights of three cameras, or the position, height, direction and tilt of one camera. A tape measure is therefore enough, and no angle needs to be measured. When the given values do not suffice, only the picture angles are filled in and the page states what is missing. When they exceed what is needed, a table compares each given value with where the calibration places it and marks those more than 10 cm or 5° off. A single wrong value shifts the others as well, so without the board lying flat the wrong one may be any of those marked. A camera turned about its line of sight by more than 5° is reported, since the room draws every camera level. Cameras placed in this way are marked as taken from the calibration until one of their values is changed by hand.

### 5. Run and inspect

Run as usual. Each participant must be named on every camera that films them, by clicking them on each camera ({ref}`step 3 <naming>`), or by clicking them on one camera and letting MACAW suggest them on the others from their head motion ({ref}`suggest-names`). The results gain measures in three dimensions, with `_3d` appended to their names. They are stored in metres; the Results page shows them in the project's unit, and exports give them in that unit as well ({ref}`scale`): wrist speed, acceleration and jerk, the distance between the hands, the distances between participants' shoulder midpoints and between their heads, the angle between the direction of a participant's head and the direction to the other participant's head (0 when the head points straight at them; this is head direction, not gaze), the angle between the direction the chest faces and the direction to the other participant (`body_toward_deg_3d`: 0 when facing them, 180 with the back to them; it requires the hips to be placed, and is not given in the picture alone, where the depth of the shoulders is unknown), gesture kinematics when the three-dimensional thresholds are set ({ref}`gestures`), and custom body measures ({ref}`measures`). A custom measure defined by a function of one's own is computed in three dimensions only when `in_3d: true` is added to it; the function then also receives the columns `<joint>_z`, with every position in metres.

On the Results page, *In 3D* shows every participant's skeleton and the cameras; drag to rotate the view and move the slider to step through the recording. The quality-control report states, for each participant, how often they were placed in three dimensions and how well the cameras agree on them, and warns when a bone keeps changing length, which indicates misplaced keypoints.

(depth)=
## Three-dimensional pose from a depth camera

A depth camera, such as an Orbbec Femto, an Azure Kinect or a RealSense, records alongside its colour video how far each of its pixels is. With one, MACAW places each participant's keypoints in three dimensions from that single camera, in metres in the colour camera's frame, and the measures of the previous section follow from them.

:::{note}
Placing participants from a depth camera has been checked on simulated depth images. Importing a RealSense recording has been checked on the maker's sample recording; importing an Azure Kinect or Orbbec recording has not yet been run on a real recording.
:::

In the setup, mark the camera as recording depth:

```yaml
cameras:
  kinect:
    body: [parent, child]
    depth: true
```

Its device folder in each recording then holds, beside the colour video, a folder `depth` with one 16-bit PNG per video frame, in frame order, whose values are distances along the depth camera's axis, and a file `depth.yaml` that describes both cameras:

```yaml
depth_size: [512, 424]                  # the depth images' width and height, in pixels
depth_matrix: [[365.1, 0, 257.8], [0, 365.1, 207.6], [0, 0, 1]]
depth_distortions: [0.09, -0.27, 0, 0, 0.09]   # k1 k2 p1 p2 k3
rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]    # the depth camera's axes in the colour camera's frame
translation: [-0.052, 0, 0]             # where the depth camera sits in the colour camera's frame, in metres
colour_size: [1920, 1080]
colour_matrix: [[1060.0, 0, 954.2], [0, 1060.0, 548.8], [0, 0, 1]]
colour_distortions: [0.02, -0.02, 0, 0, 0]
unit_mm: 1                              # a stored value of 1 is this many millimetres
```

A camera's own recording is turned into this layout by `macaw depth-import`, which leaves the alignment of depth and colour to the camera maker's library: `macaw depth-import session.bag raw/dyad01/realsense` for a RealSense recording, with `pip install pyrealsense2`, and `macaw depth-import session.mkv raw/dyad01/kinect` for an Azure Kinect or Orbbec Femto recording, with the maker's SDK installed and then `pip install pyk4a`. It writes the colour video and the depth images aligned with it, at the recording's frame rate; a frame missing from the recording repeats the picture before it with no depth. Laid out by hand, the figures of `depth.yaml` come from the camera's own calibration, which its recording software saves. When the depth images are already aligned with the colour picture, the two lenses are the same and the rotation and translation are left at none. The depth images must follow the colour video's frames one for one, at a constant frame rate. A camera that records depth cannot be combined with `pose3d.calibration` in the same project.

The step `prep.depth` stores the depth images once on the prepared video's frames, and `pose.depth` places each keypoint the colour video finds: the depth image is carried into the colour camera's view, and the keypoint is placed on the nearest surface within a few pixels of it, along the camera's line of sight through it. Two conditions follow from this. The points lie on the surface of the body facing the camera, a few centimetres in front of the joint centres, which leaves distances between participants and movement much as they are but shortens a limb seen end on. A keypoint hidden behind something nearer the camera, such as a table edge or a raised hand, is placed on that object. The quality-control report states, for each participant, the share of keypoints that found a depth, and warns when a bone keeps changing length, which points to such frames.

(model-look)=
## A model checks the review frames

The frames drawn for review ({ref}`step 7 <results>`) can also be shown to a model that reads pictures, which reports where the people drawn on a frame disagree with what it sees. A person then checks the model, both on the frames it flagged and on a few frames it passed, drawn at random.

### Setting up

The model runs on a server like the one used for language-model coding, preferably on the lab's own computer (see *A model on the lab's own computer* in {ref}`coding`), with a model that reads pictures and answers directly, such as `qwen3-vl:8b-instruct`. Name it in the settings under *A model looks at the review frames*:

```yaml
qc:
  look:
    model: qwen3-vl:8b-instruct
    people: {parent: an adult, child: a young child}
```

`people` states, for each role, what that participant looks like. It is optional. Only the participants described there are checked for carrying the right name, because a name such as `p1` gives the model nothing to go on. Any visible description serves, for example `the mother, in a red top`.

### Running

On the Review page, under *What a model saw on the review frames*, press **Ask the model about the review frames**, or run

```bash
macaw qc --look
```

This draws the review frames and asks the model, for each frame, how many people are in view and, for each person drawn, shown on their own with only their skeleton, whether it is a real person and, for the participants described, which of them it is. A frame is flagged when

- someone in view has no skeleton,
- someone is drawn twice,
- a skeleton lies on something that is not a person, such as a toy or a picture on the wall,
- the name of a described participant is on someone else.

A frame already answered is asked again only when the frame, the people drawn on it, the model or the descriptions change.

### Checking the model

The card lists each flagged frame with what the model said, followed by frames it passed, drawn at random (*Passed frames checked at random*, five per recording by default). For each, press **Right** when the model was right about the frame, that is, the problem it names is there or a frame it passed is in order, and **Wrong** otherwise. From the command line: `macaw check-look -r dyad01 --frame <file> --verdict right`. The quality-control report lists the model's findings beside these verdicts, and the Methods text states what the model was asked and how often a person agreed with it. The frames drawn at random are what estimate how many errors the model passes over.

:::{note}
Whether a skeleton follows the arms and legs of the person it is on is not asked, since models of this size do not judge it reliably. Limbs that stretch from frame to frame are flagged by the automatic checks instead.
:::

:::{important}
With a cloud provider such as `openai` or `anthropic`, the review frames, which show the participants, are sent to that provider; the command says so before it starts. Whether this is permitted is for the ethics approval to decide.
:::

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

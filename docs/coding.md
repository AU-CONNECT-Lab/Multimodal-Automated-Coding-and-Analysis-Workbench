# Coding and rating

A language model can code speech unit by unit and rate whole recordings on scales. Codes and ratings made by people can be imported and compared with the model's or with each other. Settings given as a block can be pasted into the *project.yaml* box on the settings page and adapted to the study.

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

(llm-concurrency)=
### Several requests at a time

The requests of one recording are sent one after another by default. A server that can answer several requests together finishes a recording considerably sooner when they are sent together. `coding.llm.concurrency`, in the text part of the settings, gives the number of requests kept under way at a time within one recording; a version under `coding.judges` or `coding.ratings` may set its own. The transcript check (`macaw transcript check`) and the drafting and refining of a codebook (`macaw codebook`) send their requests to the same server and use the same number. The model that reads pictures has a server of its own, and `qc.look.concurrency` sets the number for it, both for the review frames (`macaw qc --look`) and for naming people from pictures.

```yaml
coding:
  llm: {provider: ollama, model: "qwen3:14b", concurrency: 4}
```

The codes and scores are stored in the same order whatever the value, each answer keeps its own seed, and changing the value does not code or rate anything again. The first request of each recording is always sent alone, so that a context too short for the prompt, or an answer cut off before it began, stops the step after a single request. Three points determine a sensible value:

- *Recordings at once* in the *Steps* group (`macaw run -j`) runs several recordings side by side, and each sends its own requests, so the server receives up to that number multiplied by `concurrency`. The run plan states the product.
- The server must be set up to answer that many. Ollama answers as many requests at a time as its environment variable `OLLAMA_NUM_PARALLEL` allows, one by default in current versions; it should be set to at least the product, since further requests only wait in the server's queue. Each request answered in parallel is given a context of its own, so the memory the model needs grows with `OLLAMA_NUM_PARALLEL` multiplied by `OLLAMA_CONTEXT_LENGTH`, and a model that no longer fits in graphics memory runs many times more slowly. vLLM answers requests together without further settings. The llama.cpp server is started with `--parallel` at least that high, and its context (`-c`) can then be shared among the requests answered together, so it needs room for all of them. With a cloud service (`openai`, `anthropic`), the provider's rate limits apply: a request beyond them is refused, waits as long as the provider asks (up to two minutes) and is repeated up to three times before the step stops, so the value should stay within what the account allows.
- A server that answers several requests together may compute them as one batch, in which case the same seed no longer guarantees the same answer word for word. A coding or rating that must be reproducible to the letter is run with `concurrency: 1`. A value above 1 is kept with the settings in the data dictionary and noted in the quality report.

### How long a request may take

`coding.llm.timeout_s` is the time allowed for one request, from sending it to receiving its whole answer; the default is 1800 seconds. The time a request waits in the server's queue counts towards it, so a model that reasons before it answers, several requests at once and a server shared with other work each call for a larger value. A request that runs out of time is sent once more, and the step then stops with a message that names the setting. Changing the value does not rate or code anything again.

A coding or rating writes each answer down as it arrives. When the step stops part way, through a timeout or an interrupted run, its next run asks only for the answers still missing; an answer is reused only for the same text and frames sent to the same model with the same temperature, token limit, reasoning switch, sample and seed, and the quality report notes how many were reused. The answers kept in this way are removed once the step has finished, and by `macaw clean`.

For each answer, `answers.csv` ({ref}`the export of ratings <rating-export>`) gives the tokens of the text sent, the tokens generated, those of them spent on reasoning, and the seconds the request took. These show how long a rating of all recordings will take, and whether answers stop at `coding.llm.max_tokens`. Where a server returns the reasoning without counting its tokens, as Ollama does, the count is estimated from the length of the reasoning relative to the answer.

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

(notes-by)=
Where the recordings come from several tasks, a note can be given for each task, or for each value of another part of the recording name (group, session, run), in the text part of the settings. In the prompt it follows the *Note to the model*:

```yaml
coding:
  llm:
    notes: "A is the parent; B is the child."
    notes_by:
      task: {task_a: "The parent and the child plan a weekend outing together.", task_b: ""}
```

Every value the recordings take needs an entry, with `""` where no note is wanted; a run stops before it starts when one is missing and names the recordings concerned. A version under `coding.judges` or `coding.ratings` that sets its own `notes_by` uses that table instead (`notes_by: {}` gives none), and a recording's own file in the `recordings` folder can still change either note for that recording. Changing the note of one task codes or rates only that task's recordings again. The run plan, the Methods text and the data dictionary give the note of each value. `macaw codebook refine` works across recordings and is given the *Note to the model* only.

### Reliability and review

Each unit is coded several times (*Answers per unit*, five by default), and the share of identical answers is the unit's confidence. The model may also answer "cannot tell". Units below *Review below this agreement* (0.7 by default) and every "cannot tell" are placed on the Review page under *Speech acts*. An answer that gives a unit no code, because it cannot be read, uses a label that is not in the codebook, or was cut off before reaching the unit, is not taken as "cannot tell": it has no vote, it lowers the unit's confidence, a unit left without any usable answer waits on the Review page as such, and the quality-control report counts these answers by cause. Each answer is kept as it came, in `answers.parquet` beside the step's `samples.parquet`, which marks for every unit and answer what was wrong and the label given. The answers are also kept in the project's cache, so a run that asks the same question of the same model again, for example after only *Review below this agreement* changed, takes the kept answer instead of asking; a changed seed, model or prompt asks anew, and `macaw clean` removes the kept answers.

A model can give the same wrong code every time it is asked, so consistency across answers detects only some of its errors. A random sample of units the model was certain about (*Sure units checked at random*, five per recording) is therefore placed on the Review page as well, marked *random check*. Once these are decided, the quality-control report gives the proportion of them that the researcher coded differently, an estimate of how often the model's confident answers are wrong.

Agreement between the researcher's decisions, the model and imported human coding is computed as described in {ref}`own-coding`.

### Examples drawn from human coding

Once human coding has been imported ({ref}`own-coding`), a few units of each code can be drawn into a table:

```bash
macaw draw-examples examples.csv --coder anna --per-code 3
```

Each row gives the code, the speaker, the words, the preceding line and the source. Remove or edit rows as needed, and add a reason in the `why` column where the model should read one. Then select the file under *Examples people coded*. A recording is never shown its own examples, which is why the `recording_id` column must keep the recordings' names exactly: a spreadsheet program may turn `001` into `1`, and coding then stops with a message naming the ids it does not recognise. Leave the cell empty for an example from outside the project. Units used to estimate agreement should not be the ones the examples were drawn from.

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

`macaw codebook draft first.yaml --aim "how parents guide their child's play"` drafts a codebook when none exists. The model describes in a few words what each of 200 units drawn at random does, with the stated aim in mind (`--units` changes the number), and then groups these descriptions into at most twelve codes, each with a description, criteria, borderline cases and examples. A model that proposes more is asked to merge them; if it still gives more than twelve, the twelve that hold the most units are kept. Descriptions that ended up in no code are then shown to the model once more, to be placed in one of the codes or in none. `first.draft.md` lists the descriptions each code gathers, marking those placed in this second step, and those that fit no code, and says when codes were merged or left out. The draft is a starting point: selecting, merging and defining the final codes remains the researcher's work.

:::{important}
Examples in a proposed or drafted codebook are participants' own words from the transcripts. Check them before the codebook is shared or published.
:::

(ratings)=
## Rating whole recordings with a language model

Many observational coding systems score a whole interaction, or a part of it, on rating scales (for example 1 to 5, or 0 and 1) rather than coding each utterance. A language model can give such ratings from the transcript, scale by scale, and its scores can be set beside those of human raters.

:::{important}
The model reads the whole transcript of each recording. With a cloud service, the transcript is sent to that service; check that the ethics approval and the participants' consent cover this.
:::

### The scales

The scales are written in a YAML file in the project folder, for example `scales.yaml`:

```yaml
name: family_talk
scales:
  - scale: warmth
    person: parent           # a role from the project's roles, or all for everyone together
    range: [1, 5]
    definition: How much warmth and affection the parent shows the child.
    anchors:                 # any of the scores; scores between anchors lie between them
      1: cold or dismissive throughout
      3: some warmth, not sustained
      5: warm throughout
    rules:
      - Laughing together counts towards warmth.
  - scale: on_task
    person: child
    range: [0, 1]
    anchors:
      0: mostly not doing the task as asked
      1: mostly doing the task as asked
  - scale: flow
    person: all
    range: [1, 5]
    definition: How smoothly the two take turns and follow each other's lead.
```

Each scale names the person it rates, its lowest and highest score, and any of a definition, anchors and further rules. Anchors may be given for some scores only; the model is told that scores without an anchor lie between the anchors around them. A coding manual can also be copied into `definition` word for word, with anchors and rules left empty. A file that gives every scale by its name, person and range alone is rated as it stands, which shows what the model makes of the names; a file in which only some scales are described is treated as an unfinished draft, and rating stops.

### Examples rated by people

A version can show the model, after each scale's definition, examples that people rated in other conversations. They are kept in a file of their own, so that versions with and without examples share one scales file:

```yaml
coding:
  ratings:
    full_text: {scales: scales.yaml, show: []}
    full_text_examples: {scales: scales.yaml, show: [], examples: examples.yaml}
```

```yaml
# examples.yaml
examples:
  warmth:
    - lines:
        - parent: Did you have fun at the party?
        - child: Yes, we played tag.
        - parent: Oh, I'm so glad.
      score: 4
      why: Warm interest, though brief.
    - description: The parent answers the child's questions briefly and turns back to the task each time.
      score: 2
```

Each example gives either a few lines, each with the role of the person speaking (one of the project's roles, or `other`), or a description written by a person, together with the score people gave and, where they wrote one, their reason. A scale may have any number of examples or none. Which conversations the examples come from is the researcher's choice; examples taken from a recording that the version also rates show the model the people's score for that recording, so for a comparison with human ratings they are drawn from recordings outside the ones compared. Examples lengthen every request: the run plan states how many tokens they add at most, and the step stops before rating when a prompt does not fit what the server reads (see below). The examples file is part of the version's settings, so editing it rates the recordings again, while a version without one is unaffected.

### Setting up

Ratings are set up in the text part of the settings as named versions, each a step of its own. A version names its scales file and whatever it changes in the *Coding* group's model settings; everything else follows that group (model, answers per scale, what the model is shown, note, reasons).

```yaml
coding:
  llm: {provider: ollama, model: "qwen3:14b", n_samples: 5, reasons: true}
  ratings:
    names_text: {scales: scales_names.yaml, show: []}
    names_voice: {scales: scales_names.yaml, show: [timing, pitch, loudness, rate]}
    full_text: {scales: scales.yaml, show: []}
    full_voice: {scales: scales.yaml, show: [timing, pitch, loudness, rate]}
```

The versions run with the rest of the analysis, or alone with `macaw run -e coding.rate.full_text`. Before a run, the plan states how many requests each version sends per recording; on a server that answers several requests together, they can be sent several at a time ({ref}`llm-concurrency`). A version can also be shown frames of the video ({ref}`rating-frames`).

Each line of the transcript is given to the model with the role of the person speaking, as named in the project's roles. Roles that use the words of the coding manual (for example `caregiver` and `child`) therefore spare the model a translation. The roles are best chosen before the first run, since voices and cameras are named by them.

A description of the task the participants were given is entered under *Note to the model*. Where tasks differ between recordings, `notes_by` gives a note for each task ({ref}`notes by task <notes-by>`).

### What the model is asked

The model is asked about one scale at a time, so that the score on one scale does not shape the score on another. Each request holds the whole transcript in time order, followed by the one scale. Each scale is asked several times (*Answers per unit*, five by default), and the score is the median of the answers; the spread of the answers is kept beside it. An odd number of usable answers gives a whole score; where an answer is left out (see below) and an even number remain, the median can fall halfway between two points, such as 3.5. With *Model says why*, the model adds a short reason after its score.

An answer that gives no whole number within the scale's range is not corrected: it is counted, left out of the median, and reported in the quality report. The proportion of such answers per version is a column of the results.

(rating-evidence)=
With `evidence: true` in a version, each line of the transcript is numbered, and the model is asked to list first the lines its score rests on, each by its number with a few words copied from it, and only then the score; it is told to rate only from the transcript and to cite only the lines given, and, for a scale that rates one person, to cite lines that person said. Each cited line is checked mechanically: the line must exist, a scale that rates one person must cite a line said by that person, and the copied words must be found in the line once case, punctuation and spacing are set aside. Words found only across the cited line and the one before or after it are counted apart from words not in the line, since they usually mean the transcript split a sentence in two rather than that the model cited the wrong line. An answer that fails a check keeps its score and is not asked again; the quality report counts the failures, and `evidence.csv` ({ref}`the export <rating-export>`) gives every cited line with its speaker and its start and end on the recording's clock, so that the lines the model relied on can be set beside the quotations human coders noted. The check confirms that a cited line exists and holds the words; whether the line supports the score is for a person to judge. Answers that list their evidence are longer, so each request takes more time.

A model server reads only a limited number of tokens per request, and some, Ollama among them, drop the start of a longer prompt without warning. Before rating, the step compares the longest prompt with what the server reads and stops with a message if it does not fit. For Ollama the limit is raised by starting the server with, for example, `OLLAMA_CONTEXT_LENGTH=16384`.

(rating-frames)=
### Frames of the video

A version with a `frames` key is shown frames of the video between the lines of the transcript:

```yaml
coding:
  ratings:
    full_frames: {scales: scales.yaml, show: [timing], frames: {every_s: 1.0, crop: people, margin: 0.1}}
```

From the start of each part rated, one frame is taken every `every_s` seconds. Each frame follows a line giving its time on the recording's clock, as the lines of the transcript do with `timing` shown, and the lines after it are those that started before the next frame. The scale comes after the transcript and its frames, so every scale and every repeated answer about one recording begins with the same text and pictures, which a server can reuse. With excerpts ({ref}`excerpts <rating-excerpts>`), frames are taken within each marked stretch only. Frames come from the camera the setup reads most people's bodies from, else from the recording's main video; a moment the video does not reach has no frame.

With `crop: people` the frames of a recording are all cut to one box: the union of the bodies of the people named on that camera over the part rated, with the outermost 2% at each edge left out so that a stray detection does not widen it, then widened by `margin` of its width and height on each side. A person in a mirror or on a screen behind the participants then stays outside the frames. Where no one was named on the camera, the whole picture is shown and the quality report says so. `crop: whole` shows the whole picture, and `crop: [x0, y0, x1, y1]` a box in the video's pixels; a box for one recording goes in that recording's own settings (`recordings/<recording>.yaml`). The frames are sent at the camera's own resolution, and the server scales them as its model needs; `long_side: 448`, for example, first scales them so that their longer side is 448 pixels, which saves tokens only on a server that counts a picture by its size. The quality report's Decisions table gives, for each recording, the box and a picture with the box drawn on the whole frame (`frames_crop.jpg`), and `frames.parquet` beside the ratings lists every moment shown, the video it came from and its time on that video. The frames themselves are not kept: the moments and the box take them again exactly. The saved prompts name each frame by its file name in place of the picture.

Frames are sent only to a model on the user's own machine or the lab's server (Ollama, vLLM, llama.cpp); a version with frames and another provider is refused. Before the run, the step asks Ollama whether the model reads pictures. A server turns each picture into a number of tokens that depends on the model and the server, and some scale small pictures up first, so the step sends one frame on its own, counts what it took, and stops with a message giving the context the longest request needs when that exceeds what the server reads. One frame a second over a recording of several minutes amounts to hundreds of frames; where that does not fit, `every_s` is raised.

With frames and `evidence: true` ({ref}`evidence <rating-evidence>`), each frame's line gives its number as well as its time (`[frame 12, t=312.0 s]`), numbered from 1 in each request in the order shown. The model is told to rate from the transcript and the frames and to cite only the numbered lines and frames; an item of its list is either a line, by its number with a few words copied from it, or a frame, by its number with a few words on what it shows (`{"frame": 12, "seen": "child turns away"}`). For a scale that rates one person, it is asked to cite lines that person said or frames showing that person, and a reason may rest on what was seen as well as on what was said, the timing or the voice. A cited frame is checked to have been shown and to come with words on what it shows; who is in it is not checked. Words such as "people sitting at a table" given for every frame pass these checks while describing the scene rather than the frame, so within each answer the frames given the same words as another frame, once case and punctuation are set aside, are counted and the quality report says how many. `evidence.csv` gives a frame's number, its time as shown, in `seen` the model's words and in `seen_elsewhere` whether it gave them for another frame too, and `answers.csv` counts each answer's cited lines and frames apart, and the frames sharing their words. Versions without frames are asked exactly as before.

(rating-frames-from)=
A version can borrow its frames as a control for what the frames show: with `frames: {every_s: 5.0, from: borrowed.csv}`, each recording the table lists is shown instead the frames of the recording it is paired with, for example another family doing the same task:

| recording | from_recording |
|---|---|
| dyad01 | dyad07 |

The rated recording's transcript, frame times and number of frames stay as they are: each frame is taken as far from the start of the same part of the other recording (its analysed part, or its segment of the same name) and labelled with the rated recording's time. Where the other recording's part is shorter, the frames past its end are taken again from its start, so that the number of frames still matches; the quality report says how many. The other recording's frames are cut to its own box, the one its own settings give or else the one found round its people's bodies; a box in the rated recording's own settings belongs to that recording's camera and is not used. The other recording's video has to be prepared first (`prep.video`). Recordings the table does not list are shown their own frames. The quality report's Decisions table, `frames.parquet` (columns `from_recording` and `from_t`) and the Methods say which frames were borrowed and from where.

### Rating parts of a recording

Without further settings, a version rates the analysed part of each recording as a whole. `segments: [first_half, second_half]` rates the named segments of the segments table one by one instead (see {ref}`segments`); recordings without such a segment are left out of that version and the quality report says so.

(rating-excerpts)=
A version with `excerpts: excerpts.csv` rates each scale from marked stretches only, for example the moments human coders noted for that scale, so that the step from single events to a score can be examined apart from finding the events. The table has one row per stretch, on the recording's clock:

| recording | scale | start_s | end_s |
|---|---|---|---|
| dyad01 | warmth | 62.0 | 75.5 |
| dyad01 | warmth | 140.2 | 140.2 |

For each scale, the model is shown the lines of the analysed part that overlap one of that scale's stretches in the recording, whole and in time order, with `...` where one stretch ends and the next begins; with *timing* shown, the first line after `...` is marked as opening an excerpt. A stretch may be a single moment (start and end alike), which brings the line said at that moment. A scale with no stretch in a recording is not rated there, and the quality report lists such scales. The score counts as a rating of the whole recording, so that `macaw ratings agreement` sets it beside people's ratings of the whole recording. A version cannot take both excerpts and segments. Only each recording's own rows are part of its settings, so marking stretches in one recording rates only that one again. Since each scale is shown other lines, the requests about one recording no longer begin with the same transcript, and a server that reuses what it read for the previous request cannot do so; as the excerpts are short, this costs little time.

### The effect of speaker errors

Once a transcript has been corrected in ELAN (see {ref}`transcript`), every step reads the corrected one. The recogniser's transcript is kept beside it when the correction is first imported, and a version with `transcript: recognised` rates that one instead, so that the effect of the corrections on the scores can be measured:

```yaml
coding:
  ratings:
    full_text: {scales: scales.yaml, show: []}
    full_text_asr: {scales: scales.yaml, show: [], transcript: recognised}
```

Such a version rates only recordings with a corrected transcript; elsewhere the two transcripts are the same. When only part of each recording was corrected, both versions should rate that part as a named segment, so that the uncorrected remainder does not dilute the difference. To compare speaker attribution alone, the corrector moves lines between the speakers' tiers in ELAN without changing their words.

### Human ratings and agreement

People's ratings are entered in a table `ratings.csv` in the project folder, one row per score:

| recording | segment | scale | coder | score |
|---|---|---|---|---|
| dyad01 | | warmth | anna | 4 |
| dyad01 | | warmth | ben | 3 |

The segment is left empty for a rating of the whole recording. Further columns, such as a family identifier, may be added. The table is read where it lies whenever it is needed, and each line is checked: the recording must exist, the scale must be defined in a version's scales file, the score must lie in its range and be a whole number, and no coder may score the same recording, segment and scale twice. Where a final score is the mean of two coders, such as 3.5, `project.ratings_step: 0.5` admits half points; any multiple of the step is then accepted.

```bash
macaw ratings agreement --cluster group_id --out agreement
```

compares every pair of raters on each scale, over the recordings and segments both scored. Each version of the model counts as a rater under its own name. For each pair it gives the number of recordings, the intraclass correlation for absolute agreement of single raters, ICC(A,1) (McGraw and Wong, 1996), Cohen's kappa with disagreements weighted by how many points the two scores lie apart (Cohen, 1968), the proportion of identical scores, and the proportion within one point, each with a 95% percentile bootstrap interval. A model's score halfway between two points, or a person's half point, never equals a whole score, so it counts against identical scores; the column *between* gives the number of recordings where either rater gave such a score, and the output adds a note below any scale where it is above zero. For such pairs the proportion of identical scores is low by construction, and ICC(A,1) and the proportion within one point are the figures to read. The proportion within one point is not given for scales of two scores, where it is always one. `--raters anna,ben,full_text` restricts the comparison to the raters named. `--cluster` resamples whole groups in the bootstrap, so that, for example, the two recordings of a family are drawn together; it names an entity of the recording pattern (such as `group_id`) or a column added to `ratings.csv`. Scales on which nearly every recording receives the same score give unstable ICC and kappa values; the proportion of identical scores is the more informative figure there.

(rating-export)=
```bash
macaw ratings export ratings_out
```

writes `scores.csv`, every score of every version and coder with the recording's entities, the spread of the model's answers and any columns added to `ratings.csv` (a segment or scale the model was not asked about is a row with an empty score, `n_asked` 0 and `not_rated` saying why), and `answers.csv`, each of the model's answers with the recording's entities, its reason, its full text, the tokens it took and the seconds the request lasted. Where a version asks for evidence ({ref}`evidence <rating-evidence>`), `evidence.csv` gives one row per line an answer cited: the line's number, speaker, start and end on the recording's clock and words, the words the answer copied, and which checks failed. Beside each table, a data dictionary (`scores_dictionary.csv` and `answers_dictionary.csv`, each also as a JSON sidecar) gives the meaning of every column and what an empty cell means ({ref}`data-dictionary`). Its entry for `rater` lists the settings of each version of the model, with each variant and its recordings where recordings have settings of their own, and each coder; its entry for `scale` lists each scale with the person it rates, its range and its scales file.

Each rating keeps the text it sent to the model, the instructions and the transcript with the question, once for every recording, segment and scale, since every answer to a scale is asked with the same text. `macaw ratings export ratings_out --prompts` also writes these as `prompts.csv` with its data dictionary, so that what the model was shown can be set beside what a human rater saw.

(own-coding)=
## Human coding and inter-rater agreement

Coding done by people, in ELAN or another program, can be imported beside the machine codes, compared with other coders, with the language model or with the automatic gaze codes, and summarised.

### Importing

On the Review page, enter a name under *Your name*, select the recording and press **Import your coding (.eaf, .csv, .xlsx)** on the *Your coding* card. Many files can be imported at once from the command line, each file name starting with its recording (for example `dyad01_anna.eaf`):

```bash
macaw import-codes coding/anna --coder anna
```

Importing again under the same name replaces that coder's codes for the same participant and tier, and everything the same file brought in before, so that a tier first given to the wrong participant or under the wrong name leaves nothing behind once the settings are put right; the command lists what it removed. Other coders' codes are kept.

Accepted files are ELAN files (`.eaf`) and tables (`.csv`, `.tsv` or `.xlsx`) with the columns `tier`, `start`, `end` and `code`, and optionally `person`, as most coding programs can export. Start and end are given in seconds or as a clock time such as `1:02.5`. They are read on the recording's clock, which is that of the file the recording's offsets are measured against. Coding timed on another device's file is imported with `--against` and that device's name, for example `macaw import-codes coding/anna --coder anna --against side_camera`, which places each time by that file's offset; for an ELAN file, a time origin set there for that file's media is taken into account. An ELAN file written by `macaw export -f eaf` or `macaw transcript export` is on the recording's clock already and needs no `--against`.

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

The report gives Krippendorff's alpha across all coders, Cohen's kappa for each pair, and the agreement on each code for each pair, so that with two human coders and a language model, the model's agreement with each person stands beside the agreement between the people. `--out agreement` additionally writes a folder with a table of every unit, each coder's code and the words spoken, which shows which codes are confused with which and helps when revising a codebook. A researcher's decisions on the Review page take part as a coder of their own. They cover only the units put to review, which the model found hard, and a random check, so their agreement with the model is not the model's agreement over all units; the report says so.

The Agreement page of the workbench runs the same comparison. Choose the tier, how units are formed (*One coder's units*, *A time grid* or *Shared intervals*) and, if needed, a segment, then press **Compare**. The page shows the report's tables and writes the folder named under *Save to folder*. The *Your coding* card on the Review page opens it with that tier and coder selected.

```{figure} screens/agreement.png
:width: 100%
:alt: The Agreement page: the choices of tier, segment, recordings and units on the left, and on the right the result, with Krippendorff's alpha, the number of units and coders, and tables for each pair of coders and each code.

The Agreement page after **Compare**, for a tier coded by two people and a language model.
```

:::{note}
A coder who coded only part of a recording leaves the rest uncoded, which counts as disagreement. Mark the part all coders coded as a segment and compare within it with `--segment`.
:::

### Summary of code use

`macaw describe-codes` lists for each recording, coder and participant how many units and how much time each code took, in how many runs of the same code, and how many of those runs last a single unit. Many single-unit runs indicate that the coding alternates from unit to unit, which deserves a closer look before analysis. Runs are left blank for a researcher's decisions on the Review page, which cover scattered units only. `--out codes.csv` saves the table, which also gives for each sequence the share of units the model said it cannot tell (`sequence_abstained_share`) and the share left without a code by split answers or no usable answer (`sequence_uncoded_share`).

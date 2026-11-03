# MACAW

**Multimodal Automated Coding and Analysis Workbench**

MACAW derives time-resolved measures of voice, language, turn-taking, body movement, facial behaviour and gaze from audio and video recordings of dyadic and small-group interaction, and combines them in a single data set for analysis in R, Python or SPSS. Decisions that software cannot make reliably are referred to the researcher.

## Workflow

The researcher provides the recordings and a small number of settings. MACAW aligns the files from different devices, extracts the measures, codes behaviour by rule or with a language model, and checks its own output. The researcher confirms what only a person can judge, such as which participant is which, and inspects what the checks flag. Manual coding in ELAN proceeds in parallel and can be compared with the machine codes. The result is one table containing every measure for every participant over time.

```{figure} workflow.png
:width: 100%
:alt: MACAW workflow. The recordings go into Prepare, where files are matched to people, lined up and trimmed, and the researcher checks the line-up and confirms who is who. Then Extract features, Machine coding and Check quality, where the researcher decides. Alongside runs the researcher's own coding in ELAN. All of it ends in one table, ready to analyse.

Overview of the workflow (click to enlarge). Blue steps are performed by MACAW, orange marks the points at which the researcher intervenes, and green is the output.
```

The same marking is used throughout this guide: {bdg-primary}`MACAW does this` {bdg-warning}`You do this` {bdg-success}`What you get`

## Measures

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Voice
Fundamental frequency, intensity and voice quality for every turn, and the standard acoustic feature sets eGeMAPS and ComParE.
:::

:::{grid-item-card} Language
A transcript with speaker and timing, word counts, lexical diversity, syntactic complexity, and which participant introduces the words the pair come to share.
:::

:::{grid-item-card} Turn-taking
Floor holding, gaps and overlaps between speakers, interruptions, backchannels, pauses, and the predictability of turn transitions.
:::

:::{grid-item-card} Body
Pose, movement speed, interpersonal distance, and motion energy within a marked region of the picture; in three dimensions with several calibrated cameras.
:::

:::{grid-item-card} Face and gaze
Facial action units, head orientation, mouth movement and blinks; whom each participant looks at, which named objects, and mutual gaze.
:::

:::{grid-item-card} Codes
Pauses, interruptions and backchannels coded by rule, speech acts optionally coded by a language model from the researcher's codebook, and human coding imported for comparison.
:::
::::

Not yet available: three-dimensional pose from a single camera.

## Principles

- The original recordings are never modified; all output is written to a separate project folder.
- Uncertain decisions, such as which file or voice belongs to whom and low-confidence codes, are referred to the researcher instead of being guessed, and every decision is recorded.
- Quality control works on samples: the report points to the moments the checks flagged and to a random selection of ordinary ones.
- Each recording is described on its own. Statistical analysis remains with the researcher.
- Processing runs on the researcher's own computer. Transcripts and video frames leave it only when the researcher selects a cloud language model; {ref}`privacy` lists the tools that report usage statistics.
- Every run records its settings and software versions, and `macaw methods` drafts a Methods section from them.

## Documentation

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Using MACAW
:link: guide
:link-type: doc
Requirements, installation, and the seven steps from the recordings to the finished table.
:::

:::{grid-item-card} Optional features
:link: extras
:link-type: doc
Language-model coding, human coding and agreement, gaze, custom body measures, gestures, centimetres, three dimensions, a model checking the review frames, clusters.
:::

:::{grid-item-card} Questions
:link: questions
:link-type: doc
Privacy, run times, experimenters in the room, children's voices, and other special cases.
:::
::::

MACAW is at an early stage of development, so names and settings may still change. Questions and problems can be reported on the [GitHub page](https://github.com/AU-CONNECT-Lab/Multimodal-Automated-Coding-and-Analysis-Workbench/issues).

```{toctree}
:hidden:

guide
extras
questions
```

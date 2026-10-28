# MACAW

**From recordings of people talking to one table you can analyse.**

MACAW (Multimodal Automated Coding and Analysis Workbench) watches and listens to recordings of two or more people interacting. It measures their voices, words, bodies and faces over time and puts it all into one table for R, Python or SPSS. The software does the measuring. Wherever it is unsure, it stops and asks you.

## How it works

You bring the recordings and a few settings. MACAW lines up the files from different devices, measures everything it can, codes what it can by rule or with a language model, and checks its own work. You confirm the things only a person can tell, such as who is who, and look over what the checks flag. Everything ends up in one table: every measure, for every person, over time.

```{figure} workflow.png
:width: 100%
:alt: MACAW workflow. Your recordings go into Prepare, where files are matched to people, lined up and trimmed, and you check the line-up and confirm who is who. Then Extract features, Machine coding and Check quality, where you decide. Alongside runs your own coding in ELAN. All of it ends in one table, ready to analyse.

The whole path at a glance (click to enlarge). Blue boxes are done by MACAW, orange marks the places where you step in, green is what you get.
```

Each step in this guide is marked the same way: {bdg-primary}`MACAW does this` {bdg-warning}`You do this` {bdg-success}`What you get`

## What it measures

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Voice
Pitch, loudness and voice quality for every turn, plus the standard voice feature sets used in speech research.
:::

:::{grid-item-card} Words
A transcript of who said what and when, word counts, vocabulary, sentence structure, and who brings in the words the pair come to share.
:::

:::{grid-item-card} Turn-taking
Who holds the floor, gaps and overlaps between speakers, interruptions, short "mm-hm" replies, pauses, and how predictable the turn-taking is.
:::

:::{grid-item-card} Body
Each person's pose, how fast they move, how far apart people are, and the amount of movement in a part of the picture you mark.
:::

:::{grid-item-card} Face
Facial expressions as muscle movements (action units), head angle, gaze direction, mouth movement and blinks.
:::

:::{grid-item-card} Codes
Pauses, interruptions and backchannels coded by rule, and, if you want, speech acts coded by a language model from your own codebook.
:::
::::

Not there yet: who looks at whom, coding of gestures with speech, and distances in real units such as centimetres.

## Where to begin

[Using MACAW](guide.md) takes you from what you need to the finished table in seven steps. [Questions](questions.md) covers privacy, run times and special cases.

MACAW is an early version, so names and settings may still change. Questions and problems go to the [GitHub page](https://github.com/AU-CONNECT-Lab/Multimodal-Automated-Coding-and-Analysis-Workbench/issues).

```{toctree}
:hidden:

guide
questions
```

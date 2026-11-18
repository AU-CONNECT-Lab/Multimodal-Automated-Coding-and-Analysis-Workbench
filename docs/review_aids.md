# Help with the review

The aids on this page prepare decisions of {ref}`step 6 <review>`: a suggestion is shown beside the question, and the researcher still confirms it. Settings given as a block can be pasted into the *project.yaml* box on the settings page and adapted to the study.

(suggest-names)=
## Suggesting who is who on a camera

Instead of clicking every camera ({ref}`step 6 <naming>`), the researcher can let MACAW suggest who is who on the cameras nobody clicked, with *Suggest who is who where nobody clicked* under *Checks by a person* in the settings:

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

(voice-hints)=
## Assigning voices on a shared microphone

When participants share a microphone, the proposal of whose voice is which ({ref}`step 6 <review>`) can also draw on what each voice says and on how high it is. MACAW assumes nothing about the participants; the rules are written by the researcher:

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
- the name of a described participant is on someone else,
- the outline of an object followed ({ref}`objects`), shown on its own, is not on that object.

A frame already answered is asked again only when the frame, the people or outlines drawn on it, the model or the descriptions change.

### Checking the model

The card lists each flagged frame with what the model said, followed by frames it passed, drawn at random (*Passed frames checked at random*, five per recording by default). For each, press **Right** when the model was right about the frame, that is, the problem it names is there or a frame it passed is in order, and **Wrong** otherwise. From the command line: `macaw check-look -r dyad01 --frame <file> --verdict right`. The quality-control report lists the model's findings beside these verdicts, and the Methods text states what the model was asked and how often a person agreed with it. The frames drawn at random are what estimate how many errors the model passes over.

:::{note}
Whether a skeleton follows the arms and legs of the person it is on is not asked, since models of this size do not judge it reliably. Limbs that stretch from frame to frame are flagged by the automatic checks instead.
:::

:::{important}
With a cloud provider such as `openai` or `anthropic`, the review frames, which show the participants, are sent to that provider; the command says so before it starts. Whether this is permitted is for the ethics approval to decide.
:::

# 3. Fill in the settings

{bdg-warning}`You do this`

Click the gear. The choices most studies change are laid out as a form, in five groups. Everything else is below the form as text. When you save, MACAW checks the settings and tells you which finished results they would put out of date.

Project
: Where the recordings are, how many people each one has and their roles. *Task starts at* and *Task ends at*, in seconds, keep only the part you study, for example to skip the researcher's introduction.

Steps
: Which measures a run makes, and how many recordings run side by side.

Tools
: Which program does each kind of measurement, the language spoken, and whether each person has a microphone or they share one. **Body pose has no default**: choose MediaPipe or YOLO yourself (see {ref}`privacy`).

Checks by a person
: For each check in [step 5](review.md): does MACAW always wait for you, wait only when it is unsure, or never wait.

Coding
: The codebook and the language model. Only needed for [coding with a language model](language-model-coding.md).

:::{tip}
Does the task start at a different time in each recording? Each recording can have its own start and end. Put them in a small file named after the recording, such as `dyad01.yaml`, in the `recordings` folder inside the project.
:::

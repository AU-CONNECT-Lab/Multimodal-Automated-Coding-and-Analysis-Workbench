# Coding speech with a language model

A language model can code each utterance or turn of the transcript with the codes in your codebook, such as question, instruction or praise. You write the codebook: each code with a description, the criteria, examples, counterexamples and borderline cases.

Where it runs
: On a model on your own computer or lab server, or on a cloud service (OpenAI, Anthropic). With a cloud service the transcript text is sent there, so check that your ethics approval allows it.

How sure it is
: Each unit is asked several times (five by default). The share of matching answers is its confidence. The model may also answer "cannot tell".

What comes to you
: Units below 70% agreement, and every "cannot tell", appear under *Speech acts* on the Review page (see [step 5](review.md)).

Agreement
: MACAW reports how well your choices and the model's agree, as Krippendorff's alpha and Cohen's kappa.

To start, open *Coding* in the settings, add layer B, choose your codebook file and the model, then run as usual.

# Questions

(privacy)=
## Do my recordings leave my computer?

MACAW itself sends nothing anywhere. Two choices in the settings can:

- A cloud language model (OpenAI or Anthropic) receives the transcript lines it codes.
- The MediaPipe tools, one of the options for body pose and for the face, send usage statistics to Google while they run. They do not send the video, but there is no way to turn this off.

If nothing may leave the machine, choose YOLO for body pose and a language model on your own computer. The first run of each step downloads the models it needs.

## What happens when I change a setting?

Only the results that depend on that setting are made again. The settings page names them before you save, so nothing is redone by surprise.

## Can I stop halfway and continue later?

Yes. Close the window or press Stop. The next run continues where the last one stopped.

## How long does it take?

It depends mostly on how long the videos are and whether the computer has a graphics card. Transcription and the face and body steps take the longest. Run one recording first and judge from that.

## A researcher is in the room. Will their speech mix into the results?

With a shared microphone or camera sound, set *Other voices on a shared microphone* to 1 in the settings, and name that voice *none* when you check the voices. Their speech still counts as a turn in the turn-taking, but stays out of everyone else's voice and language measures.

## Children have higher voices. Is pitch measured correctly?

The pitch range can be set separately for each role, for example a higher range for the child. It is added in the text part of the settings page, with the other voice settings.

## Can I fix mistakes in the transcript?

Yes, in ELAN. See {ref}`transcript`. Everything that uses the transcript is updated on the next run.

## We have hundreds of recordings.

MACAW can run on a university computing cluster that uses Slurm, one job per recording. The README on the GitHub page explains how to set it up.

## What do I write in my paper?

Run `macaw methods`. It writes a Methods section for the steps your project actually ran, with the references to cite.

## Something went wrong.

A failed step says why at the bottom of the Run page. Fix the cause and press Run again; finished work is kept. If you are stuck, open an issue on the [GitHub page](https://github.com/AU-CONNECT-Lab/Multimodal-Automated-Coding-and-Analysis-Workbench/issues).

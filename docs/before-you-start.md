# Before you start

## You need

- Recordings: video, audio, or both. A microphone per person helps, but a single camera with its own sound works too.
- A computer with an NVIDIA graphics card for anything beyond a few recordings. Without one everything still runs, only much slower.
- Free disk space about the size of your videos, because MACAW makes one tidy copy of each video.

## Good to know

- Your recording folder is never changed. Results go to a folder of their own.
- Transcription works in many languages. Word and sentence measures are English only for now.
- Two-person recordings are the most tested. Groups of three or more are supported.

## What MACAW does not do

- Run statistics or compare groups. Each recording is described on its own, and the analysis stays yours.
- Guess silently. When it cannot decide something reliably, it asks you.

## Installing

This part is for whoever sets up the computer. MACAW is installed once per computer.

1. Install Python 3.10, 3.11 or 3.12, git, and ffmpeg.
2. Install PyTorch for the computer's graphics card, following [pytorch.org](https://pytorch.org/get-started/locally/). Doing this first keeps the graphics card in use.
3. Install MACAW with all its parts:

   ```bash
   pip install "pymacaw[audio,asr,text,vision,gui,report] @ git+https://github.com/AU-CONNECT-Lab/Multimodal-Automated-Coding-and-Analysis-Workbench.git"
   ```

Some tools are not included because of their licences: YOLO (body pose), OpenFace (face) and openSMILE (voice feature sets). Install them only if you choose them in the settings. Telling voices apart on a shared microphone uses a model on Hugging Face: accept its terms there once and log in to Hugging Face on this computer.

Once MACAW runs, the *This computer* section of its settings page shows whether it finds ffmpeg and the other programs.

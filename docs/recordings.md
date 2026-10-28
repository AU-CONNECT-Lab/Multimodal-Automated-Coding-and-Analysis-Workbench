# 1. Organise your recordings

{bdg-warning}`You do this`

Make one folder per recording. End each file's name with the person it belongs to, using your study's role names (`parent` and `child` unless you change them). A file whose name ends in anything else, such as `room`, belongs to everyone in the recording.

```text
recordings/
  dyad01/
    dyad01_parent.wav     the parent's microphone
    dyad01_child.wav      the child's microphone
    dyad01_room.mp4       a camera both share
  dyad02/
    ...
```

One camera only
: MACAW takes the sound from the video and tells the voices apart. Set *Microphones* to pyannote, the shared-microphone option, in the settings. You then say which voice is whose.

Several visits per family
: Use a folder per family with a folder per visit inside, and describe that layout once in the settings under *Recording folder pattern*.

Leaving a recording out
: It can be excluded in the settings without moving any files.

(signals)=
# Other signals

MACAW can place signals recorded alongside the interaction, currently functional near-infrared spectroscopy (fNIRS) stored as SNIRF files, on the recording's clock, show them under the video, and export them beside the behavioural measures. It does not preprocess or analyse the signals: filtering, motion correction, the general linear model and coherence between participants remain with dedicated software such as nirspipe. Any stage of a file can be shown, from the raw intensities to the cleaned haemoglobin series.

The feature is installed with `pip install "pymacaw[signals]"`.

## Listing the files

The files are listed in `signals.csv` in the project folder, one row per file, with the columns `recording`, `person`, `stage` and `file`.

```text
recording,person,stage,file
dyad01,parent,raw,../nirs/sub-p01_task-play_nirs.snirf
dyad01,parent,preproc,../nirs/derivatives/sub-p01_task-play_desc-preproc_nirs.snirf
dyad01,child,raw,../nirs/sub-c01_task-play_nirs.snirf
```

`person` is one of the project's roles; it is left empty for a file that belongs to the whole recording. `stage` is a free label such as `raw` or `preproc`, and a person may have several. A relative path is read from the folder of the table. The files stay where they are and are read once, by the step `signals.read`. Files intended for nirspipe should keep their BIDS names, since the export to nirspipe reads the participant, task and run from them.

## Placing a file on the recording's clock

{bdg-primary}`MACAW does this` {bdg-warning}`You do this`

The fNIRS device and the camera keep separate clocks. The step `signals.align` places each file on the recording's clock in one of two ways, and a person then checks the result.

**By the events the file marked.** Most recording software writes a marker into the file when the experiment sends a trigger, for instance at the start of each task. On the Signals page these events are drawn in a lane of their own, where the file's current offset puts them. To place a file by them, mark on the marks lane the moment the same event happens in the video, and give the mark the event's name, which the name field offers. Names are compared without regard to case, and `1` and `1.0` count as the same name. One mark is enough when its name occurs once in the file. When a name recurs, as with a trigger sent at every trial, the marks are paired where most of them agree, so only a few need to be marked; if they fit several places equally well, the file is flagged until a mark of another event, a look at the video or the IMU settles which. The file is placed at the mean offset of the pairs, and with three or more pairs spread over the recording a clock that runs at a slightly different rate is corrected along a line. The file is flagged when a pair lies more than 0.5 s from that offset, when a mark finds no event of its name near it, or when the IMU, where there is one, places the file more than 0.5 s elsewhere. Each file is paired on its own, so a file cut from a longer one takes its own offset.

**By the IMU.** When a file has no marked event paired and the participant's files carry the readings of a gyroscope worn on the head (an inertial measurement unit, IMU, in the SNIRF file's auxiliary channels), the step finds the offset by itself: it compares how fast the head turns on the video, from the angles of the face (`face.au`) or, failing that, from the tilt of the head among the body points (`pose.kinematics`), with how fast the gyroscope turns, and places the file where the two rise and fall together. The match is then repeated in stretches of a minute to check that they find the same offset, and a clock that runs at a slightly different rate is corrected along a line. A participant's other files take the same clock. The channels are found by name, `gyro` by default (`signals.gyro`). Some devices carry more than one IMU, for instance one on the cap and one in the unit worn on the body; the file is then left unplaced, the check lists the IMUs, and `signals.gyro` must name the one on the head, as in `signals.gyro: '^gyroscope_1_[xyz]$'`.

:::{important}
The IMU match holds only when the participant turns the head during the video and the face or the body points are found for much of it. For a participant who sits still, the correlation has no clear peak and the file is flagged with low confidence. The start times written by the two devices are shown as a hint but never applied, since they are set by hand and record only whole seconds.
:::

An offset from marked events is as accurate as the marks are placed on the video, usually within a few frames; one from the IMU is accurate to about a tenth of a second. Either is ample for haemodynamic signals that follow behaviour by several seconds, but not for signals such as EEG, which need a shared trigger recorded by both devices.

A file placed in neither way, or one flagged, waits for a person at the check *Where each signal file sits* (`signals.review`). By default every offset found waits until a person has looked at it; with `flagged` only those in doubt wait, and with `skip` none.

## The Signals page

The Signals page shows the recording's video with a strip beneath it. For each file, a carpet draws every channel of one kind (one wavelength, or HbO) as a row, less its linear trend and in its own standard deviations, from black at −3 to white at +3. A movement of the participant appears as a vertical stripe across all channels, and a poor channel as a row that stands out. Chosen channels can be drawn as lines on the same scale, and measures from the video and codes can be added as lanes of their own. The events each file marked are drawn where its offset puts them, so a misplaced file shows as events that fall beside what happens in the video. For each participant with an IMU, a lane draws the turning speed of the head on the video and that of the gyroscope together; where the file is placed correctly, their peaks coincide.

Under *Where each file sits*, **Keep** accepts the offset that was found, and typing another offset sets it by hand for all of that participant's files. Without marked events or an IMU, the offset is found by eye: a movement of the child shows as a stripe in the raw carpet, and the offset is changed until the stripe falls on the movement in the video.

The marks lane holds stretches that a person marks while watching, such as a child touching the cap. Dragging on an empty part of the lane makes a mark; a new mark takes the name of the last one, and each may carry a note. Marks are saved in `marks.csv` on the recording's clock. Unlike segments, they do not change any measure.

## Exporting

{bdg-success}`What you get`

`macaw export -f signals`
: One table per signal file, with a row for each of the file's own samples: its time on the file's clock, its time on the recording's clock, every channel, and every frame-level measure of every participant averaged over the video frames nearest that sample. A data dictionary beside each table gives the offset that was used. The measures are empty before and after the video.

`macaw export -f nirspipe`
: An events table for each run, on the file's own clock, which nirspipe reads with `--events-path`, and a table of stretches to leave out, which it reads with `--bad-spans`. The command prints both options. Which codes, marks and segments become events is set in `signals.events`; a coding tier gives one event per code, named after the participant's role where the code belongs to one (such as `child.question`), and a tier coded by several people is named as `tier/coder`. The marks listed in `signals.bad_marks` become the stretches to leave out.

A file waiting for a person is not exported. Events lasting a few seconds suit the general linear model; for wavelet coherence, a condition has to last long enough to contain several cycles of the frequency of interest, roughly half a minute or more at 0.1 Hz, so segments of the interaction are the more suitable unit there.

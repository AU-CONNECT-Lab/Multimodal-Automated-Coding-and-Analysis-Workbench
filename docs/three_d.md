# Lengths and three dimensions

Positions and movements are measured in pixels of each camera's picture unless the project converts them. The sections below convert pixels into lengths and place people in the three-dimensional room. Settings given as a block can be pasted into the *project.yaml* box on the settings page and adapted to the study.

(scale)=
## Lengths instead of pixels

Distances, speeds and areas from the video are measured in pixels. They can additionally be given in the project's unit of length (centimetres, metres or inches, chosen as *Length unit* in the settings), subject to one condition.

:::{important}
A pixel scale is valid only at the distance from the camera at which it was measured. An object twice as far away appears half as large, so a scale measured at the parent's seat makes a child seated further back appear smaller than they are. Measure an object at each participant's position and state whose position it was; each participant then receives their own scale. A measure between two participants, such as the distance between them, is converted only when both share the same scale. When the room is laid out on the Setup page, its notes give each participant's distance from the camera ({ref}`room`).
:::

There are three ways to obtain a scale:

Calibration clip filmed before the session
: With the camera in its final position, film an object of known length (a ruler or a sheet of paper) held at each participant's position. Place the clip in a `calibration` folder in the project and name it in the settings as `scale: {calibration: lab.mp4}`, so that every recording uses it. When the camera was moved for some sessions, film a new clip and name it in those recordings' own files. Moving or zooming the camera after the clip invalidates it, and a clip applies only to videos of the same size.

Measurement in the recording itself
: When an object of known length is visible in the recording, measure it there. A recording's own measurement takes precedence over a calibration clip.

A known value
: `scale: {per_px: 0.11}` for everyone, or `scale: {per_px_by_role: {child: 0.12, parent: 0.09}}` for each participant's position, in the project's unit per pixel. A value written this way takes precedence over any measurement.

To measure, open *Pixels to centimetres* (or to the unit chosen) on the Review page, stop the player on a frame where the object is in view (or choose a calibration clip in the card's corner), click the two ends of the object, enter its length, state whose position it was, and press **Save this measurement**. Measuring two or three times is recommended; the spread of the measurements shows how precise the clicks are.

The results remain in pixels. `macaw export` adds every pixel measure a second time in the project's unit, with `_cm`, `_m` or `_in` appended to its name, and writes a file beside the table with each recording's scale for each participant.

The quality-control report gives each participant's nose-to-shoulder length in the project's unit at their scale (a value far from a plausible one indicates that the participant sat nearer or further than the object), and flags a participant whose apparent size changes by more than 10% during the recording, which indicates movement toward or away from the camera. The check compares stretches of 30 seconds, which suits seated participants. When participants walk about, shorten the stretches to about 5 seconds; brief bending over is then flagged as well:

```yaml
scale:
  size_window_s: 5
```

(three-d)=
## Three-dimensional pose from several cameras

When two or more cameras film the same participants from different sides, MACAW places each participant's keypoints in the room, in metres. Distances then no longer depend on how far a participant sits from a camera, and the direction of a participant's head relative to the other participant can be measured.

Four conditions must be met:

- Every keypoint of interest is seen by at least two cameras from clearly different directions.
- The cameras record simultaneously. MACAW aligns them by their sound, as for any recording.
- The cameras are calibrated, that is, the lens of each camera and the positions of the cameras are known. This is obtained from a short clip of a printed board filmed before the session.
- The cameras are not moved or zoomed after that clip. When they are, the board is filmed again.

:::{note}
Placing participants in three dimensions has been validated on simulated cameras and on a public four-camera recording of an adult and a toddler, against that recording's own reconstruction. Finding the cameras from a board has been validated on simulated cameras only.
:::

The *3D* page of the workbench covers every step below. Its tab appears once the setup has two or more cameras, or once a calibration exists in the `calibration` folder.

### 1. Print the board

*Draw the board* on that page (or `macaw calibration-board`) writes a ChArUco board, a chessboard with small markers in its white squares, as an image in the `calibration` folder. The board sits in the lower left corner of the page, and an arrow in the margin starts at its marked corner, 15 mm from the page's left and bottom edges, and runs along its lower edge. Print it at 100%, mount it flat on a rigid backing such as foam board with the page's edges along the backing's, and measure one square with a ruler. Enter that length, in the project's unit, and press **Save the square**. The default board fills an A2 sheet. For A3, set smaller squares before drawing the board (here in centimetres; the markers keep their share of the square), then draw it with `macaw calibration-board --paper A3`:

```yaml
pose3d:
  board:
    square: 5
```

### 2. Film the board before the session

With the cameras in their final positions, clap once in view of all cameras, then move the board slowly through the space where the participants will sit, tilting it toward and away from the cameras, for about a minute. Keep it in view of at least two cameras at a time and close enough to fill a good part of the picture. Place each camera's clip in a folder named like that camera's device folder in the recordings, inside a folder for the calibration such as `calibration/lab`: the clip of the camera whose recordings sit in `front` goes in `calibration/lab/front`.

At the end of the clips, lay the board flat and still, face up, on the floor or on the table for about three seconds, in view of at least one camera. This shows which way is up and where the floor is, so that the cameras can later be placed in the room from the calibration (step 4). Laid at a known point, for example on the floor in the room's zero corner with the page's edges against the two walls and the arrow along the wall of the x axis, it also shows where the cameras stand and which way they face, and then no camera needs to be measured.

### 3. Find the cameras

Select the calibration on that page and press **Find the cameras** (or run `macaw calibrate-cameras lab`). This takes a few minutes and reports, for each camera, how far the board's corners fall from where the camera saw them (below 1 to 2 pixels is good), how often each pair of cameras saw the board together, and where the cameras stand, in the project's unit and drawn in three dimensions. The page also indicates when a clip or the board size has changed since, so that the cameras need to be found again. Then press **Use for this project**. When some sessions used another calibration, name it in those recordings' own files:

```yaml
pose3d:
  calibration: lab
```

A lab that has calibrated its cameras with Anipose can place its `calibration.toml` in the `calibration` folder and name it instead, as `calibration: lab.toml`.

### 4. Place the cameras in the room

This step is optional. The cameras' positions and directions in the room ({ref}`step 3 <setup>`) can be taken from the calibration instead of being measured. In *Board lying flat*, move the time strip to a moment when the board lies flat, check that at least one camera finds it, choose what it lay on (the floor, or a thing in the room whose height is given) and press **Save the moment**, or run `macaw calibration-floor lab --at 63.5 --on table`. When it lay at a known point, also enter where its marked corner lay, across and along in the room, and where its arrow pointed, in degrees clockwise from the y axis as a camera's direction (90 is along the x axis). With the page in the zero corner as above, the corner lies 1.5 cm from each wall: `macaw calibration-floor lab --at 63.5 --corner 1.5 1.5 --toward 90`. Then press **From the calibration** above the setup's editor, or run `macaw place-cameras`.

The calibration gives where the cameras stand relative to each other and the angle each picture spans. What ties them to the room is taken from everything the room already gives: the board lying flat and where it lay, and any camera positions, heights, directions and tilts entered. With the board lying flat at a known point, nothing else is needed; lying flat elsewhere, the positions of two cameras suffice; without it, the positions and heights of three cameras, or the position, height, direction and tilt of one camera. A tape measure is therefore enough, and no angle needs to be measured. When the given values do not suffice, only the picture angles are filled in and the page states what is missing. When they exceed what is needed, a table compares each given value with where the calibration places it and marks those more than 10 cm or 5° off. A single wrong value shifts the others as well, so without the board lying flat the wrong one may be any of those marked. A camera turned about its line of sight by more than 5° is reported, since the room draws every camera level. Cameras placed in this way are marked as taken from the calibration until one of their values is changed by hand.

### 5. Run and inspect

Run as usual. Each participant must be named on every camera that films them, by clicking them on each camera ({ref}`step 6 <naming>`), or by clicking them on one camera and letting MACAW suggest them on the others from their head motion ({ref}`suggest-names`). The results gain measures in three dimensions, with `_3d` appended to their names. They are stored in metres; the Results page shows them in the project's unit, and exports give them in that unit as well ({ref}`scale`): wrist speed, acceleration and jerk, the distance between the hands, the distances between participants' shoulder midpoints and between their heads, the angle between the direction of a participant's head and the direction to the other participant's head (0 when the head points straight at them; this is head direction, not gaze), the angle between the direction the chest faces and the direction to the other participant (`body_toward_deg_3d`: 0 when facing them, 180 with the back to them; it requires the hips to be placed, and is not given in the picture alone, where the depth of the shoulders is unknown), gesture kinematics when the three-dimensional thresholds are set ({ref}`gestures`), and custom body measures ({ref}`measures`). A custom measure defined by a function of one's own is computed in three dimensions only when `in_3d: true` is added to it; the function then also receives the columns `<joint>_z`, with every position in metres.

On the Results page, *In 3D* shows every participant's skeleton and the cameras; the recording plays beside it, and the view moves to the moment where the video is paused or the time strip is set; drag the view to rotate it. The quality-control report states, for each participant, how often they were placed in three dimensions and how well the cameras agree on them, and the results give both as `placed_3d_share` and `reprojection_error_px`, one value per participant, for leaving out participants placed poorly (with a depth camera or a body model on one camera, `placed_3d_share` is given too, and a body model also gives `model_height_m` and `height_scale`), and warns when a bone keeps changing length, which indicates misplaced keypoints.

(depth)=
## Three-dimensional pose from a depth camera

A depth camera, such as an Orbbec Femto, an Azure Kinect or a RealSense, records alongside its colour video how far each of its pixels is. With one, MACAW places each participant's keypoints in three dimensions from that single camera, in metres in the colour camera's frame, and the measures of the previous section follow from them.

:::{note}
Placing participants from a depth camera has been checked on simulated depth images and on a public recording of an adult and a toddler by two depth cameras, against that recording's own reconstruction. Importing a RealSense recording has been checked on the maker's sample recording; importing an Azure Kinect or Orbbec recording has not yet been run on a real recording.
:::

In the setup, mark the camera as recording depth:

```yaml
cameras:
  kinect:
    body: [parent, child]
    depth: true
```

Its device folder in each recording then holds, beside the colour video, a folder `depth` with one 16-bit PNG per video frame, in frame order, whose values are distances along the depth camera's axis, and a file `depth.yaml` that describes both cameras:

```yaml
depth_size: [512, 424]                  # the depth images' width and height, in pixels
depth_matrix: [[365.1, 0, 257.8], [0, 365.1, 207.6], [0, 0, 1]]
depth_distortions: [0.09, -0.27, 0, 0, 0.09]   # k1 k2 p1 p2 k3
rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]    # the depth camera's axes in the colour camera's frame
translation: [-0.052, 0, 0]             # where the depth camera sits in the colour camera's frame, in metres
colour_size: [1920, 1080]
colour_matrix: [[1060.0, 0, 954.2], [0, 1060.0, 548.8], [0, 0, 1]]
colour_distortions: [0.02, -0.02, 0, 0, 0]
unit_mm: 1                              # a stored value of 1 is this many millimetres
```

A camera's own recording is turned into this layout by `macaw depth-import`, which leaves the alignment of depth and colour to the camera maker's library: `macaw depth-import session.bag raw/dyad01/realsense` for a RealSense recording, with `pip install pyrealsense2`, and `macaw depth-import session.mkv raw/dyad01/kinect` for an Azure Kinect or Orbbec Femto recording, with the maker's SDK installed and then `pip install pyk4a`. It writes the colour video and the depth images aligned with it, at the recording's frame rate; a frame missing from the recording repeats the picture before it with no depth. Laid out by hand, the figures of `depth.yaml` come from the camera's own calibration, which its recording software saves. When the depth images are already aligned with the colour picture, the two lenses are the same and the rotation and translation are left at none. The depth images must follow the colour video's frames one for one, at a constant frame rate. A camera that records depth cannot be combined with `pose3d.calibration` in the same project.

The step `prep.depth` stores the depth images once on the prepared video's frames, and `pose.depth` places each keypoint the colour video finds: the depth image is carried into the colour camera's view, and the keypoint is placed on the nearest surface within a few pixels of it, along the camera's line of sight through it. Two conditions follow from this. The points lie on the surface of the body facing the camera, a few centimetres in front of the joint centres, which leaves distances between participants and movement much as they are but shortens a limb seen end on. A keypoint hidden behind something nearer the camera, such as a table edge or a raised hand, is placed on that object. The quality-control report states, for each participant, the share of keypoints that found a depth, and warns when a bone keeps changing length, which points to such frames.

(one-camera)=
## Three-dimensional pose from one camera

Without calibrated cameras or a depth camera, a body model fitted to each person in the picture can place them in three dimensions from the one camera that films their body. MACAW runs Multi-HMR with the Anny body model, which covers every age, in an environment of its own, and keeps for each person the fitted body whose keypoints match theirs.

:::{note}
This has been checked on a public recording of an adult and a toddler and on one of three adults, filmed by single cameras and compared with those recordings' own reconstruction.
:::

One camera does not show how large a person is: a small person near the camera and a large one further away look the same. The fitted body's size therefore sets how far each person is from the camera and from the others. For adults this holds well; a small child is given too large a body and is placed too far away, so distances to them are wrong by far more than the distances themselves. Directions, such as where a person's chest faces relative to the other person, depend much less on size. Giving each person's standing height in a table `heights.csv` beside `project.yaml` sets their size to it:

```text
recording,person,height
dyad01,parent,165
dyad01,child,98
```

The heights are in the project's unit and should be measured, since an error of a few percent moves a person by as much along the line of sight. People without a height keep the size fitted to them, and the quality-control report and the Methods text say so. A small child far from the camera may not be found by the model in many frames; the report gives, for each person, the share of frames in which they were placed. Giving the camera's lens angle in the setup's room (step 3) also helps: otherwise the model estimates it, and a wide-angle camera then places everyone too far away. Each person is placed in the frame of the camera that films their body, so two people are measured against each other only when one camera films both.

To set it up, create a separate Python environment for the model, with PyTorch for the graphics card, and in it run `pip install anny smplx roma einops pillow opencv-python trimesh pyrender "numpy<2"`. Download the Multi-HMR code with `git clone https://github.com/naver/multi-hmr`, and its checkpoint `multiHMR_672_L_anny.pt` from the Multi-HMR page into the code's `models/multiHMR` folder; its licence permits non-commercial use only. Then set two environment variables, `MACAW_MULTIHMR_PYTHON` to that environment's `python`, and `MACAW_MULTIHMR` to the code's folder, and in the project:

```yaml
pose3d:
  one_camera: multihmr
```

The step `pose.mesh` runs the model over the video of each camera that films someone's body, and `pose.one_camera` places the named people, so naming people again does not run the model again. `pose3d.one_camera_every: 2` reads every second frame, for speed. `pose3d.one_camera_threshold` (0.1) is how sure the model must be that a person is there: raising it finds fewer people, and a small child is often the one missed. A body the model finds is used only where it matches a named person's keypoints, so a lower threshold does not bring in people who are not there. Another model can be used by writing a script for it into the project's `scripts` folder and naming that script under `pose3d.one_camera`: it is called with a file describing the job and writes, for each person found in each frame, the 17 body keypoints in the camera's frame in metres, the same keypoints in the video's pixels, and the standing height of the fitted body; the job file and these arrays are described at the top of MACAW's own script for Multi-HMR, `macaw/extractors/video/pose/one_camera_runner.py`. The script's work must sit under `if __name__ == "__main__":`, since the scripts folder is also read for measure functions.

# 2. Create a project and open the workbench

{bdg-warning}`You do this`

This is the only part that needs typing. Open a terminal and run these three lines, with your own project name and the folder that holds your recordings:

```bash
macaw init parent-child-play --name "Parent-child play" --raw-dir "D:/recordings"
cd parent-child-play
macaw gui
```

The first line creates a project folder with a settings file and a results folder. The last line opens the workbench. Next time, the last two lines are enough. If you prefer a browser tab to a window, use `macaw gui --browser`.

The workbench has three pages along the top and a gear for the settings:

Run
: Choose recordings and steps, start a run and watch it.

Review
: The things waiting for you. The number on the tab says how many.

Results
: The quality page and every measure over time.

Settings (the gear)
: Everything about the project.

Steps appear under short names, such as `text.asr` for transcription and `pose.kinematics` for body movement.

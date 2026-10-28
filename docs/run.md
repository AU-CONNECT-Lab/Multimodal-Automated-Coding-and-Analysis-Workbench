# 4. Run

{bdg-primary}`MACAW does this`

On the Run page, tick the recordings and the steps you want, then press **Run**. Whatever a step needs runs with it. **Preview** lists what would run without running it, and **Stop** ends a run cleanly. Try one recording first to see that everything works.

Each step shows where it stands:

| Status | What it means |
|---|---|
| {bdg-success}`up to date` | Done with the current settings. It is not run again. |
| {bdg-info}`not run yet` | Runs the next time you press Run. |
| {bdg-warning}`to run again` | A setting or a file it depends on changed since it ran. |
| {bdg-warning}`waiting for a person` | Stopped at a check. See [step 5](review.md). |
| {bdg-danger}`failed last time` | The output at the bottom of the page says why. |

You can close the window at any time. The next run picks up where the last one stopped and does not repeat finished work.

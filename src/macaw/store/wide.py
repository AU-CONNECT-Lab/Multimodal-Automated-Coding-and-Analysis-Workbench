"""Export for downstream statistics users (R, MATLAB)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.store.schema import LONG_TABLE_SCHEMA, Scale, coerce_frame

# entity plus time columns; enough to identify a row once features become columns
DEFAULT_INDEX = [
    "recording_id",
    "group_id",
    "session_id",
    "task",
    "run",
    "subject_id",
    "subject_id_b",
    "t_start",
    "t_end",
]

RMD_TEMPLATE = """\
---
title: "MACAW export"
output: html_document
---

```{{r setup, include=FALSE}}
knitr::opts_chunk$set(echo = TRUE)
library(readr)
library(dplyr)
```

```{{r load}}
features <- read_csv("{csv_name}", show_col_types = FALSE)
manifest <- jsonlite::fromJSON("{manifest_name}")
```

The manifest describes every feature column: what produced it, at which scale, and in
what unit. Check it before interpreting anything.

```{{r overview}}
features %>%
  count(group_id, session_id, task, run) %>%
  arrange(group_id, session_id)
```

One row per recording per feature is the usual starting shape for a longitudinal model:
group_id is the grouping factor and session_id is the wave.

```{{r example}}
# library(lme4)
# lmer(value ~ session_id + (1 | group_id), data = long_features)
```
"""


def to_wide(
    df: pd.DataFrame,
    *,
    index: list[str] | None = None,
    scale: Scale | None = None,
) -> pd.DataFrame:
    """Pivot long rows to one column per feature."""
    frame = coerce_frame(df, LONG_TABLE_SCHEMA, "long table")
    if scale is not None:
        wanted = scale.value if isinstance(scale, Scale) else scale
        frame = frame[frame["scale"] == wanted]

    keys = index or DEFAULT_INDEX
    # not pivot_table: it takes the cartesian product of the index levels, and t_start x
    # t_end alone would square the row count. groupby only ever sees observed combinations,
    # and dropna=False keeps rows whose subject_id_b or task is null, which is most of them.
    grouped = frame.groupby(keys + ["feature"], dropna=False, observed=True)["value"].mean()
    pivoted = grouped.unstack("feature")
    pivoted.columns.name = None
    return pivoted.reset_index()


def export_csv(df: pd.DataFrame, path: Path, *, wide: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = to_wide(df) if wide else coerce_frame(df, LONG_TABLE_SCHEMA, "long table")
    out.to_csv(path, index=False, encoding="utf-8")
    return path


def export_rmd_template(out_dir: Path, csv_name: str, manifest_name: str) -> Path:
    """Emit a runnable .Rmd next to the CSV so R users have a starting point."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "macaw_export.Rmd"
    path.write_text(
        RMD_TEMPLATE.format(csv_name=csv_name, manifest_name=manifest_name), encoding="utf-8"
    )
    return path

"""The package `macaw new-extractor` writes: a working extractor to edit, its entry point and a
test. Whoever writes one says how to compute a variable; caching, the dependency graph, cluster
jobs, the manifest and QC come with the framework."""

from __future__ import annotations

import re
from pathlib import Path
from string import Template

from macaw.exceptions import MacawError

# a namespace part keeps a lab's names from colliding with the package's own
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")

PYPROJECT = Template('''\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "$distribution"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pymacaw"]

# this line is what makes macaw find the extractor once the package is installed
[project.entry-points."macaw.extractors"]
"$name" = "$module.extractor:$cls"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
''')

README = Template('''\
# $name

A MACAW extractor. Install it into the environment macaw runs in:

    pip install -e .

It then shows up in `macaw list-extractors` and runs with `macaw run -e $name`, or by listing
it under `pipeline.run` in project.yaml. Its parameters go under `plugins` in project.yaml:

    plugins:
      $name:
        quantiles: [0.1, 0.9]

The example reads f0 from prosody.frame and reports each person's pitch spread. Replace the
computation in `src/$module/extractor.py` and keep the shape: `requires` names what must run
first, `inputs_for` lists the files whose content decides whether a cached result is still
valid, and `run` returns long-table rows plus one FeatureSpec per feature.
''')

EXTRACTOR = Template('''\
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.extractors.base import Extractor, ExtractorResult, Phase
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.store.longtable import build_long_rows, concat_long_tables, read_long_table
from macaw.store.schema import Scale

NAME = "$name"
UPSTREAM = "prosody.frame"
DEFAULTS = {"quantiles": [0.1, 0.9]}


def pitch_spread(f0_hz: np.ndarray, low: float, high: float) -> float:
    voiced = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if voiced.size == 0:
        return float("nan")
    return float(np.quantile(voiced, high) - np.quantile(voiced, low))


class $cls(Extractor):
    name = NAME
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "$group"
    phase = Phase.DERIVE
    # naming the parameters here puts them in the cache key: change one and only this reruns
    config_sections = (f"plugins.{NAME}",)
    requires = (UPSTREAM,)

    def params(self) -> dict:
        return {**DEFAULTS, **self.config.plugins.get(NAME, {})}

    def check_available(self) -> None:
        # raise macaw.exceptions.BackendNotAvailableError here when a dependency is missing
        return None

    def _upstream_file(self, recording_id: str) -> Path:
        layout = layout_for(self.config, create=False)
        return layout.long_table_path(recording_id) / f"{UPSTREAM}.parquet"

    def inputs_for(self, recording) -> list[Path]:
        return [self._upstream_file(recording.recording_id)]

    def run(self, recording, workspace: Path) -> ExtractorResult:
        low, high = self.params()["quantiles"]
        f0 = read_long_table(self._upstream_file(recording.recording_id), features=["f0_hz"])
        frames = []
        for subject_id, rows in f0.groupby("subject_id", dropna=True):
            values = pd.DataFrame(
                {
                    "t_start": [rows["t_start"].min()],
                    "t_end": [rows["t_end"].max()],
                    "f0_spread_hz": [pitch_spread(rows["value"].to_numpy(dtype=float), low, high)],
                }
            )
            frames.append(
                build_long_rows(
                    recording_id=recording.recording_id,
                    source=self.name,
                    scale=Scale.RECORDING,
                    values=values,
                    entities=recording.entities(),
                    subject_id=str(subject_id),
                )
            )
        spec = FeatureSpec(
            name="f0_spread_hz",
            description=f"Spread of voiced f0 between its {low:g} and {high:g} quantiles.",
            unit="Hz",
            scale=Scale.RECORDING,
            group=self.group,
            extractor=self.name,
            extractor_version=self.version,
            params={"quantiles": [low, high]},
            aggregation="mean",
        )
        return ExtractorResult(rows=concat_long_tables(frames), specs=[spec])
''')

TEST = Template('''\
import numpy as np
import pytest

from $module.extractor import pitch_spread


def test_the_spread_of_a_known_range():
    assert pitch_spread(np.arange(1.0, 102.0), 0.1, 0.9) == pytest.approx(80.0)


def test_unvoiced_frames_do_not_count():
    assert pitch_spread(np.array([np.nan, 0.0, 200.0, 200.0]), 0.1, 0.9) == 0.0


def test_no_voicing_is_nan_not_zero():
    assert np.isnan(pitch_spread(np.array([np.nan, 0.0]), 0.1, 0.9))
''')


def names_for(name: str) -> dict[str, str]:
    if not NAME_PATTERN.match(name):
        raise MacawError(
            f"'{name}' is not a usable extractor name: use lowercase words joined by dots, with a "
            "namespace first, e.g. mylab.f0_spread"
        )
    module = name.replace(".", "_")
    return {
        "name": name,
        "module": module,
        "distribution": "macaw-" + module.replace("_", "-"),
        "cls": "".join(part.capitalize() for part in module.split("_")) + "Extractor",
    }


def scaffold_extractor(name: str, directory: Path, *, group: str = "custom") -> Path:
    """Write the package into directory/<distribution>; refuses to overwrite."""
    values = {**names_for(name), "group": group}
    root = Path(directory) / values["distribution"]
    if root.exists():
        raise MacawError(f"{root} already exists; choose another directory or remove it")

    files = {
        root / "pyproject.toml": PYPROJECT,
        root / "README.md": README,
        root / "src" / values["module"] / "__init__.py": Template(""),
        root / "src" / values["module"] / "extractor.py": EXTRACTOR,
        root / "tests" / "test_extractor.py": TEST,
    }
    for path, template in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.substitute(values), encoding="utf-8")
    return root

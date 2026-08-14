"""Lets a job script call `python -m macaw`, which does not depend on the console script
being on PATH after an environment activation."""

from macaw.cli.main import entrypoint

entrypoint()

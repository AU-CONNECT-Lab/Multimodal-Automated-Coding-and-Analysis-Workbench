import tomllib
from pathlib import Path

project = "MACAW"
release = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
version = release

extensions = ["myst_parser", "sphinx_design"]
myst_enable_extensions = ["colon_fence", "deflist"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "MACAW user guide"
html_show_copyright = False
html_theme_options = {
    # the blue of the workflow figure's automatic steps
    "light_css_variables": {"color-brand-primary": "#2c66a3", "color-brand-content": "#2c66a3"},
    "dark_css_variables": {"color-brand-primary": "#86b6ea", "color-brand-content": "#86b6ea"},
}

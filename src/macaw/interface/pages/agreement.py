"""How far coders agree on one tier across the recordings, people with each other and with the machine.
The Compare button runs the macaw agreement command shown under it."""

from __future__ import annotations

import json
import math
from pathlib import Path

from dash import Input, Output, State, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.interface import cli_bridge, ui
from macaw.project.segments import read_segments

LABEL = "Agreement"
PATH = "/agreement"
FOLDER = "agreement"
MODES = [
    {"label": "One coder's units", "value": "units"},
    {"label": "A time grid", "value": "grid"},
    {"label": "Shared intervals", "value": "exact"},
]
MODE_HINT = {
    "units": "Each interval of the coder below is a unit; suits people against the machine.",
    "grid": "Each step of the grid takes the code that covers most of it.",
    "exact": "Only intervals the coders marked with the same start and end are compared.",
}


def coded_tiers(config) -> dict[str, list[str]]:
    """Each tier coded in the project, and who coded it."""
    from macaw.coding.agreement import load_annotations

    try:
        annotations = load_annotations(config, resolve_recordings(config, None))
    except MacawError:   # nothing coded yet
        return {}
    pairs = annotations[["tier", "annotator"]].astype(str).drop_duplicates()
    return {str(tier): sorted(own["annotator"]) for tier, own in pairs.groupby("tier", sort=True)}


def segment_names(config) -> list[str]:
    return sorted({segment.name for found in read_segments(config).values() for segment in found})


def agreement_command(
    config_path: Path, tier: str, mode: str, coder: str | None, grid_s: float | None, segment: str | None,
    recordings: list[str] | None, samples: bool, folder: str | None,
) -> list[str]:
    flags = ["--tier", tier]
    if mode == "units":
        flags += ["--units", coder or ""]
    elif mode == "grid":
        flags += ["--grid-s", f"{grid_s:g}" if grid_s else ""]
    flags += [*(["--segment", segment] if segment else []), *(["--samples"] if samples else []), "--out", folder or FOLDER]
    return cli_bridge.build_command(["agreement"], config_path=config_path, recordings=recordings or None, extra_flags=flags)


def missing_choice(tier: str | None, mode: str, coder: str | None, grid_s: float | None) -> str | None:
    if not tier:
        return "Choose a tier first."
    if mode == "units" and not coder:
        return "Choose whose units the others are compared on."
    if mode == "grid" and not (grid_s and grid_s > 0):
        return "Give the length of a grid step in seconds."
    return None


def _number(value) -> str:
    return "-" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.3f}"


def result_view(report: dict, folder: Path, printed: str):
    """The figures of report.json as tables, its warnings, and where the unit table went."""
    parts = [
        ui.table(
            ["Units", "Rated by two or more", "Coders", "Coded by every coder", "Krippendorff's alpha"],
            [[report["n_units"], report["n_pairable"], report["n_annotators"], _number(report["coverage"]), _number(report["alpha"])]],
            numeric={"Units", "Rated by two or more", "Coders", "Coded by every coder", "Krippendorff's alpha"}, fit=True,
        ),
    ]
    if report["pairwise_kappa"]:
        columns = ["Coder", "With", "Units", "Observed agreement", "Cohen's kappa"]
        rows = [[row["annotator_a"], row["annotator_b"], row["n_units"], _number(row["observed_agreement"]), _number(row["kappa"])]
                for row in report["pairwise_kappa"]]
        parts += [html.H3("Each pair of coders"), ui.table(columns, rows, numeric=set(columns[2:]), fit=True)]
    if report["per_code_pairs"]:
        columns = ["Code", "Coder", "With", "Units", "Times the first gave it", "Times the second gave it", "Cohen's kappa"]
        rows = [[row["code"], row["annotator_a"], row["annotator_b"], row["n_units"], row["n_a"], row["n_b"], _number(row["kappa"])]
                for row in report["per_code_pairs"]]
        parts += [html.H3("Each code, each pair of coders"), ui.table(columns, rows, numeric=set(columns[3:]), fit=True)]
    if report["per_code"]:
        columns = ["Code", "Units", "Ratings", "Krippendorff's alpha"]
        rows = [[row["code"], row["n_units"], row["n_ratings"], _number(row["alpha"])] for row in report["per_code"]]
        parts += [html.H3("Each code, every coder together"), ui.table(columns, rows, numeric=set(columns[1:]), fit=True)]
    if report["abstention_rate"]:
        parts.append(ui.note("Left uncoded: " + ", ".join(f"{name} {rate:.0%}" for name, rate in report["abstention_rate"].items())))
    parts += [ui.note(warning, "warn") for warning in report["warnings"]]
    parts += [
        ui.note(f"report.json and units.csv, each unit with every coder's code and its words, are in {folder}."),
        html.Details([html.Summary("What the command printed"), html.Pre(printed, className="log")], className="more"),
    ]
    return parts


def compare(config_path: Path, argv: list[str], folder: str | None):
    code, printed = cli_bridge.run_now(argv, cwd=config_path.parent)
    where = Path(folder or FOLDER)
    where = where if where.is_absolute() else config_path.parent / where
    report = where / "report.json"
    if code != 0 or not report.is_file():
        return [ui.command(argv), ui.output(code, printed)]
    return [ui.command(argv), *result_view(json.loads(report.read_text(encoding="utf-8")), where, printed)]


def layout(config_path: Path):
    return html.Div(
        [
            ui.page_head(
                "Agreement",
                "How far people and the machine agree on one tier.",
                [ui.button("Compare", "agreement-run", icon_name="play", primary=True)],
            ),
            ui.card(
                "What to compare",
                html.Div(
                    [
                        # a tier's label names its coders, too long for one column
                        html.Div(ui.field("Tier", dcc.Dropdown(id="agreement-tier", clearable=False, className="picker"),
                                          "The codes compared, and the coders who gave them."), className="wide"),
                        ui.field("Segment", dcc.Dropdown(id="agreement-segment", placeholder="whole recordings", className="picker"),
                                 "Compare inside one named segment only; recordings without it are left out."),
                        ui.field("Recordings", dcc.Dropdown(id="agreement-recordings", multi=True, placeholder="every recording coded", className="picker")),
                    ],
                    className="field-grid",
                ),
                dcc.RadioItems(MODES, "units", id="agreement-mode", className="segmented", inline=True),
                html.P(id="agreement-mode-hint", className="hint"),
                html.Div(
                    [
                        html.Div(ui.field("On the units of", dcc.Dropdown(id="agreement-coder", clearable=False, className="picker")), id="agreement-coder-wrap"),
                        html.Div(ui.field("Grid step (s)", dcc.Input(id="agreement-grid", type="number", min=0, step="any", value=1)), id="agreement-grid-wrap"),
                        ui.field("Save to folder", dcc.Input(id="agreement-folder", value=FOLDER), "Relative to the project's folder; a comparison saved there before is replaced."),
                    ],
                    className="field-grid",
                ),
                dcc.Checklist(
                    [{"label": "Count each answer of the language model as a coder, to set it against itself", "value": "samples"}],
                    [], id="agreement-samples", className="chips",
                ),
                html.Div(id="agreement-command", className="spaced"),
            ),
            html.Div(id="agreement-message"),
            dcc.Loading(ui.card("Result", html.Div(html.P("Nothing compared yet.", className="none"), id="agreement-result"))),
            ui.card(
                "A clearer codebook",
                html.Div(
                    [
                        ui.field("Go by the codes of", dcc.Dropdown(id="agreement-refine-coder", placeholder="the one person who coded it", className="picker")),
                        ui.field("Save as", dcc.Input(id="agreement-refine-out", value=refined_name(config_path), debounce=True)),
                    ],
                    className="field-grid",
                ),
                html.Div(ui.button("Suggest a clearer codebook", "agreement-refine", icon_name="play"), className="buttons"),
                dcc.Loading(html.Div(id="agreement-refine-result")),
                desc="The model rewrites the codes coders disagree on; the codebook in use stays.",
            ),
        ]
    )


def refined_name(config_path: Path) -> str:
    """Beside the codebook in use, as the project file names it."""
    try:
        codebook = resolve_config(config_path).coding.codebook
    except (MacawError, ValueError):
        return ""
    if codebook is None:
        return ""
    path = Path(codebook)
    try:
        path = path.resolve().relative_to(config_path.parent.resolve())
    except ValueError:
        pass
    return path.with_name(f"{path.stem}_refined.yaml").as_posix()


def people_on(config, tier: str) -> list[str]:
    """Who coded a tier, leaving out the language model's codings."""
    return [coder for coder in coded_tiers(config).get(tier, []) if not coder.startswith("llm:") and coder not in config.coding.judges]


def refine_command(config_path: Path, out: str, coder: str | None, recordings: list[str] | None) -> list[str]:
    flags = [out, *(["--coder", coder] if coder else [])]
    return cli_bridge.build_command(["codebook", "refine"], config_path=config_path, recordings=recordings or None, extra_flags=flags)


def register(app, config_path: Path) -> None:
    @app.callback(
        Output("agreement-tier", "options"), Output("agreement-tier", "value"), Output("agreement-segment", "options"),
        Output("agreement-recordings", "options"), Output("agreement-message", "children"),
        Input("url", "pathname"), Input("url", "search"), State("agreement-tier", "value"),
    )
    def fill(pathname, search, current):
        if (pathname or "/").rstrip("/") != PATH:
            return (no_update,) * 5
        config = resolve_config(config_path)
        tiers = coded_tiers(config)
        options = [{"label": f"{tier} ({', '.join(coders)})", "value": tier} for tier, coders in tiers.items()]
        linked = ui.asked(search).get("tier")
        chosen = next((tier for tier in (linked, current) if tier in tiers), next(iter(tiers), None))
        message = None if tiers else ui.note("Nothing is coded yet: run a coding step, or import people's coding on the Review page.")
        return options, chosen, segment_names(config), resolve_recordings(config, None), message

    @app.callback(
        Output("agreement-coder", "options"), Output("agreement-coder", "value"),
        Input("agreement-tier", "value"), State("url", "search"), State("agreement-coder", "value"),
    )
    def coders(tier, search, current):
        found = coded_tiers(resolve_config(config_path)).get(tier, []) if tier else []
        chosen = next((coder for coder in (ui.asked(search).get("units"), current) if coder in found), found[0] if found else None)
        return found, chosen

    @app.callback(
        Output("agreement-mode-hint", "children"), Output("agreement-coder-wrap", "style"), Output("agreement-grid-wrap", "style"),
        Input("agreement-mode", "value"),
    )
    def show_mode(mode):
        hidden = {"display": "none"}
        return MODE_HINT[mode], None if mode == "units" else hidden, None if mode == "grid" else hidden

    @app.callback(
        Output("agreement-command", "children"),
        Input("agreement-tier", "value"), Input("agreement-mode", "value"), Input("agreement-coder", "value"),
        Input("agreement-grid", "value"), Input("agreement-segment", "value"), Input("agreement-recordings", "value"),
        Input("agreement-samples", "value"), Input("agreement-folder", "value"),
    )
    def show_command(tier, mode, coder, grid_s, segment, recordings, samples, folder):
        if missing_choice(tier, mode, coder, grid_s):
            return None
        argv = agreement_command(config_path, tier, mode, coder, grid_s, segment, recordings, bool(samples), folder)
        return [html.Span("The same in a terminal", className="hint"), ui.command(argv)]

    @app.callback(
        Output("agreement-result", "children"),
        Input("agreement-run", "n_clicks"),
        State("agreement-tier", "value"), State("agreement-mode", "value"), State("agreement-coder", "value"),
        State("agreement-grid", "value"), State("agreement-segment", "value"), State("agreement-recordings", "value"),
        State("agreement-samples", "value"), State("agreement-folder", "value"),
        prevent_initial_call=True,
    )
    def run_compare(_, tier, mode, coder, grid_s, segment, recordings, samples, folder):
        problem = missing_choice(tier, mode, coder, grid_s)
        if problem:
            return ui.note(problem, "warn")
        argv = agreement_command(config_path, tier, mode, coder, grid_s, segment, recordings, bool(samples), folder)
        return compare(config_path, argv, folder)

    @app.callback(Output("agreement-refine-coder", "options"), Input("url", "pathname"))
    def refine_coders(pathname):
        if (pathname or "/").rstrip("/") != PATH:
            return no_update
        config = resolve_config(config_path)
        if config.coding.codebook is None:
            return []
        from macaw.coding.codebook import read_codebook

        try:
            return people_on(config, read_codebook(config.coding.codebook).name)
        except MacawError:
            return []

    @app.callback(
        Output("agreement-refine-result", "children"),
        Input("agreement-refine", "n_clicks"),
        State("agreement-refine-out", "value"), State("agreement-refine-coder", "value"), State("agreement-recordings", "value"),
        prevent_initial_call=True,
    )
    def run_refine(_, out, coder, recordings):
        if resolve_config(config_path).coding.codebook is None:
            return ui.note("Set the codebook under Coding in the settings first.", "warn")
        if not (out or "").strip():
            return ui.note("Give a file name for the new codebook.", "warn")
        argv = refine_command(config_path, out.strip(), coder, recordings)
        # a model that reasons first takes a while for each code
        return [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent, timeout_s=3600))]

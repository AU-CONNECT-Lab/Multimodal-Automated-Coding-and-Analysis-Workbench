from __future__ import annotations

from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings

app = typer.Typer(help="Ratings of whole recordings: the model's versions under coding.ratings and people's own scores.")


def _table(config: Path, recordings: str | None):
    from macaw.coding.ratings import ratings_table

    resolved = resolve_config(config)
    known = sorted(set(resolve_recordings(resolved, None)) | set(resolved.project.excluded_recordings))
    chosen = resolve_recordings(resolved, recordings)
    return resolved, chosen, ratings_table(resolved, chosen, known)


def _folder(out: Path) -> Path:
    if out.is_file():
        raise MacawError(f"{out} is a file; give a folder")
    out.mkdir(parents=True, exist_ok=True)
    return out


@app.command("export")
def export(
    out: Path = typer.Argument(..., help="A folder for scores.csv and answers.csv."),
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
) -> None:
    """Every score, the model's and people's, one row per recording, segment, scale and rater (scores.csv), and each of the
    model's answers with its reason and its text (answers.csv), each with a data dictionary beside it."""
    from macaw.coding.ratings import RATING_DICTIONARY_COLUMNS, answers_dictionary, model_answers, scores_dictionary
    from macaw.project.dictionary import write_dictionary

    resolved, chosen, table = _table(config, recordings)
    folder = _folder(out)
    table.to_csv(folder / "scores.csv", index=False)
    answers = model_answers(resolved, chosen)
    answers.to_csv(folder / "answers.csv", index=False)
    write_dictionary(scores_dictionary(resolved, table, config.parent), folder / "scores.csv", RATING_DICTIONARY_COLUMNS)
    write_dictionary(answers_dictionary(resolved, answers, config.parent), folder / "answers.csv", RATING_DICTIONARY_COLUMNS)
    kinds = table.groupby("rater_kind")["rater"].nunique().to_dict()
    typer.echo(
        f"{len(table)} scores from {kinds.get('model', 0)} model version(s) and {kinds.get('person', 0)} coder(s), "
        f"{len(answers)} answers of the model, in {folder}"
    )


@app.command("agreement")
def agreement(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    raters: str = typer.Option(None, "--raters", help="Comma-separated raters to compare, version names and coders; default all."),
    cluster: str = typer.Option(
        None, "--cluster", help="Resample whole groups, such as the two recordings of a family: an entity like group_id, or a column of your ratings table."
    ),
    n_boot: int = typer.Option(2000, "--boot", min=0, help="Bootstrap samples for the 95% intervals; 0 gives none."),
    seed: int = typer.Option(0, "--seed", help="Fixes the bootstrap, so running it again gives the same intervals."),
    out: Path = typer.Option(None, "--out", help="A folder for agreement.csv and scores.csv."),
) -> None:
    """Agreement on each scale between each pair of raters, over the recordings and segments both scored: ICC(A,1),
    kappa weighted by how many points two scores lie apart, exact agreement and agreement within one point."""
    from macaw.coding.ratings import format_agreement, known_scales, rating_agreement

    resolved, _, table = _table(config, recordings)
    chosen = [name.strip() for name in raters.split(",") if name.strip()] if raters else None
    unknown = sorted(set(chosen or ()) - set(table["rater"].dropna()))
    if unknown:
        raise MacawError(f"no scores by {', '.join(unknown)}; the raters are: {', '.join(sorted(set(table['rater'].dropna()))) or 'none'}")
    found = rating_agreement(table, known_scales(resolved), raters=chosen, cluster=cluster, n_boot=n_boot, seed=seed)
    typer.echo(format_agreement(found))
    if out is not None:
        folder = _folder(out)
        found.to_csv(folder / "agreement.csv", index=False)
        table.to_csv(folder / "scores.csv", index=False)
        typer.echo(f"\nagreement.csv and scores.csv in {folder}")

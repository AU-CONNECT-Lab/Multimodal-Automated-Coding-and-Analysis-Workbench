"""Ratings of whole recordings side by side: the model's versions and people's own scores, and how far they agree."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.coding.llm.rate import ANSWER_COLUMNS, ANSWERS_NAME, RATE_STEP, RATINGS_NAME, WHOLE
from macaw.coding.scales import RatingScale, read_scales
from macaw.exceptions import ConfigError, MacawError
from macaw.project.layout import layout_for
from macaw.project.recording import ENTITY_NAMES, build_recording

PEOPLE_COLUMNS = ("recording", "segment", "scale", "coder", "score")
MODEL, PERSON = "model", "person"
TABLE_COLUMNS = [
    "recording_id", *ENTITY_NAMES, "segment", "scale", "person", "rater", "rater_kind", "score", "score_sd", "score_min",
    "score_max", "n_answers", "n_asked",
]
STATISTICS = ("icc", "kappa", "exact", "within_one")
_UNIT = ["recording_id", "segment"]


def known_scales(config) -> dict[str, RatingScale]:
    """Every scale the versions under coding.ratings define; one defined twice must say the same person and range."""
    found: dict[str, RatingScale] = {}
    for name, version in config.coding.ratings.items():
        for scale in read_scales(version.scales).scales:
            seen = found.get(scale.scale)
            if seen is not None and (seen.person, seen.low, seen.high) != (scale.person, scale.low, scale.high):
                raise ConfigError(
                    f"scale {scale.scale} rates {seen.person} from {seen.low} to {seen.high} in one version's scales and "
                    f"{scale.person} from {scale.low} to {scale.high} in {name}'s; the same scale has to say the same"
                )
            found.setdefault(scale.scale, scale)
    return found


def model_ratings(config, recording_ids: list[str]) -> pd.DataFrame:
    frames = _artifacts(config, recording_ids, RATINGS_NAME)
    if not frames:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    table = pd.concat(frames, ignore_index=True).assign(rater_kind=MODEL)
    return table.reindex(columns=TABLE_COLUMNS)


def model_answers(config, recording_ids: list[str]) -> pd.DataFrame:
    frames = _artifacts(config, recording_ids, ANSWERS_NAME)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ANSWER_COLUMNS)


def _artifacts(config, recording_ids: list[str], name: str) -> list[pd.DataFrame]:
    layout = layout_for(config, create=False)
    frames = []
    for rid in recording_ids:
        for version in config.coding.ratings:
            path = layout.artifacts_dir(rid, f"{RATE_STEP}{version}") / name
            if path.is_file():
                frames.append(pd.read_parquet(path))
    return [frame for frame in frames if not frame.empty]


def people_ratings(config, scales: dict[str, RatingScale], recordings: list[str]) -> pd.DataFrame:
    """project.ratings read and checked line by line; no table, no rows. Columns beyond the five are kept."""
    path = Path(config.project.ratings)
    if not path.is_file():
        return pd.DataFrame(columns=[*TABLE_COLUMNS])
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in PEOPLE_COLUMNS if column not in frame.columns]
    if missing:
        raise ConfigError(f"{path} has no {', '.join(missing)} column; it needs {', '.join(PEOPLE_COLUMNS)}")
    frame = frame.apply(lambda column: column.str.strip())
    frame = frame[(frame != "").any(axis=1)]
    versions = set(config.coding.ratings)
    scores = []
    for index, row in frame.iterrows():
        line = index + 2   # the header is line 1
        where = f"{path} line {line}"
        if row["recording"] not in recordings:
            raise ConfigError(f"{where}: {row['recording']} is not one of this project's recordings")
        scale = scales.get(row["scale"])
        if scale is None:
            raise ConfigError(f"{where}: no version under coding.ratings defines a scale {row['scale']}")
        if not row["coder"]:
            raise ConfigError(f"{where}: a score needs the coder who gave it")
        if row["coder"] in versions:
            raise ConfigError(f"{where}: {row['coder']} is the name of a version under coding.ratings; give the coder another name")
        try:
            value = float(row["score"])
        except ValueError:
            raise ConfigError(f"{where}: score '{row['score']}' is not a number") from None
        if not value.is_integer() or not scale.low <= value <= scale.high:
            raise ConfigError(f"{where}: {row['scale']} takes whole scores from {scale.low} to {scale.high}, not {row['score']}")
        scores.append(value)
    twice = frame.duplicated(subset=["recording", "segment", "scale", "coder"], keep=False)
    if twice.any():
        lines = ", ".join(str(index + 2) for index in frame.index[twice])
        raise ConfigError(f"{path}: the same coder scored the same recording, segment and scale more than once (lines {lines})")
    extra = [column for column in frame.columns if column not in PEOPLE_COLUMNS]
    out = pd.DataFrame({
        "recording_id": frame["recording"].to_numpy(), "segment": frame["segment"].to_numpy(), "scale": frame["scale"].to_numpy(),
        "person": [scales[name].person for name in frame["scale"]], "rater": frame["coder"].to_numpy(), "rater_kind": PERSON,
        "score": np.asarray(scores, dtype=float), "n_answers": 1, "n_asked": 1,
    })
    for column in extra:
        out[column] = frame[column].to_numpy()
    return out


def ratings_table(config, recording_ids: list[str], all_recordings: list[str]) -> pd.DataFrame:
    """The model's versions and people's scores, one row per recording, segment, scale and rater, with each recording's
    entities and any column people added to their table."""
    scales = known_scales(config)
    people = people_ratings(config, scales, all_recordings)
    people = people[people["recording_id"].isin(recording_ids)]
    model = model_ratings(config, recording_ids)
    extra = [column for column in people.columns if column not in TABLE_COLUMNS]
    present = [frame for frame in (model, people) if not frame.empty]
    table = pd.concat(present, ignore_index=True) if present else pd.DataFrame(columns=TABLE_COLUMNS)
    table = table.reindex(columns=[*TABLE_COLUMNS, *extra])
    table["segment"] = table["segment"].fillna(WHOLE).astype(str)
    for name in [*ENTITY_NAMES, *extra]:
        table[name] = table[name].astype(object)
    for rid in table["recording_id"].dropna().unique():
        entities = build_recording(str(rid), config).entities()
        mine = table["recording_id"] == rid
        for name in ENTITY_NAMES:
            table.loc[mine, name] = entities.get(name)
        for column in extra:
            values = set(people.loc[people["recording_id"] == rid, column])
            if len(values) > 1:
                raise ConfigError(f"{config.project.ratings}: {column} takes more than one value for {rid} ({', '.join(sorted(values))})")
            table.loc[mine, column] = values.pop() if values else None
    return table.sort_values(["recording_id", "segment", "scale", "rater_kind", "rater"], kind="stable").reset_index(drop=True)


def icc_a1(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ICC(A,1) of two raters, absolute agreement and a single rater, for each row of weights: how often each unit is drawn."""
    total = weights.sum(axis=1)
    sum_x, sum_y = weights @ x, weights @ y
    grand = (sum_x + sum_y) / (2 * total)
    rows = 2 * (weights @ ((x + y) / 2) ** 2 - grand**2 * total)
    columns = total * ((sum_x / total - grand) ** 2 + (sum_y / total - grand) ** 2)
    error = weights @ (x**2 + y**2) - 2 * grand**2 * total - rows - columns
    with np.errstate(divide="ignore", invalid="ignore"):
        ms_rows, ms_error = rows / (total - 1), error / (total - 1)
        below = ms_rows + ms_error + 2 * (columns - ms_error) / total
        return np.where(below > 0, (ms_rows - ms_error) / below, np.nan)


def linear_kappa(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cohen's kappa with disagreement weighted by how many points two scores lie apart."""
    total = weights.sum(axis=1)
    values = np.unique(np.concatenate([x, y]))
    apart = np.abs(values[:, None] - values[None, :])
    given_x = (x[:, None] == values[None, :]).astype(float)
    given_y = (y[:, None] == values[None, :]).astype(float)
    share_x, share_y = (weights @ given_x) / total[:, None], (weights @ given_y) / total[:, None]
    observed = (weights @ np.abs(x - y)) / total
    expected = np.einsum("bi,ij,bj->b", share_x, apart, share_y)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(expected > 0, 1.0 - observed / expected, np.nan)


def pair_statistics(weights: np.ndarray, x: np.ndarray, y: np.ndarray, two_point: bool) -> dict[str, np.ndarray]:
    total = weights.sum(axis=1)
    near = np.full(len(weights), np.nan) if two_point else (weights @ (np.abs(x - y) <= 1).astype(float)) / total
    return {
        "icc": icc_a1(weights, x, y), "kappa": linear_kappa(weights, x, y),
        "exact": (weights @ (x == y).astype(float)) / total, "within_one": near,
    }


def draws(groups: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """How often each unit is drawn in each bootstrap sample, whole groups at a time."""
    names, index = np.unique(groups, return_inverse=True)
    counts = rng.multinomial(len(names), np.full(len(names), 1.0 / len(names)), size=n_boot)
    return counts[:, index].astype(float)


def rating_agreement(
    table: pd.DataFrame, scales: dict[str, RatingScale], *, raters: list[str] | None = None, cluster: str | None = None,
    n_boot: int = 2000, seed: int = 0, level: float = 0.95,
) -> pd.DataFrame:
    """Per scale and pair of raters, over the recordings and segments both scored: n, ICC(A,1), linearly weighted kappa,
    exact agreement and agreement within one point, each with a percentile bootstrap interval."""
    if cluster is not None and cluster not in table.columns:
        known = ", ".join(column for column in table.columns if column in ENTITY_NAMES or column not in TABLE_COLUMNS)
        raise MacawError(f"no column {cluster} to resample by; the columns are: {known}")
    rows = []
    scored = table.dropna(subset=["score"])
    for scale_name, own in scored.groupby("scale", sort=True):
        scale = scales.get(scale_name)
        names = sorted(set(own["rater"]) & set(raters) if raters else set(own["rater"]))
        wide = own.pivot_table(index=_UNIT, columns="rater", values="score", aggfunc="first")
        groups = own.drop_duplicates(_UNIT).set_index(_UNIT)[cluster] if cluster else None
        for position, left in enumerate(names):
            for right in names[position + 1:]:
                both = wide[[left, right]].dropna()
                row = {"scale": scale_name, "person": scale.person if scale else None, "rater_a": left, "rater_b": right, "n": len(both)}
                if cluster:
                    labels = groups.reindex(both.index)
                    if labels.isna().any():
                        raise MacawError(f"{cluster} is blank for some recordings; every recording needs one to resample by")
                    row["n_clusters"] = int(labels.nunique())
                x, y = both[left].to_numpy(dtype=float), both[right].to_numpy(dtype=float)
                two_point = scale is not None and scale.high - scale.low <= 1
                found = pair_statistics(np.ones((1, len(x))), x, y, two_point) if len(x) else {key: np.array([np.nan]) for key in STATISTICS}
                spread = {}
                if len(x) >= 2 and n_boot:
                    rng = np.random.default_rng([seed, zlib.crc32(f"{scale_name}|{left}|{right}".encode())])
                    units = labels.astype(str).to_numpy() if cluster else np.arange(len(x)).astype(str)
                    spread = pair_statistics(draws(units, n_boot, rng), x, y, two_point)
                tail = (1.0 - level) / 2.0 * 100.0
                for key in STATISTICS:
                    row[key] = float(found[key][0])
                    finite = spread.get(key, np.array([]))
                    finite = finite[np.isfinite(finite)]
                    row[f"{key}_low"] = float(np.percentile(finite, tail)) if finite.size else np.nan
                    row[f"{key}_high"] = float(np.percentile(finite, 100.0 - tail)) if finite.size else np.nan
                rows.append(row)
    columns = ["scale", "person", "rater_a", "rater_b", "n", *(["n_clusters"] if cluster else [])]
    columns += [name for key in STATISTICS for name in (key, f"{key}_low", f"{key}_high")]
    return pd.DataFrame(rows, columns=columns)


def format_agreement(found: pd.DataFrame) -> str:
    if found.empty:
        return "no scale was scored by two raters on the same recordings"
    blocks = []
    for scale, own in found.groupby("scale", sort=False):
        shown = pd.DataFrame({"rater_a": own["rater_a"], "rater_b": own["rater_b"], "n": own["n"]})
        if "n_clusters" in own.columns:
            shown["clusters"] = own["n_clusters"]
        for key, title in (("icc", "ICC(A,1)"), ("kappa", "kappa"), ("exact", "exact"), ("within_one", "within 1")):
            shown[title] = [_interval(*values) for values in own[[key, f"{key}_low", f"{key}_high"]].itertuples(index=False)]
        blocks.append(f"{scale} ({own['person'].iloc[0]})\n{shown.to_string(index=False)}")
    return "\n\n".join(blocks)


def _interval(value: float, low: float, high: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.2f} [{low:.2f}, {high:.2f}]" if np.isfinite(low) else f"{value:.2f}"

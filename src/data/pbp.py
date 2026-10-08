"""Canonical NFL play-by-play ingestion.

Loads nflverse play-by-play one season at a time, keeps only the columns the
feature pipeline uses, caches each season as parquet under data/raw/pbp, and
validates the result.

Completed seasons are downloaded once and then read from the local cache.
Pass refresh=True (or refresh_seasons=[...]) to force a re-download, which is
what you want for the season currently in progress.
"""

from __future__ import annotations

from collections.abc import Iterable

import nflreadpy as nfl
import pandas as pd

from src.config import configured_path
from src.data.schedules import configured_seasons, live_season
from src.teams import normalize_team_series, validate_teams


# Columns that must exist for the EPA / team-strength pipeline.
REQUIRED_COLUMNS = (
    "game_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "play_type",
    "epa",
)

# Everything the pipeline reads. A column missing from a given season (older
# seasons lack some) is simply skipped; consumers validate what they need.
PBP_COLUMNS = (
    *REQUIRED_COLUMNS,
    "home_team",
    "away_team",
    "posteam_type",
    "qtr",
    "down",
    "yards_gained",
    "success",
    "wp",
    "score_differential",
    "penalty",
    "fumble_lost",
    "touchdown",
    # Quarterback attribution
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "qb_dropback",
    "qb_scramble",
    "qb_kneel",
    "cpoe",
    "complete_pass",
    "interception",
    "pass_touchdown",
    "sack",
    "desc",
)


def _cache_path(season: int):
    return configured_path("raw") / "pbp" / f"pbp_{season}.parquet"


def _download_season(season: int) -> pd.DataFrame:
    """Download one season and keep only PBP_COLUMNS that exist."""
    frame = nfl.load_pbp([season])

    keep = [c for c in PBP_COLUMNS if c in frame.columns]
    frame = frame.select(keep)

    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()

    return frame


def _load_season(season: int, *, refresh: bool) -> pd.DataFrame:
    path = _cache_path(season)

    if path.exists() and not refresh:
        return pd.read_parquet(path)

    print(f"  downloading {season} ...")
    frame = _download_season(season)

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)

    return frame


def load_pbp(
    seasons: Iterable[int] | None = None,
    *,
    regular_season_only: bool = False,
    refresh: bool = False,
    refresh_seasons: Iterable[int] = (),
) -> pd.DataFrame:
    """Load and standardize NFL play-by-play data."""

    if seasons is None:
        seasons = configured_seasons()

    seasons = sorted({int(s) for s in seasons})

    if not seasons:
        raise ValueError("At least one season is required.")

    # The season in progress changes weekly; completed seasons never do.
    always_refresh = {int(s) for s in refresh_seasons} | {live_season()}

    print(
        f"Loading NFL play-by-play: "
        f"{min(seasons)}-{max(seasons)}"
    )

    frames = [
        _load_season(
            season,
            refresh=refresh or season in always_refresh,
        )
        for season in seasons
    ]

    pbp = pd.concat(frames, ignore_index=True)

    missing = set(REQUIRED_COLUMNS) - set(pbp.columns)

    if missing:
        raise ValueError(
            f"PBP source missing required columns: "
            f"{sorted(missing)}"
        )

    # ---------- Canonical types ----------

    pbp["season"] = pd.to_numeric(
        pbp["season"], errors="raise"
    ).astype(int)

    pbp["week"] = pd.to_numeric(
        pbp["week"], errors="coerce"
    ).astype("Int64")

    pbp["season_type"] = (
        pbp["season_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    pbp["play_type"] = (
        pbp["play_type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # ---------- Canonical teams ----------

    for column in ("posteam", "defteam", "home_team", "away_team"):
        if column in pbp.columns:
            # older seasons mark non-plays with "" rather than null
            blank = pbp[column].astype("string").str.strip().eq("")
            pbp[column] = normalize_team_series(pbp[column]).mask(blank, pd.NA)

    validate_teams(
        pbp["posteam"].dropna(),
        source="pbp.posteam",
    )

    validate_teams(
        pbp["defteam"].dropna(),
        source="pbp.defteam",
    )

    if regular_season_only:
        pbp = pbp.loc[
            pbp["season_type"].eq("REG")
        ].copy()

    return pbp.reset_index(drop=True)


if __name__ == "__main__":
    pbp = load_pbp(
        regular_season_only=True
    )

    print()
    print("Regular-season plays:", len(pbp))

    print()
    print("Rows by season:")
    print(
        pbp.groupby("season")
        .size()
        .to_string()
    )

    print()
    print("Play types:")
    print(
        pbp["play_type"]
        .value_counts(dropna=False)
        .head(15)
        .to_string()
    )

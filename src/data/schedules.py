"""Canonical NFL schedule ingestion.

Downloads NFL schedules, normalizes team identifiers, validates the
result, and creates one standardized game table used throughout the project.
"""

from __future__ import annotations

from collections.abc import Iterable

import nflreadpy as nfl
import pandas as pd

from src.config import CONFIG, configured_path
from src.teams import normalize_team_series, validate_teams


POSTSEASON_TYPES = frozenset(
    CONFIG["seasons"]["postseason_game_types"]
)


def configured_seasons() -> list[int]:
    """Return the season range defined in config.yaml."""
    start = int(CONFIG["seasons"]["start"])
    end = int(CONFIG["seasons"]["end"])

    if end < start:
        raise ValueError(
            f"Invalid season range: start={start}, end={end}"
        )

    return list(range(start, end + 1))


def last_complete_season() -> int:
    """Last season fully played; backtests and the holdout stop here."""
    return int(CONFIG["seasons"]["last_complete"])


def live_season() -> int:
    """The season in progress (forward-test only)."""
    return int(CONFIG["seasons"]["live"])


def load_schedules(
    seasons: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Download and standardize NFL schedules."""

    if seasons is None:
        seasons = configured_seasons()

    seasons = sorted({int(s) for s in seasons})

    if not seasons:
        raise ValueError("At least one season is required.")

    print(
        f"Loading NFL schedules: "
        f"{min(seasons)}-{max(seasons)}"
    )

    games = nfl.load_schedules(seasons)

    # nflreadpy may return Polars depending on version.
    if hasattr(games, "to_pandas"):
        games = games.to_pandas()

    games = games.copy()

    required = {
        "game_id",
        "season",
        "game_type",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }

    missing = required - set(games.columns)

    if missing:
        raise ValueError(
            f"Schedule source missing required columns: "
            f"{sorted(missing)}"
        )

    # ---------- Canonical types ----------

    games["season"] = pd.to_numeric(
        games["season"], errors="raise"
    ).astype(int)

    games["week"] = pd.to_numeric(
        games["week"], errors="raise"
    ).astype(int)

    games["game_type"] = (
        games["game_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # ---------- Canonical teams ----------

    games["home_team"] = normalize_team_series(
        games["home_team"]
    )

    games["away_team"] = normalize_team_series(
        games["away_team"]
    )

    validate_teams(
        games["home_team"],
        source="schedule.home_team",
    )

    validate_teams(
        games["away_team"],
        source="schedule.away_team",
    )

    # ---------- Derived fields ----------

    games["is_postseason"] = (
        games["game_type"].isin(POSTSEASON_TYPES)
    )

    games["is_played"] = (
        games["home_score"].notna()
        & games["away_score"].notna()
    )

    games["home_win"] = pd.NA

    played = games["is_played"]

    games.loc[played, "home_win"] = (
        games.loc[played, "home_score"]
        > games.loc[played, "away_score"]
    ).astype(int)

    games["home_win"] = games["home_win"].astype("Int64")

    # ---------- Integrity checks ----------

    if games["game_id"].duplicated().any():
        duplicates = games.loc[
            games["game_id"].duplicated(keep=False),
            "game_id",
        ].tolist()

        raise ValueError(
            f"Duplicate game_id values found: "
            f"{duplicates[:10]}"
        )

    return (
        games
        .sort_values(["season", "week", "game_id"])
        .reset_index(drop=True)
    )


def postseason_games(
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Return true postseason games only."""
    return (
        games.loc[games["is_postseason"]]
        .copy()
        .reset_index(drop=True)
    )


def save_schedules(
    games: pd.DataFrame,
) -> None:
    """Save canonical schedule tables."""

    output_dir = configured_path("processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_path = output_dir / "games.parquet"
    post_path = output_dir / "postseason_games.parquet"

    games.to_parquet(all_path, index=False)

    postseason_games(games).to_parquet(
        post_path,
        index=False,
    )

    print(f"Saved: {all_path}")
    print(f"Saved: {post_path}")


if __name__ == "__main__":
    games = load_schedules()
    save_schedules(games)

    post = postseason_games(games)

    print()
    print("All games:", len(games))
    print("Postseason games:", len(post))

    print()
    print("Postseason games by season:")
    print(
        post.groupby("season")
        .size()
        .to_string()
    )

    print()
    print("Postseason games by type:")
    print(
        post.groupby(["season", "game_type"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
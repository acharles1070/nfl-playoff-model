"""EPA and efficiency features derived from NFL play-by-play."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import configured_path
from src.data.pbp import load_pbp


VALID_PLAY_TYPES = {"pass", "run"}


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Vectorized division with NaN when denominator is zero."""
    return numerator.div(
        denominator.replace(0, np.nan)
    )


def prepare_scrimmage_plays(
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Return valid run/pass plays from REG and POST games."""

    required = {
        "season",
        "season_type",
        "week",
        "game_id",
        "posteam",
        "defteam",
        "play_type",
        "epa",
    }

    missing = required - set(pbp.columns)

    if missing:
        raise ValueError(
            f"PBP missing required columns: {sorted(missing)}"
        )

    plays = pbp.loc[
        pbp["play_type"].isin(VALID_PLAY_TYPES)
        & pbp["epa"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
    ].copy()

    # Keep season type explicitly.
    plays["season_type"] = (
        plays["season_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    plays["success"] = (
        plays["epa"] > 0
    ).astype(int)

    plays["is_pass"] = (
        plays["play_type"] == "pass"
    ).astype(int)

    plays["is_rush"] = (
        plays["play_type"] == "run"
    ).astype(int)

    if "yards_gained" in plays.columns:
        yards = pd.to_numeric(
            plays["yards_gained"],
            errors="coerce",
        )

        plays["explosive_pass"] = (
            (plays["is_pass"] == 1)
            & (yards >= 20)
        ).astype(int)

        plays["explosive_rush"] = (
            (plays["is_rush"] == 1)
            & (yards >= 10)
        ).astype(int)

    else:
        plays["explosive_pass"] = 0
        plays["explosive_rush"] = 0

    return plays


def _aggregate_side(
    plays: pd.DataFrame,
    *,
    team_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Aggregate play-level metrics to one team-game row."""

    required = {
        "season",
        "season_type",
        "week",
        "game_id",
        team_column,
        "epa",
        "success",
        "is_pass",
        "is_rush",
        "explosive_pass",
        "explosive_rush",
    }

    missing = required - set(plays.columns)

    if missing:
        raise ValueError(
            f"_aggregate_side missing columns: {sorted(missing)}"
        )

    df = plays.copy()

    # Conditional EPA numerators.
    df["pass_epa"] = np.where(
        df["is_pass"].eq(1),
        df["epa"],
        0.0,
    )

    df["rush_epa"] = np.where(
        df["is_rush"].eq(1),
        df["epa"],
        0.0,
    )

    group_cols = [
        "season",
        "season_type",
        "week",
        "game_id",
        team_column,
    ]

    out = (
        df.groupby(
            group_cols,
            as_index=False,
            observed=True,
        )
        .agg(
            epa_sum=("epa", "sum"),
            plays=("epa", "size"),
            success_sum=("success", "sum"),

            pass_epa_sum=("pass_epa", "sum"),
            pass_plays=("is_pass", "sum"),

            rush_epa_sum=("rush_epa", "sum"),
            rush_plays=("is_rush", "sum"),

            explosive_passes=("explosive_pass", "sum"),
            explosive_rushes=("explosive_rush", "sum"),
        )
    )

    # Rates.
    out["epa_per_play"] = _safe_divide(
        out["epa_sum"],
        out["plays"],
    )

    out["success_rate"] = _safe_divide(
        out["success_sum"],
        out["plays"],
    )

    out["pass_epa_per_play"] = _safe_divide(
        out["pass_epa_sum"],
        out["pass_plays"],
    )

    out["rush_epa_per_play"] = _safe_divide(
        out["rush_epa_sum"],
        out["rush_plays"],
    )

    out["explosive_pass_rate"] = _safe_divide(
        out["explosive_passes"],
        out["pass_plays"],
    )

    out["explosive_rush_rate"] = _safe_divide(
        out["explosive_rushes"],
        out["rush_plays"],
    )

    # Normalize the team identifier column.
    out = out.rename(
        columns={
            team_column: "team",
        }
    )

    # Explicitly define metadata.
    # This prevents season_type (or any future identifier)
    # from accidentally being renamed as a feature.
    metadata = {
        "season",
        "season_type",
        "week",
        "game_id",
        "team",
    }

    rename_map = {
        column: f"{prefix}_{column}"
        for column in out.columns
        if column not in metadata
    }

    out = out.rename(
        columns=rename_map
    )

    # Fail immediately if metadata disappears.
    expected_metadata = {
        "season",
        "season_type",
        "week",
        "game_id",
        "team",
    }

    missing_metadata = (
        expected_metadata - set(out.columns)
    )

    if missing_metadata:
        raise AssertionError(
            "Aggregation lost metadata columns: "
            f"{sorted(missing_metadata)}"
        )

    return out


def build_team_game_epa(
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Build one EPA/efficiency row per team per game."""

    plays = prepare_scrimmage_plays(pbp)

    offense = _aggregate_side(
        plays,
        team_column="posteam",
        prefix="off",
    )

    defense = _aggregate_side(
        plays,
        team_column="defteam",
        prefix="def_allowed",
    )

    merge_keys = [
        "season",
        "season_type",
        "week",
        "game_id",
        "team",
    ]

    team_game = offense.merge(
        defense,
        on=merge_keys,
        how="outer",
        validate="one_to_one",
    )

    team_game = (
        team_game
        .sort_values(
            [
                "season",
                "team",
                "week",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )

    if team_game.duplicated(
        ["season", "game_id", "team"]
    ).any():
        raise ValueError(
            "Duplicate team-game EPA rows detected."
        )

    return team_game


def save_team_game_epa(
    team_game: pd.DataFrame,
) -> None:
    """Save standardized team-game EPA features."""

    output_dir = configured_path("processed")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / "team_game_epa.parquet"

    team_game.to_parquet(
        path,
        index=False,
    )

    print(f"Saved: {path}")


if __name__ == "__main__":

    pbp = load_pbp(
        regular_season_only=False
    )

    team_game = build_team_game_epa(pbp)

    save_team_game_epa(team_game)

    print()
    print("Team-game rows:", len(team_game))

    print()
    print("Rows by season:")
    print(
        team_game.groupby("season")
        .size()
        .to_string()
    )

    print()
    print("Columns:")
    for column in team_game.columns:
        print(" ", column)

    print()
    print("Sample:")
    print(
        team_game.head(5)
        .to_string(index=False)
    )
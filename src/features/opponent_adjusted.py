"""Leakage-free opponent-adjusted EPA features.

For each team-game, estimate opponent quality using only games the
opponent played BEFORE the current game.

Then adjust the team's observed EPA relative to the quality of the
opponent faced.

Examples:
    adjusted offensive EPA
        = team offensive EPA
          - opponent pregame defensive EPA allowed

    adjusted defensive EPA allowed
        = team defensive EPA allowed
          - opponent pregame offensive EPA

Positive adjusted offensive EPA is good.
Negative adjusted defensive EPA allowed is good.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import configured_path


KEYS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "team",
]


def add_opponent(
    team_game: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Attach each team's opponent to every team-game row."""

    required_team = {
        "season",
        "week",
        "game_id",
        "team",
    }

    missing = required_team - set(
        team_game.columns
    )

    if missing:
        raise ValueError(
            f"team_game missing columns: {sorted(missing)}"
        )

    required_games = {
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
    }

    missing = required_games - set(
        games.columns
    )

    if missing:
        raise ValueError(
            f"games missing columns: {sorted(missing)}"
        )

    home = games[
        [
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
        ]
    ].copy()

    home = home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
        }
    )

    away = games[
        [
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
        ]
    ].copy()

    away = away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
        }
    )

    lookup = pd.concat(
        [home, away],
        ignore_index=True,
    )

    if lookup.duplicated(
        ["season", "game_id", "team"]
    ).any():
        raise ValueError(
            "Duplicate opponent lookup rows."
        )

    out = team_game.merge(
        lookup,
        on=[
            "season",
            "week",
            "game_id",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )

    if out["opponent"].isna().any():
        bad = out.loc[
            out["opponent"].isna(),
            [
                "season",
                "week",
                "game_id",
                "team",
            ],
        ]

        raise ValueError(
            "Missing opponents:\n"
            + bad.head(20).to_string(
                index=False
            )
        )

    return out


def add_pregame_team_strength(
    team_game: pd.DataFrame,
) -> pd.DataFrame:
    """Create expanding pregame offense/defense EPA ratings.

    shift(1) guarantees the current game's result is excluded.
    """

    required = {
        "season",
        "team",
        "week",
        "game_id",
        "off_epa_per_play",
        "def_allowed_epa_per_play",
    }

    missing = required - set(
        team_game.columns
    )

    if missing:
        raise ValueError(
            "Missing EPA columns: "
            f"{sorted(missing)}"
        )

    out = (
        team_game
        .sort_values(
            [
                "season",
                "team",
                "week",
                "game_id",
            ]
        )
        .copy()
    )

    def expanding_prior(
        series: pd.Series,
    ) -> pd.Series:
        return (
            series
            .shift(1)
            .expanding(
                min_periods=1
            )
            .mean()
        )

    out[
        "pregame_off_epa"
    ] = (
        out.groupby(
            ["season", "team"],
            group_keys=False,
        )["off_epa_per_play"]
        .apply(expanding_prior)
    )

    out[
        "pregame_def_allowed_epa"
    ] = (
        out.groupby(
            ["season", "team"],
            group_keys=False,
        )["def_allowed_epa_per_play"]
        .apply(expanding_prior)
    )

    out[
        "games_played_before_oa"
    ] = (
        out.groupby(
            ["season", "team"]
        )
        .cumcount()
    )

    return out


def attach_opponent_strength(
    team_game: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the opponent's PRE-game strength to each row."""

    opponent_ratings = team_game[
        [
            "season",
            "week",
            "game_id",
            "team",
            "pregame_off_epa",
            "pregame_def_allowed_epa",
            "games_played_before_oa",
        ]
    ].copy()

    opponent_ratings = (
        opponent_ratings.rename(
            columns={
                "team": "opponent",
                "pregame_off_epa":
                    "opp_pregame_off_epa",
                "pregame_def_allowed_epa":
                    "opp_pregame_def_allowed_epa",
                "games_played_before_oa":
                    "opp_games_played_before_oa",
            }
        )
    )

    out = team_game.merge(
        opponent_ratings,
        on=[
            "season",
            "week",
            "game_id",
            "opponent",
        ],
        how="left",
        validate="one_to_one",
    )

    return out


def calculate_adjusted_game_epa(
    team_game: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate opponent-adjusted performance for each game."""

    out = team_game.copy()

    # Offense faced opponent defense.
    #
    # If opponent normally allows +0.10 EPA/play
    # and this team produced +0.20, adjusted offense = +0.10.
    out[
        "oa_off_epa_per_play"
    ] = (
        out["off_epa_per_play"]
        - out[
            "opp_pregame_def_allowed_epa"
        ]
    )

    # Defense faced opponent offense.
    #
    # If opponent normally creates +0.15 EPA/play
    # and this team allowed only +0.05, adjusted defense = -0.10.
    out[
        "oa_def_allowed_epa_per_play"
    ] = (
        out[
            "def_allowed_epa_per_play"
        ]
        - out[
            "opp_pregame_off_epa"
        ]
    )

    return out


def add_adjusted_snapshots(
    team_game: pd.DataFrame,
) -> pd.DataFrame:
    """Create expanding and recent pregame adjusted EPA."""

    out = (
        team_game
        .sort_values(
            [
                "season",
                "team",
                "week",
                "game_id",
            ]
        )
        .copy()
    )

    metrics = [
        "oa_off_epa_per_play",
        "oa_def_allowed_epa_per_play",
    ]

    for metric in metrics:

        grouped = out.groupby(
            ["season", "team"],
            group_keys=False,
        )[metric]

        # Full-season history before current game.
        out[
            f"{metric}_std"
        ] = grouped.transform(
            lambda s: (
                s.shift(1)
                .expanding(
                    min_periods=1
                )
                .mean()
            )
        )

        # Last 4 games before current game.
        out[
            f"{metric}_last4"
        ] = grouped.transform(
            lambda s: (
                s.shift(1)
                .rolling(
                    4,
                    min_periods=1,
                )
                .mean()
            )
        )

        # Last 8 games before current game.
        out[
            f"{metric}_last8"
        ] = grouped.transform(
            lambda s: (
                s.shift(1)
                .rolling(
                    8,
                    min_periods=1,
                )
                .mean()
            )
        )

    return out


def build_opponent_adjusted(
    team_game: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Build leakage-free opponent-adjusted EPA table."""

    out = add_opponent(
        team_game,
        games,
    )

    out = add_pregame_team_strength(
        out
    )

    out = attach_opponent_strength(
        out
    )

    out = calculate_adjusted_game_epa(
        out
    )

    out = add_adjusted_snapshots(
        out
    )

    if out.duplicated(
        [
            "season",
            "game_id",
            "team",
        ]
    ).any():
        raise ValueError(
            "Duplicate opponent-adjusted team-game rows."
        )

    return (
        out.sort_values(
            [
                "season",
                "team",
                "week",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )


def audit(
    data: pd.DataFrame,
) -> None:

    print()
    print(
        "Opponent-adjusted rows:",
        len(data),
    )

    print(
        "Duplicates:",
        data.duplicated(
            [
                "season",
                "game_id",
                "team",
            ]
        ).sum(),
    )

    print()
    print("Season types:")
    print(
        data["season_type"]
        .value_counts()
        .to_string()
    )

    usable = data.loc[
        data["games_played_before_oa"] > 1
    ]

    print()
    print(
        "Missing opponent pregame offense "
        "after team Game 2:",
        f"{usable['opp_pregame_off_epa'].isna().mean():.3%}",
    )

    print(
        "Missing opponent pregame defense "
        "after team Game 2:",
        f"{usable['opp_pregame_def_allowed_epa'].isna().mean():.3%}",
    )

    print()
    print("Adjusted EPA summary:")

    cols = [
        "oa_off_epa_per_play",
        "oa_def_allowed_epa_per_play",
        "oa_off_epa_per_play_std",
        "oa_def_allowed_epa_per_play_std",
    ]

    print(
        data[cols]
        .describe()
        .round(4)
        .to_string()
    )


def save(
    data: pd.DataFrame,
) -> None:

    output_dir = configured_path(
        "processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / "team_game_opponent_adjusted_epa.parquet"
    )

    data.to_parquet(
        path,
        index=False,
    )

    print()
    print(f"Saved: {path}")


if __name__ == "__main__":

    team_game_path = (
        configured_path("processed")
        / "team_game_epa.parquet"
    )

    games_path = (
        configured_path("processed")
        / "games.parquet"
    )

    team_game = pd.read_parquet(
        team_game_path
    )

    games = pd.read_parquet(
        games_path
    )

    adjusted = build_opponent_adjusted(
        team_game,
        games,
    )

    audit(adjusted)

    save(adjusted)
"""Leakage-free pregame team EPA snapshots.

Rate statistics are calculated from prior-game numerators and
denominators so rolling EPA/success/explosive rates are play-weighted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import configured_path


# Each rate is reconstructed from its underlying numerator/denominator.
RATE_SPECS = {
    # Offense
    "off_epa_per_play": (
        "off_epa_sum",
        "off_plays",
    ),
    "off_success_rate": (
        "off_success_sum",
        "off_plays",
    ),
    "off_pass_epa_per_play": (
        "off_pass_epa_sum",
        "off_pass_plays",
    ),
    "off_rush_epa_per_play": (
        "off_rush_epa_sum",
        "off_rush_plays",
    ),
    "off_explosive_pass_rate": (
        "off_explosive_passes",
        "off_pass_plays",
    ),
    "off_explosive_rush_rate": (
        "off_explosive_rushes",
        "off_rush_plays",
    ),

    # Defense
    "def_allowed_epa_per_play": (
        "def_allowed_epa_sum",
        "def_allowed_plays",
    ),
    "def_allowed_success_rate": (
        "def_allowed_success_sum",
        "def_allowed_plays",
    ),
    "def_allowed_pass_epa_per_play": (
        "def_allowed_pass_epa_sum",
        "def_allowed_pass_plays",
    ),
    "def_allowed_rush_epa_per_play": (
        "def_allowed_rush_epa_sum",
        "def_allowed_rush_plays",
    ),
    "def_allowed_explosive_pass_rate": (
        "def_allowed_explosive_passes",
        "def_allowed_pass_plays",
    ),
    "def_allowed_explosive_rush_rate": (
        "def_allowed_explosive_rushes",
        "def_allowed_rush_plays",
    ),
}


def _divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Safe vectorized division."""
    return numerator.div(
        denominator.replace(0, np.nan)
    )


def build_pregame_snapshots(
    team_game: pd.DataFrame,
    windows: tuple[int, ...] = (4, 8),
) -> pd.DataFrame:
    """Build leakage-free, play-weighted pregame features."""

    required = {
        "season",
        "week",
        "game_id",
        "team",
    }

    for numerator, denominator in RATE_SPECS.values():
        required.add(numerator)
        required.add(denominator)

    missing = required - set(team_game.columns)

    if missing:
        raise ValueError(
            f"team_game missing columns: {sorted(missing)}"
        )

    df = (
        team_game
        .sort_values(
            ["season", "team", "week", "game_id"]
        )
        .reset_index(drop=True)
        .copy()
    )

    group_keys = ["season", "team"]

    df["games_played_before"] = (
        df.groupby(
            group_keys,
            sort=False,
            observed=True,
        )
        .cumcount()
    )

    # ---------------------------------
    # Season-to-date weighted features
    # ---------------------------------

    for rate, (numerator, denominator) in RATE_SPECS.items():

        prior_num = (
            df.groupby(
                group_keys,
                sort=False,
                observed=True,
            )[numerator]
            .transform(
                lambda s:
                s.shift(1)
                .expanding()
                .sum()
            )
        )

        prior_den = (
            df.groupby(
                group_keys,
                sort=False,
                observed=True,
            )[denominator]
            .transform(
                lambda s:
                s.shift(1)
                .expanding()
                .sum()
            )
        )

        df[f"{rate}_std"] = _divide(
            prior_num,
            prior_den,
        )

    # ---------------------------------
    # Rolling weighted features
    # ---------------------------------

    for window in windows:

        if window <= 0:
            raise ValueError(
                f"Invalid rolling window: {window}"
            )

        for rate, (
            numerator,
            denominator,
        ) in RATE_SPECS.items():

            rolling_num = (
                df.groupby(
                    group_keys,
                    sort=False,
                    observed=True,
                )[numerator]
                .transform(
                    lambda s, w=window:
                    s.shift(1)
                    .rolling(
                        window=w,
                        min_periods=1,
                    )
                    .sum()
                )
            )

            rolling_den = (
                df.groupby(
                    group_keys,
                    sort=False,
                    observed=True,
                )[denominator]
                .transform(
                    lambda s, w=window:
                    s.shift(1)
                    .rolling(
                        window=w,
                        min_periods=1,
                    )
                    .sum()
                )
            )

            df[f"{rate}_last{window}"] = _divide(
                rolling_num,
                rolling_den,
            )

    return df


def leakage_sanity_check(
    snapshots: pd.DataFrame,
) -> None:
    """Verify that Game 1 has no pregame history."""

    feature_cols = [
        c
        for c in snapshots.columns
        if c.endswith("_std")
        or "_last4" in c
        or "_last8" in c
    ]

    first_games = snapshots.loc[
        snapshots["games_played_before"].eq(0)
    ]

    if first_games[feature_cols].notna().any().any():
        raise AssertionError(
            "Leakage detected: first games contain "
            "pregame historical values."
        )

    print("Leakage sanity check: PASSED")


def save_pregame_snapshots(
    snapshots: pd.DataFrame,
) -> None:

    output_dir = configured_path("processed")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / "team_pregame_epa_snapshots.parquet"
    )

    snapshots.to_parquet(
        path,
        index=False,
    )

    print(f"Saved: {path}")


if __name__ == "__main__":

    source = (
        configured_path("processed")
        / "team_game_epa.parquet"
    )

    team_game = pd.read_parquet(source)

    snapshots = build_pregame_snapshots(
        team_game,
        windows=(4, 8),
    )

    leakage_sanity_check(snapshots)
    save_pregame_snapshots(snapshots)

    print()
    print("Snapshot rows:", len(snapshots))
    print("Snapshot columns:", len(snapshots.columns))

    print()
    print("History distribution:")
    print(
        snapshots["games_played_before"]
        .describe()
        .to_string()
    )

    # Compact diagnostic instead of giant terminal output.
    sample = snapshots.loc[
        snapshots["games_played_before"].isin(
            [0, 4, 8, 12]
        ),
        [
            "season",
            "week",
            "team",
            "games_played_before",
            "off_epa_per_play_std",
            "off_epa_per_play_last4",
            "off_epa_per_play_last8",
            "def_allowed_epa_per_play_std",
            "def_allowed_epa_per_play_last8",
        ],
    ].head(16)

    print()
    print("Compact sample:")
    print(sample.to_string(index=False))
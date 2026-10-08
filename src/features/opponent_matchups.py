"""Attach leakage-free opponent-adjusted EPA snapshots to matchups."""

from __future__ import annotations

import pandas as pd

from src.config import configured_path


OA_FEATURES = [
    "oa_off_epa_per_play_std",
    "oa_def_allowed_epa_per_play_std",
    "oa_off_epa_per_play_last4",
    "oa_def_allowed_epa_per_play_last4",
    "oa_off_epa_per_play_last8",
    "oa_def_allowed_epa_per_play_last8",
]


def _side_features(
    oa: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    """Prepare home/away opponent-adjusted features."""

    if side not in {"home", "away"}:
        raise ValueError(
            "side must be 'home' or 'away'"
        )

    cols = [
        "season",
        "game_id",
        "team",
        *OA_FEATURES,
    ]

    missing = set(cols) - set(oa.columns)

    if missing:
        raise ValueError(
            f"OA table missing columns: {sorted(missing)}"
        )

    out = oa[cols].copy()

    rename = {
        "team": f"{side}_team",
    }

    rename.update({
        feature: f"{side}_{feature}"
        for feature in OA_FEATURES
    })

    return out.rename(columns=rename)


def build_opponent_matchups(
    matchups: pd.DataFrame,
    oa: pd.DataFrame,
) -> pd.DataFrame:
    """Attach home and away pregame OA snapshots."""

    home = _side_features(
        oa,
        "home",
    )

    away = _side_features(
        oa,
        "away",
    )

    out = matchups.merge(
        home,
        on=[
            "season",
            "game_id",
            "home_team",
        ],
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        away,
        on=[
            "season",
            "game_id",
            "away_team",
        ],
        how="left",
        validate="one_to_one",
    )

    # -------------------------------------------------
    # Matchup differences
    #
    # Positive offensive difference:
    # home offense stronger than away offense.
    #
    # For defensive EPA allowed, lower is better.
    # Therefore away - home makes positive values mean
    # the HOME defense is stronger.
    # -------------------------------------------------

    for window in [
        "std",
        "last4",
        "last8",
    ]:

        out[
            f"diff_oa_off_epa_{window}"
        ] = (
            out[
                f"home_oa_off_epa_per_play_{window}"
            ]
            - out[
                f"away_oa_off_epa_per_play_{window}"
            ]
        )

        out[
            f"diff_oa_def_epa_{window}"
        ] = (
            out[
                f"away_oa_def_allowed_epa_per_play_{window}"
            ]
            - out[
                f"home_oa_def_allowed_epa_per_play_{window}"
            ]
        )

    if out.duplicated(
        ["season", "game_id"]
    ).any():
        raise ValueError(
            "Duplicate matchup rows after OA merge."
        )

    return out


def audit(
    data: pd.DataFrame,
) -> None:

    print()
    print(
        "Opponent-adjusted matchup rows:",
        len(data),
    )

    print(
        "Columns:",
        len(data.columns),
    )

    print(
        "Duplicates:",
        data.duplicated(
            ["season", "game_id"]
        ).sum(),
    )

    print()
    print("OA difference columns:")

    diff_cols = [
        c
        for c in data.columns
        if c.startswith("diff_oa_")
    ]

    print(diff_cols)

    # Week 1 / first-game missingness is expected.
    usable = data.loc[
        (
            data[
                "home_games_played_before"
            ].fillna(0) > 1
        )
        & (
            data[
                "away_games_played_before"
            ].fillna(0) > 1
        )
    ].copy()

    print()
    print(
        "Missing OA STD after both teams Game 2:"
    )

    check = [
        "diff_oa_off_epa_std",
        "diff_oa_def_epa_std",
    ]

    for col in check:
        print(
            col,
            f"{usable[col].isna().mean():.3%}",
        )

    print()
    print("Postseason rows:")

    print(
        data.loc[
            data["is_postseason"]
        ]
        .groupby("season")
        .size()
        .to_string()
    )


if __name__ == "__main__":

    processed = configured_path(
        "processed"
    )

    matchup_path = (
        processed
        / "historical_matchups.parquet"
    )

    oa_path = (
        processed
        / "team_game_opponent_adjusted_epa.parquet"
    )

    matchups = pd.read_parquet(
        matchup_path
    )

    oa = pd.read_parquet(
        oa_path
    )

    enriched = build_opponent_matchups(
        matchups,
        oa,
    )

    audit(enriched)

    output_path = (
        processed
        / "historical_matchups_oa.parquet"
    )

    enriched.to_parquet(
        output_path,
        index=False,
    )

    print()
    print(f"Saved: {output_path}")
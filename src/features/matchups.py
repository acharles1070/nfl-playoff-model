"""Canonical historical pregame matchup dataset.

Joins each game's schedule information with the home and away
teams' leakage-free pregame EPA snapshots.

One output row = one NFL game.
"""

from __future__ import annotations

import pandas as pd

from src.config import configured_path


ID_COLS = {
    "season",
    "week",
    "game_id",
    "team",
}


def _snapshot_feature_columns(
    snapshots: pd.DataFrame,
) -> list[str]:
    """Return pregame EPA feature columns only."""

    return [
        c
        for c in snapshots.columns
        if (
            c == "games_played_before"
            or c.endswith("_std")
            or "_last4" in c
            or "_last8" in c
        )
    ]


def _prepare_side(
    snapshots: pd.DataFrame,
    *,
    side: str,
) -> pd.DataFrame:
    """Prepare snapshots for a home or away join."""

    if side not in {"home", "away"}:
        raise ValueError(
            "side must be 'home' or 'away'"
        )

    features = _snapshot_feature_columns(
        snapshots
    )

    keep = [
        "season",
        "week",
        "game_id",
        "team",
        *features,
    ]

    out = snapshots[keep].copy()

    rename = {
        "team": f"{side}_team",
    }

    rename.update({
        feature: f"{side}_{feature}"
        for feature in features
    })

    return out.rename(columns=rename)


def build_matchup_dataset(
    games: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Build one leakage-free feature row per played game."""

    game_required = {
        "game_id",
        "season",
        "week",
        "game_type",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "is_postseason",
        "is_played",
    }

    missing = game_required - set(games.columns)

    if missing:
        raise ValueError(
            f"games missing columns: {sorted(missing)}"
        )

    snapshot_required = {
        "game_id",
        "season",
        "week",
        "team",
    }

    missing = snapshot_required - set(
        snapshots.columns
    )

    if missing:
        raise ValueError(
            f"snapshots missing columns: "
            f"{sorted(missing)}"
        )

    # Only games with known outcomes can be training rows.
    matchup = games.loc[
        games["is_played"]
    ].copy()

    home = _prepare_side(
        snapshots,
        side="home",
    )

    away = _prepare_side(
        snapshots,
        side="away",
    )

    matchup = matchup.merge(
        home,
        on=[
            "season",
            "week",
            "game_id",
            "home_team",
        ],
        how="left",
        validate="one_to_one",
    )

    matchup = matchup.merge(
        away,
        on=[
            "season",
            "week",
            "game_id",
            "away_team",
        ],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------
    # Difference features
    # --------------------------------

    snapshot_features = (
        _snapshot_feature_columns(
            snapshots
        )
    )

    for feature in snapshot_features:

        # games_played_before is useful separately,
        # but its difference is also potentially useful.
        home_col = f"home_{feature}"
        away_col = f"away_{feature}"

        matchup[f"diff_{feature}"] = (
            matchup[home_col]
            - matchup[away_col]
        )

    # --------------------------------
    # Integrity checks
    # --------------------------------

    if matchup["game_id"].duplicated().any():
        raise ValueError(
            "Duplicate games detected after joins."
        )

    if len(matchup) != games["is_played"].sum():
        raise ValueError(
            "Matchup row count changed during joins."
        )

    return (
        matchup
        .sort_values(
            ["season", "week", "game_id"]
        )
        .reset_index(drop=True)
    )


def save_matchup_dataset(
    matchup: pd.DataFrame,
) -> None:

    output_dir = configured_path("processed")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / "historical_matchups.parquet"
    )

    matchup.to_parquet(
        path,
        index=False,
    )

    print(f"Saved: {path}")


def audit_matchups(
    matchup: pd.DataFrame,
) -> None:
    """Print compact integrity diagnostics."""

    print()
    print("Matchup rows:", len(matchup))
    print("Columns:", len(matchup.columns))

    print()
    print("Games by season:")
    print(
        matchup.groupby("season")
        .size()
        .to_string()
    )

    print()
    print("Postseason games by season:")
    print(
        matchup.loc[
            matchup["is_postseason"]
        ]
        .groupby("season")
        .size()
        .to_string()
    )

    print()
    print("Target distribution:")
    print(
        matchup["home_win"]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .rename("share")
        .to_string()
    )

    # After Week 1, both teams should usually have
    # historical information available.
    later = matchup.loc[
        matchup["week"] > 1
    ]

    missing_home = (
        later["home_off_epa_per_play_std"]
        .isna()
        .mean()
    )

    missing_away = (
        later["away_off_epa_per_play_std"]
        .isna()
        .mean()
    )

    print()
    print(
        "Missing home STD EPA after Week 1:",
        f"{missing_home:.3%}",
    )

    print(
        "Missing away STD EPA after Week 1:",
        f"{missing_away:.3%}",
    )


if __name__ == "__main__":

    games_path = (
        configured_path("processed")
        / "games.parquet"
    )

    snapshots_path = (
        configured_path("processed")
        / "team_pregame_epa_snapshots.parquet"
    )

    games = pd.read_parquet(
        games_path
    )

    snapshots = pd.read_parquet(
        snapshots_path
    )

    matchup = build_matchup_dataset(
        games,
        snapshots,
    )

    audit_matchups(matchup)

    save_matchup_dataset(matchup)
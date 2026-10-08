from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, CONFIG
from src.data.pbp import load_pbp
from src.teams import normalize_team


OUT_PATH = PROJECT_ROOT / "data" / "processed" / "qb_game.parquet"


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def _prepare_qb_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare valid plays attributable to a quarterback.

    Important nflverse detail:
    scramble plays can have passer_player_id/name missing even though
    rusher_player_id/name correctly identify the quarterback.

    QB identity therefore uses:
        passer identity normally
        rusher identity on QB scrambles

    Kneels and nullified/no-play plays are excluded.
    """
    df = pbp.copy()

    required = [
        "season",
        "week",
        "game_id",
        "posteam",
        "passer_player_id",
        "passer_player_name",
        "rusher_player_id",
        "rusher_player_name",
        "qb_dropback",
        "qb_scramble",
        "qb_kneel",
        "epa",
        "cpoe",
        "complete_pass",
        "interception",
        "pass_touchdown",
        "sack",
        "yards_gained",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required PBP columns: {missing}"
        )

    if "season_type" not in df.columns:
        df["season_type"] = "REG"

    df["posteam"] = df["posteam"].map(
        lambda x: normalize_team(x) if pd.notna(x) else x
    )

    numeric_cols = [
        "qb_dropback",
        "qb_scramble",
        "qb_kneel",
        "epa",
        "cpoe",
        "complete_pass",
        "interception",
        "pass_touchdown",
        "sack",
        "yards_gained",
    ]

    if "success" in df.columns:
        numeric_cols.append("success")

    for c in numeric_cols:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # Valid-play filtering
    # ---------------------------------------------------------
    #
    # nflverse marks plays erased by penalty with play_type
    # "no_play" when that column is available.
    #
    # Also use desc as a defensive fallback because schemas
    # can vary across seasons.
    #
    valid_play = pd.Series(
        True,
        index=df.index,
        dtype=bool,
    )

    if "play_type" in df.columns:
        play_type = (
            df["play_type"]
            .astype("string")
            .str.lower()
        )

        valid_play &= ~play_type.eq("no_play").fillna(False)

    if "desc" in df.columns:
        desc = (
            df["desc"]
            .astype("string")
            .str.upper()
        )

        valid_play &= ~desc.str.contains(
            "NO PLAY",
            regex=False,
            na=False,
        )

    # ---------------------------------------------------------
    # QB identity
    # ---------------------------------------------------------
    #
    # Scrambles frequently have no passer_player_id/name in the
    # nflverse data. The rusher is the QB on those plays.
    #
    is_scramble = (
        df["qb_scramble"]
        .fillna(0)
        .eq(1)
    )

    df["qb_player_id"] = df["passer_player_id"]
    df["qb_player_name"] = df["passer_player_name"]

    scramble_with_rusher_id = (
        is_scramble
        & df["rusher_player_id"].notna()
    )

    scramble_with_rusher_name = (
        is_scramble
        & df["rusher_player_name"].notna()
    )

    df.loc[
        scramble_with_rusher_id,
        "qb_player_id",
    ] = df.loc[
        scramble_with_rusher_id,
        "rusher_player_id",
    ]

    df.loc[
        scramble_with_rusher_name,
        "qb_player_name",
    ] = df.loc[
        scramble_with_rusher_name,
        "rusher_player_name",
    ]

    # ---------------------------------------------------------
    # Identify meaningful QB plays
    # ---------------------------------------------------------
    is_dropback = (
        df["qb_dropback"]
        .fillna(0)
        .eq(1)
    )

    is_kneel = (
        df["qb_kneel"]
        .fillna(0)
        .eq(1)
    )

    qb_play = (
        (is_dropback | is_scramble)
        & ~is_kneel
        & valid_play
        & df["qb_player_id"].notna()
        & df["posteam"].notna()
    )

    df = df.loc[qb_play].copy()

    # Older seasons tag depth-chart roles onto the name ("B.St. Pierre (3rd QB)").
    # The player ID is the identity; names are cosmetic, so normalize them.
    df["qb_player_name"] = (
        df["qb_player_name"]
        .astype("string")
        .str.replace(r"\s*\(\d+\w+ QB\)\s*$", "", regex=True)
    )

    # The QB on the first snap is the observed starter. PBP row order within a
    # game is chronological, so the first retained row per team-game is it.
    df["first_snap_qb_id"] = df.groupby(
        ["game_id", "posteam"],
        sort=False,
    )["qb_player_id"].transform("first")

    # Every retained row represents one QB opportunity.
    #
    # A scramble counts as a dropback-level opportunity even when
    # nflverse's passer fields were empty.
    df["dropback"] = 1

    if "success" in df.columns:
        df["success"] = pd.to_numeric(
            df["success"],
            errors="coerce",
        )
    else:
        df["success"] = (
            df["epa"] > 0
        ).astype(float)

    df["scramble"] = (
        df["qb_scramble"]
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    df["sack_ind"] = (
        df["sack"]
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    df["int_ind"] = (
        df["interception"]
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    df["pass_td_ind"] = (
        df["pass_touchdown"]
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    df["complete_ind"] = (
        df["complete_pass"]
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    # CPOE applies only where nflverse supplies it.
    df["cpoe_valid"] = (
        df["cpoe"]
        .notna()
        .astype(int)
    )

    # Separate scramble production.
    df["scramble_epa"] = np.where(
        df["scramble"].eq(1),
        df["epa"],
        0.0,
    )

    df["scramble_yards"] = np.where(
        df["scramble"].eq(1),
        df["yards_gained"],
        0.0,
    )

    return df


def build_qb_game(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Produce one row per QB/team/game.

    These are IN-GAME statistics.

    They must not be used to predict the same game. Pregame QB
    features must later be constructed exclusively from games that
    occurred before the target game.
    """
    df = _prepare_qb_plays(pbp)

    group_cols = [
        "season",
        "season_type",
        "week",
        "game_id",
        "posteam",
        "qb_player_id",
    ]

    qb = (
        df.groupby(
            group_cols,
            dropna=False,
        )
        .agg(
            qb_player_name=("qb_player_name", lambda s: s.mode().iat[0]),
            first_snap_qb_id=("first_snap_qb_id", "first"),
            dropbacks=("dropback", "sum"),
            qb_epa=("epa", "sum"),
            successes=("success", "sum"),
            completions=("complete_ind", "sum"),
            cpoe_sum=("cpoe", "sum"),
            cpoe_plays=("cpoe_valid", "sum"),
            sacks=("sack_ind", "sum"),
            interceptions=("int_ind", "sum"),
            pass_tds=("pass_td_ind", "sum"),
            scrambles=("scramble", "sum"),
            scramble_epa=("scramble_epa", "sum"),
            scramble_yards=("scramble_yards", "sum"),
            qb_yards=("yards_gained", "sum"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Per-dropback efficiency
    # ---------------------------------------------------------

    qb["epa_per_dropback"] = _safe_div(
        qb["qb_epa"],
        qb["dropbacks"],
    )

    qb["success_rate"] = _safe_div(
        qb["successes"],
        qb["dropbacks"],
    )

    qb["cpoe"] = _safe_div(
        qb["cpoe_sum"],
        qb["cpoe_plays"],
    )

    qb["sack_rate"] = _safe_div(
        qb["sacks"],
        qb["dropbacks"],
    )

    qb["int_rate"] = _safe_div(
        qb["interceptions"],
        qb["dropbacks"],
    )

    qb["pass_td_rate"] = _safe_div(
        qb["pass_tds"],
        qb["dropbacks"],
    )

    qb["scramble_rate"] = _safe_div(
        qb["scrambles"],
        qb["dropbacks"],
    )

    qb["scramble_epa_per_dropback"] = _safe_div(
        qb["scramble_epa"],
        qb["dropbacks"],
    )

    qb["scramble_yards_per_scramble"] = _safe_div(
        qb["scramble_yards"],
        qb["scrambles"],
    )

    qb["yards_per_dropback"] = _safe_div(
        qb["qb_yards"],
        qb["dropbacks"],
    )

    # The starter is the QB who took the first snap for his team.
    qb["started"] = (
        qb["qb_player_id"].eq(qb["first_snap_qb_id"])
    ).astype(int)

    qb = qb.drop(columns=["first_snap_qb_id"])

    # ---------------------------------------------------------
    # Team-game workload
    # ---------------------------------------------------------

    team_game_cols = [
        "game_id",
        "posteam",
    ]

    qb["team_game_dropbacks"] = (
        qb.groupby(team_game_cols)["dropbacks"]
        .transform("sum")
    )

    qb["dropback_share"] = _safe_div(
        qb["dropbacks"],
        qb["team_game_dropbacks"],
    )

    # ---------------------------------------------------------
    # Deterministic primary QB
    # ---------------------------------------------------------
    #
    # Old implementation:
    #
    #   dropbacks == max(dropbacks)
    #
    # allowed multiple primary QBs when dropbacks were tied.
    #
    # Rank deterministically:
    #   1. most dropbacks
    #   2. most total QB EPA
    #   3. stable player ID
    #
    # Exactly one QB will therefore be primary for every team-game.
    #
    qb["_player_sort"] = (
        qb["qb_player_id"]
        .astype("string")
        .fillna("")
    )

    qb = qb.sort_values(
        [
            "season",
            "week",
            "game_id",
            "posteam",
            "dropbacks",
            "qb_epa",
            "_player_sort",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    qb["primary_qb"] = 0

    primary_idx = (
        qb.groupby(
            team_game_cols,
            sort=False,
        )
        .head(1)
        .index
    )

    qb.loc[
        primary_idx,
        "primary_qb",
    ] = 1

    qb = qb.drop(
        columns=["_player_sort"]
    )

    return qb


def audit(qb: pd.DataFrame) -> None:
    print()
    print("=== QB GAME AUDIT ===")
    print(f"Rows: {len(qb)}")
    print(f"Columns: {len(qb.columns)}")

    print()
    print("Season types:")
    print(
        qb["season_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("Rows by season:")
    print(
        qb.groupby("season")
        .size()
        .to_string()
    )

    print()
    print("Primary QB rows by season:")
    print(
        qb.loc[qb["primary_qb"].eq(1)]
        .groupby("season")
        .size()
        .to_string()
    )

    duplicates = qb.duplicated(
        [
            "game_id",
            "posteam",
            "qb_player_id",
        ]
    ).sum()

    print()
    print(
        f"QB/team/game duplicates: "
        f"{duplicates}"
    )

    # ---------------------------------------------------------
    # Primary-QB uniqueness
    # ---------------------------------------------------------

    primary_counts = (
        qb.loc[qb["primary_qb"].eq(1)]
        .groupby(
            ["game_id", "posteam"]
        )
        .size()
    )

    multiple_primary = int(
        (primary_counts > 1).sum()
    )

    print(
        "Team-games with multiple primary QBs:",
        multiple_primary,
    )

    team_games = (
        qb[
            ["game_id", "posteam"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    primary_team_games = (
        qb.loc[qb["primary_qb"].eq(1)]
        [["game_id", "posteam"]]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "Team-games:",
        team_games,
    )

    print(
        "Team-games with a primary QB:",
        primary_team_games,
    )

    # ---------------------------------------------------------
    # Scramble audit
    # ---------------------------------------------------------

    print()
    print("=== SCRAMBLE AUDIT ===")

    print(
        "Total QB scrambles:",
        int(qb["scrambles"].sum()),
    )

    print(
        "QB rows with >=1 scramble:",
        int(qb["scrambles"].gt(0).sum()),
    )

    print(
        "Primary QB scrambles:",
        int(
            qb.loc[
                qb["primary_qb"].eq(1),
                "scrambles",
            ].sum()
        ),
    )

    print()
    print("Dropback distribution:")
    print(
        qb["dropbacks"]
        .describe()
        .to_string()
    )

    primary = (
        qb.loc[
            qb["primary_qb"].eq(1)
        ]
        .copy()
    )

    print()
    print("Primary QB efficiency:")

    efficiency_cols = [
        "epa_per_dropback",
        "success_rate",
        "cpoe",
        "sack_rate",
        "int_rate",
        "pass_td_rate",
        "scramble_rate",
        "scramble_epa_per_dropback",
        "scramble_yards_per_scramble",
        "yards_per_dropback",
        "dropback_share",
    ]

    print(
        primary[efficiency_cols]
        .describe()
        .round(4)
        .to_string()
    )

    # ---------------------------------------------------------
    # High-scramble QB sanity check
    # ---------------------------------------------------------

    print()
    print("Highest primary-QB scramble rates:")

    scramble_sample = (
        primary.loc[
            primary["dropbacks"].ge(20),
            [
                "season",
                "week",
                "posteam",
                "qb_player_name",
                "dropbacks",
                "scrambles",
                "scramble_rate",
                "scramble_epa_per_dropback",
            ],
        ]
        .sort_values(
            "scramble_rate",
            ascending=False,
        )
        .head(20)
    )

    print(
        scramble_sample.to_string(
            index=False
        )
    )

    print()
    print("Sample primary QBs:")

    sample_cols = [
        "season",
        "season_type",
        "week",
        "game_id",
        "posteam",
        "qb_player_name",
        "dropbacks",
        "dropback_share",
        "epa_per_dropback",
        "success_rate",
        "cpoe",
        "sack_rate",
        "int_rate",
        "scramble_rate",
    ]

    print(
        primary[sample_cols]
        .head(20)
        .to_string(index=False)
    )

    starters = (
        qb.loc[qb["started"].eq(1)]
        .groupby(["game_id", "posteam"])
        .size()
    )

    if not (starters.eq(1).all() and len(starters) == team_games):
        raise AssertionError(
            "Every team-game must have exactly one starting QB."
        )

    print(
        "Starter == primary QB (by dropbacks):",
        f"{qb.loc[qb['started'].eq(1), 'primary_qb'].mean():.3%}",
    )

    # Hard failures for structural problems.
    if duplicates != 0:
        raise AssertionError(
            "QB/team/game duplicates detected."
        )

    if multiple_primary != 0:
        raise AssertionError(
            "Multiple primary QBs detected "
            "for a team-game."
        )

    if primary_team_games != team_games:
        raise AssertionError(
            "At least one team-game has no "
            "primary QB."
        )


def save(
    qb: pd.DataFrame,
    path: Path = OUT_PATH,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    qb.to_parquet(
        path,
        index=False,
    )

    print()
    print(f"Saved: {path}")


if __name__ == "__main__":
    season_cfg = CONFIG["seasons"]

    if isinstance(
        season_cfg,
        dict,
    ):
        seasons = list(
            range(
                int(season_cfg["start"]),
                int(season_cfg["end"]) + 1,
            )
        )
    else:
        seasons = sorted(
            int(s)
            for s in season_cfg
        )

    print(
        f"Loading NFL play-by-play: "
        f"{min(seasons)}-{max(seasons)}"
    )

    pbp = load_pbp(
        seasons,
        regular_season_only=False,
    )

    qb = build_qb_game(pbp)

    audit(qb)
    save(qb)
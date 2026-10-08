"""Walk-forward benchmark for opponent-adjusted EPA."""

from __future__ import annotations

import pandas as pd

from src.config import configured_path
from src.models.walk_forward import (
    run_walk_forward,
    summarize_predictions,
    summarize_by_season,
)


FEATURE_SETS = {

    # Current raw EPA control.
    "std_epa": [
        "diff_off_epa_per_play_std",
        "diff_def_allowed_epa_per_play_std",
    ],

    # Opponent-adjusted EPA only.
    "oa_std": [
        "diff_oa_off_epa_std",
        "diff_oa_def_epa_std",
    ],

    # Raw EPA + opponent-adjusted EPA.
    "std_plus_oa": [
        "diff_off_epa_per_play_std",
        "diff_def_allowed_epa_per_play_std",
        "diff_oa_off_epa_std",
        "diff_oa_def_epa_std",
    ],

    # Raw + OA season-long + raw + OA recent 8.
    "std_plus_oa_recent8": [
        "diff_off_epa_per_play_std",
        "diff_def_allowed_epa_per_play_std",
        "diff_oa_off_epa_std",
        "diff_oa_def_epa_std",

        "diff_off_epa_per_play_last8",
        "diff_def_allowed_epa_per_play_last8",
        "diff_oa_off_epa_last8",
        "diff_oa_def_epa_last8",
    ],
}


if __name__ == "__main__":

    path = (
        configured_path("processed")
        / "historical_matchups_oa.parquet"
    )

    data = pd.read_parquet(path)

    # -------------------------------------------------
    # Use ONE common evaluation population.
    #
    # This is critical: OA models must not be tested
    # on a different set of games than the std baseline.
    # -------------------------------------------------

    required = sorted({
        feature
        for features in FEATURE_SETS.values()
        for feature in features
    })

    eligible = data.loc[
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

    eligible = eligible.dropna(
        subset=required
    ).copy()

    print(
        "Common eligible games:",
        len(eligible),
    )

    print(
        "Common postseason games:",
        int(
            eligible[
                "is_postseason"
            ].sum()
        ),
    )

    frames = []

    for name, features in FEATURE_SETS.items():

        print(
            f"Running: {name}"
        )

        predictions = run_walk_forward(
            eligible,
            feature_set_name=name,
            feature_columns=features,
        )

        frames.append(predictions)

    predictions = pd.concat(
        frames,
        ignore_index=True,
    )

    overall = summarize_predictions(
        predictions,
        postseason_only=False,
    )

    postseason = summarize_predictions(
        predictions,
        postseason_only=True,
    )

    by_season = summarize_by_season(
        predictions
    )

    output_dir = configured_path(
        "outputs"
    )

    predictions.to_parquet(
        output_dir
        / "oa_walk_forward_predictions.parquet",
        index=False,
    )

    overall.to_csv(
        output_dir
        / "oa_walk_forward_overall.csv",
        index=False,
    )

    postseason.to_csv(
        output_dir
        / "oa_walk_forward_postseason.csv",
        index=False,
    )

    by_season.to_csv(
        output_dir
        / "oa_walk_forward_by_season.csv",
        index=False,
    )

    print()
    print(
        "=== OA WALK-FORWARD OVERALL ==="
    )

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "=== OA WALK-FORWARD POSTSEASON ==="
    )

    print(
        postseason.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "=== OA WALK-FORWARD BY SEASON ==="
    )

    print(
        by_season.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
"""Walk-forward benchmark of every feature set against the closing market.

Rules enforced here (see src/models/evaluation.py):
  * Every model is scored on ONE common population of games.
  * Each test season is predicted using only earlier seasons.
  * Log loss is primary; the market is a comparison, never an input.

Usage:
    python -m src.models.benchmark
    python -m src.models.benchmark --min-test-season 2021
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import configured_path
from src.data.schedules import last_complete_season
from src.models.backtest import FEATURE_SETS as RAW_FEATURE_SETS
from src.models.backtest_opponent_adjusted import (
    FEATURE_SETS as OA_FEATURE_SETS,
)
from src.models.compare import compare_to_market
from src.models.walk_forward import run_walk_forward


def all_feature_sets() -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}

    for name, features in {**RAW_FEATURE_SETS, **OA_FEATURE_SETS}.items():
        if features:
            sets[name] = features

    sets["home_only"] = []

    return sets


def common_population(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """Games where both teams have history and every feature exists."""

    required = sorted(
        {c for cols in feature_sets.values() for c in cols}
    )

    eligible = data.loc[
        data["home_games_played_before"].fillna(0).gt(1)
        & data["away_games_played_before"].fillna(0).gt(1)
    ]

    return eligible.dropna(subset=required).copy()


def run_all(
    population: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    frames = []

    for name, features in feature_sets.items():
        print(f"  walk-forward: {name}")

        frames.append(
            run_walk_forward(
                population,
                feature_set_name=name,
                feature_columns=features,
            )
        )

    return pd.concat(frames, ignore_index=True)


def history_experiment(
    population: pd.DataFrame,
    features: list[str],
    *,
    first_season_options: tuple[int, ...],
    score_from: int,
) -> pd.DataFrame:
    """Does more history help? Score the SAME games with different training windows."""

    frames = []

    for first in first_season_options:
        window = population.loc[population["season"] >= first]

        pred = run_walk_forward(
            window,
            feature_set_name=f"std_epa | train from {first}",
            feature_columns=features,
        )

        frames.append(pred.loc[pred["season"] >= score_from])

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-test-season", type=int, default=None)
    args = parser.parse_args()

    processed = configured_path("processed")
    outputs = configured_path("outputs")
    outputs.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(processed / "historical_matchups_oa.parquet")
    data = data.loc[data["season"] <= last_complete_season()]
    games = pd.read_parquet(processed / "games.parquet")

    feature_sets = all_feature_sets()
    population = common_population(data, feature_sets)

    print(
        f"Common population: {len(population)} games, "
        f"{int(population['is_postseason'].sum())} postseason, "
        f"seasons {population['season'].min()}-{population['season'].max()}"
    )

    predictions = run_all(population, feature_sets)
    predictions.to_parquet(
        outputs / "benchmark_predictions.parquet",
        index=False,
    )

    table = compare_to_market(
        predictions,
        games,
        min_test_season=args.min_test_season,
    )
    table.to_csv(outputs / "benchmark_vs_market.csv", index=False)

    show = [
        "scope", "feature_set", "games", "model_log_loss",
        "market_log_loss", "delta_vs_market", "delta_ci_low",
        "delta_ci_high", "model_auc", "market_auc",
    ]

    for scope in ("all_games", "postseason"):
        print()
        print(f"=== {scope.upper()} (test seasons "
              f">= {args.min_test_season or population['season'].min() + 1}) ===")
        print(
            table.loc[table["scope"].eq(scope), show]
            .drop(columns="scope")
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    # ---- does more history help? same 2021+ games, different training windows
    print()
    print("=== HISTORY EXPERIMENT (std_epa, scored on 2021+ games) ===")

    hist = history_experiment(
        population,
        ["diff_off_epa_per_play_std", "diff_def_allowed_epa_per_play_std"],
        first_season_options=(2020, 2016, 2012, 2009),
        score_from=2021,
    )

    hist_table = compare_to_market(hist, games)
    hist_table.to_csv(outputs / "history_experiment.csv", index=False)

    print(
        hist_table.loc[
            hist_table["scope"].isin(["all_games", "postseason"]),
            ["scope", "feature_set", "games", "model_log_loss",
             "market_log_loss", "delta_vs_market", "delta_ci_low",
             "delta_ci_high"],
        ]
        .sort_values(["scope", "feature_set"])
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()

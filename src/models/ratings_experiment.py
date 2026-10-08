"""Tune the rating filter on an early window, then score it on untouched seasons.

Protocol (prevents fitting hyperparameters to the games I report):
  * TUNE_SEASONS: the dev window; its walk-forward log loss chooses parameters.
  * HOLDOUT_SEASONS: frozen parameters are scored here; nothing was tuned on
    these outcomes.
  * Season 2009 is cold start (no prior season), so it only ever trains.

Usage:
    python -m src.models.ratings_experiment
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from src.config import configured_path
from src.data.schedules import last_complete_season
from src.features.ratings import (
    RatingParams,
    attach_ratings_to_matchups,
    build_team_ratings,
)
from src.models.compare import compare_to_market, per_game_log_loss
from src.models.walk_forward import run_walk_forward


# Protocol (history extended to 1999 on 2026-10-06):
#   dev window  = test seasons 2002-2016 (~4,000 games): ideas are SELECTED here
#   holdout     = 2017-2025: scored to confirm, never used to choose
# 1999 is a cold start for the rating filter, so modeling starts in 2000.
TUNE_SEASONS = range(2002, 2017)
HOLDOUT_FIRST_SEASON = 2017
FIRST_MODEL_SEASON = 2000

NO_QB_FEATURES = ["diff_rating_off", "diff_rating_def"]
PRIMARY_FEATURES = ["diff_rating_off_total", "diff_rating_def"]

# Coarse grids: the goal is a flat, robust optimum, not a razor-sharp one.
GRIDS: dict[str, list[float]] = {
    "carryover_off": [0.30, 0.45, 0.55, 0.65, 0.80],
    "carryover_def": [0.15, 0.30, 0.40, 0.55, 0.70],
    "season_var": [0.0010, 0.0025, 0.0050, 0.0100],
    "drift_var": [0.0000, 0.0002, 0.0004, 0.0008, 0.0016],
    "obs_var": [0.025, 0.035, 0.050, 0.075],
    "carryover_qb": [0.60, 0.75, 0.85, 0.95],
    "qb_prior_var": [0.002, 0.004, 0.008, 0.016],
    "qb_season_var": [0.0005, 0.0015, 0.0030],
    "qb_drift_var": [0.0, 0.0001, 0.0003],
}

QB_GRID_NAMES = {"carryover_qb", "qb_prior_var", "qb_season_var", "qb_drift_var"}


def _qb_changed(ratings: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
    """True when either team starts a different QB than in its previous game."""
    r = ratings.sort_values(["team", "season", "week"]).copy()
    r["prev_qb"] = r.groupby("team")["pregame_qb_id"].shift(1)
    r["changed"] = r["prev_qb"].notna() & r["pregame_qb_id"].ne(r["prev_qb"])

    flags = r.set_index(["game_id", "team"])["changed"]

    home = pd.Series(
        flags.reindex(list(zip(data["game_id"], data["home_team"]))).to_numpy(),
        index=data.index,
    ).fillna(False)
    away = pd.Series(
        flags.reindex(list(zip(data["game_id"], data["away_team"]))).to_numpy(),
        index=data.index,
    ).fillna(False)

    return (home | away).astype(bool)


class Context:
    """Loads inputs once so every trial only rebuilds the ratings."""

    def __init__(self) -> None:
        processed = configured_path("processed")

        self.team_game = pd.read_parquet(processed / "team_game_epa.parquet")
        self.games = pd.read_parquet(processed / "games.parquet")
        self.played = self.games.loc[self.games["is_played"]]
        self.matchups = pd.read_parquet(
            processed / "historical_matchups_oa.parquet"
        )
        self.qb_game = pd.read_parquet(processed / "qb_game.parquet")

    def featurize(
        self,
        params: RatingParams,
        *,
        use_qb: bool = True,
    ) -> pd.DataFrame:
        ratings = build_team_ratings(
            self.team_game,
            self.played,
            params,
            self.qb_game if use_qb else None,
        )
        data = attach_ratings_to_matchups(self.matchups, ratings)

        # flag games where either starting QB differs from his team's previous start
        data["qb_changed"] = _qb_changed(ratings, data)

        return data.loc[
            data["season"].between(FIRST_MODEL_SEASON, last_complete_season())
        ].copy()


def window_log_loss(
    data: pd.DataFrame,
    features: list[str],
    seasons: range,
) -> float:
    pred = run_walk_forward(
        data,
        feature_set_name="trial",
        feature_columns=features,
    )

    pred = pred.loc[pred["season"].isin(list(seasons))]

    return float(
        per_game_log_loss(
            pred["home_win"].astype(int).to_numpy(),
            pred["p_home_win"].to_numpy(),
        ).mean()
    )


def coordinate_descent(
    ctx: Context,
    start: RatingParams,
    *,
    passes: int = 2,
) -> tuple[RatingParams, pd.DataFrame]:
    """Tune one parameter at a time on the TUNE window only."""

    best = start
    best_loss = window_log_loss(
        ctx.featurize(best), PRIMARY_FEATURES, TUNE_SEASONS
    )

    trials = [{"pass": 0, "param": "start", "value": np.nan, "loss": best_loss}]

    print(f"start  tune-window log loss = {best_loss:.5f}")

    for pass_number in range(1, passes + 1):
        for name, grid in GRIDS.items():
            scored = []

            for value in grid:
                trial = replace(best, **{name: value})
                loss = window_log_loss(
                    ctx.featurize(trial), PRIMARY_FEATURES, TUNE_SEASONS
                )
                scored.append((loss, value))
                trials.append(
                    {"pass": pass_number, "param": name, "value": value, "loss": loss}
                )

            loss, value = min(scored)

            if loss < best_loss:
                best, best_loss = replace(best, **{name: value}), loss

            curve = "  ".join(f"{v:g}:{l:.5f}" for l, v in sorted(scored, key=lambda t: t[1]))
            print(f"pass {pass_number} {name:<14} best={value:<7g} | {curve}")

    print(f"final  tune-window log loss = {best_loss:.5f}")

    return best, pd.DataFrame(trials)


def main() -> None:
    started = time.time()
    outputs = configured_path("outputs")
    outputs.mkdir(parents=True, exist_ok=True)

    ctx = Context()

    default = RatingParams()
    tuned, trials = coordinate_descent(ctx, default)
    trials.to_csv(outputs / "ratings_tuning_trials.csv", index=False)

    with open(outputs / "ratings_params_frozen.json", "w") as f:
        json.dump(asdict(tuned), f, indent=2)

    print()
    print("Default:", asdict(default))
    print("Tuned  :", asdict(tuned))

    # ------------------------------------------------------------------
    # Frozen evaluation on the holdout seasons
    # ------------------------------------------------------------------
    data_qb = ctx.featurize(tuned, use_qb=True)
    data_noqb = ctx.featurize(default, use_qb=False)

    frames = [
        run_walk_forward(
            data_qb,
            feature_set_name="ratings + QB (tuned, frozen)",
            feature_columns=PRIMARY_FEATURES,
        ),
        run_walk_forward(
            ctx.featurize(default, use_qb=True),
            feature_set_name="ratings + QB (untuned defaults)",
            feature_columns=PRIMARY_FEATURES,
        ),
        run_walk_forward(
            data_qb,
            feature_set_name="ratings + QB (QB as own feature)",
            feature_columns=["diff_rating_off", "diff_qb", "diff_rating_def"],
        ),
        run_walk_forward(
            data_noqb,
            feature_set_name="ratings, no QB (previous model)",
            feature_columns=NO_QB_FEATURES,
        ),
        # Baseline: season-to-date raw EPA (no cross-season prior).
        run_walk_forward(
            data_qb,
            feature_set_name="baseline std_epa",
            feature_columns=[
                "diff_off_epa_per_play_std",
                "diff_def_allowed_epa_per_play_std",
            ],
        ),
        run_walk_forward(
            data_qb,
            feature_set_name="home_only",
            feature_columns=[],
        ),
    ]

    data = data_qb
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_parquet(
        outputs / "ratings_experiment_predictions.parquet", index=False
    )

    games = ctx.games

    show = [
        "feature_set", "games", "model_log_loss", "market_log_loss",
        "delta_vs_market", "delta_ci_low", "delta_ci_high",
        "model_auc", "market_auc",
    ]

    results = []

    flags = data[["game_id", "qb_changed"]]

    for label, frame in (
        ("ALL GAMES incl. week 1-2", predictions),
        (
            "QB-CHANGE games (a team starts a different QB than last game)",
            predictions.merge(flags, on="game_id").pipe(
                lambda d: d.loc[d["qb_changed"]]
            ),
        ),
        (
            "ESTABLISHED (both teams >1 game this season)",
            predictions.merge(
                data[["game_id", "home_games_played_before", "away_games_played_before"]],
                on="game_id",
            ).pipe(
                lambda d: d.loc[
                    d["home_games_played_before"].fillna(0).gt(1)
                    & d["away_games_played_before"].fillna(0).gt(1)
                ]
            ),
        ),
    ):
        table = compare_to_market(
            frame, games, min_test_season=HOLDOUT_FIRST_SEASON
        )
        results.append(table.assign(population=label))

        for scope in ("all_games", "postseason"):
            print()
            print(
                f"=== HOLDOUT {HOLDOUT_FIRST_SEASON}-2025 | {label} | {scope} ==="
            )
            print(
                table.loc[table["scope"].eq(scope), show]
                .to_string(index=False, float_format=lambda x: f"{x:.4f}")
            )

    pd.concat(results).to_csv(
        outputs / "ratings_experiment_vs_market.csv", index=False
    )

    print(f"\nDone in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

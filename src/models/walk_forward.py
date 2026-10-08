"""Chronological walk-forward validation.

Every test season is predicted using ONLY seasons that occurred before it.

Example:
    2021 <- train 2020
    2022 <- train 2020-2021
    2023 <- train 2020-2022
    2024 <- train 2020-2023
    2025 <- train 2020-2024

This approximates how the model would have behaved in real deployment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from src.config import configured_path
from src.models.backtest import (
    FEATURE_SETS,
    build_logistic_model,
    evaluate_predictions,
)


def run_walk_forward(
    data: pd.DataFrame,
    *,
    feature_set_name: str,
    feature_columns: list[str],
    min_train_seasons: int = 1,
    model_factory=None,
    half_life: float | None = None,
) -> pd.DataFrame:
    """Generate chronological out-of-time predictions."""

    seasons = sorted(
        data["season"]
        .dropna()
        .astype(int)
        .unique()
    )

    rows: list[pd.DataFrame] = []

    for test_season in seasons:

        train_seasons = [
            season
            for season in seasons
            if season < test_season
        ]

        if len(train_seasons) < min_train_seasons:
            continue

        train = data.loc[
            data["season"].isin(train_seasons)
        ].copy()

        test = data.loc[
            data["season"] == test_season
        ].copy()

        if train.empty or test.empty:
            continue

        y_train = (
            train["home_win"]
            .astype(int)
            .to_numpy()
        )

        if not feature_columns:

            p_home = float(
                np.clip(
                    y_train.mean(),
                    1e-6,
                    1 - 1e-6,
                )
            )

            probabilities = np.full(
                len(test),
                p_home,
                dtype=float,
            )

        else:

            missing = (
                set(feature_columns)
                - set(data.columns)
            )

            if missing:
                raise ValueError(
                    f"{feature_set_name} missing features: "
                    f"{sorted(missing)}"
                )

            model = (model_factory or build_logistic_model)(
                feature_columns
            )

            if half_life is None:
                model.fit(train, y_train)
            else:
                # exponentially down-weight older seasons: weight 0.5 every `half_life` seasons
                age = (test_season - 1) - train["season"].to_numpy()
                model.fit(train, y_train, model__sample_weight=0.5 ** (age / half_life))

            probabilities = (
                model.predict_proba(test)[:, 1]
            )

        fold = test[
            [
                "season",
                "week",
                "game_id",
                "game_type",
                "is_postseason",
                "home_team",
                "away_team",
                "home_win",
            ]
        ].copy()

        fold["feature_set"] = feature_set_name
        fold["train_start"] = min(train_seasons)
        fold["train_end"] = max(train_seasons)
        fold["n_train_seasons"] = len(
            train_seasons
        )

        fold["p_home_win"] = probabilities

        fold["pred_home_win"] = (
            probabilities >= 0.5
        ).astype(int)

        rows.append(fold)

    if not rows:
        raise ValueError(
            "Walk-forward produced no folds."
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def calibration_stats(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Estimate calibration intercept and slope.

    Perfect calibration:
        intercept ~= 0
        slope ~= 1
    """

    y = np.asarray(
        y_true,
        dtype=int,
    )

    p = np.asarray(
        probabilities,
        dtype=float,
    )

    p = np.clip(
        p,
        1e-6,
        1 - 1e-6,
    )

    logits = np.log(
        p / (1.0 - p)
    ).reshape(-1, 1)

    # Calibration intercept:
    # fit an intercept while holding the original
    # prediction logit coefficient at 1.
    #
    # sklearn does not support offsets directly,
    # so estimate intercept numerically.
    def mean_residual(intercept: float) -> float:
        adjusted = 1.0 / (
            1.0
            + np.exp(
                -(intercept + logits[:, 0])
            )
        )

        return float(
            np.mean(y - adjusted)
        )

    lo = -10.0
    hi = 10.0

    for _ in range(100):
        mid = (lo + hi) / 2.0

        residual = mean_residual(mid)

        if residual > 0:
            lo = mid
        else:
            hi = mid

    intercept = (
        lo + hi
    ) / 2.0

    # Calibration slope:
    # logistic regression of outcome on
    # predicted log odds.
    slope_model = LogisticRegression(
        C=1e6,
        max_iter=5000,
    )

    slope_model.fit(
        logits,
        y,
    )

    slope = float(
        slope_model.coef_[0, 0]
    )

    return {
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error."""

    y = np.asarray(
        y_true,
        dtype=int,
    )

    p = np.asarray(
        probabilities,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    bin_ids = np.digitize(
        p,
        edges[1:-1],
        right=False,
    )

    ece = 0.0

    for bin_id in range(bins):

        mask = bin_ids == bin_id

        if not mask.any():
            continue

        observed = float(
            y[mask].mean()
        )

        predicted = float(
            p[mask].mean()
        )

        weight = (
            mask.sum() / len(y)
        )

        ece += weight * abs(
            observed - predicted
        )

    return float(ece)


def summarize_predictions(
    predictions: pd.DataFrame,
    *,
    postseason_only: bool = False,
) -> pd.DataFrame:
    """Summarize probability quality."""

    data = predictions.copy()

    if postseason_only:
        data = data.loc[
            data["is_postseason"]
        ].copy()

    rows = []

    for feature_set, group in data.groupby(
        "feature_set",
        sort=False,
    ):

        y = (
            group["home_win"]
            .astype(int)
            .to_numpy()
        )

        p = (
            group["p_home_win"]
            .astype(float)
            .to_numpy()
        )

        metrics = evaluate_predictions(
            y,
            p,
        )

        calibration = calibration_stats(
            y,
            p,
        )

        rows.append({
            "feature_set": feature_set,
            "games": len(group),
            **metrics,
            **calibration,
            "ece_10": expected_calibration_error(
                y,
                p,
                bins=10,
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("log_loss")
        .reset_index(drop=True)
    )


def summarize_by_season(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        feature_set,
        season,
    ), group in predictions.groupby(
        [
            "feature_set",
            "season",
        ]
    ):

        metrics = evaluate_predictions(
            group["home_win"].to_numpy(),
            group["p_home_win"].to_numpy(),
        )

        rows.append({
            "feature_set": feature_set,
            "season": int(season),
            "games": len(group),
            **metrics,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "season",
                "log_loss",
            ]
        )
        .reset_index(drop=True)
    )


if __name__ == "__main__":

    matchup_path = (
        configured_path("processed")
        / "historical_matchups.parquet"
    )

    data = pd.read_parquet(
        matchup_path
    )

    # EPA models require both teams to have
    # at least one previous game.
    epa_data = data.loc[
        (
            data[
                "home_games_played_before"
            ].fillna(0) > 0
        )
        & (
            data[
                "away_games_played_before"
            ].fillna(0) > 0
        )
    ].copy()

    frames = []

    # ------------------------------------------
    # Home-field baseline
    # ------------------------------------------

    frames.append(
        run_walk_forward(
            data,
            feature_set_name="home_only",
            feature_columns=[],
        )
    )

    # ------------------------------------------
    # EPA models
    # ------------------------------------------

    for (
        name,
        features,
    ) in FEATURE_SETS.items():

        if name == "home_only":
            continue

        frames.append(
            run_walk_forward(
                epa_data,
                feature_set_name=name,
                feature_columns=features,
            )
        )

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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        output_dir
        / "epa_walk_forward_predictions.parquet",
        index=False,
    )

    overall.to_csv(
        output_dir
        / "epa_walk_forward_overall.csv",
        index=False,
    )

    postseason.to_csv(
        output_dir
        / "epa_walk_forward_postseason.csv",
        index=False,
    )

    by_season.to_csv(
        output_dir
        / "epa_walk_forward_by_season.csv",
        index=False,
    )

    print()
    print(
        "=== WALK-FORWARD OVERALL ==="
    )

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "=== WALK-FORWARD POSTSEASON ==="
    )

    print(
        postseason.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "=== WALK-FORWARD BY SEASON ==="
    )

    print(
        by_season.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "Saved walk-forward outputs."
    )
"""Season-held-out backtesting for NFL win-probability models.

Each season is predicted by models trained without that season.

This module establishes the permanent evaluation framework used to
compare feature families throughout the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import configured_path


# ---------------------------------------------------------
# Feature families
# ---------------------------------------------------------

FEATURE_SETS: dict[str, list[str]] = {
    "home_only": [],

    "std_epa": [
        "diff_off_epa_per_play_std",
        "diff_def_allowed_epa_per_play_std",
    ],

    "recent4_epa": [
        "diff_off_epa_per_play_last4",
        "diff_def_allowed_epa_per_play_last4",
    ],

    "recent8_epa": [
        "diff_off_epa_per_play_last8",
        "diff_def_allowed_epa_per_play_last8",
    ],

    "std_plus_recent": [
        "diff_off_epa_per_play_std",
        "diff_def_allowed_epa_per_play_std",
        "diff_off_epa_per_play_last4",
        "diff_def_allowed_epa_per_play_last4",
        "diff_off_epa_per_play_last8",
        "diff_def_allowed_epa_per_play_last8",
    ],

    "expanded_epa": [
        # Overall
        "diff_off_epa_per_play_std",
        "diff_def_allowed_epa_per_play_std",

        # Pass
        "diff_off_pass_epa_per_play_std",
        "diff_def_allowed_pass_epa_per_play_std",

        # Rush
        "diff_off_rush_epa_per_play_std",
        "diff_def_allowed_rush_epa_per_play_std",

        # Success
        "diff_off_success_rate_std",
        "diff_def_allowed_success_rate_std",

        # Explosiveness
        "diff_off_explosive_pass_rate_std",
        "diff_def_allowed_explosive_pass_rate_std",
        "diff_off_explosive_rush_rate_std",
        "diff_def_allowed_explosive_rush_rate_std",

        # Recent overall form
        "diff_off_epa_per_play_last4",
        "diff_def_allowed_epa_per_play_last4",
        "diff_off_epa_per_play_last8",
        "diff_def_allowed_epa_per_play_last8",
    ],
}


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Evaluate probabilistic predictions."""

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(
            y_true,
            probabilities,
        )
    else:
        metrics["auc"] = np.nan

    return metrics


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def build_logistic_model(
    feature_columns: list[str],
) -> Pipeline:
    """Regularized logistic regression."""

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                feature_columns,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        C=1.0,
        max_iter=5000,
        random_state=7,
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                preprocessing,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ---------------------------------------------------------
# LOSO backtest
# ---------------------------------------------------------

def run_loso_backtest(
    data: pd.DataFrame,
    *,
    feature_set_name: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Leave one complete season out at a time."""

    seasons = sorted(
        data["season"]
        .dropna()
        .astype(int)
        .unique()
    )

    rows: list[pd.DataFrame] = []

    for test_season in seasons:

        train = data.loc[
            data["season"] != test_season
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

        # ---------------------------------------------
        # Home-only baseline
        # ---------------------------------------------

        if not feature_columns:

            # Estimate historical home-win probability
            # using training seasons only.
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

        # ---------------------------------------------
        # Feature model
        # ---------------------------------------------

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

            model = build_logistic_model(
                feature_columns
            )

            model.fit(
                train,
                y_train,
            )

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

        fold["feature_set"] = (
            feature_set_name
        )

        fold["p_home_win"] = (
            probabilities
        )

        fold["pred_home_win"] = (
            fold["p_home_win"] >= 0.5
        ).astype(int)

        rows.append(fold)

    if not rows:
        raise ValueError(
            "Backtest produced no folds."
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ---------------------------------------------------------
# Reporting
# ---------------------------------------------------------

def summarize_backtest(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    overall_rows = []
    season_rows = []

    for feature_set, group in predictions.groupby(
        "feature_set",
        sort=False,
    ):

        metrics = evaluate_predictions(
            group["home_win"].to_numpy(),
            group["p_home_win"].to_numpy(),
        )

        overall_rows.append({
            "feature_set": feature_set,
            "games": len(group),
            **metrics,
        })

        for season, season_group in group.groupby(
            "season"
        ):

            season_metrics = evaluate_predictions(
                season_group[
                    "home_win"
                ].to_numpy(),
                season_group[
                    "p_home_win"
                ].to_numpy(),
            )

            season_rows.append({
                "feature_set": feature_set,
                "season": int(season),
                "games": len(season_group),
                **season_metrics,
            })

    overall = pd.DataFrame(
        overall_rows
    ).sort_values(
        "log_loss"
    ).reset_index(drop=True)

    by_season = pd.DataFrame(
        season_rows
    ).sort_values(
        [
            "season",
            "log_loss",
        ]
    ).reset_index(drop=True)

    return overall, by_season


def summarize_postseason(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate only held-out postseason games."""

    post = predictions.loc[
        predictions["is_postseason"]
    ].copy()

    rows = []

    for feature_set, group in post.groupby(
        "feature_set",
        sort=False,
    ):

        metrics = evaluate_predictions(
            group["home_win"].to_numpy(),
            group["p_home_win"].to_numpy(),
        )

        rows.append({
            "feature_set": feature_set,
            "games": len(group),
            **metrics,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("log_loss")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    path = (
        configured_path("processed")
        / "historical_matchups.parquet"
    )

    data = pd.read_parquet(path)

    # Week 1 intentionally lacks within-season history.
    # Keep it for the home-only baseline but exclude it
    # from EPA models so the test measures real EPA signal
    # rather than median-imputed zero-history observations.
    epa_data = data.loc[
    (
        data["home_games_played_before"].fillna(0) > 0
    )
    & (
        data["away_games_played_before"].fillna(0) > 0
    )
].copy()

    prediction_frames = []

    # Home-only benchmark uses every game.
    prediction_frames.append(
        run_loso_backtest(
            data,
            feature_set_name="home_only",
            feature_columns=[],
        )
    )

    # EPA models use games where pregame history exists.
    for name, features in FEATURE_SETS.items():

        if name == "home_only":
            continue

        prediction_frames.append(
            run_loso_backtest(
                epa_data,
                feature_set_name=name,
                feature_columns=features,
            )
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    overall, by_season = (
        summarize_backtest(
            predictions
        )
    )

    postseason = summarize_postseason(
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
        / "epa_backtest_predictions.parquet",
        index=False,
    )

    overall.to_csv(
        output_dir
        / "epa_backtest_overall.csv",
        index=False,
    )

    by_season.to_csv(
        output_dir
        / "epa_backtest_by_season.csv",
        index=False,
    )

    postseason.to_csv(
        output_dir
        / "epa_backtest_postseason.csv",
        index=False,
    )

    print()
    print("=== OVERALL LOSO ===")
    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=== POSTSEASON ONLY ===")
    print(
        postseason.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Saved backtest outputs.")
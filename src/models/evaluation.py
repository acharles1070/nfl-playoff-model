"""Canonical evaluation policy for the NFL playoff model.

Every future feature family/model must be evaluated under the same
out-of-time rules before being promoted into the production pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationConfig:
    """Fixed project-wide evaluation rules."""

    # Earliest season in the pipeline (config.yaml seasons.start).
    first_season: int = 2009

    # Need historical training data before testing.
    first_walk_forward_test_season: int = 2010

    # Primary model-selection metric.
    primary_metric: str = "log_loss"

    # Secondary probability metric.
    secondary_metric: str = "brier"

    # Diagnostic discrimination metric.
    discrimination_metric: str = "auc"

    # Probability threshold is diagnostic only.
    classification_threshold: float = 0.50

    # Calibration reporting.
    calibration_bins: int = 10

    # Never tune feature/model choices using postseason accuracy alone.
    postseason_is_secondary: bool = True


EVALUATION = EvaluationConfig()


BASELINE_FEATURE_SET = "std_epa"

# Frozen v0 baseline: season-to-date EPA differences, logistic regression,
# walk-forward over test seasons 2010-2025 on the common population in
# src/models/benchmark.py. Any new model must beat this on log loss, and is
# judged against the market rows below, which are never model inputs.
BASELINE_VERSION = "v0-std_epa-2009"

BASELINE_WALK_FORWARD = {
    "games": 3848,
    "log_loss": 0.6439295334,
    "brier": 0.2261766297,
    "accuracy": 0.6275987526,
    "auc": 0.6714236034,
}

BASELINE_POSTSEASON = {
    "games": 188,
    "log_loss": 0.6479329069,
    "brier": 0.2281144294,
    "accuracy": 0.5904255319,
    "auc": 0.6197183099,
}

# Closing moneyline (de-vigged) on the same games. The target to close in on.
MARKET_WALK_FORWARD = {
    "games": 3848,
    "log_loss": 0.6086809778,
    "brier": 0.2106328661,
    "accuracy": 0.6676195426,
    "auc": 0.7224933264,
}

MARKET_POSTSEASON = {
    "games": 188,
    "log_loss": 0.6157759379,
    "brier": 0.2138526412,
    "accuracy": 0.6276595745,
    "auc": 0.6841218250,
}


def print_evaluation_policy() -> None:
    print("Canonical evaluation protocol")
    print("-----------------------------")
    print(
        "Walk-forward:",
        "train only on seasons earlier than test season",
    )
    print(
        "Primary metric:",
        EVALUATION.primary_metric,
    )
    print(
        "Secondary metric:",
        EVALUATION.secondary_metric,
    )
    print(
        "Discrimination:",
        EVALUATION.discrimination_metric,
    )
    print(
        "Baseline feature set:",
        BASELINE_FEATURE_SET,
    )
    print(
        "Postseason sample:",
        BASELINE_POSTSEASON["games"],
        "held-out games",
    )
    print(
        "Market reference log loss (all / postseason):",
        f"{MARKET_WALK_FORWARD['log_loss']:.4f}",
        "/",
        f"{MARKET_POSTSEASON['log_loss']:.4f}",
    )
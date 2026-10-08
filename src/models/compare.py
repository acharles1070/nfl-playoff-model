"""Paired model-vs-market comparison with bootstrap uncertainty.

With ~200 postseason games a 0.02 log-loss gap is indistinguishable from
noise. Every comparison therefore reports a paired bootstrap confidence
interval on the per-game log-loss difference (model minus market, so
NEGATIVE means the model is better).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.market import market_probabilities


EPS = 1e-6


def per_game_log_loss(
    y: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=float)

    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def paired_bootstrap_delta(
    y: np.ndarray,
    p_model: np.ndarray,
    p_reference: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Mean and 95% CI of (model log loss - reference log loss)."""

    diff = (
        per_game_log_loss(y, p_model)
        - per_game_log_loss(y, p_reference)
    )

    rng = np.random.default_rng(seed)

    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)

    return (
        float(diff.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def attach_market(
    predictions: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Add p_market to a predictions table, dropping games without a line."""

    market = market_probabilities(games)[["game_id", "p_market"]]

    out = predictions.merge(
        market,
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    return out.loc[out["p_market"].notna()].copy()


def _summarize(group: pd.DataFrame) -> dict[str, float]:
    y = group["home_win"].astype(int).to_numpy()
    p = group["p_home_win"].astype(float).to_numpy()
    q = group["p_market"].astype(float).to_numpy()

    delta, lo, hi = paired_bootstrap_delta(y, p, q)

    both_classes = len(np.unique(y)) == 2

    return {
        "games": len(group),
        "model_log_loss": float(per_game_log_loss(y, p).mean()),
        "market_log_loss": float(per_game_log_loss(y, q).mean()),
        "delta_vs_market": delta,
        "delta_ci_low": lo,
        "delta_ci_high": hi,
        "model_auc": roc_auc_score(y, p) if both_classes else np.nan,
        "market_auc": roc_auc_score(y, q) if both_classes else np.nan,
        "model_accuracy": float(((p >= 0.5) == y).mean()),
        "market_accuracy": float(((q >= 0.5) == y).mean()),
    }


def compare_to_market(
    predictions: pd.DataFrame,
    games: pd.DataFrame,
    *,
    min_test_season: int | None = None,
) -> pd.DataFrame:
    """Scope x feature_set comparison against the closing market."""

    data = attach_market(predictions, games)

    if min_test_season is not None:
        data = data.loc[data["season"] >= min_test_season]

    scopes = {
        "all_games": data,
        "postseason": data.loc[data["is_postseason"]],
        "regular_season": data.loc[~data["is_postseason"]],
    }

    rows = []

    for scope, frame in scopes.items():
        for feature_set, group in frame.groupby(
            "feature_set",
            sort=False,
        ):
            rows.append(
                {
                    "scope": scope,
                    "feature_set": feature_set,
                    **_summarize(group),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["scope", "model_log_loss"])
        .reset_index(drop=True)
    )

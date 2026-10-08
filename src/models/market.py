"""Betting-market benchmark probabilities.

The closing market is the bar every model in this project is scored against.
It is used ONLY as a comparison, never as a model input.

Primary benchmark: the de-vigged closing moneyline. It is parameter-free, so
it cannot be tuned to flatter or punish any model. Where a moneyline is
missing the closing spread is converted through a normal margin model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# Standard deviation of an NFL game margin around the closing spread.
NFL_MARGIN_SD = 13.86


def moneyline_to_implied(moneyline: pd.Series) -> pd.Series:
    """American odds -> raw implied probability (still contains the vig)."""
    odds = pd.to_numeric(moneyline, errors="coerce").astype(float)

    favorite = odds < 0

    return pd.Series(
        np.where(
            favorite,
            -odds / (-odds + 100.0),
            100.0 / (odds + 100.0),
        ),
        index=moneyline.index,
    ).where(odds.notna())


def devig_two_way(
    home_implied: pd.Series,
    away_implied: pd.Series,
) -> pd.Series:
    """Remove the vig by normalizing the two implied probabilities."""
    return home_implied / (home_implied + away_implied)


def spread_to_probability(
    spread_line: pd.Series,
    sd: float = NFL_MARGIN_SD,
) -> pd.Series:
    """nflverse spread_line > 0 means the HOME team is favored."""
    spread = pd.to_numeric(spread_line, errors="coerce").astype(float)

    return pd.Series(
        norm.cdf(spread / sd),
        index=spread_line.index,
    ).where(spread.notna())


def market_probabilities(games: pd.DataFrame) -> pd.DataFrame:
    """One row per game with the market's home-win probability."""

    required = {
        "game_id",
        "home_moneyline",
        "away_moneyline",
        "spread_line",
    }

    missing = required - set(games.columns)

    if missing:
        raise ValueError(
            f"games missing market columns: {sorted(missing)}"
        )

    p_moneyline = devig_two_way(
        moneyline_to_implied(games["home_moneyline"]),
        moneyline_to_implied(games["away_moneyline"]),
    )

    p_spread = spread_to_probability(games["spread_line"])

    out = pd.DataFrame(
        {
            "game_id": games["game_id"].to_numpy(),
            "p_market_moneyline": p_moneyline.to_numpy(),
            "p_market_spread": p_spread.to_numpy(),
        }
    )

    out["p_market"] = out["p_market_moneyline"].fillna(
        out["p_market_spread"]
    )

    if out["game_id"].duplicated().any():
        raise ValueError("Duplicate game_id in market table.")

    return out

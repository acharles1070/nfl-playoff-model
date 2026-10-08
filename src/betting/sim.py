"""Betting simulation against closing moneylines (a research exercise, not advice).

Closing-line betting is the hardest realistic test of a probability model: the price
already contains every public fact plus the bookmaker's margin (the vig).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decimal_odds(american: pd.Series | np.ndarray) -> np.ndarray:
    a = np.asarray(american, dtype=float)
    return np.where(a > 0, 1.0 + a / 100.0, 1.0 + 100.0 / np.abs(a))


def expected_value(p: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Expected profit per unit staked if the true win probability is p."""
    return p * (dec - 1.0) - (1.0 - p)


def kelly_fraction(p: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Full-Kelly fraction of bankroll: (p*dec - 1) / (dec - 1), floored at zero."""
    return np.maximum((p * dec - 1.0) / (dec - 1.0), 0.0)


def place_bets(
    games: pd.DataFrame,
    *,
    threshold: float,
    mode: str = "flat",
    kelly_mult: float = 0.25,
    cap: float = 0.05,
    bankroll0: float = 100.0,
) -> pd.DataFrame:
    """Bet the side with the larger expected value when it exceeds `threshold`.

    games needs: season, week, game_id, p_home (model), home_moneyline, away_moneyline,
    home_win. Sorted chronologically; Kelly compounds the bankroll bet by bet.
    """
    g = games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    dec_h, dec_a = decimal_odds(g["home_moneyline"]), decimal_odds(g["away_moneyline"])
    p_h = g["p_home"].to_numpy()

    ev_h, ev_a = expected_value(p_h, dec_h), expected_value(1 - p_h, dec_a)
    pick_home = ev_h >= ev_a
    ev = np.where(pick_home, ev_h, ev_a)
    dec = np.where(pick_home, dec_h, dec_a)
    p_win = np.where(pick_home, p_h, 1 - p_h)
    won = np.where(pick_home, g["home_win"], 1 - g["home_win"]).astype(float)

    take = ev > threshold
    bankroll, rows = bankroll0, []

    for i in np.flatnonzero(take):
        if mode == "flat":
            stake = 1.0
        else:
            stake = bankroll * min(kelly_mult * kelly_fraction(p_win[i], dec[i]), cap)
        pnl = stake * (dec[i] - 1.0) if won[i] else -stake
        bankroll += pnl
        rows.append({"season": g.loc[i, "season"], "game_id": g.loc[i, "game_id"], "side": "home" if pick_home[i] else "away",
                     "ev": ev[i], "stake": stake, "pnl": pnl, "ret": pnl / stake, "bankroll": bankroll})
    return pd.DataFrame(rows)

"""Opponent forecasts from the rating model: a win probability for a game weeks away, and a band for how far it will move.

The game model is the pinned champion's (`ratings_qb_logit_v1`): a logistic regression on the home-minus-away
difference in offense (team plus quarterback) and defense ratings. Nothing here refits or changes it. This module
only reads rating rows and the fitted model, so the pregame-only guarantee of the rating filter carries over.

Two things are added for a game h weeks ahead, both measured in src/opponents/horizon.py:

  shrinkage      The model was fitted on pregame ratings (h = 0). Used on ratings frozen h weeks earlier it is
                 overconfident (calibration slope 0.65-0.9). The logit is divided by 1 + a*h, with the single
                 parameter a fitted on the development window (2011-2016) and frozen here.
  revision band  How far the forecast moves between now and kickoff, by horizon: the spread of (the forecast made the
                 week of the game) minus (this forecast), on the development window. It includes what the filter
                 cannot see in advance: a quarterback change, injuries, a hot or cold streak.

The band answers "where will this forecast stand when the game arrives?", not "what is the true probability?".
The rating filter's own posterior (used by the playoff simulator) describes the second question and is much wider.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.simulation.backtest import GameModel

HORIZON_SHRINK_A = 0.0218     # fitted on test seasons 2011-2016 only (python -m src.opponents.horizon); frozen
REPLACEMENT_MAX_PRIOR_STARTS = 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def shrink_logit(logit, horizon, a: float = HORIZON_SHRINK_A):
    """Pull a forecast made `horizon` weeks ahead toward 50-50: logit / (1 + a * horizon)."""
    return np.asarray(logit, dtype=float) / (1.0 + a * np.asarray(horizon, dtype=float))


def forecast_frame(model: GameModel, frame: pd.DataFrame, neutral=None) -> pd.DataFrame:
    """Home win logit and probability for each row of a matchup frame with ratings attached.

    `frame` is the output of attach_ratings_to_matchups. `neutral` is a boolean array (neutral-site games lose
    the home-field term).
    """
    logit = np.asarray(model.home_logit(frame), dtype=float)
    if neutral is not None:
        logit = logit - model.home_logit_constant * np.asarray(neutral, dtype=float)
    return pd.DataFrame({"logit": logit, "p_home": sigmoid(logit)}, index=frame.index)


def revision_band(logit, sd, level: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    """Band on the probability scale from a revision SD measured on the logit scale."""
    z = float(norm.ppf(0.5 + level / 2))
    logit = np.asarray(logit, dtype=float)
    return sigmoid(logit - z * sd), sigmoid(logit + z * sd)


def replacement_qb(ratings: pd.DataFrame, before_season: int, first_season: int = 2010) -> float:
    """Mean rating of quarterbacks making their first starts: the what-if when a starter is out.

    Uses only seasons before `before_season`, so a historical what-if never sees its own future.
    """
    r = ratings.loc[ratings["season"].between(first_season, before_season - 1) & ratings["pregame_qb_id"].notna()]
    r = r.loc[r["pregame_qb_starts"].le(REPLACEMENT_MAX_PRIOR_STARTS)]
    return float(r["pregame_qb"].mean())


def projected_ratings(inp, season: int, week: int) -> pd.DataFrame:
    """Rating rows for every regular-season game from `week` on (and any earlier game not yet played), built from games BEFORE `week` only.

    Future games carry each team's latest rating and its last starting quarterback (the filter's own projection).
    The result is leak-free for a completed season too, because the games the filter learns from are chosen by
    week, never by is_played.
    """
    from src.features.ratings import build_team_ratings

    games = inp.games
    before = games.loc[games["season"].lt(season) | (games["season"].eq(season) & games["week"].lt(week))]
    before = before.loc[before["is_played"]]
    reg = games.loc[games["season"].eq(season) & games["game_type"].eq("REG")]
    future = reg.loc[reg["week"].ge(week) | ~reg["is_played"]]
    played_ids = set(before["game_id"])
    ratings = build_team_ratings(
        inp.team_game.loc[inp.team_game["game_id"].isin(played_ids)],
        pd.concat([before, future], ignore_index=True),
        inp.params,
        inp.qb_game.loc[inp.qb_game["game_id"].isin(played_ids)],
    )
    return ratings.loc[ratings["game_id"].isin(set(future["game_id"]))].reset_index(drop=True)

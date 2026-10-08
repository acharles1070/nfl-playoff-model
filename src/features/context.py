"""Schedule-derived game context known before kickoff.

rest_diff   home rest days minus away rest days, clipped to +-7
home_bye    home team coming off a bye / long rest (>= 13 days)
away_bye    likewise for the visitor
short_week  either team on <= 5 days rest
neutral     neutral-site game (Super Bowl, international)
is_div_game  divisional matchup
dome        roof is closed / dome (weather-proof)

Everything here is fixed by the schedule, so none of it can leak results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CONTEXT_COLUMNS = [
    "rest_diff",
    "home_bye",
    "away_bye",
    "short_week",
    "neutral",
    "is_div_game",
    "dome",
]


def build_context(games: pd.DataFrame) -> pd.DataFrame:
    g = games
    home_rest = pd.to_numeric(g["home_rest"], errors="coerce")
    away_rest = pd.to_numeric(g["away_rest"], errors="coerce")

    out = pd.DataFrame({"game_id": g["game_id"].to_numpy()})

    out["rest_diff"] = (home_rest - away_rest).clip(-7, 7).fillna(0.0).to_numpy()
    out["home_bye"] = (home_rest >= 13).astype(float).to_numpy()
    out["away_bye"] = (away_rest >= 13).astype(float).to_numpy()
    out["short_week"] = ((home_rest <= 5) | (away_rest <= 5)).astype(float).to_numpy()
    out["neutral"] = g["location"].astype(str).str.lower().eq("neutral").astype(float).to_numpy()
    out["is_div_game"] = pd.to_numeric(g["div_game"], errors="coerce").fillna(0).astype(float).to_numpy()
    out["dome"] = g["roof"].astype(str).str.lower().isin(["dome", "closed"]).astype(float).to_numpy()

    return out


def attach_context(matchups: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    ctx = build_context(games)
    out = matchups.merge(ctx, on="game_id", how="left", validate="one_to_one")

    if len(out) != len(matchups):
        raise ValueError("Row count changed while attaching context.")

    return out

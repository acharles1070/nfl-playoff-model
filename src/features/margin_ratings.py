"""Point-margin Kalman ratings (one latent strength per team, in points).

Complements the EPA/play filter in ratings.py. EPA is measured on run/pass
plays only, so it misses kicking, return units and defensive/special-teams
scores; the final margin contains them (at the price of more noise).

    margin = hfa + R_home - R_away + noise        (hfa = 0 at neutral sites)

Same leakage contract as ratings.py: a game's rating uses only earlier
(season, week) groups; teams are updated a whole week at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarginParams:
    hfa_points: float = 2.0      # fixed home edge removed before updating
    obs_var: float = 182.0       # NFL game margin variance (~13.5 pts sd)
    carryover: float = 0.60      # share of rating kept across seasons
    season_var: float = 6.0      # uncertainty injected each new season
    drift_var: float = 0.4       # uncertainty added per game played
    prior_var: float = 25.0      # uncertainty of a never-seen team


def build_margin_ratings(
    games: pd.DataFrame,
    params: MarginParams | None = None,
) -> pd.DataFrame:
    """One PREGAME row per team per game: pregame_pts, pregame_pts_var."""

    params = params or MarginParams()

    schedule = games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    rating: dict[str, float] = {}
    var: dict[str, float] = {}
    last_season: dict[str, int] = {}

    def touch(team: str, season: int) -> None:
        if team not in rating:
            rating[team], var[team], last_season[team] = 0.0, params.prior_var, season
        elif last_season[team] != season:
            rating[team] *= params.carryover
            var[team] = params.carryover ** 2 * var[team] + params.season_var
            last_season[team] = season

    rows = []

    for (season, week), wk in schedule.groupby(["season", "week"], sort=True):
        season = int(season)

        for g in wk.itertuples(index=False):
            touch(g.home_team, season)
            touch(g.away_team, season)

            for team, opp in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
                rows.append(
                    {
                        "game_id": g.game_id,
                        "team": team,
                        "pregame_pts": rating[team],
                        "pregame_pts_var": var[team],
                    }
                )

        pending = []

        for g in wk.itertuples(index=False):
            if pd.isna(g.home_score) or pd.isna(g.away_score):
                continue

            neutral = getattr(g, "location", "Home") == "Neutral"
            hfa = 0.0 if neutral else params.hfa_points

            margin = float(g.home_score - g.away_score)
            innovation = margin - hfa - (rating[g.home_team] - rating[g.away_team])
            total = var[g.home_team] + var[g.away_team] + params.obs_var

            pending.append(
                (g.home_team, g.away_team, innovation,
                 var[g.home_team] / total, var[g.away_team] / total)
            )

        for home, away, z, k_home, k_away in pending:
            rating[home] += k_home * z
            rating[away] -= k_away * z
            var[home] *= 1.0 - k_home
            var[away] *= 1.0 - k_away

        for g in wk.itertuples(index=False):
            for team in (g.home_team, g.away_team):
                var[team] += params.drift_var

    return pd.DataFrame(rows)


def attach_margin_ratings(
    matchups: pd.DataFrame,
    margin_ratings: pd.DataFrame,
) -> pd.DataFrame:
    out = matchups

    for side in ("home", "away"):
        part = margin_ratings.rename(
            columns={
                "team": f"{side}_team",
                "pregame_pts": f"{side}_pregame_pts",
                "pregame_pts_var": f"{side}_pregame_pts_var",
            }
        )
        out = out.merge(part, on=["game_id", f"{side}_team"], how="left", validate="one_to_one")

    out["diff_pts_rating"] = out["home_pregame_pts"] - out["away_pregame_pts"]
    return out

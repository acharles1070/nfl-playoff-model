"""Weekly NFL rosters (nflverse), cached per season.

The `status` column is the key: "INA" is the game-day inactive list (0.07% of INA
players took a snap that week) and "RES" is injured reserve. Coverage here starts
with the snap-count seasons (2013) because importance weights need snap shares.
"""

from __future__ import annotations

import nflreadpy as nfl
import pandas as pd

from src.data.injuries import FIRST_SNAP_SEASON, _cached, load_id_map
from src.data.schedules import configured_seasons, live_season
from src.teams import normalize_team_series


def load_rosters(seasons=None) -> pd.DataFrame:
    seasons = sorted(s for s in (seasons or configured_seasons()) if s >= FIRST_SNAP_SEASON)

    frames = [
        _cached("rosters", s, lambda s=s: nfl.load_rosters_weekly([s]), s == live_season())
        for s in seasons
    ]
    out = pd.concat(frames, ignore_index=True)

    # The roster feed leaves pfr_id blank for many players (most offensive
    # linemen), so fall back to the GSIS -> PFR map from the players table.
    id_map = load_id_map().rename(columns={"pfr_id": "pfr_id_map"})
    out = out.merge(id_map, on="gsis_id", how="left")
    out["pfr_id"] = out["pfr_id"].fillna(out["pfr_id_map"])

    out = out.loc[out["week"].notna() & out["pfr_id"].notna()].copy()
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["team"] = normalize_team_series(out["team"])
    return out[["season", "week", "team", "pfr_id", "position", "status"]]

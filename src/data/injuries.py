"""nflverse injury reports, snap counts, and the id map between them.

Injury reports carry GSIS ids; snap counts carry Pro-Football-Reference ids;
the players table maps one to the other (100% of injury rows map).

Coverage: injury reports 2009+, snap counts 2013+. Completed seasons are cached
under data/raw/injuries; the live season always re-downloads.
"""

from __future__ import annotations

import nflreadpy as nfl
import pandas as pd

from src.config import configured_path
from src.data.schedules import configured_seasons, live_season
from src.teams import normalize_team_series


# nflreadpy documents 2012, but the snap-count feed has no 2012 rows.
FIRST_INJURY_SEASON = 2009   # nflverse injury reports start here
FIRST_SNAP_SEASON = 2013


def _cached(name: str, season: int | None, loader, refresh: bool) -> pd.DataFrame:
    tag = f"{name}_{season}" if season is not None else name
    path = configured_path("raw") / "injuries" / f"{tag}.parquet"

    if path.exists() and not refresh:
        return pd.read_parquet(path)

    print(f"  downloading {tag} ...")
    frame = loader()
    frame = frame.to_pandas() if hasattr(frame, "to_pandas") else frame

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def load_injury_reports(seasons=None) -> pd.DataFrame:
    seasons = sorted(s for s in (seasons or configured_seasons()) if s >= FIRST_INJURY_SEASON)

    frames = [
        _cached("injuries", s, lambda s=s: nfl.load_injuries([s]), s == live_season())
        for s in seasons
    ]
    out = pd.concat(frames, ignore_index=True)

    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["team"] = normalize_team_series(out["team"])
    out["report_status"] = out["report_status"].astype("string").str.strip()
    return out


def load_snaps(seasons=None) -> pd.DataFrame:
    seasons = sorted(s for s in (seasons or configured_seasons()) if s >= FIRST_SNAP_SEASON)

    frames = [
        _cached("snaps", s, lambda s=s: nfl.load_snap_counts([s]), s == live_season())
        for s in seasons
    ]
    out = pd.concat(frames, ignore_index=True)

    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["team"] = normalize_team_series(out["team"])
    return out


def load_id_map() -> pd.DataFrame:
    players = _cached("players", None, nfl.load_players, False)
    return players[["gsis_id", "pfr_id"]].dropna().drop_duplicates()

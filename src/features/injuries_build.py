"""Build and save team-week injury burden (python -m src.features.injuries_build)."""

from __future__ import annotations

import pandas as pd

from src.config import configured_path
from src.data.injuries import FIRST_SNAP_SEASON, load_id_map, load_injury_reports, load_snaps
from src.features.injuries import measure_availability, team_week_burden


def main() -> None:
    reports, snaps, id_map = load_injury_reports(), load_snaps(), load_id_map()
    reports = reports.loc[reports["season"] >= FIRST_SNAP_SEASON]

    seasons = sorted(reports["season"].unique())
    p_out = {s: measure_availability(reports, snaps, id_map, s) for s in seasons}

    print("P(out | Questionable), overall, measured on the preceding 4 seasons:")
    print({s: round(p_out[s]["_overall"], 3) for s in seasons})

    burden = team_week_burden(reports, snaps, id_map, p_out)
    path = configured_path("processed") / "team_week_injury_burden.parquet"
    burden.to_parquet(path, index=False)
    print(f"Saved: {path}  ({len(burden)} team-weeks)")

    print("\nMean lost starter-equivalents per team-week by group:")
    print(burden.filter(like="lost_").mean().round(3).to_string())


if __name__ == "__main__":
    main()

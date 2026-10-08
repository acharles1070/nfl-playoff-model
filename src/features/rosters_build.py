"""Build and save roster-status burden (python -m src.features.rosters_build)."""

from __future__ import annotations

from src.config import configured_path
from src.data.injuries import load_snaps
from src.data.rosters import load_rosters
from src.features.rosters import roster_burden


def main() -> None:
    rosters, snaps = load_rosters(), load_snaps()
    burden = roster_burden(rosters, snaps)

    path = configured_path("processed") / "team_week_roster_burden.parquet"
    burden.to_parquet(path, index=False)
    print(f"Saved: {path} ({len(burden)} team-weeks, seasons {burden['season'].min()}-{burden['season'].max()})")
    print("\nMean lost starter-equivalents per team-week:")
    print(burden.filter(like="lost_").mean().round(3).to_string())


if __name__ == "__main__":
    main()

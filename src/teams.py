"""NFL team-name normalization.

All external data sources must pass through this module before joining
on team identifiers. Internal representation uses current NFL abbreviations.
"""

from __future__ import annotations

import pandas as pd


TEAM_ALIASES: dict[str, str] = {
    # Rams
    "LA": "LAR",
    "STL": "LAR",

    # Chargers
    "SD": "LAC",

    # Raiders
    "OAK": "LV",

    # Washington
    "WSH": "WAS",
}


VALID_TEAMS: frozenset[str] = frozenset({
    "ARI", "ATL", "BAL", "BUF",
    "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA",
    "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA",
    "SF", "TB", "TEN", "WAS",
})


def normalize_team(team: str | None) -> str | None:
    """Normalize one NFL team abbreviation."""
    if team is None:
        return None

    team = str(team).strip().upper()
    return TEAM_ALIASES.get(team, team)


def normalize_team_series(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series of NFL team abbreviations."""
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace(TEAM_ALIASES)
    )


def validate_teams(series: pd.Series, *, source: str = "unknown") -> None:
    """Raise an error when unexpected team abbreviations are present."""
    normalized = normalize_team_series(series)

    unknown = sorted(
        set(normalized.dropna().unique()) - VALID_TEAMS
    )

    if unknown:
        raise ValueError(
            f"Unknown NFL team abbreviations in {source}: {unknown}"
        )


# Display names (current franchise codes; historical codes map through TEAM_ALIASES).
TEAM_NAMES: dict[str, str] = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


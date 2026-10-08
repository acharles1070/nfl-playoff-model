"""NFL divisions (stable since the 2002 realignment, current franchise codes).

Relocations map through src.teams (STL->LAR, SD->LAC, OAK->LV), so the same
table covers 2002 onward.
"""

from __future__ import annotations


DIVISIONS: dict[str, dict[str, list[str]]] = {
    "AFC": {
        "East": ["BUF", "MIA", "NE", "NYJ"],
        "North": ["BAL", "CIN", "CLE", "PIT"],
        "South": ["HOU", "IND", "JAX", "TEN"],
        "West": ["DEN", "KC", "LV", "LAC"],
    },
    "NFC": {
        "East": ["DAL", "NYG", "PHI", "WAS"],
        "North": ["CHI", "DET", "GB", "MIN"],
        "South": ["ATL", "CAR", "NO", "TB"],
        "West": ["ARI", "LAR", "SEA", "SF"],
    },
}

CONFERENCE_OF = {
    team: conf
    for conf, divs in DIVISIONS.items()
    for teams in divs.values()
    for team in teams
}
